"""Numerical decisions, true multi-quantile gradients, and full framework integration."""

import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from safetensors.torch import load_model

from forecast_backbones.m4_pinball import decision as d
from forecast_backbones.m4_pinball.config import feature_columns, load_config
from forecast_backbones.m4_pinball.losses import pinball_loss
from forecast_backbones.m4_pinball.train import COLLATE, build_dataset, train
from forecast_backbones.m4_pinball.transformer_models import (
    TransformersAdaptedM4PinballModel,
)
from forecast_backbones.m4_pinball.transformer_trainer import evaluate_model
from pipelines.model_registry import build_evaluator, build_model
from pipelines.pipeline_wrapper_inference import unified_inference

torch.set_num_threads(1)


@pytest.fixture
def cfg():
    return load_config(
        {
            "inference": {"input_len": 32, "pred_len": 16},
            "model": {
                "patch_len": 8,
                "stride": 8,
                "d_model": 16,
                "d_ff": 32,
                "n_heads": 2,
                "e_layers": 1,
                "embedding_dim": 16,
                "dropout": 0.0,
                "fc_dropout": 0.0,
                "head_dropout": 0.0,
            },
            "optimization": {
                "use_cpu": True,
                "fp16": False,
                "batch_size": 2,
                "eval_batch_size": 2,
                "train_epochs": 1,
            },
        }
    )


def make_batch(B=3, history=32, future=16):
    rng = torch.Generator().manual_seed(123)
    return {
        "general_ghi": torch.rand(B, history, generator=rng) * 1000,
        "general_temp": torch.rand(B, history, generator=rng) * 30,
        "general_ghi_predict": torch.rand(B, future, generator=rng) * 1000,
        "general_temp_predict": torch.rand(B, future, generator=rng) * 30,
        "observe_power": torch.rand(B, history, generator=rng) * 30,
        "chronos": torch.rand(B, future, generator=rng) * 30,
        "cap_mw": torch.linspace(50.0, 100.0, B),
        "station": [f"s{i}" for i in range(B)],
        "timestamp_win": torch.arange(B) * 900 + 1748779200,
    }


def make_frame(B=4, history=32, future=16):
    batch = make_batch(B, history=history, future=future)
    data = {
        k: list(v.numpy()) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    data["observe_power_future"] = list(batch["general_ghi_predict"].numpy() / 25.0)
    data["timestamp_win"] = (
        pd.Timestamp("2025-06-01 09:00").value // 10**9 + np.arange(B) * 900
    )
    return pd.DataFrame(data)


def test_pinball_separate_channels_missing_labels_and_capacity():
    # One residual with tau=.1 versus .9 must have different channel gradients.
    q = torch.tensor([[[0.0, 0.0], [99.0, 99.0]]], requires_grad=True)
    y = torch.tensor([[10.0, float("nan")]])
    loss = pinball_loss(q, y, [0.1, 0.9], torch.tensor([10.0]))
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    torch.testing.assert_close(q.grad[0, 0], torch.tensor([-0.005, -0.045]))
    assert (q.grad[0, 1] == 0).all()
    with pytest.raises(ValueError, match="no finite"):
        pinball_loss(
            q, torch.full_like(y, float("nan")), [0.1, 0.9], torch.tensor([10.0])
        )


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("revin,affine", [(False, False), (True, False), (True, True)])
def test_forward_backward_and_label_independence(cfg, B, revin, affine):
    cfg["model"].update(revin=revin, affine=affine)
    model = build_model("m4_pinball", cfg).eval()
    batch = make_batch(B)
    labels = {"observe_power_future": torch.full((B, 16), 20.0)}
    labels["observe_power_future"][0, 0] = float("nan")
    output = model(batch, labels)
    assert output.logits.shape == (B, 16)
    assert output.aux["quantiles"].shape == (B, 16, 9)
    assert model.model.head.pred.out_features == 16 * 9
    assert torch.isfinite(output.logits).all()
    assert (output.logits >= 0).all() and (
        output.logits <= batch["cap_mw"][:, None]
    ).all()
    torch.testing.assert_close(output.logits, model(batch).logits)
    output.loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    batch["observe_power_future"] = torch.full((B, 16), 1e6)
    torch.testing.assert_close(output.logits, model(batch).logits)
    assert build_evaluator("m4_pinball") is evaluate_model


def test_encoder_batchnorm_matches_m4(cfg):
    model = TransformersAdaptedM4PinballModel(cfg)
    encoder = model.model.backbone.encoder
    norms = [m for m in encoder.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert len(norms) == 2 * cfg["model"]["e_layers"]
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in encoder.modules())
    assert all(m.num_features == cfg["model"]["d_model"] for m in norms)

    # Training must update running statistics; evaluation must keep them fixed.
    model.train()
    loss = model(make_batch(), {"observe_power_future": torch.full((3, 16), 20.0)}).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(m.num_batches_tracked.item() == 1 for m in norms)
    assert all(
        m.weight.grad is not None and torch.isfinite(m.weight.grad).all() for m in norms
    )
    stats = [(m.running_mean.clone(), m.running_var.clone()) for m in norms]
    model.eval()
    with torch.no_grad():
        model(make_batch(B=1))
    for m, (mean, var) in zip(norms, stats):
        torch.testing.assert_close(m.running_mean, mean, rtol=0, atol=0)
        torch.testing.assert_close(m.running_var, var, rtol=0, atol=0)
        assert m.num_batches_tracked.item() == 1


def test_train_skips_numpy_decision(cfg, monkeypatch):
    model = TransformersAdaptedM4PinballModel(cfg).train()
    monkeypatch.setattr(
        "forecast_backbones.m4_pinball.transformer_models.decide",
        lambda *a, **kw: pytest.fail("training should not run NumPy decisions"),
    )
    out = model(make_batch(), {"observe_power_future": torch.full((3, 16), 20.0)})
    assert out.logits is None and torch.isfinite(out.loss)
    out.loss.backward()


def test_default_192_step_model_and_decisions():
    model = TransformersAdaptedM4PinballModel().eval()
    batch = make_batch(1, history=672, future=192)
    output = model(batch)
    assert output.logits.shape == (1, 192)
    cal = output.aux["quantiles_cal"].numpy()
    point = output.logits.numpy()
    np.testing.assert_allclose(
        point[:, :16], d.p_star_l1(cal[:, :16], model.levels, 50.0), rtol=1e-6
    )
    np.testing.assert_allclose(
        point[:, 16:], d.p_star(cal[:, 16:], model.levels, 50.0), rtol=1e-6
    )


def test_shared_revin_inverse(cfg):
    model = TransformersAdaptedM4PinballModel(cfg).eval()
    # Fix output in normalized coordinates; each quantile must share the same mean/std.
    with torch.no_grad():
        model.model.head.pred.weight.zero_()
        model.model.head.pred.bias.copy_(torch.arange(9).repeat(16))
    batch = make_batch()
    raw = model(batch).aux["quantiles"]
    pv = torch.cat([batch["observe_power"], batch["chronos"]], dim=-1)
    expected = pv.mean(-1)[:, None, None] + torch.sqrt(
        pv.var(-1, unbiased=False) + 1e-5
    )[:, None, None] * torch.arange(9)
    torch.testing.assert_close(raw, expected.expand_as(raw))


def test_capacity_and_input_failures(cfg):
    model = TransformersAdaptedM4PinballModel(cfg).eval()
    batch = make_batch()
    del batch["cap_mw"]
    with pytest.raises(ValueError, match="Missing fixed capacity"):
        model(batch)
    batch["cap_mw"] = torch.tensor([0.0, 10.0, 20.0])
    with pytest.raises(ValueError, match="finite and positive"):
        model(batch)
    batch = make_batch()
    batch["chronos"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite forecast hint"):
        model(batch)


def test_station_capacity_mapping(cfg):
    cfg["dataset"]["capacity_by_station"] = {"s0": 80.0, "s1": 90.0, "s2": 100.0}
    model = TransformersAdaptedM4PinballModel(cfg).eval()
    batch = make_batch()
    del batch["cap_mw"]
    torch.testing.assert_close(
        model(batch).aux["cap"], torch.tensor([80.0, 90.0, 100.0])
    )
    batch["station"][0] = "unknown"
    with pytest.raises(ValueError, match="No fixed capacity"):
        model(batch)


def test_decision_matches_m8_for_valid_inputs():
    from forecast_backbones.m4_pinball.tests import m8_decision_reference as original

    rng = np.random.default_rng(0)
    q = np.sort(rng.uniform(0, 100, (3, 192, 9)), axis=-1)
    cap = np.array([100.0, 120.0, 130.0])
    for method in ("p_star", "p_star_l1", "median", "quantile:0.35"):
        np.testing.assert_allclose(
            d.select_point(q, d.MODEL_LEVELS, cap, method),
            original.select_point(q, d.MODEL_LEVELS, cap, method),
        )


def test_weighted_optima_against_direct_objective():
    q = np.array([[[5.0, 40.0, 90.0]]])
    lv, cap = [0.1, 0.5, 0.9], 100.0
    y = d.quantile_grid(q, lv)[0, 0]
    denom = np.maximum(y, 20.0)
    p = float(d.p_star(q, lv, cap)[0, 0])
    objective = lambda x: np.mean(((x - y) / denom) ** 2)
    assert objective(p) < objective(p - 0.1) and objective(p) < objective(p + 0.1)
    m = float(d.p_star_l1(q, lv, cap)[0, 0])
    objectives = np.mean(np.abs(y[:, None] - y[None, :]) / denom[None, :], axis=1)
    assert np.mean(np.abs(m - y) / denom) == pytest.approx(objectives.min())


@pytest.mark.parametrize("steps", [[-1, 3], [0.1, 3], [3, 2]])
def test_bad_step_rules_rejected(steps):
    with pytest.raises(ValueError, match="steps"):
        d.select_point(
            np.ones((1, 4, 9)),
            d.MODEL_LEVELS,
            [100.0],
            "median",
            [{"steps": steps, "method": "p_star"}],
        )


def test_rule_priority_and_default():
    q = np.tile([10.0, 20.0, 30.0], (1, 20, 1))
    point = d.select_point(
        q,
        [0.1, 0.5, 0.9],
        [100.0],
        "median",
        [
            {"steps": [0, 4], "method": "quantile:0.1"},
            {"steps": [2, 8], "method": "quantile:0.9"},
            {"steps": [30, 40], "method": "p_star"},
        ],
    )
    np.testing.assert_array_equal(point[0], [10.0] * 4 + [30.0] * 4 + [20.0] * 12)


def test_invalid_quantiles_and_parameters_fail_loudly():
    q = np.array([[[10.0, 20.0, np.nan]]])
    with pytest.raises(ValueError, match="finite"):
        d.p_star_l1(q, [0.1, 0.5, 0.9], 100.0)
    q = np.nan_to_num(q)
    with pytest.raises(ValueError, match="nondecreasing"):
        d.p_star_l1(q, [0.1, 0.5, 0.9], 100.0)
    with pytest.raises(ValueError, match="positive"):
        d.p_star(q, [0.1, 0.5, 0.9], 0.0)
    with pytest.raises(ValueError, match="tau"):
        d.level_value(q, [0.1, 0.5, 0.9], 1.5)
    with pytest.raises(ValueError, match="strictly increasing"):
        d.quantile_grid(q, [0.5, 0.1, 0.9])
    with pytest.raises(ValueError, match="step"):
        d.quantile_grid(q, [0.1, 0.5, 0.9], step=0)


def test_clipping_and_zero_shrink(cfg):
    dec = deepcopy(cfg["decision"])
    dec["point"].update(default="global_shrink", by_step=[], shrink_factor=1.5)
    q = np.full((1, 16, 9), 90.0)
    point, _ = d.decide(q, d.MODEL_LEVELS, [100.0], dec)
    assert (point == 100.0).all()
    dec["point"]["shrink_factor"] = 0.0
    assert (d.decide(q, d.MODEL_LEVELS, [100.0], dec)[0] == 0).all()
    dec["point"].update(default="median")
    # Night is NOT inferred from a zero clear-sky curve when night_zero=false.
    dec["physics"]["clear_sky_cap"] = True
    pc = np.array([[0.0, np.nan] + [20.0] * 14])
    point, _ = d.decide(q, d.MODEL_LEVELS, [100.0], dec, p_clear=pc)
    np.testing.assert_allclose(point[0], [90.0, 90.0] + [21.0] * 14)
    dec["physics"]["night_zero"] = True
    night = np.zeros((1, 16), bool)
    night[0, 0] = True
    assert (
        d.decide(q, d.MODEL_LEVELS, [100.0], dec, p_clear=pc, night=night)[0][0, 0] == 0
    )


def test_optional_physics_columns(cfg):
    cfg["decision"]["physics"].update(clear_sky_cap=True, night_zero=True)
    model = TransformersAdaptedM4PinballModel(cfg).eval()
    batch = make_batch()
    with pytest.raises(ValueError, match="p_clear"):
        model(batch)
    batch["p_clear"] = torch.full((3, 16), 20.0)
    batch["is_night"] = torch.zeros(3, 16)
    batch["is_night"][:, :2] = 1
    point = model(batch).logits
    assert (point[:, :2] == 0).all() and (point <= 21.0).all()


@pytest.mark.parametrize("head", ["flatten", "step_cov"])
def test_train_save_reload_and_unified_inference(cfg, tmp_path, head):
    cfg["model"]["prediction_head"] = head
    cfg["inference"]["pred_len"] = 192
    frame = make_frame(future=192)
    tr, va = tmp_path / "train.parquet", tmp_path / "valid.parquet"
    frame.to_parquet(tr, index=False)
    frame.to_parquet(va, index=False)
    out_dir = tmp_path / "run"
    trainer = train(cfg, tr, va, out_dir, max_steps=2, test_source=va)
    assert trainer.state.global_step == 2
    assert trainer.args.metric_for_best_model == "eval_nanwang_short_d1"
    assert trainer.args.greater_is_better
    assert trainer.state.best_metric is not None
    assert "eval_nanwang_short_d1" in trainer.evaluate()
    restored = build_model(
        "m4_pinball", load_config(out_dir / "model_result" / "model_config.yaml")
    )
    load_model(restored, str(out_dir / "model_result" / "model.safetensors"))
    restored.eval()
    trainer.model.eval()
    torch.testing.assert_close(
        restored(make_batch(future=192)).logits,
        trainer.model(make_batch(future=192)).logits,
    )
    ds = build_dataset(frame, cfg)
    result = evaluate_model(restored, ds, COLLATE, cfg)
    online = unified_inference(
        restored,
        frame.drop(columns="observe_power_future"),
        timestamp_cols=["timestamp_win"],
        entity_cols=["station"],
        feature_cols=feature_columns(cfg),
        batch_size=2,
    )
    np.testing.assert_allclose(
        np.stack(online[type(restored).__name__]),
        np.stack(result["observe_power_predict"]),
    )
    assert len(pd.read_parquet(out_dir / "result_val.parquet")) == len(frame)
    assert len(pd.read_parquet(out_dir / "result_test.parquet")) == len(frame)
    assert result["station"].tolist() == frame["station"].tolist()


def test_training_script_dispatch(cfg, tmp_path):
    from train_shell import train_script

    train_script.RUN_PATH = tmp_path
    args, device = train_script.build_train_args("m4_pinball", cfg)
    assert not args.prediction_loss_only and device == "cpu"
    cfg["inference"]["pred_len"] = 192
    frame = make_frame(future=192)
    for split in ("train", "valid", "test"):
        frame.to_parquet(tmp_path / f"{split}.parquet", index=False)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "cli_run"
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "train_shell/train_script.py"),
            "--model_name",
            "m4_pinball",
            "--model_config_path",
            str(config_path),
            "--dataset_path",
            str(tmp_path),
            "--run_path",
            str(out),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (out / "model_result/model.safetensors").is_file()
    assert (out / "result_test.parquet").is_file()
