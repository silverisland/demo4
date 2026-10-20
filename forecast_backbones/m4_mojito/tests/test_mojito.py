"""Numerical, temporal and framework contracts for the actual Torch model."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import load_model
from torch.utils.data import DataLoader

from forecast_backbones.m4_mojito.augmentation import corrupt_tail
from forecast_backbones.m4_mojito.config import TAIL_METHODS, load_config
from forecast_backbones.m4_mojito.features import (
    calendar_features,
    prediction_loss,
    prepare_features,
)
from forecast_backbones.m4_mojito.train import COLLATE, build_dataset, train
from forecast_backbones.m4_mojito.transformer_models import (
    TransformersAdaptedMojitoModel,
)
from forecast_backbones.m4_mojito.transformer_trainer import evaluate_model
from pipelines.model_registry import build_evaluator, build_model
from pipelines.pipeline_wrapper_inference import unified_inference

torch.set_num_threads(1)


@pytest.fixture
def cfg():
    return load_config(
        {
            "inference": {"input_len": 192},
            "model": {
                "d_model": 16,
                "n_heads": 2,
                "e_layers": 1,
                "d_layers": 1,
                "d_ff": 32,
                "dropout": 0.0,
            },
            "augmentation": {"probability": 0.0},
            "optimization": {
                "batch_size": 2,
                "eval_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "train_epochs": 1,
                "use_cpu": True,
            },
        }
    )


def make_batch(batch_size=3, length=192, future=16):
    generator = torch.Generator().manual_seed(17)
    return {
        "observe_power": torch.rand(batch_size, length, generator=generator) * 50,
        "general_ghi": torch.rand(batch_size, length, generator=generator) * 1000,
        "general_temp": torch.rand(batch_size, length, generator=generator) * 30,
        "general_ghi_predict": torch.rand(batch_size, future, generator=generator)
        * 1000,
        "general_temp_predict": torch.rand(batch_size, future, generator=generator)
        * 30,
        "cap_power_on": torch.linspace(80, 120, batch_size),
        "timestamp_win": torch.full((batch_size,), 1748779200, dtype=torch.int64),
    }


def make_frame(n=4):
    batch = make_batch(n)
    data = {key: list(value.numpy()) for key, value in batch.items()}
    data["station"] = [f"station-{i}" for i in range(n)]
    data["observe_power_future"] = list(batch["general_ghi_predict"].numpy() / 20)
    return pd.DataFrame(data)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_forward_backward_and_label_independence(cfg, batch_size):
    model = build_model("m4_mojito", cfg).eval()
    batch = make_batch(batch_size)
    labels = {"observe_power_future": torch.full((batch_size, 16), 30.0)}
    output = model(batch, labels)
    assert output.logits.shape == (batch_size, 16)
    torch.testing.assert_close(output.logits, model(batch).logits)
    batch["observe_power_future"] = torch.full((batch_size, 16), 100000.0)
    batch["station"] = ["never-seen"] * batch_size
    torch.testing.assert_close(output.logits, model(batch).logits)
    output.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert build_evaluator("m4_mojito") is evaluate_model


def test_default_architecture_and_missing_history():
    model = TransformersAdaptedMojitoModel().eval()
    batch = make_batch(1, 672)
    for key in (
        "observe_power",
        "general_ghi",
        "general_temp",
        "general_ghi_predict",
        "general_temp_predict",
    ):
        batch[key].fill_(float("nan"))
    features = prepare_features(batch, model.config, "cpu")
    tokens, padding = model.model.tokenizer(features)
    assert tokens.shape == (1, 94, 128)
    assert padding[:, :-1].all() and not padding[:, -1].any()
    output = model(batch, {"observe_power_future": torch.ones(1, 16)})
    assert torch.isfinite(output.logits).all()
    assert output.aux["history_unavailable"].all()
    assert output.aux["future_weather_unavailable"].all()
    output.loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("use_recent", [False, True])
def test_small_variant_without_slow_history(cfg, use_recent):
    cfg["inference"]["input_len"] = 96
    cfg["model"].update(use_slow=False, use_recent=use_recent)
    model = TransformersAdaptedMojitoModel(cfg).eval()
    batch = make_batch(length=96)
    features = prepare_features(batch, cfg, "cpu")
    assert model.model.tokenizer(features)[0].shape == (3, 73, 16)
    output = model(batch, {"observe_power_future": torch.ones(3, 16)})
    output.loss.backward()
    assert output.logits.shape == (3, 16)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_future_alignment_and_scale_equivariance(cfg):
    model = TransformersAdaptedMojitoModel(cfg).eval()
    batch = make_batch(future=192)
    batch["general_ghi_predict"].requires_grad_(True)
    prediction = model(batch).logits
    prediction.sum().backward()
    gradient = batch["general_ghi_predict"].grad
    assert (gradient[:, :16].abs() > 0).all()
    assert (gradient[:, 16:] == 0).all()
    scaled = {key: value.detach().clone() for key, value in batch.items()}
    scaled["observe_power"] *= 3
    scaled["cap_power_on"] *= 3
    torch.testing.assert_close(
        model(scaled).logits, prediction * 3, rtol=1e-5, atol=1e-4
    )


@pytest.mark.parametrize("method", TAIL_METHODS)
def test_corruption_uses_only_retained_prefix(method):
    power = torch.arange(192, dtype=torch.float32)[None, :].repeat(2, 1)
    power[
        1, -8:
    ] = -10000  # Two different true tails must produce identical corrupted inputs.
    output = corrupt_tail(
        power, torch.tensor([8, 8]), torch.tensor([TAIL_METHODS.index(method)] * 2)
    )
    torch.testing.assert_close(output[0], output[1], equal_nan=True)
    torch.testing.assert_close(output[:, :-8], power[:, :-8])
    if method == "hold":
        assert (output[:, -8:] == 183).all()
    if method == "previous_day":
        torch.testing.assert_close(
            output[0, -8:], torch.arange(88, 96, dtype=torch.float32)
        )


def test_corruption_reaches_every_history_path(cfg):
    cfg["augmentation"] = {"probability": 1.0, "lengths": [8], "methods": ["hold"]}
    batch = make_batch()
    original = batch["observe_power"].clone()
    augmented = prepare_features(batch, cfg, "cpu", training=True)
    manual = deepcopy(batch)
    manual["observe_power"][:, -8:] = original[:, -9:-8]
    expected = prepare_features(manual, cfg, "cpu")
    for key in ("history", "history_mask", "state", "recent"):
        torch.testing.assert_close(getattr(augmented, key), getattr(expected, key))
    assert augmented.history_mask[:, -8:, 0].all()  # No synthetic quality mask.
    torch.testing.assert_close(batch["observe_power"], original)
    model = TransformersAdaptedMojitoModel(cfg)
    actual_tokens = model.model.tokenizer(augmented)[0]
    torch.testing.assert_close(actual_tokens, model.model.tokenizer(expected)[0])


def test_short_history_is_left_padded_and_keeps_last_position(cfg):
    batch = make_batch(length=20)
    batch["observe_power"][:, -2:] = float("nan")
    features = prepare_features(batch, cfg, "cpu")
    assert not features.history_mask[:, :172].any()
    assert not features.history_mask[:, -2:, 0].any()
    assert (features.state[:, 1] == 2 / 192).all()
    assert torch.isfinite(features.state).all()


def test_calendar_timezone_leap_day_and_lead():
    issue = int(np.datetime64("2024-02-29T23:45:00", "s").astype(np.int64))
    past, future = calendar_features(
        torch.tensor([issue]), 192, 16, "local_wall", "cpu"
    )
    torch.testing.assert_close(future[0, 0, :2], torch.tensor([0.0, 1.0]))
    assert past[0, -1, 0] < 0
    utc_past, utc_future = calendar_features(
        torch.tensor([issue - 8 * 3600]), 192, 16, "utc", "cpu"
    )
    torch.testing.assert_close(past, utc_past)
    torch.testing.assert_close(future, utc_future)
    with pytest.raises(ValueError, match="epoch seconds"):
        calendar_features(torch.tensor([issue * 1000]), 192, 16, "utc", "cpu")


def test_masked_loss_and_capacity_validation(cfg):
    prediction = torch.zeros(3, 16, requires_grad=True)
    target = torch.full((3, 16), float("nan"))
    target[0, 0] = 10  # Error 0.1 / 0.2 = 0.5.
    target[1] = 50  # Error 0.5 / 0.5 = 1.0; rows get equal weight.
    loss = prediction_loss(prediction, target, torch.full((3, 1), 100.0))
    torch.testing.assert_close(loss, torch.tensor(0.75))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    with pytest.raises(ValueError, match="no finite labels"):
        prediction_loss(prediction, target * float("nan"), torch.ones(3, 1))
    batch = make_batch()
    batch["cap_power_on"][0] = 0
    with pytest.raises(ValueError, match="Capacity"):
        prepare_features(batch, cfg, "cpu")


def test_framework_dataset_evaluation_and_inference(cfg):
    frame = make_frame()
    model = TransformersAdaptedMojitoModel(cfg).eval()
    dataset = build_dataset(frame, cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=COLLATE)))
    assert "observe_power_future" not in batch["batch_dict"]
    result = evaluate_model(model, dataset, COLLATE, cfg)
    assert result["station"].tolist() == frame["station"].tolist()
    assert not model.training
    unlabeled = build_dataset(
        frame.drop(columns="observe_power_future"), cfg, supervised=False
    )
    without_labels = evaluate_model(model, unlabeled, COLLATE, cfg)
    assert np.isnan(np.stack(without_labels["observe_power_future"])).all()
    from forecast_backbones.m4_mojito.config import feature_columns

    inferred = unified_inference(
        model, frame, feature_cols=feature_columns(cfg), batch_size=2
    )
    assert len(inferred) == len(frame)
    np.testing.assert_allclose(
        np.stack(inferred[type(model).__name__]),
        np.stack(result["observe_power_predict"]),
        atol=1e-5,
    )


def test_training_checkpoint_roundtrip(cfg, tmp_path):
    frame = make_frame()
    frame.loc[0, "observe_power_future"][:] = np.nan
    assert len(build_dataset(frame, cfg)) == 3
    train_path, valid_path = tmp_path / "train.parquet", tmp_path / "valid.parquet"
    frame.to_parquet(train_path)
    validation = make_frame()
    validation["timestamp_win"] += 14 * 86400
    validation.to_parquet(valid_path)
    trainer = train(
        cfg, str(train_path), str(valid_path), tmp_path / "run", max_steps=2
    )
    assert trainer.state.global_step == 2
    assert trainer.state.best_model_checkpoint is not None
    restored = TransformersAdaptedMojitoModel(
        tmp_path / "run" / "model_config.yaml"
    ).eval()
    load_model(restored, str(tmp_path / "run" / "best" / "model.safetensors"))
    trainer.model.eval()
    torch.testing.assert_close(
        restored(make_batch()).logits, trainer.model(make_batch()).logits
    )


@pytest.mark.parametrize(
    "override",
    [
        {"model": {"fast_patch": 5}},
        {"model": {"n_heads": 3}},
        {"dataset": {"pv_col": "observe_power_future"}},
        {"dataset": {"weather_scales": [0, 15]}},
        {"inference": {"pred_len": 192}},
        {"model": {"typo": True}},
    ],
)
def test_invalid_configuration_fails_early(override):
    with pytest.raises(ValueError):
        load_config(override)
