"""Business-score parity and checkpoint selection when loss disagrees."""

import importlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import load_file
from transformers import EarlyStoppingCallback

from forecast_backbones.m4_pinball.config import load_config
from forecast_backbones.m4_pinball.selection import (
    d1_mask,
    decision_metrics,
    select_validation_rows,
)
from forecast_backbones.m4_pinball.tests.test_pinball import make_frame
from forecast_backbones.m4_pinball.train import COLLATE, build_dataset, train
from forecast_backbones.m4_pinball.transformer_models import (
    TransformersAdaptedM4PinballModel,
)
from forecast_backbones.m4_pinball.transformer_trainer import (
    M4Pinball_build_training_args,
    M4PinballTrainer,
)


def test_issue_selection_and_no_label_based_fallback():
    cfg = load_config({"inference": {"input_len": 32}})
    frame = make_frame(7, future=192)
    frame["station"] = ["a", "a", "a", "a", "b", "b", "c"]
    frame["timestamp_win"] = pd.to_datetime(
        [
            "2025-06-01 09:00",
            "2025-06-01 10:00",
            "2025-06-01 10:00",
            "2025-06-01 10:15",
            "2025-06-01 09:45",
            "2025-06-01 11:00",
            "2025-06-01 11:00",
        ]
    )
    ds = build_dataset(frame, cfg, drop_empty_targets=False)
    chosen = select_validation_rows(ds, cfg)
    assert chosen.column("station").tolist() == ["a", "b"]
    assert pd.DatetimeIndex(chosen.timestamp).hour.tolist() == [10, 9]
    assert pd.DatetimeIndex(chosen.timestamp).minute.tolist() == [0, 45]
    np.testing.assert_equal(
        chosen.column("observe_power_future")[0], frame["observe_power_future"].iloc[1]
    )
    # Neither a duplicate nor an earlier forecast replaces a missing selected day.
    frame.at[1, "observe_power_future"] = np.full(192, np.nan)
    chosen = select_validation_rows(
        build_dataset(frame, cfg, drop_empty_targets=False), cfg
    )
    assert chosen.column("station").tolist() == ["b"]


def test_d1_bounds_weighting_missing_labels_and_zero_floor():
    issue = pd.Timestamp("2025-06-01 10:00")
    mask = d1_mask(issue, 192)
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(55, 151))
    rows = pd.DataFrame(
        {
            "station": ["a", "a", "b"],
            "issue": [issue, issue + pd.Timedelta(days=1), issue],
        }
    )
    y = np.full((3, 192), 50.0)
    p = y.copy()
    p[:, ~mask] = 9999  # Outside D+1 must not influence selection.
    p[0, mask] = 60  # 80 points.
    p[1, mask] = 50  # 100 points, despite having just one valid label.
    y[1, 56:151] = np.nan
    p[2, mask] = 150  # Negative daily score clips to zero.
    scores = decision_metrics(p, y, [100, 100, 200], rows)
    assert scores["nanwang_short_d1"] == pytest.approx(60)
    assert scores["nanwang_short_d1_station_days"] == 3
    assert scores["nanwang_short_d1_points"] == 193
    # Floor is 0.2 * capacity, rather than dividing by capacity or zero power.
    y[:] = 0
    p[:] = 10
    scores = decision_metrics(p, y, [100, 100, 200], rows)
    assert scores["nanwang_short_d1"] == pytest.approx((50 + 50 + 75) / 3)
    with pytest.raises(ValueError, match="fixed capacity"):
        decision_metrics(p, y, [100, 101, 200], rows)


def test_no_d1_data_fails_before_training(tmp_path):
    cfg = load_config(
        {
            "inference": {"input_len": 32, "pred_len": 16},
            "model": {"patch_len": 8, "stride": 8},
            "optimization": {"use_cpu": True, "fp16": False},
        }
    )
    with pytest.raises(ValueError, match="No usable D\\+1"):
        train(cfg, make_frame(), make_frame(), tmp_path, max_steps=1)
    assert not (tmp_path / "model_result").exists()


@pytest.mark.parametrize("every", [0, 2])
def test_best_checkpoint_and_early_stop_use_score_not_loss(tmp_path, every):
    cfg = load_config(
        {
            "inference": {"input_len": 32, "pred_len": 192},
            "model": {
                "patch_len": 16,
                "stride": 16,
                "d_model": 8,
                "d_ff": 16,
                "n_heads": 2,
                "e_layers": 1,
                "embedding_dim": 8,
            },
            "optimization": {
                "use_cpu": True,
                "fp16": False,
                "batch_size": 2,
                "eval_batch_size": 2,
                "train_epochs": 5,
                "patience": 1,
                "eval_every_steps": every,
                "gradient_accumulation_steps": 2,
            },
        }
    )
    # Step validation must fire inside epochs, not just at epoch boundaries.
    frame = make_frame(16 if every else 4, future=192)
    frame["observe_power_future"] = [np.full(192, 50.0)] * len(frame)
    ds = build_dataset(frame, cfg)
    validation = select_validation_rows(ds, cfg)

    class ControlledEvaluation(M4PinballTrainer):
        def prediction_step(
            self, model, inputs, prediction_loss_only, ignore_keys=None
        ):
            loss, (points, cap), labels = super().prediction_step(
                model, inputs, False, ignore_keys
            )
            # Loss keeps improving; business score peaks on validation 2.
            evaluation = self.state.global_step // (every or 1)
            score = {1: 90, 2: 95, 3: 80}.get(evaluation, 100)
            points = labels + 50 * (1 - score / 100)
            loss = torch.tensor(4.0 - evaluation, device=points.device)
            return loss, (points, cap), labels

    trainer = ControlledEvaluation(
        model=TransformersAdaptedM4PinballModel(cfg),
        args=M4Pinball_build_training_args(cfg, tmp_path),
        data_collator=COLLATE,
        train_dataset=ds,
        eval_dataset=validation,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()
    assert trainer.state.global_step == 3 * (every or 1)
    assert trainer.state.best_metric == pytest.approx(95)
    assert trainer.state.best_model_checkpoint.endswith(
        f"checkpoint-{2 * (every or 1)}"
    )
    log = [r for r in trainer.state.log_history if "eval_nanwang_short_d1" in r]
    assert [r["eval_loss"] for r in log] == [3, 2, 1]
    np.testing.assert_allclose([r["eval_nanwang_short_d1"] for r in log], [90, 95, 80])
    saved = load_file(trainer.state.best_model_checkpoint + "/model.safetensors")
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value.cpu(), saved[name], rtol=0, atol=0)


@pytest.mark.parametrize("every,steps,best_step", [(250, 1, 1), (2, 3, 2), (2, 3, 3)])
def test_final_step_is_scored_and_export_uses_best_weights(
    tmp_path, monkeypatch, every, steps, best_step
):
    cfg = load_config(
        {
            "inference": {"input_len": 32, "pred_len": 192},
            "model": {
                "patch_len": 16,
                "stride": 16,
                "d_model": 8,
                "d_ff": 16,
                "n_heads": 2,
                "e_layers": 1,
                "embedding_dim": 8,
            },
            "optimization": {
                "use_cpu": True,
                "fp16": False,
                "batch_size": 2,
                "eval_batch_size": 2,
                "eval_every_steps": every,
            },
        }
    )
    frame = make_frame(8, future=192)
    frame["observe_power_future"] = [np.full(192, 50.0)] * len(frame)

    class ControlledEvaluation(M4PinballTrainer):
        def prediction_step(
            self, model, inputs, prediction_loss_only, ignore_keys=None
        ):
            loss, (_points, cap), labels = super().prediction_step(
                model, inputs, False, ignore_keys
            )
            score = 95 if self.state.global_step == best_step else 80
            return loss, (labels + 50 * (1 - score / 100), cap), labels

    monkeypatch.setattr(
        importlib.import_module("forecast_backbones.m4_pinball.train"),
        "M4PinballTrainer",
        ControlledEvaluation,
    )
    trainer = train(cfg, frame, frame, tmp_path, max_steps=steps)
    assert trainer.state.global_step == steps
    assert trainer.state.best_metric == pytest.approx(95)
    assert trainer.state.best_model_checkpoint.endswith(f"checkpoint-{best_step}")
    log = json.loads((tmp_path / "train_log.json").read_text())
    assert log["best_step"] == best_step and log["stopped_step"] == steps
    assert log["best"] == pytest.approx(95)
    # The final report on restored weights is not another training selection event.
    assert [r["step"] for r in log["history"]] == sorted(
        {*range(every, steps + 1, every), steps}
    )
    saved = load_file(trainer.state.best_model_checkpoint + "/model.safetensors")
    exported = load_file(str(tmp_path / "model_result" / "model.safetensors"))
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value.cpu(), saved[name], rtol=0, atol=0)
        torch.testing.assert_close(exported[name], saved[name], rtol=0, atol=0)


@pytest.mark.parametrize("every", [0, None, 250])
def test_validation_and_save_intervals_match(tmp_path, every):
    args = M4Pinball_build_training_args(
        {"optimization": {"eval_every_steps": every, "use_cpu": True}}, tmp_path
    )
    strategy = getattr(args, "eval_strategy", None) or args.evaluation_strategy
    assert strategy == args.save_strategy == ("steps" if every else "epoch")
    if every:
        assert args.eval_steps == args.save_steps == every
    assert args.load_best_model_at_end and args.greater_is_better
    assert args.metric_for_best_model == "eval_nanwang_short_d1"


@pytest.mark.parametrize(
    "key,value",
    [
        ("eval_every_steps", -1),
        ("eval_every_steps", 0.5),
        ("eval_every_steps", True),
        ("patience", 0),
        ("patience", 1.5),
    ],
)
def test_invalid_selection_options_fail_early(key, value):
    with pytest.raises(ValueError, match=key):
        load_config({"optimization": {key: value}})


def test_trainer_scores_decision_output_and_batch_size_independence(tmp_path):
    cfg = load_config(
        {
            "inference": {"input_len": 32, "pred_len": 192},
            "model": {
                "patch_len": 16,
                "stride": 16,
                "d_model": 8,
                "d_ff": 16,
                "n_heads": 2,
                "e_layers": 1,
                "embedding_dim": 8,
            },
            "optimization": {"use_cpu": True, "fp16": False, "eval_batch_size": 1},
            "decision": {
                "point": {
                    "default": "global_shrink",
                    "by_step": [],
                    "shrink_factor": 0.0,
                }
            },
        }
    )
    frame = make_frame(3, future=192)
    frame["observe_power_future"] = [np.full(192, 50.0)] * 3
    ds = select_validation_rows(build_dataset(frame, cfg), cfg)
    trainer = M4PinballTrainer(
        model=TransformersAdaptedM4PinballModel(cfg),
        args=M4Pinball_build_training_args(cfg, tmp_path),
        data_collator=COLLATE,
        eval_dataset=ds,
    )
    # Zero shrink makes decision points exactly zero regardless of raw quantiles.
    for batch_size in [1, 2, 3]:
        trainer.args.per_device_eval_batch_size = batch_size
        result = trainer.evaluate()
        assert result["eval_nanwang_short_d1"] == pytest.approx(0)
        assert result["eval_nanwang_short_d1_station_days"] == 3
        assert result["eval_nanwang_short_d1_points"] == 288
