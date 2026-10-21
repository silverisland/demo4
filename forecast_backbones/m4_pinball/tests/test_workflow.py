"""Portable checkpoints, full evaluation and M8 reporting parity."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from forecast_backbones.m4_pinball.config import load_config
from forecast_backbones.m4_pinball.data_paths import expand_paths
from forecast_backbones.m4_pinball.evaluate import evaluate, load_run, write_reports
from forecast_backbones.m4_pinball.tests.test_pinball import make_frame
from forecast_backbones.m4_pinball.train import main, train


def small_config():
    return load_config(
        {
            "inference": {"input_len": 32, "pred_len": 192},
            "model": {
                "patch_len": 8,
                "stride": 8,
                "d_model": 16,
                "n_heads": 2,
                "e_layers": 1,
                "d_ff": 32,
                "embedding_dim": 16,
            },
            "optimization": {
                "use_cpu": True,
                "fp16": False,
                "batch_size": 2,
                "eval_batch_size": 2,
                "save_every_steps": 1,
            },
        }
    )


def test_paths_glob_directory_deduplicate_and_split_overlap(tmp_path):
    path = tmp_path / "data.parquet"
    make_frame(future=192).to_parquet(path)
    assert expand_paths([path, tmp_path, str(tmp_path / "*.parquet")]) == [str(path)]
    with pytest.raises(ValueError, match="No parquet"):
        expand_paths(tmp_path / "missing*.parquet")
    with pytest.raises(ValueError, match="must not share"):
        train(small_config(), path, tmp_path, tmp_path / "run", max_steps=1)


def test_refit_is_explicit_and_has_no_best_claim(tmp_path):
    cfg = small_config()
    frame = make_frame(future=192)
    with pytest.raises(ValueError, match="Provide --valid"):
        train(cfg, frame, None, tmp_path / "bad", max_steps=2)
    with pytest.raises(ValueError, match="positive --max_steps"):
        train(cfg, frame, None, tmp_path / "bad", refit=True)
    with pytest.raises(ValueError, match="no validation"):
        train(cfg, frame, frame, tmp_path / "bad", max_steps=2, refit=True)
    out = tmp_path / "run"
    trainer = train(cfg, frame, None, out, max_steps=2, refit=True)
    log = json.loads((out / "train_log.json").read_text())
    assert log["mode"] == "refit" and log["best"] is None and log["metric"] is None
    assert trainer.state.best_model_checkpoint is None
    assert not (out / "result_val.parquet").exists()
    restored, _ = load_run(out, device="cpu")
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    first, _ = load_run(out / "checkpoints" / "step_000001", device="cpu")
    last, _ = load_run(out / "checkpoints" / "step_000002", device="cpu")
    assert any(
        not torch.equal(a, b) for a, b in zip(first.parameters(), last.parameters())
    )
    for name, value in last.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    # Preserve every issue row even when all labels are absent on one row.
    frame.at[0, "observe_power_future"] = np.full(192, np.nan, dtype=np.float32)
    test_path = tmp_path / "test.parquet"
    frame.to_parquet(test_path)
    summary = evaluate(out, test_path, tmp_path / "eval", device="cpu", point="median")
    result = pd.read_parquet(tmp_path / "eval" / "result_test.parquet")
    assert summary["n_rows"] == len(frame)
    np.testing.assert_allclose(
        np.stack(result.observe_power_predict), np.stack(result["median"])
    )
    decision = json.loads((tmp_path / "eval" / "reports" / "decision.json").read_text())
    assert decision["point"]["by_step"] == []
    assert summary["nanwang_short_D+1"] is None  # fewer than three sample days


def test_full_monthly_metrics_use_reference_denominator_and_station_mean(tmp_path):
    rows = []
    # Full 56-target daytime coverage per day, all 16 leads supplied.
    for station, error in (("a", 1.0), ("b", 3.0)):
        for day in pd.date_range("2025-06-01", periods=3):
            for issue in pd.date_range(
                day + pd.Timedelta(hours=2, minutes=15),
                day + pd.Timedelta(hours=19, minutes=45),
                freq="15min",
            ):
                rows.append(
                    {
                        "station": station,
                        "timestamp_win": issue,
                        "cap": 100.0,
                        "observe_power_future": np.full(192, 10.0),
                        "observe_power_predict": np.full(192, 10.0 + error),
                    }
                )
    result = pd.DataFrame(rows)
    summary = write_reports(result, tmp_path)
    # denominator max(10, .2*100)=20, ultra divides sum of 56 errors by 96.
    expected_ultra = np.mean(
        [round(100 * (1 - 56 / 96 * e / 20), 2) for e in (1.0, 3.0)]
    )
    assert summary["nanwang_ultra_short"] == pytest.approx(expected_ultra)
    assert summary["nanwang_short_D+1"] == pytest.approx(90.0)
    daily = pd.read_csv(tmp_path / "reports" / "d1_daily.csv")
    assert (daily.n_points == 96).all()
    assert pd.to_datetime(daily.issue_ts).dt.hour.eq(10).all()
    result.loc[0, "cap"] = 99.0
    with pytest.raises(ValueError, match="fixed"):
        write_reports(result, tmp_path / "invalid")


def test_cli_overrides_and_m8_recipe_keep_architecture(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("forecast_backbones.m4_pinball.train")
    calls = []
    monkeypatch.setattr(module, "train", lambda *a, **kw: calls.append((a, kw)))
    recipe = Path(module.__file__).with_name("config") / "m8_k2_training.yaml"
    cfg = load_config(recipe)
    assert cfg["model"] == load_config()["model"]
    meta = tmp_path / "info.csv"
    pd.DataFrame({"plant_pointname": ["a"], "GCCAPACITY": [100.0]}).to_csv(
        meta, index=False
    )
    main(
        [
            "--config",
            str(recipe),
            "--train_parquet",
            "train/*.parquet",
            "--valid",
            "val",
            "--run_id",
            "trial",
            "--max_steps",
            "6000",
            "--seed",
            "7",
            "--batch_size",
            "4",
            "--save_every",
            "200",
            "--hint_col",
            "timesfm3",
            "--station_meta",
            str(meta),
        ]
    )
    args, kwargs = calls[0]
    assert args[0]["optimization"]["random_seed"] == 7
    assert args[0]["optimization"]["warmup_steps"] == 300
    assert args[0]["optimization"]["save_every_steps"] == 200
    assert args[0]["dataset"]["tsfm_col"] == "timesfm3"
    assert args[0]["dataset"]["capacity_by_station"] == {"a": 100.0}
    assert args[4] == 6000 and kwargs["refit"] is False


def test_shell_training_selected_checkpoint_and_full_eval(tmp_path):
    root = Path(__file__).resolve().parents[3]
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(small_config()))
    frame = make_frame(future=192)
    for split in ("train", "valid", "test"):
        frame.to_parquet(tmp_path / f"{split}.parquet", index=False)
    out = tmp_path / "run"
    env = {**os.environ, "PYTHON": sys.executable, "OMP_NUM_THREADS": "1"}
    command = [
        "bash",
        str(root / "train_shell" / "train_m4_pinball.sh"),
        "--config",
        str(cfg_path),
        "--train_parquet",
        str(tmp_path / "train.parquet"),
        "--valid_parquet",
        str(tmp_path / "valid.parquet"),
        "--out",
        str(out),
        "--max_steps",
        "1",
        "--eval_every",
        "1",
    ]
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    final, _ = load_run(out, device="cpu")
    checkpoint, _ = load_run(out / "checkpoint-1", device="cpu")
    for name, value in final.state_dict().items():
        torch.testing.assert_close(value, checkpoint.state_dict()[name], rtol=0, atol=0)
    completed = subprocess.run(
        [
            "bash",
            str(root / "eval_m4_pinball.sh"),
            "--runs",
            str(out),
            "--full",
            "--test",
            str(tmp_path / "test.parquet"),
            "--out",
            str(out / "eval"),
            "--device",
            "cpu",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = pd.read_parquet(out / "eval" / "result_test.parquet")
    expected = pd.read_parquet(out / "result_val.parquet")
    np.testing.assert_allclose(
        np.stack(result.observe_power_predict),
        np.stack(expected.observe_power_predict),
        rtol=0,
        atol=0,
    )
