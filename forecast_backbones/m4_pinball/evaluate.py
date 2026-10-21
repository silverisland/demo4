"""Load a portable M4-Pinball run and report the same Nanwang metrics as M8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from safetensors.torch import load_model

from .config import load_config
from .data_paths import expand_paths
from .nanwang_metric import NanwangMetric
from .train import COLLATE, build_dataset
from .transformer_models import TransformersAdaptedM4PinballModel
from .transformer_trainer import evaluate_model

METRICS = ("nanwang_ultra_short", "nanwang_short_D+1")


def load_run(run, device=None, point=None, batch_size=None):
    path = Path(run)
    # A run directory always resolves to its final exported (selected) model.
    if (path / "model_result").is_dir():
        path = path / "model_result"
    for name in ("model.safetensors", "model_config.yaml"):
        if not (path / name).is_file():
            raise FileNotFoundError(
                f"Missing {path / name}; pass a completed run or portable checkpoint"
            )
    cfg = load_config(path / "model_config.yaml")
    if point is not None:
        # A global override must also remove the saved first-16-step rule.
        cfg["decision"]["point"].update(default=point, by_step=[])
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        cfg["optimization"]["eval_batch_size"] = batch_size
    model = TransformersAdaptedM4PinballModel(cfg)
    load_model(model, str(path / "model.safetensors"), strict=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), path


def write_reports(result, out_dir):
    """Monthly station summaries, then equal station means, matching M8 --full."""
    if result.empty or result["station"].isna().any():
        raise ValueError("Reporting requires nonempty results and explicit stations")
    if not np.isfinite(np.stack(result["observe_power_predict"])).all():
        raise ValueError("Cannot report nonfinite predictions")
    caps = {}
    for station, rows in result.groupby("station", sort=False):
        values = rows["cap"].to_numpy(float)
        if (
            not np.isfinite(values).all()
            or (values <= 0).any()
            or not np.allclose(values, values[0], rtol=1e-6, atol=0)
        ):
            raise ValueError(
                f"Station {station}: capacity must be fixed, finite and positive"
            )
        caps[station] = float(values[0])
    out = Path(out_dir)
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metric = NanwangMetric()
    daily_ultra = metric.ultra_short_daily(result, caps)
    daily_d1 = metric.short_d1_daily(result, caps)
    daily_ultra.to_csv(reports / "ultra_daily.csv", index=False)
    daily_d1.to_csv(reports / "d1_daily.csv", index=False)
    rows, monthly = [], []
    # Use the reference aggregator directly, including insufficient-month handling.
    for station, group in result.groupby("station", sort=False):
        times = pd.to_datetime(group["timestamp_win"])
        tab = metric._month_table(
            daily_ultra.loc[daily_ultra["station"] == station],
            daily_d1.loc[daily_d1["station"] == station],
            set(times.dt.normalize()),
        )
        rows.append(
            {
                "station": station,
                "cap": caps[station],
                "rows": len(group),
                **{key: tab.at[key, "mean"] for key in METRICS},
            }
        )
        monthly.append(tab.rename_axis("metric").reset_index().assign(station=station))
    table = pd.DataFrame(rows)
    table.to_csv(reports / "station_metrics.csv", index=False)
    pd.concat(monthly, ignore_index=True).to_csv(reports / "monthly.csv", index=False)
    summary = {"n_stations": len(table), "n_rows": len(result)}
    for key in METRICS:
        values = pd.to_numeric(table[key], errors="coerce")
        summary[key] = float(values.mean()) if values.notna().any() else None
        summary[f"{key}_valid_stations"] = int(values.notna().sum())
    pd.DataFrame([summary]).to_csv(reports / "summary.csv", index=False)
    (reports / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def evaluate(run, test_source, out_dir, device=None, point=None, batch_size=None):
    model, weights_dir = load_run(run, device, point, batch_size)
    cfg = model.config
    paths = expand_paths(test_source)
    dataset = build_dataset(paths, cfg, drop_empty_targets=False)
    result = evaluate_model(model, dataset, COLLATE, cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out / "result_test.parquet", index=False)
    summary = write_reports(result, out)
    (out / "eval_config.yaml").write_text(
        yaml.safe_dump(
            {
                "weights_dir": str(weights_dir.resolve()),
                "test": paths,
                "model_config": cfg,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (out / "reports" / "decision.json").write_text(
        json.dumps(cfg["decision"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Full-row M4-Pinball Nanwang evaluation"
    )
    parser.add_argument(
        "--runs",
        "--run",
        dest="run",
        required=True,
        help="One run, model_result or checkpoint directory",
    )
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Compatibility flag: evaluation always uses all rows",
    )
    parser.add_argument("--point", choices=("median", "p_star", "p_star_l1"))
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--device", help="cpu, cuda, cuda:0, ...")
    a = parser.parse_args(argv)
    evaluate(a.run, a.test, a.out, a.device, a.point, a.batch_size)


if __name__ == "__main__":
    main()
