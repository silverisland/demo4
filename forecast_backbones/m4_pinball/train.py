"""Standalone training entry point using the existing UnifiedDataset and HF Trainer."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from transformers import EarlyStoppingCallback, set_seed

from pipelines.transformer_dataset import UnifiedDataset

from .config import feature_columns, load_config
from .data_paths import expand_paths
from .selection import METRIC, select_validation_rows
from .transformer_models import TransformersAdaptedM4PinballModel
from .transformer_trainer import (
    M4Pinball_build_training_args,
    M4PinballTrainer,
    PortableCheckpointCallback,
    evaluate_model,
)

COLLATE = partial(UnifiedDataset.collate_fn, for_hf_trainer=True, drop_str_type=False)


def build_dataset(source, cfg, supervised=True, drop_empty_targets=True):
    ds = cfg["dataset"]
    target, horizon = ds["target_col"], cfg["inference"]["pred_len"]

    def valid_targets(values):
        result = []
        for row in values:
            arr = np.asarray(row, float)
            if arr.shape != (horizon,):
                raise ValueError(
                    f"{target} must contain exactly {horizon} values per row"
                )
            result.append(not drop_empty_targets or bool(np.isfinite(arr).any()))
        return result

    if isinstance(source, pd.DataFrame):
        if supervised:
            source = source.loc[valid_targets(source[target])].copy()
        columns = source.columns
    else:
        from datasets import load_dataset

        source = load_dataset(
            "parquet", data_files={"train": expand_paths(source)}, split="train"
        )
        if supervised:
            source = source.filter(lambda b: valid_targets(b[target]), batched=True)
        columns = source.column_names
    if len(source) == 0:
        raise ValueError("Dataset has no usable rows")
    if ds["capacity_by_station"] and ds["entity_col"] not in columns:
        raise ValueError(
            "capacity_by_station requires station identities in the dataset"
        )
    return UnifiedDataset(
        source,
        timestamp_cols=[ds["timestamp_col"]],
        entity_cols=[ds["entity_col"]],
        feature_cols=feature_columns(cfg),
        label_cols=[target] if supervised else [],
        epoch_unit="s",
        timestamp_output="epoch_s",
        for_hf_trainer=True,
    )


def train(
    config,
    train_source,
    valid_source,
    out_dir,
    max_steps=-1,
    test_source=None,
    refit=False,
):
    cfg = load_config(config)
    if refit and (valid_source is not None or max_steps <= 0):
        raise ValueError(
            "--refit requires positive --max_steps and no validation input"
        )
    if not refit and valid_source is None:
        raise ValueError("Provide --valid for D+1 selection, or explicitly use --refit")
    # Catch accidental split overlap before training. Temporal leakage within
    # different files still depends on how the user prepared the splits.
    sources = [
        s
        for s in (train_source, valid_source, test_source)
        if s is not None and not isinstance(s, pd.DataFrame)
    ]
    resolved = [set(expand_paths(s)) for s in sources]
    if any(a & b for i, a in enumerate(resolved) for b in resolved[i + 1 :]):
        raise ValueError("Train/valid/test inputs must not share parquet files")
    args = M4Pinball_build_training_args(cfg, out_dir, max_steps, refit=refit)
    set_seed(cfg["optimization"]["random_seed"])
    model = TransformersAdaptedM4PinballModel(cfg)
    train_ds, valid_ds = (
        build_dataset(train_source, cfg),
        build_dataset(valid_source, cfg, drop_empty_targets=False)
        if not refit
        else None,
    )
    selection_ds = (
        select_validation_rows(valid_ds, cfg) if valid_ds is not None else None
    )
    trainer = M4PinballTrainer(
        model=model,
        args=args,
        data_collator=COLLATE,
        train_dataset=train_ds,
        eval_dataset=selection_ds,
        callbacks=[PortableCheckpointCallback()]
        + (
            []
            if refit
            else [
                EarlyStoppingCallback(
                    early_stopping_patience=cfg["optimization"]["patience"]
                )
            ]
        ),
    )
    out = Path(out_dir)
    if trainer.is_world_process_zero():
        out.mkdir(parents=True, exist_ok=True)
        (out / "model_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        (out / "run.yaml").write_text(
            yaml.safe_dump(
                {
                    "model_name": "m4_pinball",
                    "mode": "refit" if refit else "validation",
                    "selection_metric": None if refit else f"eval_{METRIC}",
                    "greater_is_better": True,
                    "selection_issue_time": "10:00 (latest at or before)",
                    "selection_window": "D+1 [00:00, 24:00), 15-minute steps",
                    "selection_aggregation": "mean of clipped station-day scores",
                    "selection_rows": len(selection_ds)
                    if selection_ds is not None
                    else 0,
                    "eval_every_steps": cfg["optimization"]["eval_every_steps"],
                    "patience": cfg["optimization"]["patience"],
                    "patience_unit": "validation_count",
                    "final_step_validation": not refit,
                    "max_steps": max_steps,
                    "train": str(train_source)
                    if not isinstance(train_source, pd.DataFrame)
                    else "in-memory DataFrame",
                    "valid": str(valid_source)
                    if not isinstance(valid_source, pd.DataFrame)
                    else "in-memory DataFrame",
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    trainer.train()
    # load_best_model_at_end has already restored the highest D+1 checkpoint
    # on every rank. Never export unselected weights as the final model.
    best = trainer.state.best_model_checkpoint
    if not refit and (best is None or trainer.state.best_metric is None):
        raise RuntimeError("Training produced no D+1-selected checkpoint")
    if trainer.is_world_process_zero():
        best_step = int(Path(best).name.rsplit("-", 1)[1]) if best else None
        history = [row for row in trainer.state.log_history if f"eval_{METRIC}" in row]
        (out / "train_log.json").write_text(
            json.dumps(
                {
                    "mode": "refit" if refit else "validation",
                    "metric": None if refit else f"eval_{METRIC}",
                    "higher_better": True,
                    "best": float(trainer.state.best_metric) if best else None,
                    "best_step": best_step,
                    "best_model_checkpoint": best,
                    "stopped_step": int(trainer.state.global_step),
                    "eval_every_steps": cfg["optimization"]["eval_every_steps"],
                    "patience": cfg["optimization"]["patience"],
                    "patience_unit": "validation_count",
                    "history": history,
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        if not refit:
            print(
                f"[m4_pinball] restored best D+1={trainer.state.best_metric:.4f} "
                f"at step {best_step}; training stopped at {trainer.state.global_step}",
                flush=True,
            )
    trainer.save_model(str(out / "model_result"))
    trainer.save_state()
    if not refit:
        trainer.save_metrics("eval", trainer.evaluate())
    if trainer.is_world_process_zero():
        (out / "model_result" / "model_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        if valid_ds is not None:
            evaluate_model(model, valid_ds, COLLATE, cfg).to_parquet(
                out / "result_val.parquet", index=False
            )
        if test_source is not None:
            evaluate_model(
                model,
                build_dataset(test_source, cfg, drop_empty_targets=False),
                COLLATE,
                cfg,
            ).to_parquet(out / "result_test.parquet", index=False)
    trainer.accelerator.wait_for_everyone()
    return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train M4-Pinball with D+1 selection or explicit refit"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--train", "--train_parquet", dest="train", required=True)
    parser.add_argument("--valid", "--valid_parquet", dest="valid")
    parser.add_argument("--test")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--out")
    output.add_argument("--run_id")
    parser.add_argument("--runs_base", default="runs/m4_pinball")
    parser.add_argument(
        "--max-steps", "--max_steps", dest="max_steps", type=int, default=-1
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="No validation/early stopping; fixed step budget required",
    )
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--hint_col")
    parser.add_argument(
        "--station_meta", help="CSV mapping plant_pointname to GCCAPACITY"
    )
    parser.add_argument("--cpu", action="store_true")
    a = parser.parse_args(argv)
    cfg = load_config(a.config)
    for flag, key in {
        "batch_size": "batch_size",
        "eval_batch_size": "eval_batch_size",
        "eval_every": "eval_every_steps",
        "save_every": "save_every_steps",
        "seed": "random_seed",
        "lr": "learning_rate",
        "warmup_steps": "warmup_steps",
        "patience": "patience",
    }.items():
        value = getattr(a, flag)
        if value is not None:
            cfg["optimization"][key] = value
    if a.hint_col:
        cfg["dataset"]["tsfm_col"] = a.hint_col
    if a.station_meta:
        table = pd.read_csv(a.station_meta, dtype={"plant_pointname": str})
        if (
            table["plant_pointname"].isna().any()
            or table["plant_pointname"].duplicated().any()
        ):
            parser.error(
                "Station metadata must have unique, nonempty plant_pointname values"
            )
        cfg["dataset"]["capacity_by_station"] = dict(
            zip(table["plant_pointname"], table["GCCAPACITY"])
        )
    if a.cpu:
        cfg["optimization"].update(use_cpu=True, fp16=False)
    out = a.out or str(Path(a.runs_base) / a.run_id)
    train(cfg, a.train, a.valid, out, a.max_steps, a.test, refit=a.refit)


if __name__ == "__main__":
    main()
