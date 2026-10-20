"""Standalone training entry point using the existing UnifiedDataset and HF Trainer."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from transformers import EarlyStoppingCallback, set_seed

from pipelines.transformer_dataset import UnifiedDataset

from .config import feature_columns, load_config
from .selection import METRIC, select_validation_rows
from .transformer_models import TransformersAdaptedM4PinballModel
from .transformer_trainer import (
    M4Pinball_build_training_args,
    M4PinballTrainer,
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
            "parquet", data_files={"train": str(source)}, split="train"
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


def train(config, train_source, valid_source, out_dir, max_steps=-1, test_source=None):
    cfg = load_config(config)
    args = M4Pinball_build_training_args(cfg, out_dir, max_steps)
    set_seed(cfg["optimization"]["random_seed"])
    model = TransformersAdaptedM4PinballModel(cfg)
    train_ds, valid_ds = (
        build_dataset(train_source, cfg),
        build_dataset(valid_source, cfg, drop_empty_targets=False),
    )
    selection_ds = select_validation_rows(valid_ds, cfg)
    trainer = M4PinballTrainer(
        model=model,
        args=args,
        data_collator=COLLATE,
        train_dataset=train_ds,
        eval_dataset=selection_ds,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["optimization"]["patience"]
            )
        ],
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
                    "selection_metric": f"eval_{METRIC}",
                    "greater_is_better": True,
                    "selection_issue_time": "10:00 (latest at or before)",
                    "selection_window": "D+1 [00:00, 24:00), 15-minute steps",
                    "selection_aggregation": "mean of clipped station-day scores",
                    "selection_rows": len(selection_ds),
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
    trainer.save_model(str(out / "model_result"))
    trainer.save_state()
    trainer.save_metrics("eval", trainer.evaluate())
    if trainer.is_world_process_zero():
        (out / "model_result" / "model_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        evaluate_model(model, valid_ds, COLLATE, cfg).to_parquet(
            out / "result_val.parquet", index=False
        )
        if test_source is not None:
            evaluate_model(
                model, build_dataset(test_source, cfg), COLLATE, cfg
            ).to_parquet(out / "result_test.parquet", index=False)
    trainer.accelerator.wait_for_everyone()
    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train M4-Pinball on pre-split parquet data"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--test", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    a = parser.parse_args()
    train(a.config, a.train, a.valid, a.out, a.max_steps, a.test)


if __name__ == "__main__":
    main()
