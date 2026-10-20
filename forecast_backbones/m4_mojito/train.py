"""Train on explicit, pre-split data; reuse UnifiedDataset and Hugging Face Trainer."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from transformers import EarlyStoppingCallback, Trainer, set_seed

from pipelines.transformer_dataset import UnifiedDataset

from .config import feature_columns, load_config
from .transformer_models import TransformersAdaptedMojitoModel
from .transformer_trainer import build_training_args

COLLATE = partial(UnifiedDataset.collate_fn, for_hf_trainer=True, drop_str_type=False)


def build_dataset(source, cfg: dict, supervised=True):
    """Read only declared features; exclude rows with no usable forecast targets."""
    ds = cfg["dataset"]
    target, horizon = ds["target_col"], cfg["inference"]["pred_len"]

    def valid_targets(values):
        return [
            bool(np.isfinite(np.asarray(row, dtype=float)[:horizon]).any())
            for row in values
        ]

    if isinstance(source, pd.DataFrame):
        if supervised:
            source = source.loc[valid_targets(source[target])].copy()
    else:
        from datasets import load_dataset

        source = load_dataset("parquet", data_files={"train": source}, split="train")
        if supervised:
            source = source.filter(
                lambda batch: valid_targets(batch[target]), batched=True
            )
    if len(source) == 0:
        raise ValueError("Dataset has no usable rows")
    columns = (
        source.columns if isinstance(source, pd.DataFrame) else source.column_names
    )
    return UnifiedDataset(
        source,
        timestamp_cols=[ds["timestamp_col"]],
        entity_cols=[ds["entity_col"]] if ds["entity_col"] in columns else [],
        feature_cols=feature_columns(cfg),
        label_cols=[target] if supervised else [],
        epoch_unit="s",
        timestamp_output="epoch_s",
        for_hf_trainer=True,
    )


def train(config, train_source, valid_source, out_dir, max_steps=-1):
    cfg = load_config(config)
    args = build_training_args(cfg, out_dir, max_steps)
    set_seed(cfg["optimization"]["random_seed"])
    model = TransformersAdaptedMojitoModel(cfg)
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=COLLATE,
        train_dataset=build_dataset(train_source, cfg),
        eval_dataset=build_dataset(valid_source, cfg),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["optimization"]["patience"]
            )
        ],
    )
    if trainer.is_world_process_zero():
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
        provenance = {
            "train": str(train_source)
            if not isinstance(train_source, pd.DataFrame)
            else "in-memory DataFrame",
            "valid": str(valid_source)
            if not isinstance(valid_source, pd.DataFrame)
            else "in-memory DataFrame",
            "selection_metric": "eval_loss: weighted MAE proxy, not the business score",
            "max_steps": max_steps,
        }
        (destination / "run.yaml").write_text(
            yaml.safe_dump(provenance), encoding="utf-8"
        )
    trainer.train()
    trainer.save_model(str(Path(out_dir) / "best"))
    trainer.save_state()
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train M4-Mojito on pre-split parquet files"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()
    train(args.config, args.train, args.valid, args.out, args.max_steps)


if __name__ == "__main__":
    main()
