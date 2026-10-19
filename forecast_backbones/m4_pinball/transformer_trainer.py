"""Training arguments and ordered evaluation for the M4-Pinball adapter."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments

from .config import load_config
from .decision import level_value, quantile_cols


def M4Pinball_build_training_args(model_config, result_dir, max_steps=-1):
    opt = load_config(model_config)["optimization"]
    use_cpu = opt["use_cpu"] or not torch.cuda.is_available()
    eval_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    return TrainingArguments(
        output_dir=str(result_dir),
        per_device_train_batch_size=opt["batch_size"],
        per_device_eval_batch_size=opt["eval_batch_size"],
        gradient_accumulation_steps=opt["gradient_accumulation_steps"],
        num_train_epochs=opt["train_epochs"],
        learning_rate=opt["learning_rate"],
        weight_decay=opt["weight_decay"],
        warmup_ratio=opt["warmup_ratio"],
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        max_steps=max_steps,
        seed=opt["random_seed"],
        data_seed=opt["random_seed"],
        use_cpu=use_cpu,
        fp16=bool(opt["fp16"] and not use_cpu),
        dataloader_num_workers=opt["num_workers"],
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        label_names=["labels"],
        remove_unused_columns=False,
        report_to=[],
        logging_steps=10,
        **{eval_key: "epoch"},
    )


@torch.no_grad()
def evaluate_model(model, test_dataset, HF_collate_fn, config):
    cfg = load_config(config)
    ds, horizon = cfg["dataset"], cfg["inference"]["pred_len"]
    loader = DataLoader(
        test_dataset,
        batch_size=cfg["optimization"]["eval_batch_size"],
        shuffle=False,
        collate_fn=HF_collate_fn,
    )
    timestamps = test_dataset.timestamp
    rows, offset = [], 0
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            bd = batch.get("batch_dict", batch)
            output = model(bd)
            pred = output.logits.cpu().numpy()
            raw = output.aux["quantiles"].cpu().numpy()
            cal = output.aux["quantiles_cal"].cpu().numpy()
            cap = output.aux["cap"].cpu().numpy()
            median = level_value(cal, model.levels, 0.5)
            truth = batch.get("labels", {}).get(ds["target_col"])
            truth = (
                np.full((len(pred), horizon), np.nan)
                if truth is None
                else torch.as_tensor(truth).cpu().numpy()
            )
            stations = bd.get(ds["entity_col"], [None] * len(pred))
            physics = model._physics_inputs(bd, pred.shape)
            for i in range(len(pred)):
                station = stations[i]
                row = {
                    "timestamp_win": timestamps[offset + i],
                    "station": station.item()
                    if isinstance(station, torch.Tensor)
                    else station,
                    "cap": float(cap[i]),
                    "observe_power_predict": pred[i],
                    "observe_power_future": truth[i],
                    "median": median[i],
                    # Nested lists are portable to parquet and allow decisions without rerunning M4.
                    "quantiles_raw": raw[i].tolist(),
                }
                row.update(
                    {
                        name: cal[i, :, k]
                        for k, name in enumerate(quantile_cols(model.levels))
                    }
                )
                row.update(
                    {
                        key: value[i]
                        for key, value in physics.items()
                        if value is not None
                    }
                )
                rows.append(row)
            offset += len(pred)
    finally:
        model.train(was_training)
    return pd.DataFrame(rows)
