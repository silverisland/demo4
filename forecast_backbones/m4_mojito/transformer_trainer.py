"""Training arguments and ordered evaluation using existing framework contracts."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments

from .config import load_config


def build_training_args(model_config, result_dir, max_steps=-1):
    opt = load_config(model_config)["optimization"]
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
        use_cpu=opt["use_cpu"],
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


def evaluate_model(model, test_dataset, HF_collate_fn, config) -> pd.DataFrame:
    """Return one curve per input row; retain metadata without using it as features."""
    cfg = load_config(config)
    ds, horizon = cfg["dataset"], cfg["inference"]["pred_len"]
    loader = DataLoader(
        test_dataset,
        batch_size=cfg["optimization"]["eval_batch_size"],
        shuffle=False,
        collate_fn=HF_collate_fn,
    )
    rows = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                bd = batch.get("batch_dict", batch)
                labels = batch.get("labels", {})
                output = model(bd)
                predictions = output.logits.cpu().numpy()
                truth = labels.get(ds["target_col"])
                truth = (
                    None
                    if truth is None
                    else torch.as_tensor(truth).cpu().numpy()[:, :horizon]
                )
                timestamps = (
                    torch.as_tensor(bd[ds["timestamp_col"]]).reshape(-1).cpu().numpy()
                )
                capacity = (
                    torch.as_tensor(bd[ds["capacity_col"]]).reshape(-1).cpu().numpy()
                )
                stations = bd.get(ds["entity_col"], [None] * len(predictions))
                flags = {key: value.cpu().numpy() for key, value in output.aux.items()}
                for i, prediction in enumerate(predictions):
                    timestamp = pd.Timestamp(int(timestamps[i]), unit="s")
                    if ds["timestamp_basis"] == "utc":
                        timestamp = (
                            timestamp.tz_localize("UTC")
                            .tz_convert("Asia/Shanghai")
                            .tz_localize(None)
                        )
                    station = stations[i]
                    rows.append(
                        {
                            "timestamp_win": timestamp,
                            "station": station.item()
                            if isinstance(station, torch.Tensor)
                            else station,
                            "cap": float(capacity[i]),
                            "observe_power_predict": prediction,
                            "observe_power_future": truth[i]
                            if truth is not None
                            else np.full(horizon, np.nan),
                            **{key: bool(value[i]) for key, value in flags.items()},
                        }
                    )
    finally:
        model.train(was_training)
    return pd.DataFrame(rows)
