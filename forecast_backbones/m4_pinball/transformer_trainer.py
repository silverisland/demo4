"""Training arguments and ordered evaluation for the M4-Pinball adapter."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from safetensors.torch import save_model
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback, TrainingArguments

from .config import load_config
from .decision import level_value, quantile_cols
from .selection import METRIC, decision_metrics, metadata


class FinalStepValidationCallback(TrainerCallback):
    """Include the last optimizer step in selection, even before the first interval."""

    def on_step_end(self, args, state, control, **kwargs):
        strategy = getattr(args, "eval_strategy", None)
        if strategy is None:
            strategy = args.evaluation_strategy
        if strategy == "steps" and state.global_step >= state.max_steps:
            # HF evaluates before saving, then restores its best checkpoint at
            # train end. Both flags are needed for an off-interval final step.
            control.should_evaluate = True
            control.should_save = True
        return control


class M4PinballTrainer(Trainer):
    """Gather decision points across devices before scoring station-days."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(FinalStepValidationCallback())

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        Path(output_dir or self.args.output_dir, "model_config.yaml").write_text(
            yaml.safe_dump(self.model.config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")
        with torch.no_grad(), self.compute_loss_context_manager():
            output = model(**inputs)
        loss = output.loss.detach().mean() if output.loss is not None else None
        if prediction_loss_only:
            return loss, None, None
        target = (
            labels[self.model.labels_name[0]] if isinstance(labels, dict) else labels
        )
        # Avoid gathering quantile/embedding auxiliaries. HF/Accelerate gathers
        # points, capacities and labels together and removes padded DDP samples.
        predictions = (output.logits.detach(), output.aux["cap"].detach()[:, None])
        return loss, predictions, target.detach() if target is not None else None

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if output.label_ids is not None:
            point, capacity = output.predictions
            scores = decision_metrics(
                point,
                output.label_ids,
                capacity,
                metadata(dataloader.dataset, self.model.config),
            )
            output.metrics.update(
                {f"{metric_key_prefix}_{key}": value for key, value in scores.items()}
            )
        return output


class PortableCheckpointCallback(TrainerCallback):
    """Weight-only snapshots, independent of HF's best-model checkpoints."""

    def on_step_end(self, args, state, control, model=None, **kwargs):
        every = model.config["optimization"]["save_every_steps"]
        if (
            every
            and state.is_world_process_zero
            and (state.global_step % every == 0 or state.global_step >= state.max_steps)
        ):
            out = (
                Path(args.output_dir) / "checkpoints" / f"step_{state.global_step:06d}"
            )
            out.mkdir(parents=True, exist_ok=True)
            save_model(model, str(out / "model.safetensors"))
            (out / "model_config.yaml").write_text(
                yaml.safe_dump(model.config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        return control


def M4Pinball_build_training_args(model_config, result_dir, max_steps=-1, refit=False):
    opt = load_config(model_config)["optimization"]
    use_cpu = opt["use_cpu"] or not torch.cuda.is_available()
    eval_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    every = opt["eval_every_steps"]
    strategy = "no" if refit else ("steps" if every else "epoch")
    intervals = (
        {"eval_steps": every, "save_steps": every} if every and not refit else {}
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
        warmup_steps=opt["warmup_steps"],
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        max_steps=max_steps,
        seed=opt["random_seed"],
        data_seed=opt["random_seed"],
        use_cpu=use_cpu,
        fp16=bool(opt["fp16"] and not use_cpu),
        dataloader_num_workers=opt["num_workers"],
        save_strategy=strategy,
        save_total_limit=2,
        load_best_model_at_end=not refit,
        metric_for_best_model=f"eval_{METRIC}",
        greater_is_better=True,
        prediction_loss_only=False,
        label_names=["labels"],
        remove_unused_columns=False,
        report_to=[],
        logging_steps=10,
        **{eval_key: strategy},
        **intervals,
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
