"""M4 quantile adapter. Reuses M4 layers without changing the original M4 model."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from interfaces.modeloutput import TransformersAdaptedModelOutput

from ..m4.layers.feature_collator import HFFeatureCollator
from ..m4.layers.PatchTST_backbone import PatchTST_backbone
from ..m4.layers.RevIN import RevIN
from .config import load_config
from .decision import decide
from .losses import pinball_loss


class TransformersAdaptedM4PinballModel(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.config = load_config(configs)
        cfg = self.config
        m, inf = cfg["model"], cfg["inference"]
        self.levels = tuple(m["quantile_levels"])
        self.prediction_length = inf["pred_len"]
        self.context_length = inf["input_len"]
        self.labels_name = [cfg["dataset"]["target_col"]]
        self.featurecollator = HFFeatureCollator(cfg)
        self.revin = m["revin"]
        if self.revin:
            self.weather_revin_layer = RevIN(
                m["enc_in"] - 1, affine=m["affine"], subtract_last=m["subtract_last"]
            )
            self.pv_revin_layer = RevIN(
                1, affine=m["affine"], subtract_last=m["subtract_last"]
            )
        # Only the last projection expands: one distinct output for every (time, tau).
        self.model = PatchTST_backbone(
            c_in=m["enc_in"],
            context_window=self.context_length + self.prediction_length,
            target_window=self.prediction_length * len(self.levels),
            patch_len=m["patch_len"],
            stride=m["stride"],
            padding_patch=m["padding_patch"],
            n_layers=m["e_layers"],
            n_heads=m["n_heads"],
            d_model=m["d_model"],
            d_ff=m["d_ff"],
            dropout=m["dropout"],
            act=m["activation"],
            fc_dropout=m["fc_dropout"],
            head_dropout=m["head_dropout"],
            embedding_dim=m["embedding_dim"],
        )

    def _capacity(self, batch, batch_size, device):
        ds = self.config["dataset"]
        mapping = ds["capacity_by_station"]
        if mapping:
            if ds["entity_col"] not in batch:
                raise ValueError(f"Missing station identities: {ds['entity_col']}")
            stations = batch[ds["entity_col"]]
            if isinstance(stations, torch.Tensor):
                stations = stations.detach().cpu().reshape(-1).tolist()
            try:
                values = [mapping[str(s)] for s in stations]
            except KeyError as exc:
                raise ValueError(
                    f"No fixed capacity configured for station {exc.args[0]!r}"
                ) from exc
        else:
            if ds["capacity_col"] not in batch:
                raise ValueError(
                    f"Missing fixed capacity column {ds['capacity_col']!r}; no label-derived fallback"
                )
            values = batch[ds["capacity_col"]]
        cap = torch.as_tensor(values, device=device, dtype=torch.float32)
        if cap.ndim == 0:
            cap = cap.expand(batch_size)
        if cap.numel() != batch_size:
            raise ValueError(
                "Capacity must contain one value per sample, not a future time series"
            )
        cap = cap.reshape(batch_size)
        if not torch.isfinite(cap).all() or (cap <= 0).any():
            raise ValueError("Capacity must be finite and positive")
        return cap

    def _raw(self, batch_dict):
        x1, x2, x3, x4 = self.featurecollator(batch_dict)
        B = x3.shape[0]
        expected = (
            (B, self.config["model"]["enc_in"] - 1, self.context_length),
            (B, self.config["model"]["enc_in"] - 1, self.prediction_length),
            (B, self.context_length),
            (B, self.prediction_length),
        )
        for name, tensor, shape in zip(
            ("historical weather", "future weather", "PV history", "forecast hint"),
            (x1, x2, x3, x4),
            expected,
        ):
            if tensor.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"Nonfinite {name}; impute input features before inference"
                )
        cap = self._capacity(batch_dict, B, x3.device)
        weather = torch.cat([x1, x2], dim=-1).float()
        pv = torch.cat([x3, x4], dim=-1).float().unsqueeze(1)
        if self.revin:
            weather = self.weather_revin_layer(
                weather.transpose(1, 2), "norm"
            ).transpose(1, 2)
            pv = self.pv_revin_layer(pv.transpose(1, 2), "norm").transpose(1, 2)
        flat, embedding = self.model(weather, pv)
        q = flat.reshape(B, self.prediction_length, len(self.levels)).float()
        if self.revin:
            # [B,1,1] PV statistics broadcast identically across all times and quantiles.
            q = self.pv_revin_layer(q, "denorm")
        return q, embedding, cap

    def _physics_inputs(self, batch_dict, shape):
        ds, ph = self.config["dataset"], self.config["decision"]["physics"]

        def get(column, enabled, boolean=False):
            if not enabled:
                return None
            if column not in batch_dict:
                raise ValueError(f"Enabled physics rule requires column {column!r}")
            value = batch_dict[column]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, float)
            if value.shape != shape:
                raise ValueError(
                    f"{column} must contain only the future window {shape}"
                )
            if boolean and (
                not np.isfinite(value).all() or not np.isin(value, [0, 1]).all()
            ):
                raise ValueError(f"{column} must be a finite boolean / 0-1 night mask")
            return value.astype(bool) if boolean else value

        return {
            "p_clear": get(ds["p_clear_col"], ph["clear_sky_cap"]),
            "night": get(ds["night_col"], ph["night_zero"], True),
        }

    def forward(self, batch_dict, labels=None):
        q, embedding, cap = self._raw(batch_dict)
        loss = None
        if labels is not None:
            target = labels[self.labels_name[0]] if isinstance(labels, dict) else labels
            loss = self.compute_loss(q, target, cap)
        if self.training and labels is not None:
            # Trainer only needs the differentiable loss. Avoid GPU→CPU decisions each train step.
            return TransformersAdaptedModelOutput(loss=loss)
        physics = self._physics_inputs(batch_dict, tuple(q.shape[:2]))
        point, q_cal = decide(
            q.detach().cpu().numpy(),
            self.levels,
            cap.detach().cpu().numpy(),
            self.config["decision"],
            **physics,
        )
        point = torch.as_tensor(point, device=q.device, dtype=torch.float32)
        return TransformersAdaptedModelOutput(
            loss=loss,
            logits=point,
            predictions=point,
            embeddings=embedding,
            aux={
                "quantiles": q.detach(),
                "quantiles_cal": torch.as_tensor(
                    q_cal, device=q.device, dtype=torch.float32
                ),
                "cap": cap.detach(),
            },
        )

    def compute_loss(self, quantiles, labels, capacity):
        return pinball_loss(quantiles, labels, self.levels, capacity)

    def compute_metrics(self, eval_pred):
        predictions = eval_pred.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = eval_pred.label_ids
        if isinstance(labels, dict):
            labels = labels[self.labels_name[0]]
        predictions, labels = np.asarray(predictions), np.asarray(labels)
        if predictions.shape != labels.shape:
            raise ValueError("Point predictions and labels must have matching shapes")
        valid = np.isfinite(labels) & np.isfinite(predictions)
        if not valid.any():
            raise ValueError("No finite points for evaluation")
        error = predictions[valid] - labels[valid]
        return {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
        }
