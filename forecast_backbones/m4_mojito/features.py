"""One input transformation shared by training and inference; no station lookup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .augmentation import augment_power


@dataclass
class Features:
    history: torch.Tensor  # [B, L, C], capacity/fixed-scale normalized
    history_mask: torch.Tensor
    future: torch.Tensor  # [B, 16, C-1]
    future_mask: torch.Tensor
    history_time: torch.Tensor  # [B, L, 4]
    future_time: torch.Tensor  # [B, 16, 4]
    state: torch.Tensor  # [B, 9]
    recent: torch.Tensor  # [B, 33]
    capacity: torch.Tensor  # [B, 1]


def sequence(value, name: str, device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim != 2 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty batched sequence [B, T]")
    return tensor


def calendar_features(timestamps, length: int, horizon: int, basis: str, device):
    """Scalar issue times are epoch seconds; historical endpoint equals issue time."""
    raw = torch.as_tensor(timestamps).detach().cpu().numpy()
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw[:, 0]
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError(
            "Issue timestamps must be integer epoch seconds with shape [B]"
        )
    if ((raw < 946684800) | (raw >= 4102444800)).any():
        raise ValueError("Issue timestamps must be valid epoch seconds in 2000–2099")
    offsets = np.arange(1 - length, horizon + 1, dtype=np.int64) * 900
    seconds = raw[:, None] + offsets + (8 * 3600 if basis == "utc" else 0)
    dates = seconds.astype("datetime64[s]")
    years = dates.astype("datetime64[Y]")
    year_start = years.astype("datetime64[s]").astype(np.int64)
    year_end = (years + 1).astype("datetime64[s]").astype(np.int64)
    daily = 2 * np.pi * (seconds % 86400) / 86400
    annual = 2 * np.pi * (seconds - year_start) / (year_end - year_start)
    values = np.stack(
        [np.sin(daily), np.cos(daily), np.sin(annual), np.cos(annual)], axis=-1
    )
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    return tensor[:, :length], tensor[:, length:]


def state_features(power: torch.Tensor, mask: torch.Tensor):
    """Summaries describe finite values, not observation reliability."""
    length = power.shape[1]
    index = torch.arange(length, device=power.device)[None, :]
    last_index = index.expand_as(power).masked_fill(~mask, -1).amax(dim=1, keepdim=True)
    last = power.gather(1, last_index.clamp_min(0)).masked_fill(last_index < 0, 0)
    age = (length - 1 - last_index).float() / length
    parts = [last, age, mask.float().mean(dim=1, keepdim=True)]
    for window in (4, 16):
        x, valid = power[:, -window:], mask[:, -window:]
        count = valid.sum(dim=1, keepdim=True).clamp_min(1)
        mean = x.sum(dim=1, keepdim=True) / count
        pairs = valid[:, 1:] & valid[:, :-1]
        slope = ((x[:, 1:] - x[:, :-1]) * pairs).sum(dim=1, keepdim=True)
        slope = slope / pairs.sum(dim=1, keepdim=True).clamp_min(1)
        parts.extend([mean, slope, valid.float().mean(dim=1, keepdim=True)])
    recent = torch.cat([power[:, -16:], mask[:, -16:].float(), age], dim=1)
    return torch.cat(parts, dim=1), recent


def prepare_features(
    batch: dict, cfg: dict, device, training: bool = False
) -> Features:
    ds, inf = cfg["dataset"], cfg["inference"]
    length, horizon = inf["input_len"], inf["pred_len"]
    power = sequence(batch[ds["pv_col"]], ds["pv_col"], device)
    batch_size = power.shape[0]

    def window(value, name, future=False):
        x = sequence(value, name, device)
        if x.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch in {name}")
        if future:
            if x.shape[1] < horizon:
                raise ValueError(f"{name} needs at least {horizon} future positions")
            return x[:, :horizon]
        x = x[:, -length:]
        return F.pad(x, (length - x.shape[1], 0), value=float("nan"))

    capacity = torch.as_tensor(
        batch[ds["capacity_col"]], dtype=torch.float32, device=device
    )
    if capacity.shape not in ((batch_size,), (batch_size, 1)):
        raise ValueError("Capacity must contain one value per row: [B] or [B, 1]")
    capacity = capacity.reshape(batch_size, 1)
    if not torch.isfinite(capacity).all() or (capacity <= 0).any():
        raise ValueError(
            "Capacity must be finite and positive; no label-derived fallback"
        )
    power = window(power, ds["pv_col"]) / capacity
    if training and cfg["augmentation"]["probability"] > 0:
        power = augment_power(power, cfg["augmentation"])

    past, future = [power], []
    for hcol, fcol, offset, scale in zip(
        ds["conv_cols"],
        ds["conv_predict_cols"],
        ds["weather_offsets"],
        ds["weather_scales"],
    ):
        past.append((window(batch[hcol], hcol) - offset) / scale)
        future.append((window(batch[fcol], fcol, future=True) - offset) / scale)
    history, future = torch.stack(past, dim=-1), torch.stack(future, dim=-1)
    history_mask, future_mask = torch.isfinite(history), torch.isfinite(future)
    history = torch.where(history_mask, history, 0.0)
    future = torch.where(future_mask, future, 0.0)
    history_time, future_time = calendar_features(
        batch[ds["timestamp_col"]], length, horizon, ds["timestamp_basis"], device
    )
    if history_time.shape[0] != batch_size:
        raise ValueError("Timestamp batch size mismatch")
    state, recent = state_features(history[:, :, 0], history_mask[:, :, 0])
    return Features(
        history,
        history_mask,
        future,
        future_mask,
        history_time,
        future_time,
        state,
        recent,
        capacity,
    )


def prediction_loss(
    prediction: torch.Tensor, labels, capacity: torch.Tensor
) -> torch.Tensor:
    """Row-balanced weighted MAE on normalized power, ignoring nonfinite targets."""
    target = sequence(labels, "labels", prediction.device)
    if target.shape[0] != prediction.shape[0] or target.shape[1] < prediction.shape[1]:
        raise ValueError(
            "Labels must match the batch and cover all 16 forecast positions"
        )
    target = target[:, : prediction.shape[1]] / capacity
    valid = torch.isfinite(target)
    rows = valid.any(dim=1)
    if not rows.any():
        raise ValueError(
            "Batch contains no finite labels; remove unlabeled rows before training"
        )
    target = torch.where(valid, target, 0.0)
    error = (prediction - target).abs() / target.clamp_min(0.2)
    per_row = (error * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    return per_row[rows].mean()
