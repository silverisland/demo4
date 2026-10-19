"""Pinball supervision on distinct quantile channels, in fixed-capacity units."""

import torch


def pinball_loss(q, target, levels, capacity):
    """q [B,H,K], target [B,H], capacity [B]; ignore nonfinite labels only."""
    q, target = q.float(), target.to(device=q.device, dtype=torch.float32)
    if target.shape != q.shape[:-1]:
        raise ValueError(
            f"Target shape {target.shape} does not match quantiles {q.shape}"
        )
    if not torch.isfinite(q).all():
        raise ValueError("Nonfinite model quantiles")
    cap = torch.as_tensor(capacity, device=q.device, dtype=q.dtype)
    if cap.shape != (q.shape[0],) or not torch.isfinite(cap).all() or (cap <= 0).any():
        raise ValueError("capacity must contain one finite positive value per sample")
    taus = torch.as_tensor(levels, device=q.device, dtype=q.dtype)
    if taus.shape != (q.shape[-1],):
        raise ValueError("Quantile levels do not match the output head")
    valid = torch.isfinite(target)
    if not valid.any():
        raise ValueError("Batch has no finite training targets")
    # Replace missing labels BEFORE arithmetic: NaN * 0 would still be NaN.
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    error = (safe_target[..., None] - q) / cap[:, None, None]
    loss = torch.maximum(taus * error, (taus - 1) * error)
    return loss[valid].mean()
