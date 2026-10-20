"""Causal tail corruption; synthetic quality labels never enter the network."""

from __future__ import annotations

import torch

from .config import TAIL_METHODS


def corrupt_tail(
    power: torch.Tensor, lengths: torch.Tensor, methods: torch.Tensor
) -> torch.Tensor:
    """Replace each row's tail using only its retained prefix; zero length is clean."""
    batch, length = power.shape
    if lengths.shape != (batch,) or methods.shape != (batch,):
        raise ValueError("Tail lengths and method IDs must have shape [B]")
    if (lengths < 0).any() or (lengths >= length).any():
        raise ValueError("Tail length must be in [0, history length)")
    if (methods < 0).any() or (methods >= len(TAIL_METHODS)).any():
        raise ValueError("Unknown tail corruption method")
    index = torch.arange(length, device=power.device)[None, :].expand(batch, -1)
    cutoff = length - lengths[:, None]
    tail = index >= cutoff
    retained = torch.isfinite(power) & ~tail
    last_index = index.masked_fill(~retained, -1).amax(dim=1, keepdim=True)
    previous_index = index.masked_fill(~retained | (index >= last_index), -1).amax(
        dim=1, keepdim=True
    )
    last = power.gather(1, last_index.clamp_min(0)).masked_fill(
        last_index < 0, float("nan")
    )
    previous = power.gather(1, previous_index.clamp_min(0))
    slope = (last - previous) / (last_index - previous_index).clamp_min(1)
    slope = torch.where(previous_index >= 0, slope, torch.zeros_like(slope))
    day_index = index - 96
    yesterday = power.gather(1, day_index.clamp_min(0))
    yesterday = yesterday.masked_fill(
        (day_index < 0) | (day_index >= cutoff), float("nan")
    )
    replacements = (
        torch.full_like(power, float("nan")),
        last.expand_as(power),
        torch.zeros_like(power),
        yesterday,
        last + slope * (index - last_index),
    )
    out = power.clone()
    for method, replacement in enumerate(replacements):
        out = torch.where(tail & (methods[:, None] == method), replacement, out)
    return out


def augment_power(power: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Sample one method and length per row, retaining a configured clean fraction."""
    batch = power.shape[0]
    device = power.device
    lengths = torch.tensor(cfg["lengths"], device=device)
    lengths = lengths[torch.randint(len(lengths), (batch,), device=device)]
    lengths = lengths * (torch.rand(batch, device=device) < cfg["probability"])
    methods = torch.tensor(
        [TAIL_METHODS.index(m) for m in cfg["methods"]], device=device
    )
    methods = methods[torch.randint(len(methods), (batch,), device=device)]
    return corrupt_tail(power, lengths, methods)
