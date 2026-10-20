"""Joint multiscale history attention and weather-conditioned horizon queries."""

from __future__ import annotations

import torch
from torch import nn

from .features import Features


class HistoryTokenizer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        model, inf = cfg["model"], cfg["inference"]
        width = model["d_model"]
        self.history_length = inf["input_len"]
        self.scales = [(model["fast_len"], model["fast_patch"])]
        if model["use_slow"]:
            self.scales.insert(0, (self.history_length, model["slow_patch"]))
        self.projections = nn.ModuleList(
            nn.Linear(2 * patch, width) for _, patch in self.scales
        )
        self.variable = nn.Embedding(1 + len(cfg["dataset"]["conv_cols"]), width)
        self.scale = nn.Embedding(len(self.scales), width)
        self.time = nn.Linear(5, width)
        self.state = nn.Linear(9, width)

    def forward(self, features: Features):
        tokens, padding = [], []
        for scale, ((length, patch), projection) in enumerate(
            zip(self.scales, self.projections)
        ):
            values = (
                features.history[:, -length:].transpose(1, 2).unfold(-1, patch, patch)
            )
            mask = (
                features.history_mask[:, -length:]
                .transpose(1, 2)
                .unfold(-1, patch, patch)
            )
            encoded = projection(torch.cat([values, mask.float()], dim=-1))
            clock = (
                features.history_time[:, -length:].unfold(1, patch, patch).mean(dim=-1)
            )
            positions = torch.arange(
                1 - length, 1, device=values.device, dtype=values.dtype
            )
            positions = (
                positions.unfold(0, patch, patch).mean(dim=-1) / self.history_length
            )
            positions = positions[None, :, None].expand(clock.shape[0], -1, -1)
            time = self.time(torch.cat([clock, positions], dim=-1))
            encoded = (
                encoded
                + self.variable.weight[None, :, None, :]
                + self.scale.weight[scale]
                + time[:, None, :, :]
            )
            tokens.append(encoded.flatten(1, 2))
            padding.append(~mask.any(dim=-1).flatten(1, 2))
        # Keep a summary key to prevent all-masked attention, even with no observations.
        tokens.append(self.state(features.state)[:, None, :])
        padding.append(
            torch.zeros(
                features.history.shape[0], 1, dtype=torch.bool, device=values.device
            )
        )
        return torch.cat(tokens, dim=1), torch.cat(padding, dim=1)


class Mojito(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        model = cfg["model"]
        width = model["d_model"]
        weather = len(cfg["dataset"]["conv_predict_cols"])
        self.horizon = cfg["inference"]["pred_len"]
        self.use_recent = model["use_recent"]
        self.tokenizer = HistoryTokenizer(cfg)
        layer_args = dict(
            d_model=width,
            nhead=model["n_heads"],
            dim_feedforward=model["d_ff"],
            dropout=model["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Construct layers separately rather than cloning identical initial parameters.
        self.encoder = nn.ModuleList(
            nn.TransformerEncoderLayer(**layer_args) for _ in range(model["e_layers"])
        )
        self.decoder = nn.ModuleList(
            nn.TransformerDecoderLayer(**layer_args) for _ in range(model["d_layers"])
        )
        self.memory_norm = nn.LayerNorm(width)
        self.output_norm = nn.LayerNorm(width)
        self.query = nn.Sequential(
            nn.Linear(2 * weather + 5, width), nn.GELU(), nn.Linear(width, width)
        )
        self.horizon_embedding = nn.Embedding(self.horizon, width)
        head_input = width + 2 * weather + 4 + (33 if self.use_recent else 0)
        self.head = nn.Sequential(
            nn.Linear(head_input, model["head_dim"]),
            nn.GELU(),
            nn.Linear(model["head_dim"], 1),
        )

    def forward(self, features: Features) -> torch.Tensor:
        memory, padding = self.tokenizer(features)
        for layer in self.encoder:
            memory = layer(memory, src_key_padding_mask=padding)
        memory = self.memory_norm(memory)
        known = torch.cat(
            [features.future, features.future_mask.float(), features.future_time],
            dim=-1,
        )
        index = torch.arange(self.horizon, device=memory.device)
        lead = ((index + 1) / self.horizon)[None, :, None].expand(
            memory.shape[0], -1, -1
        )
        decoded = self.query(torch.cat([known, lead], dim=-1)) + self.horizon_embedding(
            index
        )
        for layer in self.decoder:
            decoded = layer(decoded, memory, memory_key_padding_mask=padding)
        head_inputs = [self.output_norm(decoded), known]
        if self.use_recent:
            head_inputs.append(features.recent[:, None, :].expand(-1, self.horizon, -1))
        return self.head(torch.cat(head_inputs, dim=-1)).squeeze(-1)
