"""Per-step quantile decoder with direct covariates and a residual MLP."""

import torch
from torch import nn
from torch.nn import functional as F


class StepCovHead(nn.Module):
    """Shared across future steps; output weights are distinct for each quantile.

    M4 encodes one token per channel and patch. Concatenate the encoded channels
    for the patch covering each future step, then append its within-patch
    position and input covariates (weather and the external PV forecast).
    """

    def __init__(
        self,
        channels,
        d_model,
        patch_num,
        patch_len,
        stride,
        context_window,
        prediction_length,
        quantiles,
        hidden_dim=256,
        position_dim=32,
        dropout=0.1,
    ):
        super().__init__()
        if not 0 < prediction_length <= context_window:
            raise ValueError("prediction_length must be within the input window")
        self.channels = channels
        self.d_model = d_model
        self.patch_num = patch_num
        self.prediction_length = prediction_length
        # For overlaps choose the latest-starting available patch covering t.
        steps = torch.arange(context_window - prediction_length, context_window)
        patch_index = (steps // stride).clamp(max=patch_num - 1)
        offsets = steps - patch_index * stride
        if ((offsets < 0) | (offsets >= patch_len)).any():
            raise ValueError(
                "step_cov requires every future step to be covered by a patch; "
                "adjust patch_len/stride or use padding_patch='end'"
            )
        self.register_buffer("patch_index", patch_index, persistent=False)
        self.register_buffer("position_index", offsets, persistent=False)
        self.position = nn.Embedding(patch_len, position_dim)
        nn.init.normal_(self.position.weight, std=0.02)
        width = channels * d_model + position_dim + channels
        self.fc1 = nn.Linear(width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, quantiles)
        self.skip = nn.Linear(width, quantiles)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoded, inputs):
        # encoded [B,C,D,N], inputs [B,C,L+H], using the same RevIN units.
        batch = encoded.shape[0]
        patch_features = encoded.permute(0, 3, 1, 2).reshape(
            batch, self.patch_num, self.channels * self.d_model
        )
        h = patch_features[:, self.patch_index]
        position = self.position(self.position_index).unsqueeze(0).expand(batch, -1, -1)
        covariates = inputs[:, :, -self.prediction_length :].transpose(1, 2)
        features = torch.cat([h, position, covariates], dim=-1)
        hidden = self.fc1(features)
        quantiles = self.fc2(self.dropout(F.gelu(hidden))) + self.skip(features)
        # Preserve the backbone's flattened output interface: [B,H*K].
        return quantiles.flatten(1), hidden.mean(dim=1)
