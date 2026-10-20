"""Registry adapter: Model(config), forward(batch_dict, labels=None)."""

from __future__ import annotations

from torch import nn

from interfaces.modeloutput import TransformersAdaptedModelOutput

from .config import load_config
from .features import prediction_loss, prepare_features
from .model import Mojito


class TransformersAdaptedMojitoModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = load_config(config)
        self.prediction_length = self.config["inference"]["pred_len"]
        self.model = Mojito(self.config)

    def forward(self, batch_dict: dict, labels: dict | None = None):
        features = prepare_features(
            batch_dict,
            self.config,
            next(self.parameters()).device,
            training=self.training,
        )
        normalized = self.model(features)
        predictions = normalized * features.capacity
        loss = None
        if labels is not None:
            loss = prediction_loss(
                normalized,
                labels[self.config["dataset"]["target_col"]],
                features.capacity,
            )
        return TransformersAdaptedModelOutput(
            loss=loss,
            logits=predictions,
            predictions=predictions,
            aux={
                "history_unavailable": ~features.history_mask[:, :, 0].any(dim=1),
                "future_weather_unavailable": ~features.future_mask.flatten(1).any(
                    dim=1
                ),
            },
        )
