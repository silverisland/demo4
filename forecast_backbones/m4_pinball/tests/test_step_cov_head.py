"""Step alignment, direct covariate path and quantile learning."""

import pytest
import torch

from forecast_backbones.m4_pinball.config import load_config
from forecast_backbones.m4_pinball.layers.step_cov_head import StepCovHead
from forecast_backbones.m4_pinball.tests.test_pinball import make_batch
from forecast_backbones.m4_pinball.transformer_models import (
    TransformersAdaptedM4PinballModel,
)


@pytest.mark.parametrize(
    "patch_len,stride,patch_num,window,horizon,expected",
    [
        (4, 4, 3, 12, 6, [1, 1, 2, 2, 2, 2]),  # Future begins within a patch.
        (4, 2, 5, 12, 6, [3, 3, 4, 4, 4, 4]),  # Overlapping patches.
        (4, 4, 3, 10, 4, [1, 1, 2, 2]),  # Replicated end padding.
    ],
)
def test_future_steps_use_the_correct_encoded_patch(
    patch_len, stride, patch_num, window, horizon, expected
):
    head = StepCovHead(
        channels=2,
        d_model=1,
        patch_num=patch_num,
        patch_len=patch_len,
        stride=stride,
        context_window=window,
        prediction_length=horizon,
        quantiles=3,
        hidden_dim=4,
        position_dim=2,
        dropout=0.0,
    )
    with torch.no_grad():
        for param in head.parameters():
            param.zero_()
        # Read encoded channel 0 and channel 1 separately, proving flatten order.
        head.skip.weight[0, 0] = 1
        head.skip.weight[1, 1] = 1
    encoded = torch.stack(
        [
            torch.arange(patch_num),
            torch.arange(patch_num) + 10,
        ]
    ).float()[None, :, None, :]
    q, _ = head(encoded, torch.zeros(1, 2, window))
    q = q.reshape(1, horizon, 3)
    torch.testing.assert_close(q[0, :, 0], torch.tensor(expected).float())
    torch.testing.assert_close(q[0, :, 1], torch.tensor(expected).float() + 10)
    assert (q[0, :, 2] == 0).all()


def test_future_covariates_directly_affect_only_the_corresponding_step():
    head = StepCovHead(3, 2, 3, 4, 4, 12, 4, 9, hidden_dim=8, position_dim=2, dropout=0)
    with torch.no_grad():
        for param in head.parameters():
            param.zero_()
        head.skip.weight[:, -3] = 1  # First weather channel.
        head.skip.weight[:, -1] = 2  # External PV forecast channel.
    encoded = torch.zeros(1, 3, 2, 3)
    inputs = torch.zeros(1, 3, 12)
    inputs[0, 0, 9] = 5
    inputs[0, 2, 10] = 7
    q, _ = head(encoded, inputs)
    expected = torch.tensor([0.0, 5.0, 14.0, 0.0])[:, None].expand(4, 9)
    torch.testing.assert_close(q.reshape(4, 9), expected)


@pytest.mark.parametrize("revin,affine", [(False, False), (True, False), (True, True)])
def test_step_head_quantile_gradients_and_label_independence(revin, affine):
    cfg = load_config(
        {
            "inference": {"input_len": 32, "pred_len": 16},
            "model": {
                "prediction_head": "step_cov",
                "patch_len": 8,
                "stride": 8,
                "d_model": 16,
                "d_ff": 32,
                "n_heads": 2,
                "e_layers": 1,
                "step_head_hidden": 24,
                "step_head_position_dim": 4,
                "dropout": 0.0,
                "head_dropout": 0.0,
                "revin": revin,
                "affine": affine,
            },
        }
    )
    model = TransformersAdaptedM4PinballModel(cfg).eval()
    batch = make_batch()
    target = torch.full((3, 16), 1e4)
    target[0, 0] = float("nan")
    out = model(batch, {"observe_power_future": target})
    assert out.aux["quantiles"].shape == (3, 16, 9)
    assert out.embeddings.shape == (3, 24)
    torch.testing.assert_close(out.logits, model(batch).logits)
    out.loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    # Every output gets its own tau-weighted gradient (target above all outputs).
    grad = model.model.head.fc2.bias.grad
    torch.testing.assert_close(grad / grad[-1], torch.arange(1, 10) / 9)
    assert (model.model.head.skip.weight.grad.abs().sum(dim=1) > 0).all()


def test_default_192_step_head_and_invalid_geometry():
    cfg = load_config({"model": {"prediction_head": "step_cov"}})
    model = TransformersAdaptedM4PinballModel(cfg).eval()
    with torch.no_grad():
        out = model(make_batch(1, history=672, future=192))
    assert out.aux["quantiles"].shape == (1, 192, 9)
    assert out.logits.shape == (1, 192)
    assert model.model.head.fc1.in_features == 3 * 256 + 32 + 3
    # Stride leaves uncovered future steps.
    with pytest.raises(ValueError, match="covered by a patch"):
        StepCovHead(3, 8, 2, 4, 8, 16, 8, 9)
    with pytest.raises(ValueError, match="prediction_head"):
        load_config({"model": {"prediction_head": "unknown"}})
    with pytest.raises(ValueError, match="step_head_hidden"):
        load_config({"model": {"step_head_hidden": 0}})
