"""Resolve and validate the small, versioned Mojito configuration."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("config") / "configs.yaml"
TAIL_METHODS = ("missing", "hold", "zero", "previous_day", "trend")


def load_config(source=None) -> dict:
    """Merge section overrides without mutating the caller's configuration."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if isinstance(source, (str, Path)):
        source = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    for section, values in deepcopy(source or {}).items():
        if section not in cfg or not isinstance(values, dict):
            raise ValueError(f"Unknown or invalid config section: {section}")
        unknown = set(values) - set(cfg[section])
        if unknown:
            raise ValueError(f"Unknown {section} options: {sorted(unknown)}")
        cfg[section].update(values)

    ds, inf, model, aug, opt = (
        cfg[k]
        for k in ("dataset", "inference", "model", "augmentation", "optimization")
    )
    if inf["pred_len"] != 16:
        raise ValueError("M4-Mojito predicts exactly 16 steps at 15-minute intervals")
    integer_options = {
        "input_len": inf["input_len"],
        **{
            k: model[k]
            for k in (
                "d_model",
                "n_heads",
                "e_layers",
                "d_layers",
                "d_ff",
                "fast_len",
                "fast_patch",
                "slow_patch",
                "head_dim",
            )
        },
        **{
            k: opt[k]
            for k in (
                "batch_size",
                "eval_batch_size",
                "gradient_accumulation_steps",
                "patience",
            )
        },
    }
    for key, value in integer_options.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 16 <= model["fast_len"] <= inf["input_len"]:
        raise ValueError("Require 16 <= fast_len <= input_len")
    if model["fast_len"] % model["fast_patch"]:
        raise ValueError("fast_patch must divide fast_len")
    if model["use_slow"] and inf["input_len"] % model["slow_patch"]:
        raise ValueError("slow_patch must divide input_len")
    if model["d_model"] % model["n_heads"]:
        raise ValueError("n_heads must divide d_model")
    if not 0 <= model["dropout"] < 1 or not 0 <= aug["probability"] <= 1:
        raise ValueError("Invalid dropout or augmentation probability")
    if not aug["lengths"] or any(
        type(n) is not int or not 0 < n < inf["input_len"] for n in aug["lengths"]
    ):
        raise ValueError("Tail lengths must be positive integers below input_len")
    if not aug["methods"] or set(aug["methods"]) - set(TAIL_METHODS):
        raise ValueError(f"Tail methods must be drawn from {TAIL_METHODS}")
    if ds["timestamp_basis"] not in ("local_wall", "utc"):
        raise ValueError("timestamp_basis must be local_wall or utc")
    n_weather = len(ds["conv_cols"])
    if n_weather == 0 or any(
        len(ds[k]) != n_weather
        for k in (
            "conv_predict_cols",
            "weather_offsets",
            "weather_scales",
        )
    ):
        raise ValueError(
            "Weather columns, offsets and scales must have equal nonzero lengths"
        )
    if any(not math.isfinite(v) or v <= 0 for v in ds["weather_scales"]):
        raise ValueError("Weather scales must be finite and positive")
    if any(not math.isfinite(v) for v in ds["weather_offsets"]):
        raise ValueError("Weather offsets must be finite")
    columns = [
        ds["pv_col"],
        ds["capacity_col"],
        *ds["conv_cols"],
        *ds["conv_predict_cols"],
    ]
    reserved = [ds["target_col"], ds["timestamp_col"], ds["entity_col"]]
    if len(set(columns)) != len(columns) or set(columns) & set(reserved):
        raise ValueError(
            "Feature columns must be distinct and exclude target/time/entity columns"
        )
    return cfg


def feature_columns(cfg: dict) -> list[str]:
    ds = cfg["dataset"]
    return [
        ds["pv_col"],
        *ds["conv_cols"],
        *ds["conv_predict_cols"],
        ds["capacity_col"],
    ]
