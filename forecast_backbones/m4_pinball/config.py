"""Independent M4-Pinball configuration; does not mutate M4 or caller dictionaries."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from .decision import validate_levels, validate_point_config

DEFAULT_CONFIG = Path(__file__).with_name("config") / "configs.yaml"


def load_config(source=None):
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if isinstance(source, (str, Path)):
        source = yaml.safe_load(Path(source).read_text(encoding="utf-8"))

    def merge(base, overrides):
        for key, value in overrides.items():
            if key not in base:
                raise ValueError(f"Unknown m4_pinball option: {key}")
            if isinstance(base[key], dict) and base[key]:
                if not isinstance(value, dict):
                    raise ValueError(f"{key} must be a mapping")
                merge(base[key], value)
            else:
                base[key] = deepcopy(value)

    merge(cfg, source or {})
    ds, inf, m = (cfg[k] for k in ("dataset", "inference", "model"))
    for key, value in {
        **inf,
        **{
            k: m[k]
            for k in (
                "enc_in",
                "patch_len",
                "stride",
                "d_model",
                "n_heads",
                "e_layers",
                "d_ff",
                "embedding_dim",
            )
        },
    }.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if m["patch_len"] > inf["input_len"] + inf["pred_len"]:
        raise ValueError("patch_len exceeds the combined input window")
    if m["d_model"] % m["n_heads"]:
        raise ValueError("n_heads must divide d_model")
    if not ds["conv_cols"] or len(ds["conv_cols"]) != len(ds["conv_predict_cols"]):
        raise ValueError(
            "Historical and future weather columns must have equal nonzero lengths"
        )
    if m["enc_in"] != len(ds["conv_cols"]) + 1:
        raise ValueError(
            "enc_in must equal the weather channel count plus one PV channel"
        )
    if m["loss_type"] != "pinball":
        raise ValueError("m4_pinball only supports loss_type=pinball")
    validate_levels(m["quantile_levels"])
    validate_point_config(cfg["decision"]["point"])
    ph = cfg["decision"]["physics"]
    if not np.isfinite(ph["clear_sky_cap_ratio"]) or ph["clear_sky_cap_ratio"] <= 0:
        raise ValueError("clear_sky_cap_ratio must be finite and positive")
    mapping = ds["capacity_by_station"]
    if not isinstance(mapping, dict):
        raise TypeError("capacity_by_station must be a mapping")
    if any(not np.isfinite(v) or v <= 0 for v in mapping.values()):
        raise ValueError("Station capacities must be finite and positive")
    ds["capacity_by_station"] = {str(k): float(v) for k, v in mapping.items()}
    features = feature_columns(cfg)
    if len(features) != len(set(features)) or set(features) & {
        ds["target_col"],
        ds["timestamp_col"],
        ds["entity_col"],
    }:
        raise ValueError(
            "Input columns must be distinct and exclude target/time/entity columns"
        )
    return cfg


def feature_columns(cfg):
    ds, ph = cfg["dataset"], cfg["decision"]["physics"]
    cols = [*ds["conv_cols"], *ds["conv_predict_cols"], ds["pv_col"], ds["tsfm_col"]]
    if not ds["capacity_by_station"]:
        cols.append(ds["capacity_col"])
    if ph["clear_sky_cap"]:
        cols.append(ds["p_clear_col"])
    if ph["night_zero"]:
        cols.append(ds["night_col"])
    return cols
