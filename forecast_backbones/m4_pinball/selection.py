"""Standalone D+1 checkpoint scoring, matching M8's station-day mean."""

from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Subset

METRIC = "nanwang_short_d1"
ISSUE_TIME = pd.Timedelta(hours=10)
STEP = pd.Timedelta(minutes=15)


class ValidationRows(Subset):
    """Indexed view preserving row metadata for numpy and Arrow datasets."""

    @property
    def timestamp(self):
        return self.dataset.timestamp[self.indices]

    @property
    def entity_cols(self):
        return self.dataset.entity_cols

    def column(self, name):
        return self.dataset.column(name)[self.indices]


def metadata(dataset, config):
    station_col = config["dataset"]["entity_col"]
    if station_col not in dataset.entity_cols:
        raise ValueError("D+1 selection requires an explicit station column")
    times = pd.DatetimeIndex(dataset.timestamp)
    stations = np.asarray(dataset.column(station_col))
    if times.hasnans or pd.isna(stations).any():
        raise ValueError(
            "D+1 selection requires finite timestamps and station identities"
        )
    return pd.DataFrame({"station": stations, "issue": times})


def d1_mask(issue, horizon):
    """Element zero is issue + 15 minutes, not the issue itself."""
    targets = pd.date_range(issue + STEP, periods=horizon, freq=STEP)
    start = issue.normalize() + pd.Timedelta(days=1)
    return (targets >= start) & (targets < start + pd.Timedelta(days=1))


def select_validation_rows(dataset, config):
    """Select before checking labels: never fall back to an earlier, easier row."""
    rows = metadata(dataset, config)
    rows["index"] = np.arange(len(rows))
    rows["day"] = rows["issue"].dt.normalize()
    rows = rows.loc[(rows["issue"] - rows["day"]) <= ISSUE_TIME]
    rows = rows.sort_values(["issue", "index"], kind="stable")
    rows = rows.drop_duplicates(["station", "issue"], keep="first")
    rows = rows.groupby(["station", "day"], sort=False).tail(1)
    target_col = config["dataset"]["target_col"]
    indices = []
    for row in rows.itertuples(index=False):
        target = np.asarray(dataset[int(row.index)]["labels"][target_col])
        if np.isfinite(target[d1_mask(row.issue, len(target))]).any():
            indices.append(int(row.index))
    if not indices:
        raise ValueError(
            "No usable D+1 validation rows: need an issue at/before 10:00, "
            "a horizon reaching the next day, and finite next-day labels"
        )
    return ValidationRows(dataset, sorted(indices))


def decision_metrics(predictions, targets, capacities, rows):
    """Score already-selected rows; finite D+1 points only, like M8."""
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    capacities = np.asarray(capacities).reshape(-1)
    if (
        predictions.shape != targets.shape
        or predictions.ndim != 2
        or len(predictions) != len(rows)
        or len(capacities) != len(rows)
    ):
        raise ValueError("D+1 predictions, labels, capacities and metadata must align")
    if not np.isfinite(capacities).all() or (capacities <= 0).any():
        raise ValueError("D+1 capacities must be finite and positive")
    if not np.isfinite(predictions).all():
        raise ValueError("D+1 predictions must be finite")
    frame = rows.copy()
    frame["cap"] = capacities
    for _, group in frame.groupby("station", sort=False):
        if not np.allclose(group["cap"], group["cap"].iloc[0], rtol=1e-6, atol=0):
            raise ValueError("D+1 selection requires a fixed capacity per station")
    scores, counts = [], []
    for i, issue in enumerate(rows["issue"]):
        mask = d1_mask(issue, predictions.shape[1]) & np.isfinite(targets[i])
        if not mask.any():
            continue
        y, p = targets[i, mask].astype(float), predictions[i, mask].astype(float)
        error = (y - p) / np.maximum(y, 0.2 * capacities[i])
        scores.append(max(0.0, 100.0 * (1.0 - np.sqrt(np.mean(error**2)))))
        counts.append(int(mask.sum()))
    if not scores:
        raise ValueError("No usable D+1 validation targets")
    return {
        METRIC: float(np.mean(scores)),
        "nanwang_short_d1_station_days": len(scores),
        "nanwang_short_d1_points": sum(counts),
    }
