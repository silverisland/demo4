# -*- coding: utf-8 -*-
"""evaluate.py —— 南网逐日指标、配对检验（按天分块 bootstrap）、按天气档分解。

主指标按预报类型：超短期 nanwang_ultra_short（MAE 型，前 16 步），短期 nanwang_short_D+1（RMSE 型，10:00 起报的 D+1）。
配对单位是"站 × 日"；bootstrap 按天整块重抽（同一天各站共享天气，不是独立样本）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (str(HERE), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fields as F  # noqa: E402
from ops import clear_sky_ghi  # noqa: E402

METRIC_BY_TYPE = {"ultra_short": "nanwang_ultra_short", "short": "nanwang_short_D+1"}


def primary_metric(cfg: dict, predict_type: str) -> str:
    return (cfg.get("evaluation") or {}).get("primary_by_type", METRIC_BY_TYPE).get(predict_type, METRIC_BY_TYPE[predict_type])


def result_frame(table: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    r = table[["timestamp_win", "station"]].copy()
    r["station"] = r["station"].astype(str)
    r["observe_power_predict"] = list(np.asarray(pred, dtype=np.float32))
    r[F.LABEL] = table[F.LABEL].to_numpy()
    return r


def gccap_of(table: pd.DataFrame) -> Dict[str, float]:
    g = table.groupby(table["station"].astype(str))["cap_power_on_val"].max()
    return {k: float(v) for k, v in g.items()}


def daily_metric(result: pd.DataFrame, gccap: Dict[str, float], metric: str) -> pd.DataFrame:
    """-> DataFrame[station, date, accuracy_pct]"""
    from nanwang_metric import NanwangMetric      # 本目录副本（复制自 pipelines/nanwang_metric.py）
    nm = NanwangMetric()
    if metric == "nanwang_short_D+1":
        t = nm.short_d1_daily(result, gccap)
    elif metric == "nanwang_ultra_short":
        t = nm.ultra_short_daily(result, gccap)
    else:
        raise ValueError(f"未知指标 {metric!r}")
    t = t[["station", "date", "accuracy_pct"]].copy()
    t["station"] = t["station"].astype(str)
    t["date"] = pd.to_datetime(t["date"]).dt.normalize()
    return t


def fleet_mean(daily: pd.DataFrame) -> Optional[float]:
    """先站内逐日平均，再跨站平均（与月度表"每站 mean 再看全站"同向，便于横向比较）。"""
    if daily is None or not len(daily):
        return None
    return float(daily.groupby("station")["accuracy_pct"].mean().mean())


def paired(base: pd.DataFrame, exp: pd.DataFrame, rules: Optional[dict] = None, seed: int = 0) -> dict:
    """exp − base 的逐站逐日配对比较。rules: min_mean_gain / min_station_win_frac / bootstrap / ci。"""
    rules = rules or {}
    m = base.merge(exp, on=["station", "date"], suffixes=("_a", "_b"))
    out = {"n_pairs": int(len(m)), "n_days": int(m["date"].nunique()) if len(m) else 0,
           "n_stations": int(m["station"].nunique()) if len(m) else 0}
    if not len(m):
        return {**out, "mean_gain": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "station_win_frac": np.nan,
                "passed": False, "reason": "无配对样本"}
    m["diff"] = m["accuracy_pct_b"] - m["accuracy_pct_a"]
    mean_gain = float(m["diff"].mean())
    # 按天分块 bootstrap：每次有放回抽天，抽到的天带上它所有站
    by_day = m.groupby("date")["diff"].agg(["sum", "count"])
    sums, cnts = by_day["sum"].to_numpy(), by_day["count"].to_numpy()
    rng = np.random.default_rng(seed)
    nb = int(rules.get("bootstrap", 2000))
    idx = rng.integers(0, len(sums), size=(nb, len(sums)))
    boot = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    ci = float(rules.get("ci", 0.95))
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    st = m.groupby("station")["diff"].mean()
    win = float((st >= 0).mean())
    need_gain = float(rules.get("min_mean_gain", 0.3))
    need_win = float(rules.get("min_station_win_frac", 0.65))
    reasons = []
    if mean_gain < need_gain:
        reasons.append(f"平均提升 {mean_gain:.3f} < {need_gain}")
    if not lo > 0:
        reasons.append(f"置信区间下界 {lo:.3f} ≤ 0")
    if win < need_win:
        reasons.append(f"站胜率 {win:.2f} < {need_win}")
    return {**out, "mean_gain": mean_gain, "ci_lo": float(lo), "ci_hi": float(hi), "station_win_frac": win,
            "passed": not reasons, "reason": "；".join(reasons) if reasons else "通过"}


# ----------------------------------------------------------------------------- 天气档（诊断用，用真值算）
def day_classes(table: pd.DataFrame, edges=(0.35, 0.75), min_clear: float = 50.0, tz_offset_h: float = 8.0) -> pd.DataFrame:
    """每站每个目标日的实测晴空指数 = Σ实测功率 / Σ(容量 × 晴空辐照 / 1000)，只算晴空辐照 ≥ min_clear 的时刻。
    -> DataFrame[station, date, kt_obs, weather]（weather ∈ 阴 / 多云 / 晴）。只用于结果分解，不进模型。"""
    t = table.sort_values(["station", "timestamp_win"])
    step = t.groupby("station")["timestamp_win"].diff().median()
    lead_n = 1 if (pd.notna(step) and step <= pd.Timedelta(minutes=15)) else 96
    recs = []
    for st, g in t.groupby(t["station"].astype(str)):
        lab = np.stack([np.asarray(v, float)[:lead_n] for v in g[F.LABEL].to_numpy()])
        S = g["timestamp_win"].values.astype("datetime64[s]").astype("int64")[:, None]
        ts = (S + np.arange(1, lead_n + 1)[None, :] * 900).astype("datetime64[s]")
        lat = np.full(len(g), g["lat"].iloc[0] if "lat" in g else np.nan, float)
        lon = np.full(len(g), g["lon"].iloc[0] if "lon" in g else np.nan, float)
        if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
            continue
        cs = clear_sky_ghi(ts, lat, lon, tz_offset_h)
        cap = g["cap_power_on_val"].to_numpy(float)[:, None] * np.ones_like(cs)
        recs.append(pd.DataFrame({"station": st, "ts": ts.ravel(), "y": lab.ravel(),
                                  "pc": (cap * cs / 1000.0).ravel(), "cs": cs.ravel()}))
    if not recs:
        return pd.DataFrame(columns=["station", "date", "kt_obs", "weather"])
    d = pd.concat(recs).drop_duplicates(["station", "ts"])
    d = d[(d["cs"] >= min_clear) & np.isfinite(d["y"])]
    d["date"] = pd.to_datetime(d["ts"]).dt.normalize()
    agg = d.groupby(["station", "date"]).agg(y=("y", "sum"), pc=("pc", "sum")).reset_index()
    agg["kt_obs"] = agg["y"] / agg["pc"].where(agg["pc"] > 0)
    lo, hi = edges
    agg["weather"] = np.where(agg["kt_obs"] < lo, "阴", np.where(agg["kt_obs"] < hi, "多云", "晴"))
    return agg[["station", "date", "kt_obs", "weather"]]


def breakdown(base: pd.DataFrame, exp: pd.DataFrame, classes: pd.DataFrame, predict_type: str) -> pd.DataFrame:
    """配对差按天气档分解。短期日指标的 date 是起报日 D，打分的是 D+1，这里对齐到 D+1。"""
    m = base.merge(exp, on=["station", "date"], suffixes=("_a", "_b"))
    if not len(m) or classes is None or not len(classes):
        return pd.DataFrame(columns=["weather", "n", "mean_gain", "acc_base", "acc_exp"])
    m["target_date"] = m["date"] + (pd.Timedelta(days=1) if predict_type == "short" else pd.Timedelta(0))
    m = m.merge(classes.rename(columns={"date": "target_date"}), on=["station", "target_date"], how="left")
    m["weather"] = m["weather"].fillna("未知")
    m["diff"] = m["accuracy_pct_b"] - m["accuracy_pct_a"]
    return (m.groupby("weather").agg(n=("diff", "size"), mean_gain=("diff", "mean"),
                                     acc_base=("accuracy_pct_a", "mean"), acc_exp=("accuracy_pct_b", "mean"))
            .reset_index())
