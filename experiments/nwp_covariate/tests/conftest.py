# -*- coding: utf-8 -*-
"""合成 DS 轨迹表（只含白名单 128 列 + update_time），供本目录测试使用。

物理上大致合理：Haurwitz 晴空 × 逐日云量 -> 真实 GHI -> 功率；各源 = 真实 GHI × 偏差 + 噪声，
9 个格点在源值上再加扰动。故意制造：observe_power 尾部缺失、某站某天 EC 格点整源缺失。
为控制体积，测试用 context_len=288（真实 672），短期未来段保持 480 点以验证截断。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fields as F  # noqa: E402
from ops import clear_sky_ghi  # noqa: E402

C = 288
LAT, LON = 23.0, 108.5
STEP = pd.Timedelta(minutes=15)
SRC_BIAS = {"EC": (1.05, 40.0), "GDFS": (0.90, 80.0), "SS": (1.10, 110.0)}
SG_BIAS = (1.0, 60.0)


def _smooth(x, k=6):
    return np.convolve(x, np.ones(k) / k, mode="same")


def make_series(t0, n, seed):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(t0, periods=n, freq="15min").values.astype("datetime64[s]")
    clear = clear_sky_ghi(ts[None, :], np.array([LAT]), np.array([LON]))[0]
    days = pd.DatetimeIndex(ts).normalize()
    uday = days.unique()
    kd = dict(zip(uday, rng.uniform(0.15, 1.0, len(uday))))
    kt = np.array([kd[d] for d in days]) * (1 + _smooth(rng.normal(0, 0.15, n)))
    ghi = np.clip(clear * np.clip(kt, 0.05, 1.1), 0, None)
    hours = pd.DatetimeIndex(ts).hour.to_numpy() + pd.DatetimeIndex(ts).minute.to_numpy() / 60
    temp = 24 + 6 * np.sin((hours - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.5, n)
    s = {"ts": ts, "clear": clear, "ghi": ghi, "temp": temp}
    for src, (b, sd) in SRC_BIAS.items():
        base = np.clip(ghi * b + _smooth(rng.normal(0, sd, n)), 0, None) * (clear > 0)
        s[f"{src}_ghi_pts"] = [np.clip(base * (1 + rng.normal(0, 0.03)) + _smooth(rng.normal(0, sd / 3, n)), 0, None)
                               * (clear > 0) for _ in F.POINTS]
        s[f"{src}_temp_pts"] = [temp + rng.normal(0, 0.8) + rng.normal(0, 0.3, n) for _ in F.POINTS]
    s["SG_ghi"] = np.clip(ghi * SG_BIAS[0] + _smooth(rng.normal(0, SG_BIAS[1], n)), 0, None) * (clear > 0)
    s["SG_temp"] = temp + rng.normal(0, 0.5, n)
    s["SG_other"] = {c: rng.uniform(0, 10, n) for c in ("WS_SOLARGIS", "WD_SOLARGIS", "PREC_SOLARGIS",
                                                         "PWAT_SOLARGIS", "SDWE_SOLARGIS")}
    s["SG_other"]["WD_SOLARGIS"] = rng.uniform(0, 360, n)
    s["ghi_real"] = np.clip(ghi + rng.normal(0, 15, n), 0, None) * (clear > 0)
    return s


def make_ds(issue_times, stations, pf_len, seed=0, ctx=C, nan_tail_every=7, drop_ec_day=None):
    """issue_times: 起报时刻列表；每站每个起报时刻一行。返回 DataFrame（列 = 白名单 + update_time）。"""
    t_min, t_max = min(issue_times), max(issue_times)
    t0 = t_min - (ctx - 1) * STEP
    n = int((t_max - t0) / STEP) + pf_len + 1
    rows = []
    for si, st in enumerate(stations):
        cap = 50.0 + 25 * si
        s = make_series(t0, n, seed + si)
        power = cap * s["ghi"] / 1000 * 0.85 * (1 - 0.004 * (s["temp"] + s["ghi"] / 25 - 25))
        power = np.clip(power, 0, None)
        for k, S in enumerate(issue_times):
            i = int((S - t0) / STEP)
            h = slice(i - ctx + 1, i + 1); f = slice(i + 1, i + 1 + pf_len)
            r = {"station": st, "timestamp_win": S, "update_time": S}
            op = power[h].astype(np.float32).copy()
            if nan_tail_every and k % nan_tail_every == 3:
                op[-3:] = np.nan
            r["observe_power"] = op
            r["GHI_real"] = s["ghi_real"][h].astype(np.float32)
            r["cap_power_on"] = np.full(ctx, cap, np.float32)
            r["observe_power_future"] = power[f].astype(np.float32)

            def put(col, arr):
                r[col] = arr[h].astype(np.float32)
                r[col + F.FUTURE_SUFFIX] = arr[f].astype(np.float32)
            put("GHI_SOLARGIS", s["SG_ghi"]); put("TEMP_SOLARGIS", s["SG_temp"])
            for c, v in s["SG_other"].items():
                put(c, v)
            for src in F.GRID_SOURCES:
                for j, p in enumerate(F.POINTS):
                    put(F.grid_col(src, "ghi", p), s[f"{src}_ghi_pts"][j])
                    put(F.grid_col(src, "temp", p), s[f"{src}_temp_pts"][j])
            if drop_ec_day is not None and st == stations[0] and S.normalize() == drop_ec_day:
                for p in F.POINTS:
                    for v in ("ghi", "temp"):
                        c = F.grid_col("EC", v, p)
                        r[c] = np.full(ctx, np.nan, np.float32)
                        r[c + F.FUTURE_SUFFIX] = np.full(pf_len, np.nan, np.float32)
            rows.append(r)
    df = pd.DataFrame(rows)
    return df[["update_time"] + [c for c in F.ALLOWED_COLUMNS]]


def write_ds(df, root: Path, predict_type: str):
    for d, g in df.groupby(df["timestamp_win"].dt.normalize()):
        out = root / f"predict_type={predict_type}" / f"date={d.date()}"
        out.mkdir(parents=True, exist_ok=True)
        g.to_parquet(out / "part.parquet", index=False)


def short_issue_times(start="2026-08-01", days=14):
    out = []
    for d in pd.date_range(start, periods=days, freq="1D"):
        out += [d + pd.Timedelta(hours=4), d + pd.Timedelta(hours=10)]
    return out


def ultra_issue_times(start="2026-08-01", days=3):
    out = []
    for d in pd.date_range(start, periods=days, freq="1D"):
        out += list(pd.date_range(d + pd.Timedelta(hours=5), d + pd.Timedelta(hours=20), freq="15min"))
    return out


@pytest.fixture(scope="session")
def ds_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds")
    write_ds(make_ds(short_issue_times(), ["st_a", "st_b", "st_c"], 480, seed=1,
                     drop_ec_day=pd.Timestamp("2026-08-05")), root, "short")
    write_ds(make_ds(ultra_issue_times(), ["st_a", "st_b"], 192, seed=11), root, "ultra_short")
    return root


@pytest.fixture
def cfg(ds_root, tmp_path):
    c = yaml.safe_load(open(HERE / "config.yaml", encoding="utf-8"))
    c = copy.deepcopy(c)
    c["run"].update({"out_dir": str(tmp_path / "runs" / "{predict_type}"), "backend": "mock"})
    c["data"].update({"ds_glob": str(ds_root / "predict_type={predict_type}" / "date=*" / "*.parquet"),
                      "feature_store": str(tmp_path / "fs" / "{predict_type}"), "context_len": C,
                      "station_meta": None, "default_latlon": [LAT, LON], "area": "province_x"})
    c["evaluation"]["keep_rule"]["bootstrap"] = 200
    c["rank"].update({"max_rows": 50000, "n_repeats": 2})
    c["rank"]["lgbm"].update({"min_data_in_leaf": 20, "num_boost_round": 120, "early_stopping_rounds": 20})
    return c
