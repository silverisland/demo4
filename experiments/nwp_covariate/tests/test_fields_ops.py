# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

import fields as F
import ops
from conftest import C, make_ds, short_issue_times
from features import load_config, make_ctx


def test_whitelist_is_128_and_paired():
    cols = F.ALLOWED_COLUMNS
    assert len(cols) == 128 and len(set(cols)) == 128
    for c in cols:
        if c in F.ID_COLS or c in (F.TARGET, F.LABEL, F.CAP) or c in F.HISTORY_ONLY or c.endswith(F.FUTURE_SUFFIX):
            continue
        assert c + F.FUTURE_SUFFIX in cols, c


@pytest.mark.parametrize("col,ho", [("observe_power_future", False), ("EC_JTKRD7RO_GHI_p13_predict", False),
                                    ("EC_JTKRD7RO_v_hl_1000_p13", False), ("GHI_real", False),
                                    ("GHI_SOLARGIS", True), ("cap_power_on", False)])
def test_check_raw_rejects(col, ho):
    with pytest.raises(ValueError):
        F.check_raw(col, ho)


def test_agg_matches_online_rules(cfg):
    """格点 9 点中位数 + general_GHI/TEMP 跨源中位数，与线上 feature_aggregation + RULES 逐位一致。"""
    from nanwang_online.feature_aggregation import aggregate_features
    from nanwang_online.weather_aggregation_rules_v2_1 import RULES
    t = short_issue_times(days=6)
    df = make_ds(t, ["st_a", "st_b"], 192, seed=5, drop_ec_day=pd.Timestamp("2026-08-03")).drop(columns="update_time")
    # 再造一行"只剩 1 个格点有效"（min_valid=2 应整行置 NaN）
    for p in F.POINTS[1:]:
        df.at[0, F.grid_col("GDFS", "ghi", p)] = np.full(C, np.nan, np.float32)
    online = aggregate_features(df, RULES)
    cfg["data"]["pred_len"] = 192
    ctx = make_ctx(df.reset_index(drop=True), cfg)
    for mine, theirs in [("general_ghi", "general_GHI"), ("general_temp", "general_TEMP"),
                         ("ghi_ec", "EC_JTKRD7RO_GHI"), ("ghi_gdfs", "GDFS_JTHFGNED_MID_JTNWQISX_GHI"),
                         ("temp_ss", "solar_southern_TEMP")]:
        p = ctx.get(mine)
        h = np.stack([np.asarray(v, float) if v is not None and np.ndim(v) else np.full(C, np.nan)
                      for v in online[theirs]])
        f = np.stack([np.asarray(v, float) if v is not None and np.ndim(v) else np.full(192, np.nan)
                      for v in online[theirs + "_predict"]])
        np.testing.assert_allclose(p.hist, h, rtol=1e-6, equal_nan=True, err_msg=mine)
        np.testing.assert_allclose(p.fut, f, rtol=1e-6, equal_nan=True, err_msg=mine)
    assert np.isnan(ctx.get("ghi_gdfs").hist[0]).all()


def test_clear_sky_and_kt():
    ts = ops.window_timestamps(np.array(["2026-06-21T12:00:00"], dtype="datetime64[s]"), 96, 0)
    cs = ops.clear_sky_ghi(ts, np.array([23.0]), np.array([108.5]))[0]
    hours = pd.DatetimeIndex(ts[0]).hour
    assert cs[hours == 3].max() == 0 and cs.max() > 900
    noon = int(np.argmax(cs)); assert 11 <= hours[noon] <= 13


def test_inv_error_prefers_accurate_source():
    B, Cn, P = 4, 96 * 3, 96
    ts = np.array(["2026-08-10T10:00:00"] * B, dtype="datetime64[s]")
    full_ts = ops.window_timestamps(ts, Cn, P)
    clear = ops.clear_sky_ghi(full_ts, np.full(B, 23.0), np.full(B, 108.5))
    truth = clear * 0.7
    rng = np.random.default_rng(0)
    good = truth + rng.normal(0, 10, truth.shape)
    bad = truth * 1.6 + rng.normal(0, 80, truth.shape)
    cols = {"good": ops.Pair.from_full(good, Cn), "bad": ops.Pair.from_full(bad, Cn),
            "ref": ops.Pair.from_full(truth, Cn)}
    ctx = ops.Ctx(ts_win=ts, context_len=Cn, pred_len=P, lat=np.full(B, 23.0), lon=np.full(B, 108.5),
                  cap=np.full(B, 50.0), load_raw=None, cols=dict(cols))
    inv = ops.op_combine(ctx, ["good", "bad"], method="inv_error", reference="ref")
    mean = ops.op_combine(ctx, ["good", "bad"], method="mean")
    tf = truth[:, Cn:]
    err = lambda a: np.sqrt(np.nanmean((a - tf) ** 2))  # noqa: E731
    assert err(inv.fut) < 0.5 * err(mean.fut)
    bias = ops.op_combine(ctx, ["good", "bad"], method="bias_mean", reference="ref")
    assert err(bias.fut) < err(mean.fut)


def test_trim_mean_and_std():
    st = np.array([[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]], [[100.0, np.nan]]])
    tm = ops.agg_members(st, "trim_mean")
    assert tm[0, 0] == pytest.approx(2.5) and tm[0, 1] == pytest.approx(2.0)
    assert ops.agg_members(st, "std")[0, 1] == pytest.approx(np.std([1, 2, 3]))


def test_inv_error_warns_with_location():
    B, Cn, P = 3, 96 * 2, 96
    ts = np.array(["2026-08-10T10:00:00"] * B, dtype="datetime64[s]")
    full_ts = ops.window_timestamps(ts, Cn, P)
    clear = ops.clear_sky_ghi(full_ts, np.full(B, 23.0), np.full(B, 108.5))
    ref = ops.Pair.from_full(clear * 0.7, Cn)
    ref.hist[1] = np.nan                                                # 第 2 行参考全缺
    a = ops.Pair.from_full(clear * 0.7, Cn); b = ops.Pair.from_full(clear * 1.3, Cn)
    ctx = ops.Ctx(ts_win=ts, context_len=Cn, pred_len=P, lat=np.full(B, 23.0), lon=np.full(B, 108.5),
                  cap=np.full(B, 50.0), load_raw=None, cols={"a": a, "b": b, "ref": ref},
                  specs={"x": {"op": "combine", "members": ["a", "b"], "method": "inv_error", "reference": "ref"}},
                  stations=np.array(["s0", "s1", "s2"]))
    out = ctx.get("x")
    assert len(ctx.warnings) == 1 and ctx.warnings[0].startswith("[x] inv_error: 1/3") and "s1@" in ctx.warnings[0]
    mid = (out.fut[1] - (a.fut[1] + b.fut[1]) / 2)                     # 退化行 = 等权均值
    assert np.nanmax(np.abs(mid)) < 1e-9
    assert np.nanmean(np.abs(out.fut[0] - a.fut[0])) < np.nanmean(np.abs(out.fut[0] - b.fut[0]))
