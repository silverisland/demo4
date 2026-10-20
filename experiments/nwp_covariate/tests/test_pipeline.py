# -*- coding: utf-8 -*-
import copy
import json
import logging
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import fields as F
from backends import M5Backend, MockBackend, postprocess, repair_tail
from features import build, feature_store_dir, read_feature_store, write_feature_store


def _build(cfg, pt):
    table, rep = build(cfg, pt)
    write_feature_store(table, feature_store_dir(cfg, pt), rep)
    return table, rep


@pytest.mark.parametrize("pt", ["short", "ultra_short"])
def test_build_feature_store(cfg, pt):
    table, rep = _build(cfg, pt)
    assert rep["failed"] == {}, rep["failed"]
    assert rep["missing_allowed_columns"] == []
    assert len(np.asarray(table["general_ghi"].iloc[0])) == cfg["data"]["context_len"]
    assert len(np.asarray(table["general_ghi_predict"].iloc[0])) == cfg["data"]["pred_len"]
    assert "ghi_real_predict" not in table.columns                      # history_only 不生成未来列
    assert (table["cap_power_on_val"] > 0).all()
    back = read_feature_store(cfg, pt)
    assert len(back) == len(table)


def test_postprocess_night_zero(cfg):
    ts = np.array(["2026-08-01T10:00:00"], dtype="datetime64[s]")
    out = postprocess(np.full((1, 192), -1.0) + 2, ts, cfg)
    stamps = pd.date_range("2026-08-01 10:15", periods=192, freq="15min")
    hm = stamps.strftime("%H:%M")
    assert (out[0][(hm >= "20:00") | (hm <= "06:00")] == 0).all()
    assert (out[0][(hm > "06:00") & (hm < "20:00")] == 1).all()


class _FakeEvaluator:
    """代替 timesfm3.TimesFM3Evaluator：有未来协变量时输出第一个协变量的未来段 × 0.05，否则输出历史最后值。"""
    def __init__(self):
        self.calls = []

    def predict_batch(self, inputs, horizon, past_future_covariates=None, past_only_covariates=None, **kw):
        self.calls.append({"n": len(inputs), "pf": None if past_future_covariates is None else
                           past_future_covariates[0].shape, "po": None if past_only_covariates is None else
                           past_only_covariates[0].shape})
        outs = []
        for i, x in enumerate(inputs):
            if past_future_covariates is not None:
                fc = past_future_covariates[i][0, -horizon:] * 0.05
            else:
                fc = np.full(horizon, float(x[0, -1]))
            outs.append(SimpleNamespace(forecast=fc, quantiles=None))
        return iter(outs)


def _real_m5(cfg):
    pytest.importorskip("torch"); pytest.importorskip("transformers")
    from forecast_backbones.m5.transformer_models import TransformersAdaptedTimesFM3Model as M5
    from backends import m5_model_config
    mc = m5_model_config(cfg)
    mc["timesfm3"]["eager_init"] = False
    m = M5(mc)
    m._backbone, m._backbone_device = _FakeEvaluator(), "cpu"
    return m


def test_m5_backend_through_unified_inference(cfg):
    """真实 M5 适配类 + unified_inference（线上同一路径），只把 timesfm3 推理器换成假的。"""
    table, _ = _build(cfg, "short")
    model = _real_m5(cfg)
    be = M5Backend(cfg, model=model, batch_size=16)
    pf = [("general_ghi", "general_ghi_predict"), ("general_temp", "general_temp_predict")]
    pred = be.predict(table, pf, ["ghi_real"])
    C, P = cfg["data"]["context_len"], cfg["data"]["pred_len"]
    assert pred.shape == (len(table), P) and np.isfinite(pred).all()
    call = model._backbone.calls[0]
    assert call["pf"] == (2, C + P) and call["po"] == (1, C)
    # 行对齐：预测 = general_ghi_predict × 0.05（NaN 已按 interp 补）
    exp = np.stack([np.asarray(v, float) for v in table["general_ghi_predict"]]) * 0.05
    ok = np.isfinite(exp).all(axis=1)
    np.testing.assert_allclose(pred[ok], exp[ok], rtol=1e-4, atol=1e-4)
    # 单变量（尾部修补用）
    seqs = np.stack([np.asarray(v, float)[-100:] for v in table[F.TARGET]])
    seqs = np.nan_to_num(seqs)
    u = be.predict_univariate(seqs, table)
    np.testing.assert_allclose(u[:, 0], seqs[:, -1], rtol=1e-5)


def test_m5_backend_chunked_progress(cfg, caplog):
    """分块推理（每块打进度）与一次推完逐行一致。"""
    table, _ = _build(cfg, "short")
    pf = [("general_ghi", "general_ghi_predict")]
    whole = M5Backend(cfg, model=_real_m5(cfg), batch_size=len(table), log_every=1).predict(table, pf, [])
    with caplog.at_level(logging.INFO, logger="nwp_cov.backends"):
        chunked = M5Backend(cfg, model=_real_m5(cfg), batch_size=4, log_every=1).predict(table, pf, [])
    np.testing.assert_array_equal(chunked, whole)
    msgs = [r.getMessage() for r in caplog.records if "推理进度" in r.getMessage()]
    assert len(msgs) == -(-len(table) // 4) and f"{len(table)}/{len(table)}" in msgs[-1]


def test_repair_tail(cfg):
    table, _ = _build(cfg, "short")
    df = table[["timestamp_win", "station", F.TARGET]].copy()
    n_nan = sum(np.isnan(np.asarray(v)).any() for v in df[F.TARGET])
    assert n_nan > 0
    fixed, n = repair_tail(df, MockBackend(cfg), fix_len=600)
    assert n == n_nan
    assert not any(np.isnan(np.asarray(v)).any() for v in fixed[F.TARGET])


@pytest.mark.parametrize("pt", ["short", "ultra_short"])
def test_run_experiments_mock(cfg, pt):
    from run_experiments import run, run_dir
    s = run(cfg, ["C0", "C1_online", "A_inverr", "D_obs", "D_phys"], pt, "mock", rebuild=True)
    assert set(s["exp"]) >= {"C0", "C1_online", "A_inverr", "D_obs", "D_phys"}
    assert s["fleet_mean"].notna().all(), s
    row = s.set_index("exp").loc["C1_online"]
    assert row["baseline"] == "C0" and np.isfinite(row["mean_gain"])
    out = run_dir(cfg, pt)
    assert (out / "report.md").exists() and (out / "summary_by_weather.csv").exists()
    n_cache = len(list((out / "preds").glob("*.parquet")))
    run(cfg, ["C0", "C1_online"], pt, "mock")                             # 第二次全命中缓存
    assert len(list((out / "preds").glob("*.parquet"))) == n_cache


def test_extends_and_past_only(cfg):
    from run_experiments import resolve_experiment, split_covariates
    e = resolve_experiment(cfg, "D_obs")
    assert e["covariates"] == ["general_ghi", "general_temp", "ghi_real"]
    pf, po = split_covariates(cfg, e["covariates"])
    assert po == ["ghi_real"] and ("general_ghi", "general_ghi_predict") in pf


def test_rank_lgbm(cfg):
    pytest.importorskip("lightgbm")
    from rank_lgbm import baseline_pred, rank
    from run_experiments import load_or_build
    table = load_or_build(cfg, "short", rebuild=True)
    res = rank(table, cfg, "short", baseline_pred(cfg, table, "short", "mock"))
    r = res["ranking"]
    assert len(r) > 10 and res["info"]["target"] == "residual"
    assert r["perm_mean"].notna().all() and r["shap_mean_abs"].notna().all()
    top = r[~r["below_noise"]]["feature"].tolist()
    assert any(f.startswith(("ghi", "general_ghi", "p_phys", "kt")) for f in top[:5]), r.head(8)
    assert len(res["groups"]) > 3


def test_search_m5_mock(cfg):
    from search_m5 import Search
    from run_experiments import load_or_build
    from backends import MockBackend
    cfg["search"].update({"order_from": "list", "order": ["ghi_ec", "ghi_sg", "ghi_gdfs", "general_temp"],
                          "fusion_methods": ["median", "inv_error"]})
    cfg["evaluation"]["keep_rule"].update({"min_mean_gain": 0.0, "min_station_win_frac": 0.0})
    table = load_or_build(cfg, "short", rebuild=True)
    s = Search(cfg, table, MockBackend(cfg), "short")
    res = s.run([], cfg["search"]["order"])
    assert s.trials and res["selected"], res
    acts = {t["action"].split(":")[0] for t in s.trials}
    assert "add" in acts and "fuse" in acts
    json.dumps(res, default=str)


def test_cache_key_changes_with_feature_definition(cfg):
    from run_experiments import signature
    pf = [("general_ghi", "general_ghi_predict")]
    a = signature("mock", cfg, "short", pf, [])
    cfg["features"]["ghi_ec"]["min_valid"] = 5                         # 改的是 general_ghi 的成员
    assert signature("mock", cfg, "short", pf, []) != a
    cfg["data"]["time_range"] = {"start": "2026-08-01", "end": "2026-08-03"}
    b = signature("mock", cfg, "short", pf, [])
    cfg["data"]["time_range"] = None
    assert signature("mock", cfg, "short", pf, []) == b


def test_require_explains_missing_repo_module(monkeypatch):
    import paths
    with pytest.raises(ModuleNotFoundError, match="run.sys_path"):
        paths.require("pipelines.no_such_module_xyz")
    with pytest.raises(ModuleNotFoundError, match="pip install"):
        paths.require("no_such_package_xyz.sub")
    paths.setup(["."])                                                    # 相对路径按仓库根解析
    assert str(paths.DEFAULT_ROOT) in sys.path


def test_local_metric_copy_matches_repo(cfg):
    """本目录 nanwang_metric.py 副本与 pipelines/ 原版逐站逐日结果一致（原版改了要同步副本）。"""
    try:
        from pipelines.nanwang_metric import NanwangMetric as Repo
    except ModuleNotFoundError:
        pytest.skip("脱离仓库运行，没有 pipelines/")
    from nanwang_metric import NanwangMetric as Local
    import evaluate as E
    table, _ = _build(cfg, "short")
    pred = MockBackend(cfg).predict(table, [("general_ghi", "general_ghi_predict")], [])
    res = E.result_frame(table, pred)
    gc = E.gccap_of(table)
    pd.testing.assert_frame_equal(Local().short_d1_daily(res, gc), Repo().short_d1_daily(res, gc))
    pd.testing.assert_frame_equal(Local().ultra_short_daily(res, gc), Repo().ultra_short_daily(res, gc))


def test_without_ghi_real(cfg, tmp_path):
    """DS 没有 GHI_real（缺列或全为空）：相关特征构建失败，显式实验跳过而不中断，精排的 inv_error 融合退回 power。"""
    from conftest import make_ds, short_issue_times, write_ds
    from features import build
    from run_experiments import run
    from search_m5 import Search
    from backends import MockBackend

    for mode in ("drop", "nan"):
        df = make_ds(short_issue_times()[:24], ["st_a", "st_b"], 480, seed=3)
        if mode == "drop":
            df = df.drop(columns=["GHI_real"])
        else:
            df["GHI_real"] = [np.full(len(v), np.nan, np.float32) for v in df["GHI_real"]]
        root = tmp_path / mode
        write_ds(df, root, "short")
        c = copy.deepcopy(cfg)
        c["data"]["ds_glob"] = str(root / "predict_type={predict_type}" / "date=*" / "*.parquet")
        c["run"]["out_dir"] = str(tmp_path / f"runs_{mode}" / "{predict_type}")
        table, report = build(c, "short")
        assert {"ghi_real", "ghi_inverr", "ghi_bias"} <= set(report["failed"])
        assert "ghi_inverr_p" in report["built"] and "ghi_real" not in table.columns

        be = MockBackend(c)
        summary = run(c, ["C0", "C1_online", "A_inverr", "A_inverr_p", "D_obs"], "short", "mock", table=table, backend=be)
        s = summary.set_index("exp")
        assert s.loc["A_inverr", "reason"].startswith("跳过") and s.loc["D_obs", "reason"].startswith("跳过")
        assert np.isfinite(s.loc["A_inverr_p", "fleet_mean"])

        c["search"].update({"fusion_methods": ["inv_error"], "max_consecutive_fail": 5})
        srch = Search(c, table, be, "short")
        srch.run(["ghi_ec"], ["ghi_real", "ghi_gdfs"])
        fz = [v for v in srch.fused.values() if v["method"] == "inv_error"]
        assert fz and fz[0]["spec"]["reference"] == "power"
