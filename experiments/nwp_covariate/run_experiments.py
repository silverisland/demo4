# -*- coding: utf-8 -*-
"""run_experiments.py —— 跑 config.experiments 里的显式协变量组合，与基线做配对比较。

用法（cwd=仓库根）：
    python experiments/nwp_covariate/run_experiments.py --all
    python experiments/nwp_covariate/run_experiments.py --exp C0 C1_online A_inverr --backend m5
    python experiments/nwp_covariate/run_experiments.py --all --backend mock      # 本机跑通流程

输出 {run.out_dir}/：
    preds/<签名>.parquet       每个协变量组合的预测（缓存，组合不变就不重跑）
    daily/<实验>.csv           逐站逐日南网准确率
    summary.csv / report.md    全站均值、与基线的配对差、置信区间、站胜率、是否通过、按天气档拆解
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fields as F  # noqa: E402
import evaluate as E  # noqa: E402
from backends import infer_cols, make_backend, postprocess  # noqa: E402
from features import (build, feature_meta, feature_store_dir, load_config, read_feature_store,  # noqa: E402
                      run_dir, write_feature_store)

log = logging.getLogger("nwp_cov.run")


# ----------------------------------------------------------------------------- 实验解析
def resolve_experiment(cfg: dict, name: str, _seen=None) -> dict:
    exps = cfg.get("experiments") or {}
    if name not in exps:
        raise KeyError(f"实验 {name!r} 不在 config.experiments 里")
    _seen = set(_seen or ()) | {name}
    e = dict(exps[name] or {})
    covs: List[str] = []
    if e.get("extends"):
        if e["extends"] in _seen:
            raise ValueError(f"实验 extends 循环: {name}")
        covs += resolve_experiment(cfg, e["extends"], _seen)["covariates"]
    covs += list(e.get("covariates") or [])
    covs = list(dict.fromkeys(covs))
    meta = feature_meta(cfg)
    for c in covs:
        if c not in meta:
            raise KeyError(f"实验 {name}: 协变量 {c!r} 没在 config.features 里定义")
    base = e.get("baseline", (cfg.get("evaluation") or {}).get("baseline"))
    return {"name": name, "covariates": covs, "baseline": None if base == name else base, "note": e.get("note", "")}


def split_covariates(cfg: dict, covs: Sequence[str], meta: Optional[dict] = None) -> Tuple[list, list]:
    """特征名 -> (past_future_cols, past_only_cols)。history_only 的进 past_only。"""
    meta = meta or feature_meta(cfg)
    pf, po = [], []
    for c in covs:
        if meta.get(c, {}).get("history_only"):
            po.append(c)
        else:
            pf.append((c, c + F.FUTURE_SUFFIX))
    return pf, po


def spec_closure(cfg: dict, names) -> dict:
    """协变量及其递归依赖的特征定义；任何一个定义变了，缓存签名就变。"""
    from ops import dependencies
    specs = cfg.get("features") or {}
    out, todo = {}, list(names)
    while todo:
        n = todo.pop()
        if n in out or n not in specs:
            continue
        out[n] = specs[n]
        todo += dependencies(specs[n])
    return out


def signature(backend_name: str, cfg: dict, predict_type: str, pf, po, extra: str = "") -> str:
    """缓存键：后端、协变量、它们的定义、影响特征数值的 data 设置、M5 参数、后处理。
    time_range / stations 只决定行，不进键（命中时还要求每一行都在缓存里）。"""
    d = {k: v for k, v in cfg["data"].items() if k not in ("time_range", "stations", "feature_store")}
    names = [p[0] for p in pf] + list(po)
    s = json.dumps({"b": backend_name, "pt": predict_type, "data": d, "pf": [list(p) for p in pf], "po": list(po),
                    "spec": spec_closure(cfg, names), "m5": cfg.get("m5"), "pp": cfg.get("postprocess"),
                    "x": extra}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------------------------- 跑一组协变量
def predict_set(table: pd.DataFrame, backend, cfg: dict, predict_type: str, pf, po,
                cache_dir: Optional[Path] = None, extra_sig: str = "") -> np.ndarray:
    """-> [B, P] 后处理后的预测，行与 table 对齐。按签名缓存到 cache_dir。"""
    missing = [c for c in infer_cols(pf, po) if c not in table.columns]
    if missing:
        raise KeyError(f"特征表缺列 {missing}；改了 config.features 后请 --rebuild")
    sig = signature(backend.name, cfg, predict_type, pf, po, extra_sig)
    cache = cache_dir / f"{sig}.parquet" if cache_dir else None
    key = table[["timestamp_win", "station"]].astype({"station": str})
    if cache is not None and cache.exists():
        c = pd.read_parquet(cache)
        c["timestamp_win"] = pd.to_datetime(c["timestamp_win"])
        m = key.merge(c, on=["timestamp_win", "station"], how="left")
        if m["pred"].notna().all():
            log.info("命中预测缓存 %s", cache.name)
            return np.stack([np.asarray(v, np.float32) for v in m["pred"].to_numpy()])
    raw = backend.predict(table, pf, po)
    pred = postprocess(raw, table["timestamp_win"].to_numpy(), cfg)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        out = key.copy()
        out["pred"] = list(pred)
        out.to_parquet(cache, index=False)
        with open(cache.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump({"past_future_cols": [list(p) for p in pf], "past_only_cols": list(po),
                       "backend": backend.name, "extra": extra_sig}, f, ensure_ascii=False, indent=2)
    return pred


def score_set(table: pd.DataFrame, pred: np.ndarray, cfg: dict, predict_type: str) -> pd.DataFrame:
    res = E.result_frame(table, pred)
    return E.daily_metric(res, E.gccap_of(table), E.primary_metric(cfg, predict_type))


# ----------------------------------------------------------------------------- 主流程
def load_or_build(cfg: dict, predict_type: str, rebuild: bool = False) -> pd.DataFrame:
    fs = feature_store_dir(cfg, predict_type)
    if rebuild or not list(fs.glob("station=*.parquet")):
        table, report = build(cfg, predict_type)
        write_feature_store(table, fs, report)
        if report["failed"]:
            log.warning("构建失败的特征: %s", report["failed"])
    table = read_feature_store(cfg, predict_type)
    if F.LABEL not in table.columns:
        raise ValueError("特征库里没有标签 observe_power_future，无法评估")
    return table


def run(cfg: dict, names: List[str], predict_type: str, backend_name: str, rebuild: bool = False,
        table: Optional[pd.DataFrame] = None, backend=None) -> pd.DataFrame:
    out_dir = run_dir(cfg, predict_type)
    (out_dir / "daily").mkdir(parents=True, exist_ok=True)
    table = load_or_build(cfg, predict_type, rebuild) if table is None else table
    backend = backend or make_backend(cfg, backend_name)
    meta = feature_meta(cfg)
    exps = {n: resolve_experiment(cfg, n) for n in names}
    for e in list(exps.values()):                      # 基线也要跑
        b = e["baseline"]
        if b and b not in exps:
            exps[b] = resolve_experiment(cfg, b)
    skipped: Dict[str, str] = {}                      # 缺列（如 DS 没有 GHI_real）的实验跳过，不中断整批
    for n, e in exps.items():
        pf, po = split_covariates(cfg, e["covariates"], meta)
        miss = [c for c in infer_cols(pf, po) if c not in table.columns]
        if miss:
            skipped[n] = f"跳过：特征库缺 {miss}"
    for n, e in exps.items():
        if n not in skipped and e["baseline"] in skipped:
            skipped[n] = f"跳过：基线 {e['baseline']} 无法运行"
    for n, why in skipped.items():
        log.warning("实验 %s %s", n, why)
    order = sorted((n for n in exps if n not in skipped), key=lambda n: (exps[n]["baseline"] is not None, n))
    daily: Dict[str, pd.DataFrame] = {}
    for n in order:
        e = exps[n]
        pf, po = split_covariates(cfg, e["covariates"], meta)
        log.info("实验 %s: past_future=%s past_only=%s", n, [p[0] for p in pf], po)
        pred = predict_set(table, backend, cfg, predict_type, pf, po, out_dir / "preds")
        daily[n] = score_set(table, pred, cfg, predict_type)
        daily[n].to_csv(out_dir / "daily" / f"{n}.csv", index=False)

    ev = cfg.get("evaluation") or {}
    classes = E.day_classes(table, tuple(ev.get("weather_edges", (0.35, 0.75))),
                            tz_offset_h=float(cfg["data"].get("tz_offset_h", 8)))
    rows, parts = [], []
    for n in order:
        e = exps[n]
        r = {"exp": n, "baseline": e["baseline"] or "", "covariates": ",".join(e["covariates"]),
             "fleet_mean": E.fleet_mean(daily[n]), "note": e["note"]}
        if e["baseline"]:
            p = E.paired(daily[e["baseline"]], daily[n], ev.get("keep_rule"))
            r.update({k: p[k] for k in ("mean_gain", "ci_lo", "ci_hi", "station_win_frac", "n_days",
                                         "n_stations", "passed", "reason")})
            bd = E.breakdown(daily[e["baseline"]], daily[n], classes, predict_type)
            bd.insert(0, "exp", n)
            parts.append(bd)
        rows.append(r)
    for n, why in skipped.items():
        e = exps[n]
        rows.append({"exp": n, "baseline": e["baseline"] or "", "covariates": ",".join(e["covariates"]),
                     "fleet_mean": np.nan, "note": e["note"], "passed": False, "reason": why})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    weather = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    weather.to_csv(out_dir / "summary_by_weather.csv", index=False)
    write_report(out_dir / "report.md", summary, weather, cfg, predict_type, backend.name)
    return summary


def _md(df: pd.DataFrame) -> str:
    if df is None or not len(df):
        return "（无）\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.3f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_report(path: Path, summary: pd.DataFrame, weather: pd.DataFrame, cfg: dict, predict_type: str, backend: str):
    kr = (cfg.get("evaluation") or {}).get("keep_rule") or {}
    txt = [f"# 协变量组合实验 {cfg['run']['name']} / {predict_type}\n",
           f"- 后端：{backend}；主指标：{E.primary_metric(cfg, predict_type)}",
           f"- 保留规则：平均提升 ≥ {kr.get('min_mean_gain', 0.3)}，按天分块 bootstrap {kr.get('ci', 0.95):.0%} 下界 > 0，"
           f"站胜率 ≥ {kr.get('min_station_win_frac', 0.65)}\n",
           "## 汇总\n", _md(summary.drop(columns=["note"], errors="ignore")),
           "\n## 按天气档拆解（配对差，目标日实测晴空指数分档）\n", _md(weather)]
    path.write_text("\n".join(txt), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--predict_type", default=None)
    ap.add_argument("--exp", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--backend", choices=["m5", "mock"], default=None)
    ap.add_argument("--rebuild", action="store_true", help="重新构建特征库")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    pt = a.predict_type or cfg["data"]["predict_type"]
    names = list(cfg["experiments"]) if a.all or not a.exp else a.exp
    summary = run(cfg, names, pt, a.backend or cfg["run"].get("backend", "m5"), a.rebuild)
    pd.set_option("display.width", 200)
    print(summary.drop(columns=["note", "covariates"], errors="ignore").to_string(index=False))
    print(f"\n-> {run_dir(cfg, pt)}")


if __name__ == "__main__":
    main()
