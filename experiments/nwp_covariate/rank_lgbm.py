# -*- coding: utf-8 -*-
"""rank_lgbm.py —— LightGBM 粗排：哪些候选特征对"C0 没解释掉的误差"有信息量。

做法：
  1. 样本 = (起报行, 预报步)。短期只取每站每天 ≤10:00 的最后一行、第 55..150 步（与 D+1 指标同口径）；
     超短期取全部行的第 0..15 步。只留目标时刻晴空辐照 ≥ min_clear 的白天样本。
  2. 目标 = (标签 − C0 预测)/cap（C0 预测从 run_experiments 的缓存读，没有就现跑；--no_baseline 时用 标签/cap）。
  3. 特征 = 每个候选在目标时刻的未来值（history_only 的取起报时刻的最后值；power 类除以 cap）
           + 背景特征（步长、目标时刻晴空辐照、最近实测功率比、前一天同时刻功率比）+ 高斯噪声参照列。
  4. 按日期切分：前 train_frac 的天训练，其余天验证；早停。
  5. 在验证集上算：单列置换重要性（RMSE 增量，重复 n_repeats 次）、簇级 / 来源级 / 物理量级分组置换、
     TreeSHAP 平均 |贡献|、LightGBM gain。重要性 ≤ 噪声列最大值的标为 below_noise。

输出 {run.out_dir}/rank/：ranking.csv、groups.csv、corr_clusters.csv、ranking_order.txt（给 search_m5.py 用）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fields as F  # noqa: E402
from features import feature_meta, load_config, run_dir, stack_col  # noqa: E402

log = logging.getLogger("nwp_cov.rank")


# ----------------------------------------------------------------------------- 样本
def issue_rows(table: pd.DataFrame, issue_time: Optional[str]) -> pd.DataFrame:
    """每站每天取 timestamp_win ≤ issue_time 的最后一行（与 NanwangMetric.short_d1_daily 同口径）。"""
    if not issue_time:
        return table
    t = table.copy()
    t["_d"] = t["timestamp_win"].dt.normalize()
    hh, mm = map(int, issue_time.split(":"))
    ok = (t["timestamp_win"].dt.hour * 60 + t["timestamp_win"].dt.minute) <= hh * 60 + mm
    t = t[ok].sort_values("timestamp_win").groupby(["station", "_d"], as_index=False).tail(1)
    return t.drop(columns="_d")


def candidates_of(cfg: dict, table: pd.DataFrame) -> List[str]:
    meta = feature_meta(cfg)
    cand = (cfg.get("rank") or {}).get("candidates", "auto")
    names = [n for n, m in meta.items() if m["candidate"]] if cand == "auto" else list(cand)
    ok = [n for n in names if n in table.columns]
    if len(ok) < len(names):
        log.warning("特征库里没有这些候选（构建失败或未构建）：%s", sorted(set(names) - set(ok)))
    return ok


def build_samples(table: pd.DataFrame, cfg: dict, predict_type: str, cands: List[str],
                  base_pred: Optional[np.ndarray]) -> pd.DataFrame:
    r = cfg.get("rank") or {}
    C, P = int(cfg["data"]["context_len"]), int(cfg["data"]["pred_len"])
    lo, hi = r.get("leads", {}).get(predict_type, [0, 15] if predict_type == "ultra_short" else [55, 150])
    leads = np.arange(int(lo), min(int(hi), P - 1) + 1)
    meta = feature_meta(cfg)
    cap = table["cap_power_on_val"].to_numpy(float)
    lab = stack_col(table, F.LABEL, P, "head")[:, leads]
    clear = stack_col(table, "ghi_clear_sky_predict", P, "head")[:, leads]
    obs = stack_col(table, F.TARGET, C, "tail")
    last = obs[:, -1]
    yday = np.full((len(table), len(leads)), np.nan)
    for j, l in enumerate(leads):                      # 目标时刻 − 1 天（−96 步）在历史段里的位置
        k = C - 1 + (l + 1) - 96
        while k >= C:
            k -= 96
        yday[:, j] = obs[:, k] if k >= 0 else np.nan
    B, L = lab.shape
    df = pd.DataFrame({
        "row": np.repeat(np.arange(B), L), "lead": np.tile(leads, B),
        "date": np.repeat(table["timestamp_win"].dt.normalize().to_numpy(), L),
        "station": np.repeat(table["station"].astype(str).to_numpy(), L),
        "clear_target": clear.ravel(),
        "last_ratio": np.repeat(last / np.maximum(cap, 1e-6), L),
        "yday_ratio": (yday / np.maximum(cap, 1e-6)[:, None]).ravel(),
    })
    y = lab / np.maximum(cap, 1e-6)[:, None]
    if base_pred is not None:
        y = y - base_pred[:, leads] / np.maximum(cap, 1e-6)[:, None]
    df["y"] = y.ravel()
    for n in cands:
        if meta[n]["history_only"]:
            v = np.repeat(stack_col(table, n, C, "tail")[:, -1], L)
        else:
            v = stack_col(table, n + F.FUTURE_SUFFIX, P, "head")[:, leads]
            if meta[n]["quantity"] == "power":
                v = v / np.maximum(cap, 1e-6)[:, None]
            v = v.ravel()
        df[n] = v
    df = df[(df["clear_target"] >= float(r.get("min_clear", 50))) & np.isfinite(df["y"])]
    mx = int(r.get("max_rows", 400000))
    if len(df) > mx:
        df = df.sample(mx, random_state=0)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- 重要性
def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def corr_clusters(X: pd.DataFrame, cols: List[str], thr: float) -> Dict[str, int]:
    """|Spearman| ≥ thr 的候选并成一簇（并查集）。"""
    if len(cols) < 2:
        return {c: i for i, c in enumerate(cols)}
    rho = X[cols].rank().corr().abs().fillna(0).to_numpy()
    parent = list(range(len(cols)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if rho[i, j] >= thr:
                parent[find(i)] = find(j)
    roots = {}
    return {c: roots.setdefault(find(i), len(roots)) for i, c in enumerate(cols)}


def permutation(booster, X: pd.DataFrame, y: np.ndarray, groups: Dict[str, List[str]], n_repeats: int,
                seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = _rmse(booster.predict(X), y)
    rows = []
    for g, cols in groups.items():
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            for c in cols:                              # 组内各列用同一个置换，保持组内相关结构
                Xp[c] = X[c].to_numpy()[perm]
            inc.append(_rmse(booster.predict(Xp), y) - base)
        inc = np.asarray(inc)
        rows.append({"name": g, "members": ",".join(cols), "perm_mean": float(inc.mean()),
                     "perm_std": float(inc.std(ddof=1)) if len(inc) > 1 else 0.0,
                     "perm_rel": float(inc.mean() / base) if base > 0 else np.nan})
    return pd.DataFrame(rows)


def rank(table: pd.DataFrame, cfg: dict, predict_type: str, base_pred: Optional[np.ndarray]) -> dict:
    import lightgbm as lgb
    r = cfg.get("rank") or {}
    meta = feature_meta(cfg)
    issue = (r.get("issue_time") or {}).get(predict_type)
    sel = issue_rows(table, issue)
    bp = None
    if base_pred is not None:
        bp = base_pred[sel.index.to_numpy()]
    sel = sel.reset_index(drop=True)
    cands = candidates_of(cfg, sel)
    if not cands:
        raise ValueError("没有可排序的候选特征")
    S = build_samples(sel, cfg, predict_type, cands, bp)
    ctx_cols = [c for c in r.get("context_features", ["lead", "clear_target", "last_ratio", "yday_ratio"]) if c in S]
    rng = np.random.default_rng(0)
    noise = [f"noise_{i}" for i in range(int(r.get("noise_columns", 2)))]
    for c in noise:
        S[c] = rng.normal(size=len(S))
    cols = ctx_cols + cands + noise
    days = np.sort(S["date"].unique())
    if len(days) < 3:
        raise ValueError(f"只有 {len(days)} 天样本，无法按天切分训练/验证")
    cut = days[max(1, min(len(days) - 1, int(round(len(days) * float(r.get("train_frac", 0.7))))))]
    tr, va = S[S["date"] < cut], S[S["date"] >= cut]
    p = dict(r.get("lgbm") or {})
    nbr = int(p.pop("num_boost_round", 600)); es = int(p.pop("early_stopping_rounds", 50))
    dtr = lgb.Dataset(tr[cols], tr["y"], free_raw_data=False)
    dva = lgb.Dataset(va[cols], va["y"], reference=dtr, free_raw_data=False)
    booster = lgb.train(p, dtr, num_boost_round=nbr, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
    Xv, yv = va[cols], va["y"].to_numpy()
    nrep = int(r.get("n_repeats", 5))

    single = permutation(booster, Xv, yv, {c: [c] for c in cands + noise}, nrep).set_index("name")
    contrib = booster.predict(Xv, pred_contrib=True)[:, :-1]
    shap = pd.Series(np.abs(contrib).mean(axis=0), index=cols)
    gain = pd.Series(booster.feature_importance("gain"), index=cols)
    noise_max = float(single.loc[noise, "perm_mean"].max()) if noise else 0.0
    noise_shap = float(shap[noise].max()) if noise else 0.0
    clusters = corr_clusters(Xv, cands, float(r.get("corr_threshold", 0.95)))

    rows = []
    for c in cands:
        m = meta[c]
        s = single.loc[c]
        rows.append({"feature": c, "quantity": m["quantity"], "source": m["source"] or "", "fusible": m["fusible"],
                     "history_only": m["history_only"], "perm_mean": s["perm_mean"], "perm_std": s["perm_std"],
                     "perm_rel": s["perm_rel"], "shap_mean_abs": float(shap[c]), "gain": float(gain[c]),
                     "cluster": clusters[c], "below_noise": bool(s["perm_mean"] <= noise_max and shap[c] <= noise_shap),
                     "nan_frac": float(np.isnan(S[c]).mean())})
    ranking = pd.DataFrame(rows).sort_values(["below_noise", "perm_mean", "shap_mean_abs"],
                                             ascending=[True, False, False]).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    groups: Dict[str, List[str]] = {}
    for c in cands:
        groups.setdefault(f"source:{meta[c]['source'] or 'none'}", []).append(c)
        groups.setdefault(f"quantity:{meta[c]['quantity']}", []).append(c)
    for c, k in clusters.items():
        groups.setdefault(f"cluster:{k}", []).append(c)
    groups = {g: v for g, v in groups.items() if not (g.startswith("cluster:") and len(v) < 2)}
    grp = permutation(booster, Xv, yv, groups, nrep).sort_values("perm_mean", ascending=False)

    info = {"n_samples": int(len(S)), "n_train": int(len(tr)), "n_valid": int(len(va)),
            "split_date": str(pd.Timestamp(cut).date()), "best_iteration": int(booster.best_iteration or nbr),
            "valid_rmse": _rmse(booster.predict(Xv), yv), "valid_rmse_zero": _rmse(np.zeros_like(yv), yv),
            "target": "residual" if base_pred is not None else "level", "noise_perm_max": noise_max,
            "noise_shap_max": noise_shap, "context_features": ctx_cols}
    clus = pd.DataFrame({"feature": list(clusters), "cluster": list(clusters.values())}).sort_values("cluster")
    return {"ranking": ranking, "groups": grp, "clusters": clus, "info": info}


def baseline_pred(cfg: dict, table: pd.DataFrame, predict_type: str, backend_name: Optional[str]) -> Optional[np.ndarray]:
    r = cfg.get("rank") or {}
    if r.get("target", "residual") != "residual" or backend_name == "none":
        return None
    from backends import make_backend
    from run_experiments import predict_set, resolve_experiment, split_covariates
    e = resolve_experiment(cfg, r.get("baseline_exp", "C0"))
    pf, po = split_covariates(cfg, e["covariates"])
    be = make_backend(cfg, backend_name or cfg["run"].get("backend", "m5"))
    return predict_set(table, be, cfg, predict_type, pf, po, run_dir(cfg, predict_type) / "preds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--predict_type", default=None)
    ap.add_argument("--backend", choices=["m5", "mock", "none"], default=None,
                    help="算 C0 预测（残差目标）用的后端；none = 目标改为 标签/cap")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    pt = a.predict_type or cfg["data"]["predict_type"]
    from run_experiments import load_or_build
    table = load_or_build(cfg, pt, a.rebuild)
    res = rank(table, cfg, pt, baseline_pred(cfg, table, pt, a.backend))
    out = run_dir(cfg, pt) / "rank"
    out.mkdir(parents=True, exist_ok=True)
    res["ranking"].to_csv(out / "ranking.csv", index=False)
    res["groups"].to_csv(out / "groups.csv", index=False)
    res["clusters"].to_csv(out / "corr_clusters.csv", index=False)
    (out / "ranking_order.txt").write_text("\n".join(res["ranking"]["feature"]) + "\n", encoding="utf-8")
    (out / "info.json").write_text(json.dumps(res["info"], ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(json.dumps(res["info"], ensure_ascii=False, indent=2))
    print(res["ranking"][["rank", "feature", "quantity", "source", "perm_mean", "shap_mean_abs", "cluster",
                          "below_noise"]].to_string(index=False))
    print("\n分组置换：")
    print(res["groups"][["name", "perm_mean", "perm_std", "members"]].to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
