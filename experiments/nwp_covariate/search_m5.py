# -*- coding: utf-8 -*-
"""search_m5.py —— M5 精排：按粗排顺序逐个尝试候选，贪心保留。

每一步对当前候选 c 尝试两类动作，全部与"当前已选组合"做配对比较：
  add          把 c 作为一个新协变量并排加入（M5 自己学怎么用）；
  fuse:<方法>  若 c 可融合（fusible）且已选里有同 quantity 的可融合特征 s（默认只取第一个），
               把 s 的成员 ∪ c 的成员合成一列替换 s（方法取 search.fusion_methods：median/mean/trim_mean/inv_error…）。
每个候选最多 1 + len(fusion_methods) 次 M5 推理（fusion_with=first 时）。
通过保留规则（平均提升、按天分块 bootstrap 下界 > 0、站胜率）的动作里取平均提升最大的接受；
都不通过记一次失败。连续失败 max_consecutive_fail 次或已选达到 max_selected 时停止。

用法（cwd=仓库根）：
    python experiments/nwp_covariate/search_m5.py [--backend m5|mock] [--predict_type short]
输出 {run.out_dir}/search/：trials.csv（每次尝试）、selected.json（最终组合与融合定义）、report.md
"""
from __future__ import annotations

import argparse
import hashlib
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
import evaluate as E  # noqa: E402
from backends import make_backend  # noqa: E402
from features import feature_meta, feature_specs, load_config, run_dir, table_ctx  # noqa: E402
from run_experiments import (load_or_build, predict_set, resolve_experiment, score_set,  # noqa: E402
                             split_covariates, _md)

log = logging.getLogger("nwp_cov.search")


def base_members(cfg: dict, name: str) -> List[str]:
    """一个特征的"融合成员"：combine 定义的取其 members，其余就是它自己。"""
    s = feature_specs(cfg).get(name, {})
    if s.get("op") == "combine" and s.get("method") in ("median", "mean", "trim_mean", "inv_error",
                                                         "bias_mean", "weighted_mean"):
        return list(s["members"])
    return [name]


def order_of(cfg: dict, predict_type: str) -> List[str]:
    s = cfg.get("search") or {}
    if s.get("order_from", "rank") == "rank":
        p = run_dir(cfg, predict_type) / "rank" / "ranking.csv"
        if not p.exists():
            raise FileNotFoundError(f"没有粗排结果 {p}；先跑 rank_lgbm.py，或把 search.order_from 改成 list")
        r = pd.read_csv(p)
        if s.get("drop_below_noise", True):
            r = r[~r["below_noise"].astype(bool)]
        names = r["feature"].tolist()
    else:
        names = list(s.get("order") or [])
    return names[: int(s.get("top_k", len(names)))]


class Search:
    def __init__(self, cfg: dict, table: pd.DataFrame, backend, predict_type: str):
        self.cfg, self.backend, self.pt = cfg, backend, predict_type
        self.table = table.copy()
        self.meta = feature_meta(cfg)
        self.s = cfg.get("search") or {}
        self.rule = (cfg.get("evaluation") or {}).get("keep_rule")
        self.out = run_dir(cfg, predict_type)
        self.fused: Dict[str, dict] = {}                 # 动态融合特征名 -> {members, method, quantity}
        self.trials: List[dict] = []
        self._ref_warned = False

    # ---- 特征信息
    def quantity(self, n: str) -> str:
        return self.fused[n]["quantity"] if n in self.fused else self.meta.get(n, {}).get("quantity", "other")

    def members(self, n: str) -> List[str]:
        return list(self.fused[n]["members"]) if n in self.fused else base_members(self.cfg, n)

    def fusible(self, n: str) -> bool:
        if n in self.fused:
            return True
        return bool(self.meta.get(n, {}).get("fusible")) or len(base_members(self.cfg, n)) > 1

    # ---- 动态融合
    def make_fused(self, members: List[str], method: str, quantity: str) -> str:
        key = f"{method}|" + "|".join(members)
        name = f"fz_{quantity}_{method}_{hashlib.md5(key.encode()).hexdigest()[:6]}"
        if name in self.fused:
            return name
        ref = self.s.get("fusion_reference", "ghi_real")
        if ref != "power" and ref not in self.table.columns:
            if not self._ref_warned:
                log.warning("参考真值 %s 不在特征库里（DS 缺 GHI_real 或全为空），inv_error / bias_mean 改用 power", ref)
                self._ref_warned = True
            ref = "power"
        spec = {"op": "combine", "members": members, "method": method, "min_valid": min(2, len(members))}
        if method in ("inv_error", "bias_mean"):
            spec.update({"reference": ref, "min_clear": 50, "window_steps": int(self.cfg["data"]["context_len"])})
        ctx = table_ctx(self.table, self.cfg, members + ([ref] if ref != "power" else []))
        ctx.specs = {**ctx.specs, name: spec}
        pair = ctx.get(name)
        self.table[name] = list(pair.hist.astype(np.float32))
        self.table[name + F.FUTURE_SUFFIX] = list(pair.fut.astype(np.float32))
        self.fused[name] = {"members": members, "method": method, "quantity": quantity, "spec": spec}
        return name

    # ---- 评估
    def daily(self, covs: List[str]) -> pd.DataFrame:
        pf, po = split_covariates(self.cfg, covs, self.meta)
        extra = json.dumps({n: self.fused[n]["spec"] for n in covs if n in self.fused}, sort_keys=True)
        pred = predict_set(self.table, self.backend, self.cfg, self.pt, pf, po, self.out / "preds", extra)
        return score_set(self.table, pred, self.cfg, self.pt)

    def actions(self, selected: List[str], c: str) -> List[tuple]:
        """候选 c 的所有尝试动作。融合时 c 也展开成基础成员，与目标特征的成员取并集（不嵌套融合特征）；
        search.fusion_with=first（默认）只与该物理量的第一个已选特征融合，all 则与每一个都试。"""
        acts = [("add", selected + [c])]
        if not self.fusible(c) or self.meta.get(c, {}).get("history_only"):
            return acts
        q = self.quantity(c)
        targets = [i for i, s in enumerate(selected) if s != c and self.quantity(s) == q and self.fusible(s)]
        if self.s.get("fusion_with", "first") == "first":
            targets = targets[:1]
        for i in targets:
            s = selected[i]
            mem = self.members(s)
            new = mem + [m for m in self.members(c) if m not in mem]
            if len(new) == len(mem):
                continue                                   # c 的成员已全部在 s 里，融合不会带来新信息
            for m in self.s.get("fusion_methods", ["median", "mean"]):
                try:
                    fz = self.make_fused(new, m, q)
                except (KeyError, ValueError) as e:
                    log.warning("融合 %s+%s(%s) 失败: %s", s, c, m, e)
                    continue
                acts.append((f"fuse:{m}:{s}", selected[:i] + [fz] + selected[i + 1:]))
        return acts

    def run(self, start: List[str], order: List[str]) -> dict:
        selected = list(start)
        cur = self.daily(selected)
        start_mean = E.fleet_mean(cur)
        log.info("起点 %s：全站均值 %.3f", selected, start_mean or float("nan"))
        fails, step = 0, 0
        used = {m for s in selected for m in self.members(s)} | set(selected)
        for c in order:
            if c in used:
                if c not in selected:
                    log.info("候选 %s 已是已选特征 %s 的成员，跳过", c, selected)
                continue
            if c not in self.table.columns:
                log.warning("候选 %s 不在特征库里（构建失败或缺列），跳过", c)
                continue
            if len(selected) >= int(self.s.get("max_selected", 10)):
                log.info("已选达到上限，停止"); break
            if fails >= int(self.s.get("max_consecutive_fail", 2)):
                log.info("连续 %d 次失败，停止", fails); break
            step += 1
            best = None
            for act, covs in self.actions(selected, c):
                d = self.daily(covs)
                p = E.paired(cur, d, self.rule)
                rec = {"step": step, "candidate": c, "action": act, "covariates": ",".join(covs),
                       "fleet_mean": E.fleet_mean(d), **{k: p[k] for k in ("mean_gain", "ci_lo", "ci_hi",
                                                                             "station_win_frac", "passed", "reason")}}
                self.trials.append(rec)
                log.info("  %-28s %-36s gain=%+.3f ci=[%+.3f,%+.3f] win=%.2f %s", c, act, p["mean_gain"],
                         p["ci_lo"], p["ci_hi"], p["station_win_frac"], "✔" if p["passed"] else "✘")
                if p["passed"] and (best is None or p["mean_gain"] > best[2]["mean_gain"]):
                    best = (act, covs, p, d)
            if best:
                selected, cur, fails = best[1], best[3], 0
                used |= {m for s in selected for m in self.members(s)} | {c}
                log.info("接受 %s (%s) -> %s", c, best[0], selected)
            else:
                fails += 1
                used.add(c)
        return {"start": start, "start_fleet_mean": start_mean, "selected": selected,
                "final_fleet_mean": E.fleet_mean(cur),
                "fused": {n: {k: v for k, v in f.items()} for n, f in self.fused.items() if n in selected}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--predict_type", default=None)
    ap.add_argument("--backend", choices=["m5", "mock"], default=None)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    pt = a.predict_type or cfg["data"]["predict_type"]
    table = load_or_build(cfg, pt, a.rebuild)
    be = make_backend(cfg, a.backend or cfg["run"].get("backend", "m5"))
    s = cfg.get("search") or {}
    start = resolve_experiment(cfg, s.get("start", "C0"))["covariates"]
    order = order_of(cfg, pt)
    log.info("候选顺序: %s", order)
    srch = Search(cfg, table, be, pt)
    res = srch.run(start, order)
    out = run_dir(cfg, pt) / "search"
    out.mkdir(parents=True, exist_ok=True)
    trials = pd.DataFrame(srch.trials)
    trials.to_csv(out / "trials.csv", index=False)
    (out / "selected.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    txt = [f"# M5 精排 {cfg['run']['name']} / {pt}\n",
           f"- 起点：{res['start']}（全站均值 {res['start_fleet_mean']}）",
           f"- 结果：{res['selected']}（全站均值 {res['final_fleet_mean']}）",
           f"- 融合定义：`{json.dumps({k: {'members': v['members'], 'method': v['method']} for k, v in res['fused'].items()}, ensure_ascii=False)}`\n",
           "## 每次尝试\n", _md(trials.drop(columns=["covariates"], errors="ignore"))]
    (out / "report.md").write_text("\n".join(txt), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("selected", "start_fleet_mean", "final_fleet_mean")}, ensure_ascii=False, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
