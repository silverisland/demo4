# -*- coding: utf-8 -*-
"""backends.py —— 推理后端（与线上 multi_infer 的 M5 调用方式对齐）。

M5Backend：
  - 模型类与线上相同（forecast_backbones.m5.transformer_models.TransformersAdaptedTimesFM3Model）；
  - 调用 pipelines.pipeline_wrapper_inference.unified_inference，entity_cols=(timestamp_win, station)，
    feature_cols=[observe_power, 协变量历史列, 协变量未来列...]，与线上超短期分支完全一致；
  - 模型只加载一次，换实验时只改 past_future_cols / past_only_cols（这两个是普通属性，不触发重载）。
  - 线上短期分支用 rolling_inference 滚到 480 点；南网 D+1 只用到第 55..150 步，192 点足够，这里统一 192。
MockBackend：本机无 timesfm3 时跑通流程用。有辐照类协变量时按辐照折算功率，否则用前一天同时刻持续。

后处理与线上一致：负值置 0，目标时刻落在 20:00~06:00（含）的点置 0。
"""
from __future__ import annotations

import hashlib
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (str(HERE), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fields as F  # noqa: E402
from paths import require  # noqa: E402

log = logging.getLogger("nwp_cov.backends")


def m5_model_config(cfg: dict, past_future: Sequence[Sequence[str]] = (), past_only: Sequence[str] = ()) -> dict:
    d = cfg["data"]
    t = dict(cfg.get("m5") or {})
    t.update({"context_len": int(d["context_len"]), "pred_len": int(d["pred_len"]),
              "past_future_cols": [list(p) for p in past_future], "past_only_cols": list(past_only)})
    return {"dataset": {"pv_col": F.TARGET}, "timesfm3": t}


def infer_cols(past_future: Sequence[Sequence[str]], past_only: Sequence[str]) -> List[str]:
    cols = [F.TARGET]
    for h, f in past_future:
        cols += [h, f]
    cols += list(past_only)
    return list(dict.fromkeys(cols))


# ----------------------------------------------------------------------------- 后端
class M5Backend:
    name = "m5"

    def __init__(self, cfg: dict, device: Optional[str] = None, batch_size: Optional[int] = None, model=None,
                 log_every: Optional[int] = None):
        self.cfg = cfg
        self.batch_size = int(batch_size or cfg["run"].get("batch_size", 512))
        self.log_every = max(1, int(log_every or cfg["run"].get("log_every_batches", 8)))   # 每多少个 batch 打一次进度
        self.pred_len = int(cfg["data"]["pred_len"])
        if model is None:
            mod = require("forecast_backbones.m5.transformer_models")
            M5 = getattr(mod, "TransformersAdaptedTimesFM3Model", None) or getattr(mod, "TransformersAdaptedM5Model")
            mc = m5_model_config(cfg)
            if device:
                mc["timesfm3"]["device"] = device
            model = M5(mc)
        self.model = model

    def _set_covariates(self, past_future, past_only):
        self.model.past_future_cols = [tuple(p) for p in past_future]
        self.model.past_only_cols = list(past_only)
        if hasattr(self.model, "_warned_pairs"):
            self.model._warned_pairs = set()

    def _run(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        unified_inference = require("pipelines.pipeline_wrapper_inference").unified_inference
        infer = df[["timestamp_win", "station"] + cols].copy()
        # 分块调用 unified_inference（共享层不动），每块结束打一行进度
        n, step = len(infer), self.batch_size * self.log_every
        parts, t0 = [], time.time()
        for s in range(0, n, step):
            parts.append(unified_inference(self.model, infer.iloc[s:s + step], timestamp_cols=("timestamp_win",),
                                           entity_cols=("timestamp_win", "station"), feature_cols=cols,
                                           batch_size=self.batch_size))
            done, dt = min(s + step, n), time.time() - t0
            rate = done / dt if dt > 0 else float("inf")
            log.info("推理进度 %d/%d 行 (%.1f%%)  %.1f 行/秒  已用 %.0fs  预计剩余 %.0fs",
                     done, n, 100.0 * done / n, rate, dt, (n - done) / rate if rate > 0 else float("nan"))
        out = pd.concat(parts, ignore_index=True)
        pred_col = [c for c in out.columns if c not in ("timestamp_win", "station")][0]
        out["timestamp_win"] = pd.to_datetime(out["timestamp_win"])
        out["station"] = out["station"].astype(str)
        key = df[["timestamp_win", "station"]].copy()
        key["station"] = key["station"].astype(str)
        m = key.merge(out[["timestamp_win", "station", pred_col]], on=["timestamp_win", "station"], how="left")
        P = self.pred_len
        res = np.full((len(m), P), np.nan, dtype=np.float32)
        for i, v in enumerate(m[pred_col].tolist()):
            if v is not None:
                a = np.asarray(v, dtype=np.float32).ravel()[:P]
                res[i, :a.size] = a
        return res

    def predict(self, table: pd.DataFrame, past_future, past_only) -> np.ndarray:
        self._set_covariates(past_future, past_only)
        return self._run(table, infer_cols(past_future, past_only))

    def predict_univariate(self, seqs: np.ndarray, ids: pd.DataFrame) -> np.ndarray:
        """seqs [n, L] 同长度的功率序列 -> [n, P]（线上 fix_observe_power_tail_miss 的单变量推理）。"""
        self._set_covariates([], [])
        df = ids[["timestamp_win", "station"]].copy().reset_index(drop=True)
        df[F.TARGET] = list(seqs.astype(np.float32))
        return self._run(df, [F.TARGET])


class MockBackend:
    """本机自测：有辐照/功率类协变量 -> 折算功率；否则前一天同时刻持续。确定性（同一协变量组合结果相同）。"""
    name = "mock"

    def __init__(self, cfg: dict, seed: int = 0):
        from features import feature_meta
        self.cfg = cfg
        self.pred_len = int(cfg["data"]["pred_len"])
        self.meta = feature_meta(cfg) if cfg.get("features") else {}
        self.seed = seed

    def _quantity(self, col: str) -> str:
        if col in self.meta:
            return self.meta[col]["quantity"]
        return "ghi" if "GHI" in col.upper() else "other"

    def predict(self, table: pd.DataFrame, past_future, past_only) -> np.ndarray:
        P = self.pred_len
        cap = table["cap_power_on_val"].to_numpy(float)[:, None]
        hist = np.stack([np.asarray(v, float) for v in table[F.TARGET].to_numpy()])
        ghi = [np.stack([np.asarray(v, float)[:P] for v in table[f].to_numpy()])
               for h, f in past_future if self._quantity(h) == "ghi"]
        pw = [np.stack([np.asarray(v, float)[:P] for v in table[f].to_numpy()])
              for h, f in past_future if self._quantity(h) == "power"]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if pw:
                base = np.nanmean(np.stack(pw), axis=0) * 0.82
            elif ghi:
                base = np.nanmean(np.stack(ghi), axis=0) / 1000.0 * cap * 0.82
        if not pw and not ghi:
            day = hist[:, -96:]
            base = np.tile(day, (1, int(np.ceil(P / 96))))[:, :P]
        sig = "|".join(f"{h},{f}" for h, f in past_future) + "#" + ",".join(past_only)
        rng = np.random.default_rng(int(hashlib.md5(sig.encode()).hexdigest()[:8], 16) + self.seed)
        noise = rng.normal(0, 0.02, base.shape) * cap
        return np.clip(np.nan_to_num(base) + noise, 0, None).astype(np.float32)

    def predict_univariate(self, seqs: np.ndarray, ids: pd.DataFrame) -> np.ndarray:
        P = self.pred_len
        last = seqs[:, -96:] if seqs.shape[1] >= 96 else np.repeat(seqs[:, -1:], 96, axis=1)
        return np.tile(np.nan_to_num(last), (1, int(np.ceil(P / 96))))[:, :P].astype(np.float32)


def make_backend(cfg: dict, name: str, device: Optional[str] = None):
    if name == "mock":
        return MockBackend(cfg)
    if name == "m5":
        return M5Backend(cfg, device=device or cfg["run"].get("device"))
    raise ValueError(f"未知后端 {name!r}")


# ----------------------------------------------------------------------------- 线上同款前后处理
def repair_tail(df: pd.DataFrame, backend, fix_len: int = 600) -> Tuple[pd.DataFrame, int]:
    """复刻线上 fix_observe_power_tail_miss：找 observe_power 第一个 NaN 的位置 pos，
    取其前 fix_len 点做单变量预测，从 pos 起填 min(剩余长度, pred_len) 个点。没有 NaN 的行不动。"""
    df = df.copy()
    arrs = [np.asarray(v, dtype=float).ravel() if v is not None else np.array([]) for v in df[F.TARGET].tolist()]
    todo: Dict[int, List[Tuple[int, int]]] = {}          # 上下文长度 -> [(行号, 第一个 NaN 位置)]
    for i, a in enumerate(arrs):
        nan_pos = np.nonzero(np.isnan(a))[0]
        if a.size and len(nan_pos):
            pos = int(nan_pos[0])
            L = pos - max(0, pos - fix_len)
            if L >= 32:                                   # 上下文太短不补（线上会让模型报错后兜底）
                todo.setdefault(L, []).append((i, pos))
    n = 0
    for L, items in todo.items():
        idx = [i for i, _ in items]
        seqs = np.stack([arrs[i][pos - L:pos] for i, pos in items])
        pred = backend.predict_univariate(seqs, df.iloc[idx])
        for k, (i, pos) in enumerate(items):
            a = arrs[i].copy()
            m = min(a.size - pos, pred.shape[1])
            a[pos:pos + m] = pred[k, :m]
            arrs[i] = a
            n += 1
    df[F.TARGET] = [a.astype(np.float32) for a in arrs]
    if n:
        log.info("observe_power 尾部修补 %d 行", n)
    return df, n


def target_times(ts_win, pred_len: int) -> np.ndarray:
    s = pd.to_datetime(pd.Series(ts_win)).values.astype("datetime64[s]").astype("int64")[:, None]
    return (s + np.arange(1, pred_len + 1)[None, :] * 900).astype("datetime64[s]")


def postprocess(pred: np.ndarray, ts_win, cfg: dict) -> np.ndarray:
    pp = cfg.get("postprocess") or {}
    out = np.asarray(pred, dtype=np.float32).copy()
    if pp.get("clip_negative", True):
        out = np.where(np.isfinite(out), np.clip(out, 0, None), out)
    night = pp.get("night_zero")
    if night:
        start, end = night
        tt = pd.DatetimeIndex(target_times(ts_win, out.shape[1]).reshape(-1))
        hm = np.array(tt.strftime("%H:%M")).reshape(out.shape)
        mask = ((hm >= start) & (hm <= end)) if start <= end else ((hm >= start) | (hm <= end))
        out[mask] = 0.0
    return out
