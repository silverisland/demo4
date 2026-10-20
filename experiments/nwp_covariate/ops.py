# -*- coding: utf-8 -*-
"""ops.py —— 特征算子库：每个特征是一对数组 (hist [B, C], fut [B, P])，用 Pair 表示。

算子签名统一为 op(ctx, **args) -> Pair。args 里引用其他特征时写特征名，算子内部 ctx.get(name) 递归构建。
config.features 里每个特征的 `op` 就是 OPS 注册表里的键。

聚合语义与线上 nanwang_online/feature_aggregation.py 一致（历史段、未来段分别聚合）：
  - 一个成员在某行整段全 NaN 才算"该行缺失"；有效成员数 < min_valid 的行整行置 NaN；
  - 其余逐时刻对有效成员做 nanmedian / nanmean / nanmin / nanmax。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import fields as F

STEP_MIN = 15
HAURWITZ_A, HAURWITZ_B = 1098.0, 0.059


# ============================================================================ 数据结构
@dataclass
class Pair:
    hist: np.ndarray            # [B, C]
    fut: np.ndarray             # [B, P]；history_only 的量为全 NaN

    def full(self) -> np.ndarray:
        return np.concatenate([self.hist, self.fut], axis=1)

    @staticmethod
    def from_full(arr: np.ndarray, context_len: int) -> "Pair":
        return Pair(arr[:, :context_len].astype(np.float64), arr[:, context_len:].astype(np.float64))

    def map(self, fn: Callable[[np.ndarray], np.ndarray]) -> "Pair":
        return Pair(fn(self.hist), fn(self.fut))


@dataclass
class Ctx:
    """一批起报窗的公共上下文 + 惰性特征构建。"""
    ts_win: np.ndarray                          # [B] datetime64[s]，起报时刻 S
    context_len: int
    pred_len: int
    lat: np.ndarray                             # [B]，缺 -> NaN
    lon: np.ndarray
    cap: np.ndarray                             # [B] 装机容量（cap_power_on_val）
    load_raw: Callable[[str, bool], Pair]       # (原始历史列名, history_only) -> Pair
    specs: Dict[str, dict] = field(default_factory=dict)
    cols: Dict[str, Pair] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    tz_offset_h: float = 8.0
    stations: Optional[np.ndarray] = None       # [B] 站名，只用于告警里定位
    _building: set = field(default_factory=set)
    _stack: list = field(default_factory=list)
    _clear: Optional[Pair] = None

    def where(self, mask: np.ndarray, limit: int = 5) -> str:
        """把行掩码描述成 "站@起报时刻, ..."，便于回查数据。"""
        idx = np.flatnonzero(mask)
        st = self.stations if self.stations is not None else np.array(["?"] * len(self.ts_win))
        items = [f"{st[i]}@{pd.Timestamp(self.ts_win[i]):%Y-%m-%d %H:%M}" for i in idx[:limit]]
        return ", ".join(items) + (f" 等 {len(idx)} 行" if len(idx) > limit else "")

    def warn(self, msg: str):
        if self._stack:
            msg = f"[{self._stack[-1]}] {msg}"
        if msg not in self.warnings:
            self.warnings.append(msg)

    def has_spec(self, name: str) -> bool:
        return name in self.cols or name in self.specs

    def get(self, name: str) -> Pair:
        if name in self.cols:
            return self.cols[name]
        if name not in self.specs:
            raise KeyError(f"特征 {name!r} 没有定义（config.features 里加上它）")
        if name in self._building:
            raise ValueError(f"特征定义有循环引用: {name}")
        self._building.add(name)
        self._stack.append(name)
        try:
            spec = dict(self.specs[name])
            op = spec.pop("op")
            for k in ("quantity", "fusible", "note", "candidate"):
                spec.pop(k, None)
            if op != "grid":                            # grid 的 source 是参数，其余算子的 source 只是元信息
                spec.pop("source", None)
            if op != "raw":
                spec.pop("history_only", None)
            self.cols[name] = run_op(self, op, spec)
        finally:
            self._building.discard(name)
            self._stack.pop()
        return self.cols[name]

    def timestamps(self) -> np.ndarray:
        return window_timestamps(self.ts_win, self.context_len, self.pred_len)

    def clear_sky(self) -> Pair:
        if self._clear is None:
            ts = self.timestamps()
            have = np.isfinite(self.lat) & np.isfinite(self.lon)
            out = np.zeros(ts.shape)
            if have.any():
                out[have] = clear_sky_ghi(ts[have], self.lat[have], self.lon[have], self.tz_offset_h)
            if (~have).any():
                self.warn(f"{int((~have).sum())} 行缺站点坐标，晴空辐照退化为时钟几何（06:15-20:00 常数 800）")
                out[~have] = clock_clear_sky(ts[~have])
            self._clear = Pair.from_full(out, self.context_len)
        return self._clear


# ============================================================================ 太阳几何 / 晴空
def window_timestamps(ts_win, context_len: int, pred_len: int, step_min: int = STEP_MIN) -> np.ndarray:
    """[B, C+P]：历史最后一点 = S；未来第 i 点（0 起）= S + (i+1)·15min。"""
    s = np.asarray(pd.to_datetime(ts_win).values.astype("datetime64[s]")).astype("int64").reshape(-1, 1)
    offs = np.concatenate([np.arange(-(context_len - 1), 1), np.arange(1, pred_len + 1)]) * (step_min * 60)
    return (s + offs[None, :]).astype("datetime64[s]")


def _spencer(doy: np.ndarray):
    B = 2.0 * np.pi * (doy - 1.0) / 365.0
    decl = (0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B) - 0.006758 * np.cos(2 * B)
            + 0.000907 * np.sin(2 * B) - 0.002697 * np.cos(3 * B) + 0.001480 * np.sin(3 * B))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B)
                    - 0.014615 * np.cos(2 * B) - 0.040849 * np.sin(2 * B))
    return decl, eot


def cos_zenith(ts: np.ndarray, lat: np.ndarray, lon: np.ndarray, tz_offset_h: float = 8.0) -> np.ndarray:
    """ts [B, L] 北京时间墙钟、lat/lon [B] -> cosθz [B, L]（地平线下为负）。Spencer(1971)。"""
    t = pd.DatetimeIndex(ts.reshape(-1))
    doy = t.dayofyear.to_numpy(float).reshape(ts.shape)
    clock_h = (t.hour.to_numpy(float) + t.minute.to_numpy(float) / 60.0).reshape(ts.shape)
    decl, eot = _spencer(doy)
    phi = np.deg2rad(lat.astype(float))[:, None]
    solar_h = clock_h + (4.0 * (lon.astype(float)[:, None] - 15.0 * tz_offset_h) + eot) / 60.0
    omega = np.deg2rad(15.0 * (solar_h - 12.0))
    return np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(omega)


def clear_sky_ghi(ts: np.ndarray, lat: np.ndarray, lon: np.ndarray, tz_offset_h: float = 8.0) -> np.ndarray:
    """Haurwitz 晴空 GHI W/m²，夜间 0。"""
    cz = cos_zenith(ts, lat, lon, tz_offset_h)
    out = np.zeros_like(cz)
    day = cz > 0
    out[day] = HAURWITZ_A * cz[day] * np.exp(-HAURWITZ_B / cz[day])
    return out


def clock_clear_sky(ts: np.ndarray, day_start: str = "06:15", day_end: str = "20:00", level: float = 800.0) -> np.ndarray:
    t = pd.DatetimeIndex(ts.reshape(-1))
    tod = (t.hour * 60 + t.minute).to_numpy().reshape(ts.shape)
    lo = int(day_start[:2]) * 60 + int(day_start[3:]); hi = int(day_end[:2]) * 60 + int(day_end[3:])
    return np.where((tod >= lo) & (tod <= hi), level, 0.0)


# ============================================================================ 成员聚合（线上语义）
ONLINE_METHODS = ("median", "mean", "min", "max")
AGG_METHODS = ONLINE_METHODS + ("std", "trim_mean")


def agg_members(stack: np.ndarray, method: str, min_valid: int = 1) -> np.ndarray:
    """stack [M, B, L] -> [B, L]。成员在某行整段全 NaN 视为缺失；有效成员 < min_valid 的行置 NaN。"""
    stack = np.asarray(stack, dtype=np.float64)
    M, B, L = stack.shape
    field_valid = ~np.isnan(stack).all(axis=2)                       # [M, B]
    n_valid = field_valid.sum(axis=0)                                 # [B]
    v = np.where(field_valid[:, :, None], stack, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if method == "median":
            out = np.nanmedian(v, axis=0)
        elif method == "mean":
            out = np.nanmean(v, axis=0)
        elif method == "min":
            out = np.nanmin(v, axis=0)
        elif method == "max":
            out = np.nanmax(v, axis=0)
        elif method == "std":
            out = np.nanstd(v, axis=0)
        elif method == "trim_mean":                                   # 去掉一个最大一个最小再平均；成员 < 3 时退化为均值
            n = np.isfinite(v).sum(axis=0)
            s = np.nansum(v, axis=0)
            trimmed = (s - np.nanmax(v, axis=0) - np.nanmin(v, axis=0)) / np.maximum(n - 2, 1)
            out = np.where(n >= 3, trimmed, np.nanmean(v, axis=0))
        else:
            raise ValueError(f"未知聚合方法 {method!r}；可用 {AGG_METHODS}")
    out = np.where((n_valid >= max(min_valid, 1))[:, None], out, np.nan)
    return out


def weighted_members(stack: np.ndarray, w: np.ndarray, min_valid: int = 1) -> np.ndarray:
    """stack [M, B, L]，w [M] 或 [M, B]；逐时刻在非 NaN 成员上重新归一化权重。"""
    stack = np.asarray(stack, dtype=np.float64)
    M, B, L = stack.shape
    w = np.asarray(w, dtype=np.float64)
    w = np.broadcast_to(w.reshape(M, -1, 1) if w.ndim == 2 else w.reshape(M, 1, 1), (M, B, L))
    field_valid = ~np.isnan(stack).all(axis=2)
    n_valid = field_valid.sum(axis=0)
    ok = np.isfinite(stack) & field_valid[:, :, None]
    ww = np.where(ok, w, 0.0)
    den = ww.sum(axis=0)
    out = np.where(den > 1e-12, (np.where(ok, stack, 0.0) * ww).sum(axis=0) / np.where(den > 1e-12, den, 1.0), np.nan)
    return np.where((n_valid >= max(min_valid, 1))[:, None], out, np.nan)


def _seg_agg(pairs: Sequence[Pair], method: str, min_valid: int) -> Pair:
    """历史段、未来段分别聚合（线上 RULES 也是 hist 与 _predict 各一条规则）。"""
    return Pair(agg_members(np.stack([p.hist for p in pairs]), method, min_valid),
                agg_members(np.stack([p.fut for p in pairs]), method, min_valid))


# ============================================================================ 原始字段
def op_raw(ctx: Ctx, col: str, history_only: bool = False, scale: float = 1.0, offset: float = 0.0) -> Pair:
    F.check_raw(col, history_only)
    p = ctx.load_raw(col, history_only)
    if scale != 1.0 or offset != 0.0:
        p = p.map(lambda a: a * scale + offset)
    return p


def op_grid(ctx: Ctx, source: str, var: str, method: str = "median", min_valid: int = 2,
            point: Optional[int] = None, points: Optional[Sequence[int]] = None,
            scale: float = 1.0, offset: float = 0.0) -> Pair:
    """格点源（EC / GDFS / SS）的 9 个格点 -> 一列。method=point 取单个格点（默认中心点 13）。"""
    if source not in F.GRID_SOURCES:
        raise ValueError(f"source={source!r} 不是格点源；可用 {sorted(F.GRID_SOURCES)}")
    if method == "point":
        p = op_raw(ctx, F.grid_col(source, var, int(point or F.CENTER_POINT)))
    else:
        pts = list(points or F.POINTS)
        members = []
        for k in pts:                                   # 线上 missing="skip"：表里没有的格点列直接跳过
            try:
                members.append(op_raw(ctx, F.grid_col(source, var, int(k))))
            except KeyError as e:
                ctx.warn(f"grid {source}/{var}: {e}，跳过该格点")
        if not members:
            raise KeyError(f"grid {source}/{var}: 9 个格点列都不在 DS 表里")
        p = _seg_agg(members, method, min_valid)
    if scale != 1.0 or offset != 0.0:
        p = p.map(lambda a: a * scale + offset)
    return p


# ============================================================================ 跨源聚合
def _day_mask_hist(ctx: Ctx, min_clear: float) -> np.ndarray:
    return ctx.clear_sky().hist >= min_clear


def _reference_hist(ctx: Ctx, reference: str) -> np.ndarray:
    """技巧/偏差估计的参考真值（只用历史段）。reference='power' 时用功率折算的辐照 observe_power/cap*1000。"""
    if reference == "power":
        p = ctx.get(F.TARGET).hist
        return p / np.maximum(ctx.cap[:, None], 1e-6) * 1000.0
    return ctx.get(reference).hist


def op_combine(ctx: Ctx, members: Sequence[str], method: str = "median", min_valid: int = 1,
               weights: Optional[Sequence[float]] = None, reference: str = "ghi_real",
               err_power: float = 1.0, min_clear: float = 50.0, window_steps: Optional[int] = None,
               bias_clip: Sequence[float] = (0.5, 2.0), min_points: int = 8) -> Pair:
    """多源合成一列。

    method:
      median / mean / min / max / std / trim_mean  —— 逐时刻聚合（线上语义），std 即多源分歧度；
      weighted_mean —— 固定权重 weights（与 members 一一对应）；
      inv_error     —— 每行按历史段白天对参考真值的 RMSE^err_power 倒数加权（只用起报前数据）；
      bias_mean     —— 每个成员先按历史段白天 sum(参考)/sum(成员) 做乘性订正（截到 bias_clip），再取均值。
    两者都要求某成员在窗口内至少有 min_points 个"白天且成员、参考都有值"的点才估计它的权重/订正系数；
    不足的成员：inv_error 权重置 0（所有成员都不足时整行退化为等权），bias_mean 不订正（系数 1）。
    """
    pairs = [ctx.get(m) for m in members]
    if method in AGG_METHODS:
        return _seg_agg(pairs, method, min_valid)
    H = np.stack([p.hist for p in pairs]); Fu = np.stack([p.fut for p in pairs])     # [M, B, L]
    if method == "weighted_mean":
        if weights is None or len(weights) != len(members):
            raise ValueError("weighted_mean 需要与 members 等长的 weights")
        w = np.asarray(weights, float)
        return Pair(weighted_members(H, w, min_valid), weighted_members(Fu, w, min_valid))
    ref = _reference_hist(ctx, reference)
    day = _day_mask_hist(ctx, min_clear)
    if window_steps:
        day = day.copy(); day[:, :-int(window_steps)] = False
    ok = day[None] & np.isfinite(H) & np.isfinite(ref)[None]                          # [M, B, C]
    n = ok.sum(axis=2)                                                                # [M, B] 可用点数
    enough = n >= int(min_points)
    if method == "inv_error":
        mse = np.where(ok, (H - ref[None]) ** 2, 0.0).sum(axis=2) / np.maximum(n, 1)
        rmse = np.sqrt(mse)
        w = np.where(enough, 1.0 / np.maximum(rmse, 1e-3) ** float(err_power), np.nan)
        no_skill = ~np.isfinite(w).any(axis=0)                                        # 所有成员都估不出：等权
        if no_skill.any():
            ref_empty = no_skill & ~(np.isfinite(ref) & day).any(axis=1)
            ctx.warn(f"inv_error: {int(no_skill.sum())}/{len(no_skill)} 行无法估计权重，退化为等权"
                     f"（其中 {int(ref_empty.sum())} 行窗口内白天参考真值 {reference} 全缺）："
                     f"{ctx.where(no_skill)}")
        w = np.where(np.isfinite(w), w, 0.0)
        w[:, no_skill] = 1.0
        return Pair(weighted_members(H, w, min_valid), weighted_members(Fu, w, min_valid))
    if method == "bias_mean":
        num = np.where(ok, ref[None], 0.0).sum(axis=2)
        den = np.where(ok, H, 0.0).sum(axis=2)
        k = np.where(enough & (den > 1e-6), num / np.where(den > 1e-6, den, 1.0), 1.0)
        none = ~enough.any(axis=0)
        if none.any():
            ctx.warn(f"bias_mean: {int(none.sum())}/{len(none)} 行参考真值 {reference} 不足，不做订正（等同均值）："
                     f"{ctx.where(none)}")
        k = np.clip(k, float(bias_clip[0]), float(bias_clip[1]))[:, :, None]          # [M, B, 1]
        return Pair(agg_members(H * k, "mean", min_valid), agg_members(Fu * k, "mean", min_valid))
    raise ValueError(f"combine 不支持 method={method!r}")


# ============================================================================ 衍生量
def op_clear_sky(ctx: Ctx) -> Pair:
    return ctx.clear_sky()


def op_kt(ctx: Ctx, ghi: str, min_clear: float = 50.0, kt_max: float = 1.3) -> Pair:
    """晴空指数 = ghi / 晴空辐照；晴空辐照 < min_clear 的时刻置 0。"""
    g, c = ctx.get(ghi), ctx.clear_sky()
    def f(a, b):
        ok = b >= min_clear
        return np.clip(np.where(ok, a / np.where(ok, b, 1.0), 0.0), 0.0, kt_max)
    return Pair(f(g.hist, c.hist), f(g.fut, c.fut))


def op_p_phys(ctx: Ctx, ghi: str, temp: Optional[str] = None, wind: Optional[str] = None,
              gamma: float = -0.004, u0: float = 25.0, u1: float = 6.84, ws_default: float = 2.0,
              t_ref: float = 25.0) -> Pair:
    """理论功率 = cap × GHI/1000 × [1 + γ(T_cell − 25)]，T_cell 用 Faiman：T_amb + GHI/(u0 + u1·ws)。"""
    g = ctx.get(ghi)
    t = ctx.get(temp) if temp else None
    ws = ctx.get(wind) if wind else None
    cap = ctx.cap.astype(float)[:, None]
    def f(gg, tt, ww):
        gg = np.clip(gg, 0.0, None)
        eff = 1.0
        if tt is not None:
            w_ = np.full_like(gg, ws_default) if ww is None else np.nan_to_num(np.clip(ww, 0, None), nan=ws_default)
            tc = tt + gg / (u0 + u1 * w_)
            eff = 1.0 + gamma * (tc - t_ref)
        return np.clip(cap * np.clip(gg / 1000.0, 0.0, 1.2) * eff, 0.0, None)
    return Pair(f(g.hist, t.hist if t else None, ws.hist if ws else None),
                f(g.fut, t.fut if t else None, ws.fut if ws else None))


def op_threshold(ctx: Ctx, x: str, thr: float, above: bool = True) -> Pair:
    """二值化：x > thr 为 1（above=False 时 x < thr 为 1），NaN 保持 NaN。"""
    def f(a):
        b = (a > thr) if above else (a < thr)
        return np.where(np.isfinite(a), b.astype(float), np.nan)
    return ctx.get(x).map(f)


def op_scale(ctx: Ctx, x: str, k: float = 1.0, offset: float = 0.0) -> Pair:
    return ctx.get(x).map(lambda a: a * k + offset)


def op_diff(ctx: Ctx, a: str, b: str) -> Pair:
    pa, pb = ctx.get(a), ctx.get(b)
    return Pair(pa.hist - pb.hist, pa.fut - pb.fut)


def op_trig(ctx: Ctx, x: str, fn: str = "sin") -> Pair:
    """角度（度）-> sin/cos，用于风向这类环形量。"""
    f = np.sin if fn == "sin" else np.cos
    return ctx.get(x).map(lambda a: f(np.deg2rad(a)))


OPS: Dict[str, Callable[..., Pair]] = {
    "raw": op_raw, "grid": op_grid, "combine": op_combine,
    "clear_sky": op_clear_sky, "kt": op_kt, "p_phys": op_p_phys,
    "threshold": op_threshold, "scale": op_scale, "diff": op_diff, "trig": op_trig,
}


def run_op(ctx: Ctx, op: str, args: Optional[dict] = None) -> Pair:
    if op not in OPS:
        raise KeyError(f"未知算子 {op!r}；可用: {sorted(OPS)}")
    return OPS[op](ctx, **dict(args or {}))


def dependencies(spec: dict) -> List[str]:
    """一个特征定义直接引用的其他特征名（用于校验与推断可融合成员）。"""
    deps = []
    for k in ("ghi", "temp", "wind", "x", "a", "b"):
        v = spec.get(k)
        if isinstance(v, str):
            deps.append(v)
    if spec.get("op") == "combine":
        deps += list(spec.get("members") or [])
        ref = spec.get("reference", "ghi_real")
        if spec.get("method") in ("inv_error", "bias_mean") and ref != "power":
            deps.append(ref)
    return deps
