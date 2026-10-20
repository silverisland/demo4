# forecast_backbones/m8/decision.py（2026-10-13 起自带一份，不再引用 m6）
"""南网口径决策层：分位函数 → 物理钳位 → 最优上报点。纯 numpy。
短期（RMSE 型）最优点 p* = Σ y w2 / Σ w2，w2 = 1/max(y,0.2C)²；
超短期（MAE 型）最优点 = w1 加权中位数，w1 = 1/max(y,0.2C)。"""
from __future__ import annotations

import numpy as np

MODEL_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CAL_LEVELS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)


def _interp_weights(levels, taus):
    lv = np.asarray(levels, float)
    idx = np.clip(np.searchsorted(lv, taus, side="right") - 1, 0, len(lv) - 2)
    lo, hi = idx, idx + 1
    w = np.clip((taus - lv[lo]) / (lv[hi] - lv[lo]), 0.0, 1.0)      # 两端平延
    return lo, hi, w


def quantile_grid(q, levels, step: float = 0.005) -> np.ndarray:
    """[..., K] 分位 → [..., G] 等概率网格上的分位函数值（线性插值，尾部平延）。"""
    q = np.asarray(q, float)
    taus = np.arange(step / 2, 1.0, step)
    lo, hi, w = _interp_weights(levels, taus)
    return q[..., lo] * (1.0 - w) + q[..., hi] * w


def level_value(q, levels, tau: float) -> np.ndarray:
    q = np.asarray(q, float)
    lo, hi, w = _interp_weights(levels, np.asarray([tau], float))
    return (q[..., lo] * (1.0 - w) + q[..., hi] * w)[..., 0]


def _cap_like(cap, grid: np.ndarray) -> np.ndarray:
    """cap 广播成 grid.shape[:-1] 再加尾维。"""
    return np.broadcast_to(np.asarray(cap, float), grid.shape[:-1])[..., None]


def p_star(q, levels, cap, floor_ratio: float = 0.2, step: float = 0.005) -> np.ndarray:
    y = quantile_grid(q, levels, step)
    w = 1.0 / np.maximum(y, floor_ratio * _cap_like(cap, y)) ** 2
    return (y * w).sum(-1) / w.sum(-1)


def p_star_l1(q, levels, cap, floor_ratio: float = 0.2, step: float = 0.005) -> np.ndarray:
    y = quantile_grid(q, levels, step)                       # 沿网格非降
    w = 1.0 / np.maximum(y, floor_ratio * _cap_like(cap, y))
    cw = np.cumsum(w, axis=-1)
    k = (cw >= 0.5 * cw[..., -1:]).argmax(axis=-1)
    return np.take_along_axis(y, k[..., None], axis=-1)[..., 0]


def global_shrink_factor(y, p, cap, floor_ratio: float = 0.2) -> float:
    """f* = Σ y p / d² / Σ p² / d²，d = max(y, 0.2C)。cap 标量或与 y 同长的逐点数组；只用 y/p/cap 三者都有限的点。
    可用点为空时抛 ValueError；den<=0（如全部预测为 0）时退化返回 1.0（无收缩信息）。"""
    y = np.asarray(y, float).reshape(-1); p = np.asarray(p, float).reshape(-1)
    capv = np.broadcast_to(np.asarray(cap, float).reshape(-1) if np.ndim(cap) else np.asarray(cap, float), y.shape)
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(capv)
    if not ok.any():
        raise ValueError("global_shrink_factor: no usable points")
    d = np.maximum(y[ok], floor_ratio * capv[ok])
    den = np.sum(p[ok] ** 2 / d ** 2)
    return float(np.sum(y[ok] * p[ok] / d ** 2) / den) if den > 0 else 1.0


def enforce_monotone(q) -> np.ndarray:
    return np.sort(np.asarray(q, float), axis=-1)


def physics_clip(q, night, p_clear, cap, clear_sky_cap_ratio: float = 1.05, capacity_cap: bool = True,
                 night_ceiling: bool = True, clip_negative: bool = True) -> np.ndarray:
    """白天 clip 到 min(clear_sky_cap_ratio·p_clear, cap)，night 掩码为 True 的步归零。
    p_clear 非有限值（缺测/NaN）时该步退化为只用装机容量 cap 钳位（不受 capacity_cap 开关影响），
    不设晴空上界，避免 NaN 传播到预报。
    night_ceiling=True（默认，m6 口径）：p_clear<=0 的步 upper=0 → 预报被钳成 0。注意这等于隐式的夜间归零：
      夜间晴空 GHI 恒为 0，所以就算 night 掩码全 False、clear_sky_cap_ratio 再大，夜间也会被钳成 0。
    night_ceiling=False（M8 night_zero=false 的口径）：p_clear<=0 的步与非有限值同样处理，只用 cap 钳位，
      夜间 / 退化站窗外的输出由模型自己给，决策层不做任何形式的夜间归零。
    clip_negative=False：负值也不截到 0（严格裸对比用；负功率物理上不可能，默认仍截）。"""
    q = np.asarray(q, float).copy()
    pc = np.asarray(p_clear, float)
    capv = np.broadcast_to(np.asarray(cap, float), pc.shape)
    upper = clear_sky_cap_ratio * pc
    if capacity_cap:
        upper = np.minimum(upper, capv)
    free = ~np.isfinite(pc) if night_ceiling else (~np.isfinite(pc) | (pc <= 0.0))
    upper = np.where(free, capv, upper)
    q = np.minimum(q, upper[..., None])
    if clip_negative:
        q = np.maximum(q, 0.0)
    q[np.asarray(night, bool)] = 0.0
    return q


def _point_by_method(q, levels, cap_bt, method: str, shrink_factor: float, floor_ratio: float, step: float):
    if method == "p_star":
        return p_star(q, levels, cap_bt, floor_ratio, step)
    if method == "p_star_l1":
        return p_star_l1(q, levels, cap_bt, floor_ratio, step)
    if method == "median":
        return level_value(q, levels, 0.5)
    if method == "global_shrink":
        return shrink_factor * level_value(q, levels, 0.5)
    if method.startswith("quantile:"):
        return level_value(q, levels, float(method.split(":", 1)[1]))
    raise ValueError(f"未知 decision.point.method: {method!r}")


def select_point(q, levels, cap, default: str, by_step=(), shrink_factor: float = 1.0,
                 floor_ratio: float = 0.2, step: float = 0.005) -> np.ndarray:
    """q [B,T,K] → [B,T]。by_step: [{'steps':[lo,hi), 'method':...}]，先匹配者胜；其余用 default。
    default 只在 by_step 规则未覆盖的步上计算（不对全部 T 都跑一遍再被覆盖丢弃）。"""
    q = np.asarray(q, float)
    B, T, _ = q.shape
    cap_bt = np.broadcast_to(np.asarray(cap, float).reshape(B, 1), (B, T))
    out = np.empty((B, T), float)
    assigned = np.zeros(T, bool)
    for rule in by_step or ():
        lo, hi = int(rule["steps"][0]), int(rule["steps"][1])
        sl = np.arange(lo, min(hi, T))
        sl = sl[~assigned[sl]]
        if sl.size == 0:
            continue
        out[:, sl] = _point_by_method(q[:, sl], levels, cap_bt[:, sl], rule["method"],
                                      shrink_factor, floor_ratio, step)
        assigned[sl] = True
    remaining = np.flatnonzero(~assigned)
    if remaining.size:
        out[:, remaining] = _point_by_method(q[:, remaining], levels, cap_bt[:, remaining], default,
                                             shrink_factor, floor_ratio, step)
    return out


def quantile_cols(levels) -> list[str]:
    """结果表里各分位列的列名（q_0.1 … q_0.9）。"""
    return [f"q_{float(l):g}" for l in levels]
