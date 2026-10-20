# -*- coding: utf-8 -*-
"""fields.py —— 允许使用的 DS 列（白名单）与成对约定。

线上 ds_trajectory_input parquet 读入后（drop update_time）就是这些列。实验只能从这里取字段：
  - 历史列 <col>（长 672，最后一点落在起报时刻）必须与未来列 <col>_predict（长 192 或 480）成对使用；
  - GHI_real 只有历史段，只能作 past-only 协变量或参考真值；
  - observe_power_future 是标签，任何特征都不许读它。
"""
from __future__ import annotations

from typing import Iterable, List

FUTURE_SUFFIX = "_predict"
ID_COLS = ("station", "timestamp_win")
TARGET = "observe_power"
LABEL = "observe_power_future"
CAP = "cap_power_on"
HISTORY_ONLY = ("GHI_real",)

# 9 个格点：5×5 网格中间的 3×3，编号 7..19，中心点是 13
POINTS = (7, 8, 9, 12, 13, 14, 17, 18, 19)
CENTER_POINT = 13

# 格点化的气象源：source -> (前缀, {量: 原始字段名})
GRID_SOURCES = {
    "EC": ("EC_JTKRD7RO_", {"ghi": "GHI", "temp": "t2m"}),
    "GDFS": ("GDFS_JTHFGNED_MID_JTNWQISX_", {"ghi": "GHI", "temp": "tem_hl_10"}),
    "SS": ("solar_southern_", {"ghi": "GHI", "temp": "t2m"}),
}
# SOLARGIS：站点级单列
SOLARGIS = ("GHI_SOLARGIS", "TEMP_SOLARGIS", "WS_SOLARGIS", "WD_SOLARGIS",
            "PREC_SOLARGIS", "PWAT_SOLARGIS", "SDWE_SOLARGIS")


def grid_col(source: str, var: str, point: int) -> str:
    prefix, vars_ = GRID_SOURCES[source]
    return f"{prefix}{vars_[var]}_p{point}"


def _allowed() -> List[str]:
    cols = list(ID_COLS) + [TARGET, "GHI_real", CAP]
    hist = list(SOLARGIS)
    for src in GRID_SOURCES:
        for var in ("ghi", "temp"):
            hist += [grid_col(src, var, p) for p in POINTS]
    cols += hist + [h + FUTURE_SUFFIX for h in hist] + [LABEL]
    return cols


ALLOWED_COLUMNS = tuple(_allowed())
_ALLOWED = frozenset(ALLOWED_COLUMNS)


def is_allowed(col: str) -> bool:
    return col in _ALLOWED


def check_raw(col: str, history_only: bool = False) -> None:
    """一个原始历史列能否当特征用：在白名单里、不是标签/ID、且（非 history_only 时）有对应 _predict 列。"""
    if col not in _ALLOWED:
        raise ValueError(f"列 {col!r} 不在允许的字段清单里（见 fields.ALLOWED_COLUMNS）")
    if col in (LABEL,) or col in ID_COLS or col == CAP:
        raise ValueError(f"列 {col!r} 是标签/ID/容量列，不能当协变量")
    if col.endswith(FUTURE_SUFFIX):
        raise ValueError(f"请写历史列名 {col[:-len(FUTURE_SUFFIX)]!r}，未来列自动配对")
    if history_only:
        if col not in HISTORY_ONLY and col != TARGET:
            raise ValueError(f"列 {col!r} 有未来段，不应按 history_only 使用")
        return
    if col in HISTORY_ONLY:
        raise ValueError(f"列 {col!r} 只有历史段，只能 history_only=true 作 past-only 协变量")
    if col + FUTURE_SUFFIX not in _ALLOWED:
        raise ValueError(f"列 {col!r} 没有对应的 {col + FUTURE_SUFFIX!r}")


def missing_in(columns: Iterable[str]) -> List[str]:
    """白名单里有、但这批 DS 表里没有的列（用于数据体检）。"""
    have = set(columns)
    return [c for c in ALLOWED_COLUMNS if c not in have]


def extra_in(columns: Iterable[str]) -> List[str]:
    """DS 表里有、但不在白名单里的列：实验不会读它们。"""
    return sorted(c for c in columns if c not in _ALLOWED and c != "update_time")
