"""
气象预测融合模型 — 特征聚合模块

遵循 feature_aggregation_specification.md v1.0 规范。

核心职责：接收 DataFrame 和聚合规则列表，按规则顺序（级联）执行聚合，
将多个气象源字段融合为单一鲁棒输出列。

支持两种列类型：
  - 标量列：每个单元格为 float，聚合输出 (n_rows,) 一维数组
  - 向量列：每个单元格为 list / np.ndarray，聚合输出 (n_rows, vec_dim)

设计原则：
  1. 聚合函数为纯函数，不感知 DataFrame / 配置 / 字段解析
  2. 字段解析、值提取、权重提取各自独立
  3. aggregate_features() 仅做编排，细节全部委托给内部函数
  4. 通过注册表（AGGREGATORS）新增聚合方法无需改动其他代码
  5. 输入校验尽早失败（fail-fast），给出明确错误信息
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


# ============================================================================
# 聚合函数 — 纯函数，仅依赖 numpy；同时支持 2D 标量和 3D 向量输入
# ============================================================================

def _weighted_mean(
    values: np.ndarray, weights: np.ndarray, min_valid: int
) -> np.ndarray:
    """行级加权平均，含 min_valid 约束。"""
    if values.ndim == 3:
        return _weighted_mean_3d(values, weights, min_valid)
    return _weighted_mean_2d(values, weights, min_valid)


def _weighted_mean_2d(
    values: np.ndarray, weights: np.ndarray, min_valid: int
) -> np.ndarray:
    """标量加权平均：(n_rows, n_fields) → (n_rows,)"""
    mask = ~np.isnan(values)
    valid_count = mask.sum(axis=1)

    w = weights[np.newaxis, :]  # (1, n_fields)
    w_masked = np.where(mask, w, 0.0)
    w_sum = w_masked.sum(axis=1)

    v_safe = np.where(mask, values, 0.0)
    numerator = (v_safe * w_masked).sum(axis=1)

    result = np.full(values.shape[0], np.nan)
    valid = (valid_count >= min_valid) & (w_sum > 1e-15)
    result[valid] = numerator[valid] / w_sum[valid]
    return result


def _weighted_mean_3d(
    values: np.ndarray, weights: np.ndarray, min_valid: int
) -> np.ndarray:
    """向量加权平均：(n_rows, n_fields, vec_dim) → (n_rows, vec_dim)"""
    n_rows, n_fields, vec_dim = values.shape

    is_nan = np.isnan(values)  # 只算一次
    field_valid = ~is_nan.all(axis=2)  # (n_rows, n_fields)
    valid_count = field_valid.sum(axis=1)  # (n_rows,)

    w = weights.reshape(1, n_fields, 1)  # (1, n_fields, 1)
    w_masked = np.where(field_valid[:, :, np.newaxis], w, 0.0)
    w_sum = w_masked.sum(axis=1)  # (n_rows, vec_dim)

    v_safe = np.where(is_nan, 0.0, values)
    numerator = (v_safe * w_masked).sum(axis=1)  # (n_rows, vec_dim)

    result = np.full((n_rows, vec_dim), np.nan)
    valid_rows = valid_count >= min_valid
    valid_mask = valid_rows[:, np.newaxis] & (w_sum > 1e-15)
    result[valid_mask] = numerator[valid_mask] / w_sum[valid_mask]
    return result


def _make_unweighted_agg(
    np_func: Callable[[np.ndarray, int], np.ndarray],
) -> Callable[[np.ndarray, int], np.ndarray]:
    """工厂函数，将 np.nan* 包装为支持 min_valid 且兼容 2D/3D 输入。"""

    def wrapper(values: np.ndarray, min_valid: int) -> np.ndarray:
        if values.ndim == 3:
            return _unweighted_3d(values, min_valid, np_func)
        return _unweighted_2d(values, min_valid, np_func)

    return wrapper


def _unweighted_2d(
    values: np.ndarray,
    min_valid: int,
    np_func: Callable[[np.ndarray, int], np.ndarray],
) -> np.ndarray:
    """标量聚合：(n_rows, n_fields) → (n_rows,)"""
    mask = ~np.isnan(values)
    valid_count = mask.sum(axis=1)
    result = np.full(values.shape[0], np.nan)
    valid = valid_count >= min_valid
    if valid.any():
        result[valid] = np_func(values[valid], axis=1)
    return result


def _unweighted_3d(
    values: np.ndarray,
    min_valid: int,
    np_func: Callable[[np.ndarray, int], np.ndarray],
) -> np.ndarray:
    """向量聚合：(n_rows, n_fields, vec_dim) → (n_rows, vec_dim)"""
    n_rows, _n_fields, vec_dim = values.shape

    field_mask = ~np.isnan(values).all(axis=2)  # (n_rows, n_fields)
    valid_count = field_mask.sum(axis=1)

    result = np.full((n_rows, vec_dim), np.nan)
    valid = valid_count >= min_valid
    if not valid.any():
        return result

    if field_mask.all():
        # 全字段有效 → 无需拷贝
        result[valid] = np_func(values[valid], axis=1)
    else:
        invalidated = values.copy()
        invalidated[~field_mask] = np.nan
        result[valid] = np_func(invalidated[valid], axis=1)

    return result


# 聚合方法注册表
AGGREGATORS: dict[str, tuple[Callable, bool]] = {
    "mean": (_make_unweighted_agg(np.nanmean), False),
    "weighted_mean": (_weighted_mean, True),
    "median": (_make_unweighted_agg(np.nanmedian), False),
    "max": (_make_unweighted_agg(np.nanmax), False),
    "min": (_make_unweighted_agg(np.nanmin), False),
    "sum": (_make_unweighted_agg(np.nansum), False),
}

# 合法的缺失策略
_VALID_MISSING_STRATEGIES: frozenset[str] = frozenset({"skip", "raise", "fill_nan"})


# ============================================================================
# 内部辅助函数 — 各司其职
# ============================================================================

def _is_field_available(df: pd.DataFrame, field: str) -> bool:
    """判断字段是否有效：存在于 DataFrame 中，且不全为 NaN。"""
    if field not in df.columns:
        return False
    return not df[field].isna().all()


def _resolve_fields(
    df: pd.DataFrame,
    raw_fields: list[dict[str, Any]],
    missing_strategy: str,
    rule_index: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """单次遍历，将字段分为可用和缺失。"""
    available: list[dict[str, Any]] = []
    missing_names: list[str] = []

    for spec in raw_fields:
        if _is_field_available(df, spec["field"]):
            available.append(spec)
        else:
            missing_names.append(spec["field"])

    if missing_strategy == "raise" and missing_names:
        raise KeyError(
            f"Rule {rule_index}: required fields missing or all-NaN: "
            f"{missing_names}. Available columns: {list(df.columns)}"
        )

    if missing_strategy == "fill_nan":
        return raw_fields, missing_names

    return available, missing_names


def _detect_value_type(
    df: pd.DataFrame,
    field_specs: list[dict[str, Any]],
    missing_names: list[str],
) -> tuple[str, int | None]:
    """检测值类型：标量 or 向量。同时校验类型一致性。

    Raises:
        ValueError: 向量维度不一致，或标量/向量字段混用。
    """
    seen_scalar: bool = False
    vec_dim: int | None = None

    for spec in field_specs:
        field = spec["field"]
        if field in missing_names:
            continue
        series = df[field]
        for val in series:
            if val is None:
                continue
            # 0 维 ndarray → 视作标量
            if isinstance(val, np.ndarray) and val.ndim == 0:
                if vec_dim is not None:
                    raise ValueError(
                        f"Mixed value types: field '{field}' is scalar (0-d array) "
                        f"but another field is vector (dim={vec_dim}). "
                        f"All fields must be consistently scalar or vector."
                    )
                seen_scalar = True
                break
            # 向量（list 或多维 ndarray）
            if isinstance(val, (list, np.ndarray)):
                dim = len(val)
                if seen_scalar:
                    raise ValueError(
                        f"Mixed value types: field '{field}' is vector (dim={dim}) "
                        f"but another field is scalar. "
                        f"All fields must be consistently scalar or vector."
                    )
                if vec_dim is None:
                    vec_dim = dim
                elif dim != vec_dim:
                    raise ValueError(
                        f"Vector dimension mismatch: field '{field}' has dim={dim}, "
                        f"but expected dim={vec_dim}. "
                        f"All vector fields must have the same dimension."
                    )
                break
            # 标量 float
            if pd.isna(val):
                continue
            if vec_dim is not None:
                raise ValueError(
                    f"Mixed value types: field '{field}' is scalar "
                    f"but another field is vector (dim={vec_dim}). "
                    f"All fields must be consistently scalar or vector."
                )
            seen_scalar = True
            break

    return ("vector", vec_dim) if vec_dim is not None else ("scalar", None)


def _build_scalar_matrix(
    df: pd.DataFrame,
    field_specs: list[dict[str, Any]],
    missing_names: list[str],
    n_rows: int,
) -> np.ndarray:
    """标量模式：(n_rows, n_fields)"""
    arrays: list[np.ndarray] = []
    for spec in field_specs:
        field = spec["field"]
        if field in missing_names:
            arrays.append(np.full(n_rows, np.nan))
        else:
            arrays.append(np.asarray(df[field].values, dtype=float))
    return np.column_stack(arrays)


def _build_vector_matrix(
    df: pd.DataFrame,
    field_specs: list[dict[str, Any]],
    missing_names: list[str],
    n_rows: int,
    vec_dim: int,
) -> np.ndarray:
    """向量模式：(n_rows, n_fields, vec_dim)"""
    matrices: list[np.ndarray] = []
    for spec in field_specs:
        field = spec["field"]
        if field in missing_names:
            matrices.append(np.full((n_rows, vec_dim), np.nan))
            continue

        series = df[field]
        mat = np.full((n_rows, vec_dim), np.nan)
        non_null = series.notna()

        if non_null.any():
            valid_vals = series[non_null]
            try:
                mat[non_null.values] = np.array(valid_vals.tolist(), dtype=float)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Field '{field}': cannot convert to uniform float array "
                    f"(expected dim={vec_dim}). "
                    f"Ensure all rows have the same vector length. "
                    f"Original error: {e}"
                ) from e

        matrices.append(mat)

    return np.stack(matrices, axis=1)  # (n_fields, n_rows, vec_dim) → (n_rows, n_fields, vec_dim)


def _build_value_matrix(
    df: pd.DataFrame, field_specs: list[dict[str, Any]], missing_names: list[str]
) -> np.ndarray:
    """自动检测类型并构建数值矩阵。"""
    n_rows = len(df)
    value_type, vec_dim = _detect_value_type(df, field_specs, missing_names)

    if value_type == "vector":
        return _build_vector_matrix(df, field_specs, missing_names, n_rows, vec_dim)
    return _build_scalar_matrix(df, field_specs, missing_names, n_rows)


def _extract_weights(
    field_specs: list[dict[str, Any]],
    normalize: bool,
) -> np.ndarray:
    """提取权重数组，可选归一化。"""
    weights = np.array([s.get("weight", 1.0) for s in field_specs], dtype=float)
    if normalize:
        total = weights.sum()
        if total > 1e-15:
            weights = weights / total
    return weights


def _store_result(df: pd.DataFrame, col: str, result: np.ndarray) -> None:
    """将聚合结果写入 DataFrame 列。"""
    ndim = result.ndim
    if ndim == 0:
        df[col] = float(result)
    elif ndim == 1:
        df[col] = result
    elif ndim == 2:
        df[col] = list(result)
    else:
        df[col] = result


# ============================================================================
# 公共 API
# ============================================================================

def aggregate_features(
    df: pd.DataFrame,
    rules: list[dict[str, Any]],
) -> pd.DataFrame:
    """按规则列表顺序对 DataFrame 执行特征聚合（级联）。

    支持标量列和向量（array/list）列。

    Args:
        df: 输入 DataFrame，包含气象特征列。
        rules: 聚合规则列表，每条规则为字典：

            - agg_output_col (str)       — 输出列名
            - agg_method (str)           — 聚合方法名称
            - input_cols (dict)          — {"fields": [{"field": ..., "weight": ...}, ...]}
            - missing (str, optional)    — "skip"(默认) | "raise" | "fill_nan"
            - min_valid (int, optional)  — 最少有效字段数，默认 1
            - normalize_weights (bool)   — 是否归一化权重，默认 True

    Returns:
        新 DataFrame，额外包含聚合输出列。不修改原始 DataFrame。

    Raises:
        ValueError: 配置非法（不支持的 method/missing、min_valid<0、向量维度不一致等）。
        KeyError: missing="raise" 且存在不可用字段时。
    """
    df = df.copy()

    for i, rule in enumerate(rules):
        # ── 解析 + 默认值 ──
        agg_output_col: str = rule["agg_output_col"]
        agg_method: str = rule["agg_method"]
        missing_strategy: str = rule.get("missing", "skip")
        min_valid: int = int(rule.get("min_valid", 1))
        normalize_weights: bool = bool(rule.get("normalize_weights", True))

        # ── 合法性校验（fail-fast） ──
        if agg_method not in AGGREGATORS:
            raise ValueError(
                f"Rule {i}: unsupported agg_method '{agg_method}'. "
                f"Supported: {sorted(AGGREGATORS.keys())}"
            )
        if missing_strategy not in _VALID_MISSING_STRATEGIES:
            raise ValueError(
                f"Rule {i}: invalid missing strategy '{missing_strategy}'. "
                f"Must be one of: {sorted(_VALID_MISSING_STRATEGIES)}"
            )
        if min_valid < 0:
            raise ValueError(
                f"Rule {i}: min_valid must be >= 0, got {min_valid}"
            )

        input_cols = rule.get("input_cols")
        if not isinstance(input_cols, dict) or "fields" not in input_cols:
            raise ValueError(
                f"Rule {i}: 'input_cols' must be a dict with a 'fields' key, "
                f"got: {type(input_cols).__name__}"
            )
        raw_fields: list[dict[str, Any]] = input_cols["fields"]
        if not isinstance(raw_fields, list):
            raise ValueError(
                f"Rule {i}: 'input_cols.fields' must be a list, "
                f"got: {type(raw_fields).__name__}"
            )

        if not raw_fields:
            df[agg_output_col] = np.nan
            continue

        # ── 1. 字段解析 ──
        resolved_fields, missing_names = _resolve_fields(
            df, raw_fields, missing_strategy, i
        )

        if not resolved_fields:
            df[agg_output_col] = np.nan
            continue

        # ── 2. 构建数值矩阵 ──
        values = _build_value_matrix(df, resolved_fields, missing_names)

        # ── 3. 执行聚合 ──
        agg_func, needs_weights = AGGREGATORS[agg_method]
        if needs_weights:
            weights = _extract_weights(resolved_fields, normalize_weights)
            result = agg_func(values, weights, min_valid)
        else:
            result = agg_func(values, min_valid)

        # ── 4. 写入输出列 ──
        _store_result(df, agg_output_col, result)

    return df