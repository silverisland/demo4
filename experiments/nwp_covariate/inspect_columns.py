# -*- coding: utf-8 -*-
"""inspect_columns.py —— 扫描 DS 轨迹表，输出每列的：数组长度、NaN 比例、取值范围、是否在允许字段白名单里。
主要用来核对：白名单 128 列是否齐全、历史 672 / 未来 192|480 长度对不对、各源温度单位是否一致（K 还是 ℃）。

用法（cwd=仓库根）：
    python experiments/nwp_covariate/inspect_columns.py --config experiments/nwp_covariate/config.yaml \
        [--predict_type ultra_short] [--max_files 3] [--out columns_report.csv]
也可以直接给文件：
    python experiments/nwp_covariate/inspect_columns.py --files a.parquet b.parquet
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import sys

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fields as F  # noqa: E402
KNOWN_PREFIXES = ("EC_JTKRD7RO_", "GDFS_JTHFGNED_MID_JTNWQISX_", "solar_southern_", "observe_weather_",
                  "SOLARGIS_", "general_", "GHI_", "TEMP_", "ssrd_pos_", "t2m_pos_", "observe_power")


def _cell_stats(v):
    """单元格 -> (是否数组, 长度, nan 比例, min, max)。"""
    if v is None:
        return False, 0, 1.0, np.nan, np.nan
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.asarray(v, dtype=float).ravel()
        if arr.size == 0:
            return True, 0, 1.0, np.nan, np.nan
        fin = np.isfinite(arr)
        return True, int(arr.size), float(1 - fin.mean()), (float(arr[fin].min()) if fin.any() else np.nan), (float(arr[fin].max()) if fin.any() else np.nan)
    try:
        f = float(v)
        return False, 1, float(not np.isfinite(f)), f, f
    except (TypeError, ValueError):
        return False, 1, 0.0, np.nan, np.nan


def guess_prefix(col: str) -> str:
    for p in KNOWN_PREFIXES:
        if col.startswith(p):
            return p
    return col.split("_")[0] + "_" if "_" in col else ""


def inspect(files: list[str], sample_rows: int = 50) -> pd.DataFrame:
    rows = {}
    for f in files:
        df = pd.read_parquet(f)
        head = df.head(sample_rows)
        for c in df.columns:
            st = rows.setdefault(c, {"column": c, "prefix": guess_prefix(c),
                                     "is_future": c.endswith("_predict") or c.endswith("_future"),
                                     "dtype": str(df[c].dtype), "n_files": 0, "is_array": False,
                                     "lengths": set(), "nan_frac": [], "vmin": np.inf, "vmax": -np.inf,
                                     "example_file": Path(f).name})
            st["n_files"] += 1
            for v in head[c].tolist():
                is_arr, n, nanf, lo, hi = _cell_stats(v)
                st["is_array"] |= is_arr
                st["lengths"].add(n)
                st["nan_frac"].append(nanf)
                if np.isfinite(lo):
                    st["vmin"] = min(st["vmin"], lo); st["vmax"] = max(st["vmax"], hi)
    out = []
    for c, st in rows.items():
        out.append({
            "column": c, "prefix": st["prefix"], "is_future": st["is_future"], "is_array": st["is_array"],
            "lengths": "|".join(str(x) for x in sorted(st["lengths"])),
            "nan_frac_mean": round(float(np.mean(st["nan_frac"])), 4) if st["nan_frac"] else np.nan,
            "vmin": None if not np.isfinite(st["vmin"]) else round(st["vmin"], 4),
            "vmax": None if not np.isfinite(st["vmax"]) else round(st["vmax"], 4),
            "dtype": st["dtype"], "n_files": st["n_files"], "example_file": st["example_file"],
            "allowed": F.is_allowed(c),
        })
    rep = pd.DataFrame(out).sort_values(["prefix", "is_future", "column"]).reset_index(drop=True)
    return rep


def resolve_files(a) -> list[str]:
    if a.files:
        return a.files
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    pt = a.predict_type or cfg["data"]["predict_type"]
    pat = cfg["data"]["ds_glob"].format(predict_type=pt)
    if not Path(pat).is_absolute():
        pat = str(REPO_ROOT / pat)
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"没有匹配到 DS parquet: {pat}")
    # 首、中、尾各取一个，覆盖时段差异
    pick = sorted({0, len(files) // 2, len(files) - 1})[: a.max_files]
    return [files[i] for i in pick]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--predict_type", default=None)
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--max_files", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).with_name("columns_report.csv")))
    a = ap.parse_args()
    files = resolve_files(a)
    print("扫描文件:"); [print("  ", f) for f in files]
    rep = inspect(files)
    rep.to_csv(a.out, index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 1000); pd.set_option("display.max_colwidth", 48)
    print(rep.to_string(index=False))
    print(f"\n共 {len(rep)} 列 -> {a.out}")
    print("按前缀计数:"); print(rep.groupby("prefix")["column"].count().sort_values(ascending=False).to_string())
    miss = F.missing_in(rep["column"])
    print(f"\n白名单 {len(F.ALLOWED_COLUMNS)} 列，表里缺 {len(miss)} 列：{miss}")
    print(f"表里有但不在白名单（实验不会读）：{F.extra_in(rep['column'])}")


if __name__ == "__main__":
    main()
