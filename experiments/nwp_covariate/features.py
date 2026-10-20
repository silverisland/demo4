# -*- coding: utf-8 -*-
"""features.py —— DS 轨迹表 -> 特征库（与线上 DS 同格式：每行一个站一个起报窗，list 列）。

步骤（与线上 multi_infer.handle 对齐）：
  1. 读 ds_trajectory_input/*.parquet，drop update_time；按时段/站点过滤，去省级行；
  2. cap_power_on_val = cap_power_on 的最后一个元素（线上兜底 1，这里可在 config 改）；
  3. （可选）observe_power 尾部缺失用 M5 单变量补齐（线上 fix_observe_power_tail_miss，见 backends.repair_tail）；
  4. 按 config.features 构建所有特征：<name> 为历史 672 点，<name>_predict 为未来 pred_len 点；
  5. 按站写 parquet：feature_store/{predict_type}/station=*.parquet + build_report.json。

用法（cwd=仓库根）：
    python experiments/nwp_covariate/features.py --config experiments/nwp_covariate/config.yaml \
        [--predict_type short] [--backend m5|mock]
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths  # noqa: E402,F401  (设置 sys.path)
import fields as F  # noqa: E402
from ops import Ctx, Pair, dependencies  # noqa: E402

log = logging.getLogger("nwp_cov.features")
META_KEYS = ("quantity", "source", "fusible", "note", "candidate")


# ----------------------------------------------------------------------------- config / io
def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    paths.setup((cfg.get("run") or {}).get("sys_path"))
    return cfg


def abs_path(p) -> Path:
    p = Path(str(p))
    return p if p.is_absolute() else REPO_ROOT / p


def run_dir(cfg: dict, predict_type: str) -> Path:
    r = cfg["run"]
    return abs_path(r["out_dir"].replace("${run.name}", r["name"]).replace("{predict_type}", predict_type))


def feature_store_dir(cfg: dict, predict_type: str) -> Path:
    return abs_path(cfg["data"]["feature_store"].format(predict_type=predict_type))


def ds_files(cfg: dict, predict_type: str) -> List[str]:
    pat = cfg["data"]["ds_glob"].format(predict_type=predict_type)
    files = sorted(glob.glob(str(abs_path(pat)) if not Path(pat).is_absolute() else pat))
    if not files:
        raise FileNotFoundError(f"没有匹配到 DS parquet: {pat}")
    return files


def load_ds(cfg: dict, predict_type: str, files: Optional[List[str]] = None) -> pd.DataFrame:
    d = cfg["data"]
    files = files or ds_files(cfg, predict_type)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop(columns=[c for c in ("update_time",) if c in df.columns])
    if "observe_power_repaired" in df.columns:          # 线上落盘的表带修补列时直接用（与 multi_infer 一致）
        df[F.TARGET] = df["observe_power_repaired"]
    df["timestamp_win"] = pd.to_datetime(df["timestamp_win"])
    tr = d.get("time_range")
    if tr:
        df = df[(df["timestamp_win"] >= pd.Timestamp(tr["start"])) & (df["timestamp_win"] <= pd.Timestamp(tr["end"]))]
    if d.get("drop_province_row", True):
        area = d.get("area", "province_")
        st = df["station"].astype(str)
        df = df[~st.str.startswith("province_") & (st != area)]
    stations = d.get("stations", "all")
    if isinstance(stations, list):
        df = df[df["station"].astype(str).isin([str(s) for s in stations])]
    df = df.sort_values(["station", "timestamp_win"]).drop_duplicates(["station", "timestamp_win"], keep="last")
    extra = F.extra_in(df.columns)
    if extra:
        log.info("DS 表里有 %d 列不在白名单内，实验不会读取：%s", len(extra), extra[:12])
    log.info("DS 表: %d 文件, %d 行, %d 站, %s .. %s", len(files), len(df), df["station"].nunique(),
             df["timestamp_win"].min(), df["timestamp_win"].max())
    return df.reset_index(drop=True)


def load_station_meta(path: Optional[str]) -> pd.DataFrame:
    """station, lat, lon。兼容 info.csv（plant_pointname/LATITUDE/LONGITUDE）与 m6 station_meta.csv。"""
    empty = pd.DataFrame(columns=["station", "lat", "lon"])
    if not path or not abs_path(path).exists():
        return empty
    m = pd.read_csv(abs_path(path))
    low = {c.lower(): c for c in m.columns}
    def pick(*names):
        for n in names:
            if n in low:
                return m[low[n]]
        return None
    st, lat, lon = pick("station", "plant_pointname", "plantid"), pick("lat", "latitude"), pick("lon", "lng", "longitude")
    if st is None or lat is None or lon is None:
        return empty
    return pd.DataFrame({"station": st.astype(str), "lat": pd.to_numeric(lat, errors="coerce"),
                         "lon": pd.to_numeric(lon, errors="coerce")}).drop_duplicates("station")


# ----------------------------------------------------------------------------- 数组列
def stack_col(df: pd.DataFrame, col: str, length: int, take: str) -> np.ndarray:
    """object/list 列 -> [B, length] float64。take='tail' 取末尾（历史），'head' 取开头（未来）。不足补 NaN。"""
    out = np.full((len(df), length), np.nan)
    for i, v in enumerate(df[col].tolist()):
        if v is None or np.ndim(v) == 0:
            continue
        a = np.asarray(v, dtype=np.float64).ravel()
        if a.size == 0:
            continue
        if take == "tail":
            a = a[-length:]; out[i, -a.size:] = a
        else:
            a = a[:length]; out[i, :a.size] = a
    return out


def cap_values(df: pd.DataFrame, cap_col: str, default: float) -> np.ndarray:
    """线上：cap_power_on_val = cap_power_on[-1]（list 时），否则兜底。这里取最后一个有效正值。"""
    out = np.full(len(df), float(default))
    if cap_col not in df.columns:
        return out
    for i, v in enumerate(df[cap_col].tolist()):
        if v is None:
            continue
        a = np.asarray(v, dtype=float).ravel()
        a = a[np.isfinite(a) & (a > 0)]
        if a.size:
            out[i] = a[-1]
    return out


# ----------------------------------------------------------------------------- 特征定义
def feature_specs(cfg: dict) -> Dict[str, dict]:
    specs = {k: dict(v) for k, v in (cfg.get("features") or {}).items()}
    for name, s in specs.items():
        if "op" not in s:
            raise ValueError(f"特征 {name!r} 缺 op")
        for dep in dependencies(s):
            if dep not in specs and dep != F.TARGET:
                raise ValueError(f"特征 {name!r} 引用了未定义的特征 {dep!r}")
    return specs


def feature_meta(cfg: dict) -> Dict[str, dict]:
    """name -> {quantity, source, fusible, history_only, candidate}，给排序分组与贪心融合用。"""
    out = {}
    for name, s in feature_specs(cfg).items():
        out[name] = {"quantity": s.get("quantity", "other"), "source": s.get("source"),
                     "fusible": bool(s.get("fusible", False)),
                     "history_only": bool(s.get("history_only", False)),
                     "candidate": bool(s.get("candidate", True))}
    return out


def make_ctx(df: pd.DataFrame, cfg: dict, specs: Optional[Dict[str, dict]] = None,
             meta_df: Optional[pd.DataFrame] = None, preload: Optional[Dict[str, Pair]] = None) -> Ctx:
    d = cfg["data"]
    C, P = int(d["context_len"]), int(d["pred_len"])
    if meta_df is None:
        meta_df = load_station_meta(d.get("station_meta"))
    mm = df[["station"]].astype(str).merge(meta_df, on="station", how="left")
    dl = d.get("default_latlon")
    if dl:
        miss = mm["lat"].isna() | mm["lon"].isna()
        mm.loc[miss, "lat"], mm.loc[miss, "lon"] = float(dl[0]), float(dl[1])
    if "cap_power_on_val" in df.columns:
        cap = df["cap_power_on_val"].to_numpy(float)
    else:
        cap = cap_values(df, d.get("cap_col", F.CAP), d.get("cap_default", 1.0))

    def load_raw(col: str, history_only: bool) -> Pair:
        if col not in df.columns:
            raise KeyError(f"DS 表缺列 {col!r}")
        h = stack_col(df, col, C, "tail")
        if history_only:
            if not np.isfinite(h).any():
                raise KeyError(f"DS 表列 {col!r} 全为空")
            return Pair(h, np.full((len(df), P), np.nan))
        fc = col + F.FUTURE_SUFFIX
        if fc not in df.columns:
            raise KeyError(f"DS 表缺列 {fc!r}")
        f = stack_col(df, fc, P, "head")
        if not (np.isfinite(h).any() or np.isfinite(f).any()):
            raise KeyError(f"DS 表列 {col!r} 与 {fc!r} 全为空")
        return Pair(h, f)

    ctx = Ctx(ts_win=df["timestamp_win"].values.astype("datetime64[s]"), context_len=C, pred_len=P,
              lat=mm["lat"].to_numpy(float), lon=mm["lon"].to_numpy(float), cap=cap,
              load_raw=load_raw, specs=specs if specs is not None else feature_specs(cfg),
              tz_offset_h=float(d.get("tz_offset_h", 8.0)), stations=df["station"].astype(str).to_numpy())
    ctx.cols[F.TARGET] = Pair(stack_col(df, F.TARGET, C, "tail"), np.full((len(df), P), np.nan))
    for k, v in (preload or {}).items():
        ctx.cols[k] = v
    return ctx


# ----------------------------------------------------------------------------- 构建
def build(cfg: dict, predict_type: Optional[str] = None, files: Optional[List[str]] = None,
          repairer=None) -> tuple:
    """-> (DataFrame 特征表, report)。repairer(df) -> df 可选，用来补 observe_power 尾部缺失。"""
    d = cfg["data"]
    predict_type = predict_type or d["predict_type"]
    C, P = int(d["context_len"]), int(d["pred_len"])
    df = load_ds(cfg, predict_type, files)
    if df.empty:
        raise ValueError("过滤后 DS 表为空")
    df["cap_power_on_val"] = cap_values(df, d.get("cap_col", F.CAP), d.get("cap_default", 1.0))
    report = {"predict_type": predict_type, "rows": int(len(df)), "stations": int(df["station"].nunique()),
              "missing_allowed_columns": F.missing_in(df.columns), "built": [], "failed": {}, "warnings": [],
              "repaired_rows": 0, "nan_frac": {}}
    if repairer is not None:
        df, n = repairer(df)
        report["repaired_rows"] = int(n)

    specs = feature_specs(cfg)
    meta_df = load_station_meta(d.get("station_meta"))
    ctx = make_ctx(df, cfg, specs, meta_df)
    for name in specs:
        try:
            ctx.get(name)
            report["built"].append(name)
        except (KeyError, ValueError) as e:
            report["failed"][name] = str(e)
            log.warning("特征 %s 构建失败: %s", name, e)
    report["warnings"] = list(ctx.warnings)

    out = df[["timestamp_win", "station", "cap_power_on_val"]].copy()
    out["lat"], out["lon"] = ctx.lat, ctx.lon
    out[F.TARGET] = list(ctx.cols[F.TARGET].hist.astype(np.float32))
    if F.LABEL in df.columns:
        out[F.LABEL] = list(stack_col(df, F.LABEL, P, "head").astype(np.float32))
    clear = ctx.clear_sky()
    out["ghi_clear_sky"] = list(clear.hist.astype(np.float32))
    out["ghi_clear_sky" + F.FUTURE_SUFFIX] = list(clear.fut.astype(np.float32))
    meta = feature_meta(cfg)
    for name in report["built"]:
        pair = ctx.cols[name]
        out[name] = list(pair.hist.astype(np.float32))
        if not meta[name]["history_only"]:
            out[name + F.FUTURE_SUFFIX] = list(pair.fut.astype(np.float32))
            report["nan_frac"][name] = {"hist": round(float(np.isnan(pair.hist).mean()), 4),
                                        "fut": round(float(np.isnan(pair.fut).mean()), 4)}
        else:
            report["nan_frac"][name] = {"hist": round(float(np.isnan(pair.hist).mean()), 4)}
    return out.reset_index(drop=True), report


def write_feature_store(table: pd.DataFrame, out_dir: Path, report: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in glob.glob(str(out_dir / "station=*.parquet")):
        Path(old).unlink()
    for st, g in table.groupby("station", sort=True):
        g.reset_index(drop=True).to_parquet(out_dir / f"station={st}.parquet", index=False)
    with open(out_dir / "build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info("特征库写入 %s：%d 站, %d 行", out_dir, table["station"].nunique(), len(table))


def read_feature_store(cfg: dict, predict_type: str) -> pd.DataFrame:
    fs = feature_store_dir(cfg, predict_type)
    files = sorted(glob.glob(str(fs / "station=*.parquet")))
    if not files:
        raise FileNotFoundError(f"特征库为空，先跑 features.py: {fs}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp_win"] = pd.to_datetime(df["timestamp_win"])
    return df


def table_ctx(table: pd.DataFrame, cfg: dict, names: List[str]) -> Ctx:
    """从特征表重建 Ctx（已有特征直接预载），用于贪心搜索时临时合成新的融合特征。
    names 要包含融合用到的成员和参考列（如 ghi_real）。"""
    d = cfg["data"]
    C, P = int(d["context_len"]), int(d["pred_len"])
    pre = {}
    for n in set(names):
        if n in table.columns:
            h = stack_col(table, n, C, "tail")
            fc = n + F.FUTURE_SUFFIX
            f = stack_col(table, fc, P, "head") if fc in table.columns else np.full((len(table), P), np.nan)
            pre[n] = Pair(h, f)
    meta_df = table[["station", "lat", "lon"]].drop_duplicates("station").astype({"station": str})
    ctx = make_ctx(table, cfg, feature_specs(cfg), meta_df, pre)
    return ctx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--predict_type", default=None)
    ap.add_argument("--backend", choices=["m5", "mock", "none"], default=None,
                    help="observe_power 尾部修补用的模型；默认按 config.data.repair_tail")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    pt = a.predict_type or cfg["data"]["predict_type"]
    repairer = None
    rt = cfg["data"].get("repair_tail") or {}
    backend = a.backend or ("m5" if rt.get("enabled") else "none")
    if backend != "none":
        from backends import make_backend, repair_tail
        be = make_backend(cfg, backend)
        repairer = lambda df: repair_tail(df, be, int(rt.get("fix_len", 600)))  # noqa: E731
    table, report = build(cfg, pt, repairer=repairer)
    write_feature_store(table, feature_store_dir(cfg, pt), report)
    print(json.dumps({k: report[k] for k in ("rows", "stations", "built", "failed", "warnings", "repaired_rows")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
