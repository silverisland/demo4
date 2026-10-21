# Vendored from NextM8/pipelines/nanwang_metric.py on 2026-10-21.
# Kept unchanged for evaluation parity, including MIN_DAYS=3 and rounding.
#!/usr/bin/env python3
"""南网指标计算（独立模块，不依赖 pvcore）。

输入 result_df 的每行是一个起报窗：
    timestamp_win          起报时刻 S（str/datetime 均可）
    station                站号
    observe_power_predict  长度 192 的预测序列，第 i 个元素对应目标时刻 S+(i+1)*15min
    observe_power_future   长度 192 的真值序列，对齐方式与预测相同

两个指标（口径均与原 station_analysis_*.py 一致，全点含夜间，0.2C 分母下限；
日指标（及站级整体值）为负时一律 cap 到 0，因此月度值 / mean 列恒 >= 0）：

超短期 nanwang_ultra_short（MAE 型）：
    目标时刻 t 的 16 个预报值取起报 S=t-k*15min 行预测的第 k 个元素（k=1..16），
    真值取 t-15min 行真值序列第 0 个元素。
    日指标 = (1 - mean_t mean_k |P_real(t) - p_k(t)| / max(P_real(t), 0.2*GCCAPCITY)) * 100，
    t 只取当日 06:15~20:00 且 16 个 lead 全齐的时刻（起报行 2:15~19:45 连续时正好 56 个点）。

短期 nanwang_short_D+1（RMSE 型，即原 nanwang_official）：
    每个起报日 D 取一行（默认 issue_time="10:00"：用当日 <= 10:00 的最晚一行，即南网日前
    起报；issue_time=None 用当日最早一行），从该行 192 点里切出目标时刻落在
    [D+1 00:00, D+2 00:00) 的 96 个点（10:00 起报时为 predict[55..150]），
    日指标 = (1 - sqrt(mean(((P_real - P_pred) / max(P_real, 0.2*GCCAPCITY))^2))) * 100。

月度汇总 monthly_table：
    列 0~11 对应 1~12 月，最后一列 mean；行 yearmonth / daycount（该月日历天数）/
    sample_count（该月有数据的天数，数据全时 = daycount）/ nanwang_ultra_short /
    nanwang_short_D+1。月值 = 该月日指标之和 / 有日指标的天数；mean 列 = 有值月份的
    均值，分母是该站实际有数据的月份数（不固定 12）。每站一个 sheet，各站时间范围
    可不同，不做跨站汇总。

用法：
    from nanwang_metric import NanwangMetric
    m = NanwangMetric()
    m.ultra_short(result_df, gccap={"st01": 20.0})            # 站级整体超短期
    m.ultra_short_daily(result_df, gccap=20.0)                # 逐站逐日超短期
    m.short_d1_daily(result_df, gccap=20.0)                   # 逐站逐日短期 D+1
    m.monthly_table(result_df, gccap=20.0, out_xlsx="nanwang_monthly.xlsx")
"""

import datetime as dt

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=15)
N_LEADS = 16
TABLE_ROWS = ["yearmonth", "daycount", "sample_count", "nanwang_ultra_short", "nanwang_short_D+1"]

# 月度均值口径：剔除该月样本天数 <= 2 天的月份后，各月等权求均值。
# 与 eval_shell/make_models_summary.py / pipelines/metric.py 的 ODS 口径一致。
# 判定用 sample_count 行（该月有数据/有日指标的天数），而非日历天数 daycount。
MIN_DAYS = 3


class NanwangMetric:
    def __init__(self, step: pd.Timedelta = STEP, n_leads: int = N_LEADS,
                 day_start: str = "06:15", day_end: str = "20:00",
                 ts_col: str = "timestamp_win", station_col: str = "station",
                 pred_col: str = "observe_power_predict", true_col: str = "observe_power_future"):
        self.step, self.n_leads = step, n_leads
        self.day_start = dt.time.fromisoformat(day_start)
        self.day_end = dt.time.fromisoformat(day_end)
        self.ts_col, self.station_col = ts_col, station_col
        self.pred_col, self.true_col = pred_col, true_col

    # ------------------------------------------------ 公共
    def _prep(self, result_df: pd.DataFrame) -> pd.DataFrame:
        df = result_df.copy()
        if self.station_col not in df.columns:
            print(f"  [info] no '{self.station_col}' column -> treating all rows as one station 'all'")
            df[self.station_col] = "all"
        df[self.ts_col] = pd.to_datetime(df[self.ts_col])
        return df.sort_values(self.ts_col).drop_duplicates([self.station_col, self.ts_col], keep="first")

    def _gc(self, gccap, st):
        gc = gccap.get(st, gccap.get(str(st))) if isinstance(gccap, dict) else gccap
        if gc is None or not (gc > 0):
            print(f"  [warn] station {st}: gccap missing/non-positive -> skipped")
            return None
        return float(gc)

    # ------------------------------------------------ 超短期：对齐
    def build_lead_matrix(self, df_station: pd.DataFrame):
        """单站行集 -> (truth Series, leads DataFrame)。

        truth: index=目标时刻 t，值取起报 t-15min 行真值序列第 0 个元素。
        leads: index 同 truth，列 p1..p16；lead k（起报 t-k*step 行预测第 k 个元素）
               放列 p{17-k}，即 p16 最新鲜、p1 最旧，与原 ultra_loaders 一致。缺行为 NaN。
        目标集合 = 所有 起报时刻+15min（即真值可取的时刻），不做时段过滤。"""
        preds, trues = {}, {}
        for _, row in df_station.iterrows():
            s = pd.Timestamp(row[self.ts_col])
            p = np.asarray(row[self.pred_col], float)
            t = np.asarray(row[self.true_col], float)
            if p.size < self.n_leads or t.size < 1:
                continue
            preds[s], trues[s] = p, t
        if not preds:
            return pd.Series(dtype=float), pd.DataFrame(columns=[f"p{j}" for j in range(1, self.n_leads + 1)])
        targets = sorted(s + self.step for s in preds)
        truth = pd.Series({t: trues[t - self.step][0] for t in targets}, dtype=float)
        leads = pd.DataFrame(np.nan, index=truth.index,
                             columns=[f"p{j}" for j in range(1, self.n_leads + 1)])
        for t in truth.index:
            for k in range(1, self.n_leads + 1):
                s = t - k * self.step
                if s in preds:
                    leads.at[t, f"p{self.n_leads + 1 - k}"] = preds[s][k - 1]
        return truth, leads

    def _ultra_per_target(self, df_station: pd.DataFrame, gccap: float,
                          full_leads_only: bool = True) -> pd.Series:
        """-> 每个入选目标时刻的 mean_k |P_real - p_k| / max(P_real, 0.2C)（Series, index=t）。
        入选 = 当日 day_start~day_end 内、（默认）16 个 lead 全部有值、真值有限。"""
        truth, leads = self.build_lead_matrix(df_station)
        if not len(truth):
            return pd.Series(dtype=float)
        tod = np.array([ts.time() for ts in truth.index])
        keep = (tod >= self.day_start) & (tod <= self.day_end)
        if full_leads_only:
            keep &= leads.notna().all(axis=1).to_numpy()
        truth, leads = truth[keep], leads.loc[keep]
        t = truth.to_numpy(float)
        P = leads.to_numpy(float)
        if t.size == 0:
            return pd.Series(dtype=float)
        r = np.abs(t[:, None] - P) / np.maximum(t, 0.2 * gccap)[:, None]
        ok = np.isfinite(r)
        valid = np.flatnonzero(ok.any(axis=1))
        return pd.Series([r[j][ok[j]].mean() for j in valid], index=truth.index[valid], dtype=float)

    # ------------------------------------------------ 超短期：指标
    def ultra_short(self, result_df: pd.DataFrame, gccap, full_leads_only: bool = True) -> pd.DataFrame:
        """站级整体超短期准确率（所有入选时刻一把平均）
        -> DataFrame[station, accuracy_pct, n_targets, gccap]。
        gccap: 标量或 {station: 容量}；查不到/非正的站跳过并打 warn。"""
        rows = []
        for st, g in self._prep(result_df).groupby(self.station_col, sort=True):
            gc = self._gc(gccap, st)
            if gc is None:
                continue
            per_t = self._ultra_per_target(g, gc, full_leads_only)
            if not len(per_t):
                print(f"  [warn] station {st}: no valid (truth, lead) pair -> skipped")
                continue
            rows.append({"station": st,
                         "accuracy_pct": round(max(0.0, (1.0 - float(per_t.sum())/(96*per_t.index.normalize().nunique())) * 100.0), 2),
                         "n_targets": int(len(per_t)), "gccap": gc})
        return pd.DataFrame(rows, columns=["station", "accuracy_pct", "n_targets", "gccap"])

    def ultra_short_daily(self, result_df: pd.DataFrame, gccap,
                          full_leads_only: bool = True) -> pd.DataFrame:
        """逐站逐日超短期准确率 -> DataFrame[station, date, accuracy_pct, n_targets]。
        日 = 目标时刻所在日（06:15~20:00 窗内目标与起报行同日）。"""
        rows = []
        for st, g in self._prep(result_df).groupby(self.station_col, sort=True):
            gc = self._gc(gccap, st)
            if gc is None:
                continue
            per_t = self._ultra_per_target(g, gc, full_leads_only)
            if not len(per_t):
                continue
            for d, chunk in per_t.groupby(per_t.index.normalize()):
                rows.append({"station": st, "date": d,
                             "accuracy_pct": max(0.0, (1.0 - float(chunk.sum())/96.0) * 100.0),
                             "n_targets": int(len(chunk))})
        return pd.DataFrame(rows, columns=["station", "date", "accuracy_pct", "n_targets"])

    # ------------------------------------------------ 短期 D+1
    @staticmethod
    def nanwang_official(p_real, p_pred, gccap):
        """南网「两个细则」短期准确率（RMSE 型），与原 pvcore.metrics.nanwang_official 同式：
        (1 - sqrt(mean(((p_real - p_pred) / max(p_real, 0.2*gccap))^2))) * 100。
        只用两边都有限的点；gccap 非正或无点 -> None。
        注意：本函数返回原始值（可为负，与 pvcore 逐位一致），cap 到 0 在调用方做。"""
        if gccap is None or not (gccap > 0):
            return None
        a = np.asarray(p_real, float)
        p = np.asarray(p_pred, float)
        ok = np.isfinite(a) & np.isfinite(p)
        if not ok.any():
            return None
        r = (a[ok] - p[ok]) / np.maximum(a[ok], 0.2 * gccap)
        return (1.0 - float(np.sqrt(np.mean(r ** 2)))) * 100.0

    def short_d1_daily(self, result_df: pd.DataFrame, gccap, issue_time="10:00", horizon_days=1) -> pd.DataFrame:
        """逐站逐起报日的短期 D+1 准确率
        -> DataFrame[station, date(起报日 D), issue_ts(所用行), accuracy_pct, n_points]。

        每个起报日 D 用一行：issue_time="HH:MM"（默认 "10:00"，南网日前起报时刻）取当日
        timestamp_win <= 该时刻的最晚一行——有 10:00 行就精确用它，缺了回退到更早的最近
        一行；当日全在该时刻之后则跳过该日并打 warn。issue_time=None 取当日最早一行。
        打分点 = 该行 192 个目标时刻中落在 [D+1 00:00, D+2 00:00) 的点（10:00 起报时
        为 predict[55..150] 共 96 个），与南网 D+1 全天窗对齐。"""
        cutoff = dt.time.fromisoformat(issue_time) if issue_time is not None else None
        rows = []
        for st, g in self._prep(result_df).groupby(self.station_col, sort=True):
            gc = self._gc(gccap, st)
            if gc is None:
                continue
            for D, gd in g.groupby(g[self.ts_col].dt.normalize()):
                if cutoff is None:
                    row = gd.iloc[0]
                else:
                    ok = gd[gd[self.ts_col].dt.time <= cutoff]
                    if ok.empty:
                        print(f"  [warn] station {st} {D.date()}: no row at/before {issue_time} -> day skipped")
                        continue
                    row = ok.iloc[-1]
                S = pd.Timestamp(row[self.ts_col])
                p = np.asarray(row[self.pred_col], float)
                a = np.asarray(row[self.true_col], float)
                n = min(p.size, a.size)
                stamps = pd.date_range(S + self.step, periods=n, freq=self.step)
                win = (stamps >= D + pd.Timedelta(days=horizon_days)) & (stamps < D + pd.Timedelta(days=horizon_days+1))
                fin = win & np.isfinite(p[:n]) & np.isfinite(a[:n])
                acc = self.nanwang_official(a[:n][fin], p[:n][fin], gc)
                if acc is None:
                    print(f"  [warn] station {st} {D.date()}: no valid D+1 point -> day skipped")
                    continue
                rows.append({"station": st, "date": D, "issue_ts": S,
                             "accuracy_pct": max(0.0, acc), "n_points": int(fin.sum())})
        return pd.DataFrame(rows, columns=["station", "date", "issue_ts", "accuracy_pct", "n_points"])

    # ------------------------------------------------ 月度汇总表
    @staticmethod
    def _month_table(ultra_daily: pd.DataFrame, short_daily: pd.DataFrame, sample_dates) -> pd.DataFrame:
        """单站（或 fleet）的月度表：列 0~11 + mean，行 TABLE_ROWS。
        ultra_daily/short_daily: DataFrame[date, accuracy_pct]；sample_dates: 有数据的日期集合。"""
        tab = pd.DataFrame(np.nan, index=TABLE_ROWS, columns=list(range(12)) + ["mean"], dtype=object)
        periods = sorted({pd.Period(d, "M") for d in sample_dates})
        used = {}
        for per in periods:
            col = per.month - 1
            if col in used:
                print(f"  [warn] month {per.month} appears twice ({used[col]} and {per}) -> {per} ignored")
                continue
            used[col] = per
            days = {d for d in sample_dates if pd.Period(d, "M") == per}
            tab.at["yearmonth", col] = str(per)
            tab.at["daycount", col] = per.days_in_month
            tab.at["sample_count", col] = len(days)
            for name, daily in (("nanwang_ultra_short", ultra_daily), ("nanwang_short_D+1", short_daily)):
                if len(daily):
                    v = daily.loc[pd.PeriodIndex(daily["date"], freq="M") == per, "accuracy_pct"]
                    if len(v):
                        tab.at[name, col] = round(float(v.mean()), 2)
        # mean 列：南网两个指标行剔除 sample_count<=2 的月份后各月等权
        # （与 ODS / 汇总口径一致）；daycount/sample_count 信息行保持原全列均值。
        metric_names = ("nanwang_ultra_short", "nanwang_short_D+1")
        for name in TABLE_ROWS[1:]:
            cols = list(range(12))
            if name in metric_names:
                sc = pd.to_numeric(tab.loc["sample_count", cols], errors="coerce")
                cols = [c for c in cols if pd.notna(sc.get(c)) and float(sc[c]) >= MIN_DAYS]
            v = pd.to_numeric(tab.loc[name, cols], errors="coerce").dropna()
            if len(v):
                tab.at[name, "mean"] = round(float(v.mean()), 2)
        return tab

    def monthly_table(self, result_df: pd.DataFrame, gccap, out_xlsx=None,
                      issue_time="10:00", full_leads_only: bool = True) -> dict:
        """月度汇总 -> {站号: DataFrame}；给 out_xlsx 时同时写 Excel，每站一个 sheet。
        各站时间范围可以不同（如一站 2025 年、一站 2026 年），互不影响；正因如此不做
        跨站汇总 sheet。月值 = 该月日指标之和 / 有日指标的天数（数据全时分母 = daycount）；
        mean 列 = 有值月份的均值——分母是该站实际有数据的月份数（5 个月就除以 5），不固定 12。"""
        df = self._prep(result_df)
        ud = self.ultra_short_daily(df, gccap, full_leads_only)
        sd = self.short_d1_daily(df, gccap, issue_time)
        tables = {}
        stations = sorted(set(df[self.station_col]), key=str)
        for st in stations:
            dates = set(df.loc[df[self.station_col] == st, self.ts_col].dt.normalize())
            tables[str(st)] = self._month_table(ud[ud["station"] == st], sd[sd["station"] == st], dates)
        if out_xlsx:
            with pd.ExcelWriter(out_xlsx) as xw:
                for name, tab in tables.items():
                    tab.to_excel(xw, sheet_name=str(name)[:31])
            print(f"[monthly_table] written: {out_xlsx}  ({len(tables)} sheet)")
        return tables
