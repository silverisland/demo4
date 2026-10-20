# nwp_covariate —— M5 气象协变量组合实验（LightGBM 粗排 + M5 精排）

所有可变项都在 `config.yaml`。输入是线上 `multi_infer` 落盘的 DS 轨迹表
（`ds_trajectory_input/*.parquet`，读入后 drop `update_time`），只允许使用其中 128 列（`fields.py` 白名单）。

## 字段规则（代码强制）

- 历史列 `X`（672 点，最后一点 = 起报时刻）与 `X_predict`（短期 480 / 超短期 192 点，截前 `pred_len`）必须成对使用。
- `GHI_real` 只有历史段，只能当 past-only 协变量或 `inv_error`/`bias_mean` 的参考真值。
- `observe_power_future` 是标签，不进任何特征；`cap_power_on` 只用来取容量（线上口径：最后一个元素）。

## 与线上对齐的地方

| 环节 | 线上 | 本目录 |
|---|---|---|
| 格点聚合 | `RULES`：每源 9 格点中位数，`min_valid=2` | `op: grid`，测试里与 `aggregate_features(df, RULES)` 逐位比对 |
| 跨源聚合 | `general_GHI/TEMP` = GDFS、EC、SOLARGIS 中位数，`min_valid=2` | 特征 `general_ghi` / `general_temp` |
| M5 推理 | `TransformersAdaptedTimesFM3Model` + `unified_inference`，`entity_cols=(timestamp_win, station)` | `backends.M5Backend`，同一个类、同一个函数；模型只加载一次，换组合只改协变量属性 |
| 尾部修补 | `fix_observe_power_tail_miss`（M5 单变量，`fix_len=600`） | `backends.repair_tail`，`data.repair_tail.enabled` 打开 |
| 后处理 | 负值置 0，20:00~06:00（含两端）置 0 | `backends.postprocess` |
| 指标 | 南网短期 D+1 / 超短期 | 本目录 `nanwang_metric.py`（`pipelines/nanwang_metric.py` 的原样副本）逐站逐日 |

短期线上用 `rolling_inference` 滚到 480 点；D+1 只用第 55..150 步，这里统一推 192 点。

## 用法（cwd = 仓库根）

```bash
P=experiments/nwp_covariate
python $P/inspect_columns.py                 # 核对 128 列是否齐全、长度、各源温度单位
python $P/features.py                        # DS -> 特征库（features 段全部候选）
python $P/run_experiments.py --all           # 显式实验：C0、C1_online（线上现状）、各聚合方式 ...
python $P/rank_lgbm.py                       # 粗排：C0 残差上的置换重要性 + TreeSHAP + 噪声参照
python $P/search_m5.py                       # 精排：按粗排顺序贪心，"并排加入"或"与同类特征融合"
# 本机没有 timesfm3 权重时加 --backend mock 跑通流程；--predict_type ultra_short 切超短期
```

输出在 `runs/m5/nwp_cov/<run.name>/<predict_type>/`：`summary.csv`、`report.md`、`summary_by_weather.csv`、
`rank/ranking.csv`、`search/trials.csv`、`search/selected.json`。预测按协变量组合缓存在 `preds/`，组合不变不重跑。

DS 没有 `GHI_real`（缺列或全为空）也能跑：`ghi_real / ghi_inverr / ghi_bias` 构建失败并记在 `build_report.json`，
用到它们的实验（`A_inverr`、`A_bias`、`D_obs`）在 `summary.csv` 里标"跳过"，精排的 `inv_error` 融合自动改用 `power` 作参考。

## 新增一个组合 / 聚合方式

只改 `config.yaml`：在 `features` 里加一条（`op` 见 `ops.py`：`raw / grid / combine / clear_sky / kt / p_phys /
threshold / scale / diff / trig`），在 `experiments` 里引用它，`--rebuild` 重建特征库。
`combine` 支持 `median / mean / trim_mean / min / max / std / weighted_mean / inv_error / bias_mean`。

## 仓库内依赖

`nanwang_metric.py` 已复制进本目录。真实 M5 推理还需要仓库里的 4 个文件（找不到时配 `run.sys_path`）：
`pipelines/pipeline_wrapper_inference.py`、`pipelines/transformer_dataset.py`、
`forecast_backbones/m5/transformer_models.py`、`interfaces/modeloutput.py`。

## 测试

```bash
python -m pytest experiments/nwp_covariate/tests -q
```
合成数据只含白名单 128 列。覆盖：白名单与成对规则、与线上 `aggregate_features` 的逐位一致性、特征库构建、
真实 M5 适配类经 `unified_inference` 推理（只把 timesfm3 推理器换成假的）、尾部修补、显式实验、粗排、精排。
