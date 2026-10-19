# M4-Pinball

保留 M4 的 PatchTST、天气 / 历史功率 / 外部预测输入及 RevIN，新增独立的
`m4_pinball` 模型。只读复用 `m4/layers`，不修改 M4 源码、配置或权重。
末层由 `64 → H` 改为 `64 → H×K`，输出 `[B,H,K]`，默认 H=192、K=9。

## 训练与推理

- 每个分位通道分别用对应的 Pinball loss 监督；不是同一个预测重复计算 9 次损失。
- 各分位共享 PV RevIN 的统计量。恢复真实功率单位后，预测和标签同时按固定容量归一化计算损失。
- 标签 NaN/Inf 被掩码；训练入口会过滤整行标签全缺失的样本。输入天气、历史功率和外部提示必须已完成缺测处理，非有限输入直接报错。
- 训练不使用旧 M4 的 MAE / 南网混合损失，不从标签估算容量，不对分位做物理裁剪后再计算损失。
- 有标签且 `model.train()` 时只返回 loss，避免每批执行 NumPy 决策；验证和无标签推理返回完整输出。
- 推理：恢复功率 → 分位排序 → 物理裁剪 → 分段选点 → 再检查最终点预测的物理范围。
- `logits` 和 `predictions` 均为 `[B,H]` 上报曲线。`aux` 包含 `quantiles`、`quantiles_cal`、`cap`。

`decision.py` 基于 M8 决策层独立复制，保持默认网格下的 `p_star` / `p_star_l1` 数值口径。
补充了分位、容量和分段参数检查，支持晴空裁剪独立开关、零缩放系数，缩放后再次约束点预测。
M8 不受影响。短期 `p_star` 最小化归一化平方损失代理，不保证严格最大化期望日 RMSE 分数。

默认决策：前 16 步 `p_star_l1`，其余 `p_star`；负值截 0、容量封顶；晴空上限和夜间归零默认关闭。
物理裁剪只在推理执行。开启晴空上限需要 `[B,H]` 的 `p_clear`（功率单位）；开启夜间归零需要
`[B,H]` 的 `is_night`（bool 或 0/1）。列名可配置，当前模型不自行计算太阳几何。

## 数据与配置

默认输入列：`general_ghi`、`general_temp`、`observe_power`（672 点），
`general_ghi_predict`、`general_temp_predict`、`chronos`（192 点），以及 `cap_mw`（每行一个容量）。
监督列 `observe_power_future`（192 点），时间列 `timestamp_win`，站点列 `station`。

**容量必须来自固定装机容量，并与评测口径一致。** 默认读取 `cap_mw`；也可以在配置里提供
`dataset.capacity_by_station: {站名: 容量}`，此时按 `station` 查容量，不再需要 `cap_mw` 列。
没有匹配容量时明确报错，不使用未来标签兜底。列值与功率必须使用同一单位。
如果显式把 `capacity_col` 改成 `cap_power_on`，需要自行确认它是固定容量而非随时间变化的开机容量。

默认外部提示列与 M4 配置一致为 `chronos`。数据实际使用 `observe_power_chronos_pred` 时，设置
`dataset.tsfm_col: observe_power_chronos_pred`。保留 M4 主干所需的外部预测，不能省略此列。

配置入口为 `config/configs.yaml`。独立训练入口支持只包含修改项的 YAML，例如：

```yaml
dataset:
  tsfm_col: observe_power_chronos_pred
  capacity_by_station: {示例站A: 100.0, 示例站B: 80.0}  # 替换成真实站名和容量
optimization:
  batch_size: 64          # 每设备批量
  eval_batch_size: 32
```

## 启动

在项目根目录 `NextGenPVForecast` 中运行：

```bash
python -m forecast_backbones.m4_pinball.train \
  --config /path/to/overrides.yaml \
  --train /path/to/train.parquet \
  --valid /path/to/valid.parquet \
  --out runs/m4_pinball/exp1
```

可选 `--test /path/to/test.parquet`，训练结束后生成测试结果；可选 `--max-steps 2` 做小规模链路验证。
不传 `--config` 使用默认配置。CPU 自动关闭 fp16。支持 `torchrun`，批量是每设备的值。

也已接入公共入口：

```bash
python train_shell/train_script.py --model_name m4_pinball \
  --model_config_path /path/to/overrides.yaml \
  --dataset_path /path/to/splits \
  --run_path runs/m4_pinball/exp1
```

公共入口沿用其要求：切分目录需要 `train.parquet`、`valid.parquet`、`test.parquet`。
专用入口只需要 train / valid。模型和 evaluator 由现有 registry 自动发现，不用修改注册表。

最优模型按验证集容量归一化 Pinball loss 选择，patience 按验证次数计；不是直接按南网日分选模型。
输出 `model_result/model.safetensors` 和同目录 `model_config.yaml`，以及 `result_val.parquet`、
可选 `result_test.parquet`。结果包含上报曲线、中位数、裁剪后的分位列、原始分位和容量，便于后续比较决策策略。

## 加载与切换决策

```python
from safetensors.torch import load_model
from pipelines.model_registry import build_model
from forecast_backbones.m4_pinball.config import load_config

cfg = load_config("runs/m4_pinball/exp1/model_result/model_config.yaml")
model = build_model("m4_pinball", cfg)
load_model(model, "runs/m4_pinball/exp1/model_result/model.safetensors")
model.eval()
output = model(batch_dict)  # 使用已配置的决策，真实功率单位

# 同一份分位模型改用中位数，无需重训；清空分段规则才能覆盖所有步。
model.config["decision"]["point"].update(default="median", by_step=[])
median_output = model(batch_dict)
```

兼容现有 `pipelines.pipeline_wrapper_inference.unified_inference`；可用
`feature_columns(model.config)` 提供精确输入列。原 M4 checkpoint 的输出头形状不兼容新模型，
需要训练新的分位模型；未提供隐式忽略权重不匹配的加载方式。

## 验证

```bash
python -m pytest forecast_backbones/m4_pinball/tests -q
```

测试包括独立分位梯度、RevIN 共享反归一化、缺测标签、容量错误、M8 选点数值对齐、分段优先级、
物理开关、零缩放和越界修复，以及本地 Parquet → Trainer → 保存重载 → 统一推理的完整链路。
