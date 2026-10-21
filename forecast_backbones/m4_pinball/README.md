# M4-Pinball

保留 M4 的 PatchTST、天气 / 历史功率 / 外部预测输入及 RevIN，新增独立的
`m4_pinball` 模型。PatchTST、RevIN 和输入整理代码均自带于 `m4_pinball/layers`，
不引用 `m4` 或其他模型包；共享的 `interfaces`、`pipelines` 仍为框架依赖。
原 M4 源码、配置和权重不受影响。
默认 Flatten 预测头的末层由 `64 → H` 改为 `64 → H×K`，输出 `[B,H,K]`，默认 H=192、K=9。
Encoder 显式使用与 M4 一致的 BatchNorm；预测头内原有的 LayerNorm 保持不变。

## 可选的逐时刻预测头

`model.prediction_head` 默认仍为 `flatten`，可切换为 `step_cov` 做对照实验。
后者借鉴 M8 的“逐时刻特征拼接 + 残差 MLP”，实现完全位于本模型内，不导入 M8。

| 项目 | flatten（基线） | step_cov（实验） |
| --- | --- | --- |
| 解码输入 | 所有通道、所有 Patch 展平 | 对应 Patch 的各通道编码 + Patch 内位置 + 该步协变量 |
| 默认结构 | 6912 → 256 → 256 → 64 → 1728 | 803 → 256 → 9，加 803 → 9 线性跳接 |
| 时间维度 | 一次输出整个 192 步窗口 | 同一预测头应用于每个未来时刻 |
| 分位数 | 共享隐藏层，末层权重各自独立 | 共享隐藏层，末层权重各自独立 |

M4 编码保留通道维，因此将同一 Patch 的 3 个通道各 256 维编码拼成 768 维，
再加 32 维位置嵌入与 3 维逐步输入，共 803 维。逐步输入包括未来天气和外部 PV 预测，
沿用进入 backbone 前的 RevIN 单位；没有新增数据列，也不读取未来标签。
残差 MLP 为 `Linear → GELU → Dropout → Linear` 加线性跳接。
不额外添加单调或非负输出约束，继续在原始分位数上计算 Pinball loss。

默认 patch_len/stride 都保持 96，以便单独验证换头的影响。重叠 Patch 时选择覆盖该步且
起点最晚的可用 Patch；未来步未被任何 Patch 覆盖则报错，可调整 stride 或启用 `padding_patch: end`。
`aux.quantiles` 仍为 `[B,H,K]`；`embeddings` 在 step_cov 下为逐步隐藏向量的均值，
宽度为 `step_head_hidden`，不再由 `embedding_dim` 控制。

直接试用配置：

```bash
python -m forecast_backbones.m4_pinball.train \
  --config forecast_backbones/m4_pinball/config/variants/step_cov.yaml \
  --train /path/to/train.parquet --valid /path/to/valid.parquet \
  --out runs/m4_pinball/step_cov
```

若已有数据列、容量或训练参数的自定义 YAML，在原 YAML 的 `model` 下加入以下字段即可，
其余参数与基线保持一致：

```yaml
model:
  prediction_head: step_cov
  step_head_hidden: 256
  step_head_position_dim: 32
```

两种头使用相同的 Pinball loss、decision 和南网 D+1 选模；实验效果需用相同数据切分、
训练预算和随机种子对照。新头需要重新训练，不能直接加载 Flatten 头的完整权重。
已有 BatchNorm + Flatten 权重和配置仍可严格加载，未声明 prediction_head 的旧配置默认使用 flatten。

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

训练损失仍为容量归一化 Pinball loss；**最优模型和早停改按决策后的南网短期 D+1
评分 `eval_nanwang_short_d1`（越高越好）**。默认每 250 个优化器步评估并保存一次，连续
8 次评分未提升则早停，与 M8 K2 的验证节奏一致。`patience` 按连续验证次数计，不是 epoch
或微批次数；梯度累积时，完成一次参数更新才计一步。训练最后一步即使不足验证间隔，也会
评估、保存并参与最佳权重选择。达到 `--max-steps` 或 epoch 预算同样结束训练。

可在自定义 YAML 中覆盖（不用修改网络或损失）：

```yaml
optimization:
  eval_every_steps: 250   # 0/null 改回每个 epoch 验证、保存
  patience: 8            # 连续多少次 D+1 分数未创新高就停止
```

若要复现此前按 epoch 选模的训练流程，显式设置 `eval_every_steps: 0` 和 `patience: 10`。
早停与最佳权重都按严格更高的原始 D+1 分数判断，不平滑、不用 eval_loss，不修改 decision
配置。验证集必需且不得使用测试集代替；`--test` 只用于恢复最佳权重后的最终结果导出。
评分口径与 M8 默认选模一致，但实现自带于本模型：

- 每站每个起报日取 10:00 及之前的最后一行；重复站点/起报时刻保留第一行。
- 第 0 个预测对应起报后 15 分钟，只评分次日 `[00:00, 24:00)` 内的点。
  10:00 起报、192 点预测时，使用下标 `[55:151]` 共 96 点。
- 日分为 `max(0, 100 * (1 - sqrt(mean(((y-p)/max(y,0.2C))**2))))`，
  `p` 使用当前 decision 配置处理后的上报点，`C` 使用每站固定装机容量。
- 缺测标签仅排除对应点；全缺测的站日跳过，不回退到更早的起报。
  允许部分有效点参与日分，与 M8 一致。最终分数为所有有效站日分数的等权平均，
  不是全点 RMSE，也不是站均值或月均值再平均。
- 时间沿用数据集时间轴（与 M8 相同），不自动转换时区；起报列应与业务评分的钟表时间一致。
  缺少站点标识或没有可评分的 D+1 数据会明确报错，不静默改回 loss 选模。

验证只运行上述被选中的起报行；日志同时记录评分、有效站日数和有效点数。
`eval_loss` 仍是这些起报行整个预测窗口的 Pinball 诊断值，不参与选模。
多卡评估先汇总各卡预测、去掉补齐样本，再统一计算日分。
训练结束自动恢复评分最高的 checkpoint，随后才导出 `model_result` 并评估；最终结果导出仍覆盖整个验证集。
`train_log.json` 记录每次训练期验证、最佳分数 `best`、最佳步数 `best_step`、
实际停止步数 `stopped_step` 和最佳 checkpoint 路径。最后一步分数变差时仍导出较早的最佳权重，
不会把最后一步当成最佳步；没有选出有效 checkpoint 时直接报错。
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
选模测试覆盖起报筛选、D+1 边界、缺测和日分截零、批量大小，以及 loss 与业务评分方向相反时的
最佳权重保存、恢复和早停，以及按优化步验证、非整间隔最后一步选模、短于一个验证间隔的训练。

独立性测试在仅含本模型和框架依赖的临时目录运行，并禁止导入其他模型包。
层代码迁入本目录本身不改变计算；但随后修复了 Encoder 漏传归一化参数的问题，
从误用的 LayerNorm 改为与 M4 一致的 BatchNorm。
修复前的 LayerNorm 版 m4_pinball 权重无法严格加载到当前结构，需要重新训练；
不可通过忽略缺失权重来视为等价迁移。
