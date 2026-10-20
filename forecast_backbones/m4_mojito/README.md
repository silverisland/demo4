# M4-Mojito

面向未来 4 小时、16 个点的共享光伏预测模型。保留 M4 的跨变量联合注意力，采用多尺度历史编码、天气条件 query 和共享逐时刻预测头。新站与已有站使用同一输入流程，不依赖站点 ID、经纬度、太阳位置或外部大模型预测。

## 结构与文件

默认历史为 672 点，功率/辐照度/温度共三个变量；慢尺度 patch=96，快尺度取最近 96 点、patch=4。93 个历史 token 加一个 STATE token，经 3 层、128 维、4 头编码器。未来天气构造 16 个 query，经 2 层解码器，再与天气及最近 16 点功率旁路共同预测。输出 `[B, 16]`，恢复为实际功率单位。

| 文件 | 职责 |
|---|---|
| `config.py`、`config/configs.yaml` | 默认配置、覆盖与校验 |
| `features.py` | 容量/天气尺度、时间、mask、摘要、监督损失 |
| `augmentation.py` | 仅使用保留历史的尾部扰动 |
| `model.py` | tokenizer、共享编码器、条件解码器与输出头 |
| `transformer_models.py` | 统一 ModelOutput 和模型发现接口 |
| `transformer_trainer.py` | 训练参数及有序评估 |
| `train.py` | 复用 UnifiedDataset、HF Trainer 的训练入口 |

不复制旧 M4 的网络实现或通用数据加载器；旧 M4 及共享框架保持原样。注册名为 `m4_mojito`，展示名为 M4-Mojito。

## 数据契约

每行是一个实体、一个起报时刻，所有序列间隔 15 分钟。字段可在配置中映射。

| 默认字段 | 单行形状与含义 |
|---|---|
| `observe_power` | 历史序列，末点对应起报时刻 |
| `general_ghi`、`general_temp` | 同时间轴的历史天气 |
| `general_ghi_predict`、`general_temp_predict` | 未来序列，第一个点是起报后 15 分钟 |
| `cap_power_on` | 有限正数；与功率单位一致，不从标签推算 |
| `timestamp_win` | 标量起报时间，经 UnifiedDataset 转成 int64 秒 |
| `observe_power_future` | 监督目标，仅训练/评价需要 |
| `station` | 可选元数据，仅保留用于评价，不参与网络 |

历史超过窗口时取末端，短于窗口时在左侧补缺失；未来天气和标签超过 16 点时取前 16 点，不足时明确报错。数据必须预先按真实时间位置对齐，不能把旧观测向右移动。天气序列允许 NaN/Inf 表示缺失，但不能缺少配置要求的整列；整列不可用时填等长 NaN。温度单位为 ℃，辐照度为 W/m²，其他单位先在上游转换。

`timestamp_basis: local_wall` 适配现有数据的 Asia/Shanghai 无时区墙钟时间。若传入的秒数是真实 Unix 时间，配置为 `utc`，模型将加 8 小时计算本地日历特征。禁止混用这两种语义；不做时间单位自动猜测。输入历史天气及未来 NWP 必须是起报时已经可获得的版本。

站级使用本站容量与天气；省级使用业务口径一致的总功率、容量和预先聚合天气，分别训练权重。站点聚合、容量治理、缺站覆盖率不在模型内部猜测，也不新增一套数据管道。

## 训练与推理

Python 3.10+。依赖 PyTorch、Transformers、Accelerate、NumPy、pandas、PyYAML、datasets、PyArrow；测试另外需要 pytest。代码按仓库的 Transformers 4.40.1 接口接入。已在 Python 3.12、PyTorch 2.5.1、Transformers 4.40.1、Accelerate 0.30.1 的 CPU 环境验证，未验证 GPU 或多卡训练。

从项目根目录运行，显式提供已经切分好的数据：

```bash
python -m forecast_backbones.m4_mojito.train \
  --config forecast_backbones/m4_mojito/config/configs.yaml \
  --train /path/to/train.parquet \
  --valid /path/to/valid.parquet \
  --out /path/to/runs/m4-mojito
```

默认 fp32、AdamW、cosine、学习率 3e-4。每设备 batch=64、梯度累积=4；单设备有效 batch 为 256，多设备还要乘设备数。种子在模型创建前设置。CPU 验证可将 `optimization.use_cpu` 设为 true，`--max-steps 2` 可用于检查训练链路。

第一版以验证集 weighted MAE（`eval_loss`）保存最优权重、早停；这不是已确认的正式业务分数。训练入口不负责时间/站点切分，调用方须防止标签跨切分边界，并另外评价完整留出站。全部未来标签无效的行会被训练入口排除；直接调用模型遇到整个 batch 无标签时明确报错，避免零损失仍触发优化器更新。

输出包含 `model_config.yaml`、`run.yaml`、HF 检查点与训练状态、`eval_results.json`、`best/model.safetensors`。配置中保存列顺序、尺度、时间语义和训练种子；输入文件路径写入 run.yaml，实际数据版本/切分清单由上游保存。

```python
from safetensors.torch import load_model

from forecast_backbones.m4_mojito.config import feature_columns
from pipelines.model_registry import build_model
from pipelines.pipeline_wrapper_inference import unified_inference

model = build_model("m4_mojito", "/path/to/run/model_config.yaml")
load_model(model, "/path/to/run/best/model.safetensors")
model.eval()
result = unified_inference(
    model,
    input_frame,
    feature_cols=feature_columns(model.config),
    entity_cols=["timestamp_win", "station"],
)
```

直接前向为 `model(batch_dict, labels=None)`，每个序列 tensor 必须有 batch 维。纯预测与携带标签的前向在 eval 模式输出一致。`model.eval()` 同时关闭 dropout 和训练扰动。

全历史功率或全未来天气不可用时，前向仍保持数值有限，但通过 `.aux` 的 `history_unavailable` / `future_weather_unavailable` 标记。`evaluate_model` 会保留这些列；通用 `unified_inference` 仅取 logits，生产侧需要这些标记时应直接读取 ModelOutput 或使用 evaluator。这种极端输入的预测效果未验证，不能把有限输出当成质量保证。

## 扰动与消融

默认 25% 的训练行抽取 1/2/4/8 点尾部，随机使用显式缺失、前值保持、零填补、前一日同时刻或保留历史趋势外推。概率是试验起点，并非线上缺失率。隐性填补不提供真实扰动 mask，所有 patch、STATE 和旁路都从同一份受扰动历史派生。验证/推理不随机扰动。

- 干净基线：`augmentation.probability: 0.0`。
- 单尺度：`model.use_slow: false`。
- 去近期旁路：`model.use_recent: false`。
- 可通过 `corrupt_tail` 构造固定方法、固定长度的压力测试输入，然后在 eval 模式评价；未将压力分数包装为预期线上收益。

```bash
python -m pytest forecast_backbones/m4_mojito/tests -q
ruff check forecast_backbones/m4_mojito --select E,F,I
ruff format --check forecast_backbones/m4_mojito
```

第一版没有实现站点 ID 分支、太阳位置、外部预测先验、空间网络或自动业务降级，也不兼容旧 M4 权重。结构正确性测试不能替代真实数据的效果评估。
