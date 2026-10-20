# M8 架构复盘：DeltaNet 不是瓶颈，分数在先验、协议与基线里（深度调研报告）

日期：2026-10-13　对象：`forecast_backbones/m8`（PVRex）　前置综述：`docs/survey_covariate_pv_models.md`（2026-10-08）
检索与核实：5 个并行视角、约 150 次检索、约 125 篇在 arXiv / DOI / 出版社页面逐条打开核对；只引用页面上读到的数字，
读不到的写「未读到数字」；无法打开的条目一律不引用。标 [unconfirmed] 的是检索后确认没有文献的方向。

## 摘要

M8 用 Reverso 式「门控长卷积 + Gated DeltaNet」交替骨干处理 54 个 patch token，训练数据只有 17 个站、约 2,700 站日，NWP 只有
GHI 与气温两个变量。站内三轮消融显示 attention、sLSTM、conv+DeltaNet 三种骨干的南网 D+1 都在 ±0.6 的种子噪声内，而 attention
训练快 2.4 倍。本报告追问三件事：DeltaNet 在这个规模上是否多此一举；哪些杠杆有证据能换来 1 分以上；两个掉队站与「半个 epoch 见顶、
之后震荡」的训练现象该怎么治。文献给出的判断与站内证据一致：**DeltaNet 的全部设计动机（线性时间、长上下文联想记忆）在 54 个
token 上不成立，且唯一的小规模受控对比把标准 Gated DeltaNet 排在 xLSTM 与 Mamba-2 之后；但把它换成 attention 也只是等价，不会
涨分。** 2024–2026 年带 NWP 的光伏日前预测证据反复表明：模型类之间的差距约 1 分，而 NWP 质量与物理结构化（−5% 到 −44%）、
容量漂移与站点先验的在线修正（−10%）、多站联合训练（−15% 到 −40%）、集成（−3% 到 −4%）才是可复现的大效应。对 M8 而言，
最有把握的改动依次是：把水平面 GHI 晴空上限换成按站历史估计的包络、无坐标站用历史日形而非钟表几何、把 TimesFM-3 预测作为
提示通道加残差基准、训练协议改为「小学习率 + EMA + 5 种子集成 + 固定评估协议」、补一个带工程特征的 LightGBM 全局基线。
这些改动没有一项要求换骨干；若要换，换成 patched attention 以节省训练时间，并补测 a6_no_mixer 下界。

## 1. 引言与研究问题

M8 的出发点是把零样本时序基础模型的骨干（TiRex 的 sLSTM、Reverso 的 conv+DeltaNet）搬到带 NWP 的光伏场景从零训练。
前一份综述（2026-10-08）回答的是「协变量该在哪里融合、混合器怎么选」，当时的证据主要来自 2048 点上下文、千万级序列的
基础模型消融，和 M8 的实际规模（54 token、17 站）相差两个数量级。三轮站内实验之后，用户提出的质疑很具体：DeltaNet 是
linear attention 的进化版，目的是把 O(N²) 降到 O(N)，而我们根本没有这个需求，是不是多此一举？模型效果不理想，架构要不要改？
有没有更好的办法？

本报告冻结三个研究问题：

- **RQ1**：在约 50–100 个 token、小数据、从零训练的条件下，DeltaNet / Gated DeltaNet / sLSTM 这类线性递推混合器相对 softmax
  attention 与「无混合器」基线有没有精度或效率证据？DeltaNet 的「上下文内在线回归」归纳偏置在短序列上兑现过吗？
- **RQ2**：在 17 站、两变量 NWP、15 分钟日前预测这个规模上，哪些杠杆有超过 1 分（种子噪声 ±0.6）的证据？有没有哪一类模型
  （GBDT、微调基础模型、TabPFN-TS、iTransformer、TSFM+修正器）能稳定胜过调好的 5M 混合模型？
- **RQ3**：站内观测到的三类失效——物理先验错位（倾斜/跟踪/双面阵列的晴空上限、缺坐标站的钟表几何）、半个 epoch 见顶后
  验证分震荡 1.5–3 分、超短期指标落后——文献里各有什么被证据支持的处方？

报告按「M8 现状与站内证据 → 方法 → 分类体系 → 五个分支 → 跨分支讨论 → 开放问题 → 结论」组织；结论逐条回答 RQ1–RQ3，
并给出按证据强度排序的实验清单。

## 2. 研究对象：M8 的方案与实施现状

### 2.1 方案怎么落地的（代码事实）

| 层 | 实现 | 位置 |
|---|---|---|
| 逐步通道 x[B,864,17] | y/cap、y_mask、is_future、4 个 NWP 通道（GHI、气温；两个 Solargis 通道在现有数据里恒为 0）、k_nwp（NWP 晴空指数）、elev/90、p_clear/cap（pvlib Ineichen 水平面 GHI 换算）、is_night、tod/doy 正余弦、nwp_resid（历史白天 y − GHI_pred/1000）、cov_missing | `features.py` |
| 静态向量 + 站点嵌入 | log cap、经纬高、degraded、窗内统计 3 个；Embedding(4097, 256)，槽 0 为未知站，训练时 slot_dropout 0.3 | `features.py` / `model.py` |
| token 化 | patch 16 步 → 42 历史 + 12 未来 = 54 token；ResidualMLP(16×17→256) + 静态投影 + 站嵌入 + 位置嵌入 | `model.py` |
| 骨干 | 4 个 pre-RMSNorm Block：[conv, deltanet, conv, deltanet]；GatedLongConv 核长 54；GatedDeltaNet 4 头 64×64 状态、q/k 短卷积 + L2 归一、β 写入门、α 遗忘门初值 ≈0.98，**按 54 步 Python 循环递推** | `layers.py` |
| 输出头 | step_cov：每步拼 [token 隐状态, patch 内位置嵌入, 该步原始 17 通道] → ResidualMLP → 9 档分位 | `model.py` |
| 损失与增强 | pinball（未来段 + CPM 掩码段；夜间同权，2026-10-12 起）；CPM p_row 0.5、协变量 dropout 0.1、幅度调制 0.3 | `transformer_models.py` / `features.py` |
| 决策层（无参数） | 单调化 → 1.05·p_clear 晴空上限（夜间不生效）+ 容量上限 → D+1 段 p*、前 16 步 p*_L1 | `transformer_models.py` + `m6.decision` |
| 规模 | 4.96M 参数，其中站点嵌入 1.05M；batch 64 → 约 4,000 步/epoch | — |

### 2.2 站内证据（三轮验证 + 测试集）

| 证据 | 数字 | 含义 |
|---|---|---|
| 骨干消融（第一轮，旧切分，单种子） | a1 conv+deltanet 76.01 / a3 attention 75.88 / a2 sLSTM 73.9–74.0（第二轮）；训练 19.0 vs **8.05** vs 41–45 分钟 | 三种骨干精度不可分辨；attention 快 2.4 倍，sLSTM 慢 2.2 倍 |
| 第二轮（m8_v1，3 种子） | a1 74.38±0.37；c5_large（13.6M）74.13；b1 双流 73.75；b3 token 头 73.82；c2 recent_peak 73.82 | 加参数、改融合位置、改归一化都不涨 |
| 唯一超出噪声的项 | d2 p* 决策 +1.05；c8 调度 +0.26；3 种子集成 +0.3–0.5 | 收益来自决策层与集成，不来自结构 |
| NWP 通路总价值 | b4_no_nwp 73.44 vs a1 76.01 | 整条协变量通路只值 2.6 分 |
| 训练曲线 | a1 在 1,500–2,500 步（0.4–0.6 epoch）见顶，之后验证分 73.2–75.8 来回 | 过拟合快、验证噪声大 |
| 测试集（e1_final ×3，--full） | 逐站均值 76.13；对 M5（TimesFM-3 零样本+协变量）17 站中 15 站更高；对 M4 多数站高约 3 分 | 整体并不差 |
| 两个掉队站 | 华能北方润达（双面 + 跟踪，40.3°N，12 月）晴空上限 0.51·cap 压峰、bias 0.66，上限放宽后 +1.7；国能和煦无坐标 → 钟表几何、白天过宽，比 M5 低 12.2 | 单站损失 6–12 分，是物理先验错位，不是骨干 |
| 从未跑过 | a4_deltanet_only、a5_conv_only、**a6_no_mixer**、b2_bidirectional、d3/d4（TimesFM）、h1–h3（晴空比） | 「无序列混合」下界未知 |

这些数字本身已经把问题缩小了：骨干不可分辨，先验错位可损 6–12 分，决策与集成各值约 1 分。文献要回答的是，这个格局是不是普遍规律，
以及还有哪些没试过的杠杆。

## 3. 方法

五个检索视角并行（每个视角两轮，宽→窄）：P1 混合器在短序列小数据上的证据；P2 2024–2026 年带 NWP 的光伏日前基准与竞赛；
P3 小数据、重叠窗口的训练协议；P4 物理先验、归一化、无元数据晴空包络、TSFM 残差学习；P5 微调基础模型、TabPFN-TS、GBDT、
变量 token 模型等替代模型类。时间范围以 2024–2026 为主，经典锚点（DLinear、PatchTST、N-BEATS、M4）保留。纳入标准：有可读数字的
受控对比或竞赛/基准结果；排除：无法打开、只有二手摘要、或天气输入为再分析/实测却未标注的工作（这类工作会高估协变量价值，
下文凡引用都注明输入类型）。最终语料约 125 篇，分支内三篇以上为覆盖底线。

证据分级按 CS/AI 惯例：同管线消融 > 公开基准/竞赛 > 自报告；措辞按梯度：多源一致用「表明/证实」，两三项用「提示/支持」，
单项或机理推断用「可能」。

## 4. 分类体系

按「杠杆」分五个分支，分支内再按证据条件分：

| 分支 | 回答 | 关键条件变量 | 空格（= 发现） |
|---|---|---|---|
| A 混合器 / 骨干 | RQ1 | token 数、参数规模、通道数、地平线 | ≤100 token + 小数据 + 从零训练的直接消融：**没有** |
| B 训练协议 | RQ2、RQ3b | 数据规模、评估协议、集成规模 | 重叠窗口有效样本量：**没有**；TS 上 EMA/SWA 的有/无对照表：**没有** |
| C 物理先验与输入 | RQ3a | 阵列类型、元数据完整性、历史长度 | 双面/跟踪电站上晴空上限失效的实测：**没有** |
| D 外部预测作先验（TSFM 残差/提示） | RQ2、RQ3 | 基座强度、残差学习器是否持有 TSFM 缺的信息 | 光伏上「零样本 TSFM 作输入通道」的同行评审研究：**没有** |
| E 替代模型类 | RQ2 | NWP 类型（业务/再分析）、站数、特征工程 | 多站 15 分钟业务 NWP 下 FT-TSFM vs GBDT vs 从零混合模型的同协议对比：**没有** |

跨分支的「骑墙」工作有三篇，值得单独说明：Za'ter & Hodge（ERCOT 电力系统基础模型评测）[63] 同时给出微调 TSFM、从零 Transformer
与未见站点结果，落在 D 与 E；Xu 等的 PVDAQ 专家融合 [64] 同时含 LightGBM、微调 Chronos-2 与集成，落在 D 与 E；NVIDIA 的
Cross-Unet [69] 是唯一一篇多站 15 分钟 WRF NWP 下深度模型大幅胜过 GBDT 的工作，同时对 A（NWP 怎么进网络）与 E 发言。

## 5. 分支 A：混合器在 54 个 token 上值不值

### 5.1 效率动机在这里不存在

DeltaNet 与 Gated DeltaNet 的原始论文把全部效率论证建立在序列长度上：chunkwise 算法「随序列长度 L 和头维 d_head 增大获得更大
加速」，而「所有线性时间模型在更长序列训练上优于 Transformer」[1]；Gated DeltaNet 的作者自己写道「Transformer++ 在 2K 上下文
窗口内取得最好的吞吐」（FlashAttention-2）[2]。2026 年的线性注意力架构综述给出的交叉点同样在千级 token：4k 时纯 Gated DeltaNet
吞吐为 softmax 的 1.22 倍，32k 时 3.5 倍 [3]。Based 的作者则直接指出「线性注意力实现常常慢于优化过的标准注意力」，24 倍吞吐
只在定制 IO 感知核之后出现 [4]。文献中没有任何低于 1K token 的计时 [unconfirmed: no work retrieved with timings below 1K tokens]。
M8 的 GatedDeltaNet 是 54 步 Python 循环，没有 chunk 核；站内数字（a1 19.0 分钟 vs a3 8.05 分钟）与上述方向一致。
**在效率这一维上，DeltaNet 在 M8 里是净损失，这一点证据充分。**

### 5.2 精度：短序列上差距缩到噪声，小参数上标准 GDN 反而垫底

仅有的两个含 attention 对照臂的时序基础模型消融方向一致但量级不同。Reverso 在 2048 个逐点 token、2.6M 参数上测 9 种混合器：
DeltaNet 0.732、Gated DeltaNet 0.730、长卷积 0.735、attention（RoPE）0.750、conv+DeltaNet 0.725 MASE，种子噪声 0.002–0.003 [5]；
线性递推家族内部差 ≤0.005，attention 落后 0.02–0.03。TiRex 是最接近 M8 的设置（64 个 patch token、9 档分位）：sLSTM 0.411 vs
Transformer 0.422 总 CRPS，但**短地平线只差 0.006（0.455 vs 0.461），在 Chronos-ZS 的 MASE 上 Transformer 反而更好（0.768 vs
0.776）**；同一表里 CPM 掩码值 0.013–0.034 CRPS，大于换骨干 [6]。两项工作都出自提出该架构的实验室，权重要相应打折。

更关键的是唯一一个匹配 M8 参数量级的受控比较。Hartl 等用 TiRex 协议在 1M–80M 参数、256 个 token 上只换混合器 [7]：1M 时
xLSTM 0.768、Mamba-2 0.795、**Gated DeltaNet 0.882** MASE；10M 时 0.733 / 0.767 / 0.817；40M 时 0.716 / 0.721 / 0.792；到 80M
才收敛（0.715 / 0.715 / 0.735）。允许负特征值的 GDN[−1,1] 把差距收回大半（1M 0.792，10M 0.764）。M8 的实现用 β = σ(·) ∈ [0,1]，
正是被排在最后的标准变体。这个结果没有 Transformer 对照臂，所以不能说 attention 会赢 xLSTM，但它足以说明：**在 5M 这个量级，
没有证据支持 Gated DeltaNet 是最优的递推选择，有一项证据说它是最差的。** TempoPFN 的 Householder 步数消融（H=1 即 DeltaNet
0.573 → H=4 0.561 CRPS）[8] 与 DeltaProduct [9] 讲的是同一件事：delta 规则的表达力要靠加强版才兑现，而且都在 2048 长度上。

反向证据也存在：「Revisiting the Generic Transformer」[10] 与 Moirai 2.0 [11] 用普通 patch attention 在受控配方下达到 SOTA；从零训练
的短回看实验里，普通 xLSTM 在 ETTh（回看 96）上比 attention 类模型差 0.24–0.30 MSE [12]，Mamba+attention 的混合体在 ETTh1 上比单层
DLinear 差 0.24 MSE [13]。这些工作的共同条件是**小数据 + 短回看 + 从零训练**，正是 M8 的处境。

### 5.3 「无混合器」基线有多强

多项独立证据表明，在小数据、少通道的条件下，不做时间轴混合的模型与 2–6M 参数的序列模型只差几个百分点甚至更好：供热负荷基准
（13.6 万小时点、72 token）上 8k 参数 FCN 的 RMSE 与 2.1M 参数 xLSTM 相差 13–16%，并在 24 小时地平线上优于 3.7M 的 Transformer
编码器 [14]；调好的 Ridge 回归在 8 个 LTSF 基准中的 6 个上优于 Transformer/MLP/CNN [15]；TFB 在 8,068 条单变量序列上让线性回归与
随机森林排第一，归因于每序列数据有限 [16]；DLinear [17] 与 PatchTST [18] 的经典对照说明 patch 化而非混合器类型才是修复
Transformer 的关键。光伏上的对应证据是 Markovics & Mayer 的 24 模型比较（16 站、2 年、15 分钟、NWP 输入）：核岭回归与 MLP 居首，
加入太阳位置角与处理过的辐照把 RMSE 降 13.1%，与模型之间的全部差距同量级 [20]；TabPFN-TS 完全不做序列值的时间混合也能
在协变量预测上达到 SOTA [21]。

但混合器并非一律无用。TFB 的条件分析指出 Transformer 类在「明显季节性」数据集上胜过线性模型 [16]；ms-Mamba 在回看 96 时，
7 通道 ETT 上 SSM / attention / 线性模型差 ≤0.01 MSE，而在 321/862 通道的 Electricity/Traffic 上跨变量混合值 0.05–0.2 MSE [19]——
混合器的价值在变量轴，不在时间轴。光伏有强日周期（支持保留某种混合）、却只有几个通道（不支持跨变量注意力）。

### 5.4 DeltaNet 的归纳偏置在短上下文上兑现过吗

前一份综述提出的机理假设——delta 规则等价于「NWP 特征 → 实测功率」的上下文内在线回归——在文献里只有语言模型规模的回忆
任务作支撑：固定状态模型「在回忆上吃力」[4]，门控卷积比 attention 差「至多 2.1 困惑度点」且 82% 来自联想回忆 [22]，而循环模型
的检索缺陷被定位为少数头的「尖锐选择」能力而非状态大小 [23]。后者意味着这个缺陷在短上下文上也存在。没有任何时序工作在
整个上下文能装进状态矩阵（d_head² ≥ token 数 × 键维）的条件下测过 delta 规则的收益 [unconfirmed: no work retrieved]。
站内的 a4_deltanet_only 与 a6_no_mixer 从未跑过，所以这个假设在 M8 上既没被证实也没被证伪。

**分支结论**：DeltaNet 在 M8 里的效率理由不成立（证据充分）；精度上与 attention 不可分辨（两项同管线消融在短地平线上差 ≤0.006，
与站内 ±0.6 内的结果一致）；在 5M 参数量级上标准 GDN 被一项受控研究排在最后（单项证据）。文献支持的改动不是换一种花哨的混合器，
而是把消融预算花在配方上，并补一个 attention 或无混合器的对照。

## 6. 分支 B：小数据训练协议

### 6.1 架构差异落在噪声里是文献常态

「No Champions」在 8 模型 × 14 数据集、约 5,000 次训练上做 Friedman 检验，统计量 1.14、p = 0.95，而超参搜索本身值约 15%
（DLinear/Weather/H96 MSE 0.198 → 0.168）[24]；BasicTS+ 发现同一模型在不同论文里差 20–33% [25]；TFB 发现仅「drop last」一项
就能让 ETTh2/H336 的 MSE 差 0.04–0.07 [16]；TSPP 用 K=256 个种子重复训练后指出「换个随机种子就会出现精度离散」[26]。
M8 观测到的 1.5–3 分评估间震荡大于 0.37 的种子标准差，说明**按单次验证挑 checkpoint 本身就是在挑噪声**。

### 6.2 模型该更小、学习率该更低

时序缩放律的结论是：最优模型与回看长度随数据量增长，「数据稀缺时大模型会过拟合」[27]。剪枝研究在 ETTm2（1.7 万样本）上把
嵌入缩到 1/10，Transformer 的 MSE 从 0.502 降到 0.310，Informer 0.405 → 0.336，并观察到「模型收敛极快，暗示模型过大」；换到
80 万样本的 ENTSO-E 后方向反转 [28]。超参流水线发现 PatchTST 在 ETTh1 上「更小的学习率（≤1e-4）配更多 epoch（≥5）」最好 [29]；
PatchTST 自己在小数据集上用 D=16、dropout 0.2 [18]。M8 的 257k 窗口来自约 2,700 站日，每站约 1.5 万个时间步，正是 ETTh1 量级；
5M 参数、lr 3e-4、dropout 0.1 在这个量级上偏大偏激进。站内 c5_large 不涨、c8 降 lr 略涨，与此一致。

### 6.3 集成是最有资历的正则化器

N-BEATS 的结论是「集成是比常见替代品强得多的正则化手段」：dropout/L2 改善单模型却伤害集成，180 个成员缩到 18 个 OWA 只损失
不到 0.5%（0.797 vs 0.803）[30]；M4 竞赛 17 个最准方法里 12 个是组合 [31]；水文领域 TFT 的种子集成把 CRPS 从 0.83 降到 0.73 [32]。
光伏上的异构集成给 1–4%，多数未过显著性检验 [41]。M8 的 3 种子 +0.3–0.5 在这条曲线的起点，5–10 种子大概率还能再要到几个十分位。
一个常被忽略的推论：按单模型调的 dropout/weight decay 会侵蚀集成收益，正则化应当按集成分数调。

### 6.4 EMA、增强、自监督：证据强度递减

权重 EMA 在视觉领域被证实改善泛化与「预测一致性」[33]，但时序上只有 TSPP 一个来源且自相矛盾（正文「结果不一」，附录「相当
有用」，34 个最优配置里 27 个选了 EMA）[26] [unconfirmed: no TS with/without table]。它针对的正是评估间震荡，是最便宜的 A/B。
频域增强在 5 种子控制下给 ETT 量级 3–10% MSE（Dominant Shuffle：TiDE 0.479 → 0.441，LightTS 0.565 → 0.499）[34]，FrAug 在 1%
数据上收益最大（ETTh2/DLinear 0.572 → 0.351）但单次运行、有多处回退 [35]；最新的 11 种增强横评指出**只增强输入不增强目标
「显著伤害性能」**，强模型收益最小 [36]；策略搜索也能在小数据集上产生 −104% 的灾难项 [43]。没有任何光伏增强研究
[unconfirmed]。同数据自监督预训练是矛盾的：PatchTST 自己的掩码预训练在最小的两个数据集 ETTh1/h2 上反而更差（0.430 vs 0.417，
0.355 vs 0.331）[18]，SimMTM [37] 与 TimeSiam [38] 的收益建立在偏弱的随机初始化基线上（TimeSiam 的 PatchTST 基线 0.473 比原文
0.417 差 13%）；光伏上有证据的是跨站预训练（SolarBench：本地数据需求减少 50% 以上）[40]，不是同站自监督。
选择性学习（按不确定性掩掉损失中不可靠的时间步）在 NeurIPS 2025 上报告 6.5–37% MSE [39]，与云过境时刻的光伏噪声天然匹配，单项证据。

### 6.5 重叠窗口

没有工作量化深度预报模型在重叠窗口上的有效样本量或步长消融 [unconfirmed]。能用的证据是泄漏：窗口级随机切分能把真实 20–60%
的准确率抬到 99.9% [42]，Bergmeir 等证明只有在残差不相关时 K 折才有效 [84]。M8 的按时间块切分已经是安全做法；
推论是把「一个 epoch」理解为 2,700 站日而不是 257k 窗口，于是 0.4–0.6 epoch 见顶就不奇怪了。

**分支结论**：文献对这个规模的处方高度一致——小模型、低学习率、更长的退火、EMA、5–10 种子集成、固定评估协议（≥5 种子、
配对检验、不按单次验证挑点）。增强与自监督是二阶且有回退风险。

## 7. 分支 C：物理先验要鲁棒，不要硬

### 7.1 归一化：固定物理尺度优于窗口统计，晴空比作目标有风险

CycleNet 的附录给出最直接的证据：在 Solar-Energy（137 站、10 分钟）上去掉 RevIN 让 MSE 从 0.250 降到 0.208（H=96）、0.351 → 0.258
（H=720），作者归因于「夜间连续零值」扭曲窗口均值 [44]。这支持 M8 的容量归一化而不是 M4 的 RevIN。但把晴空指数当**目标**
的唯一直接对比里它输给了原始功率（Sun 等，15 分钟 nowcast）[45]，Voyant 等在 2025 年进一步指出晴空模型「对低太阳高度角下气溶胶
光学厚度误差高度敏感」，无晴空模型的 ELM 持平或优于 McClear 归一的 ARMA [46]。条件是清楚的：晴空量作**输入**在 NWP 驱动的日前
场景且历史足够时有收益（Mayer 与 Yang：加入物理计算的预测因子 MAE −5.2% 到 −10.4% [47]）；作**目标或硬上限**时，一旦
晴空模型错（低高度角、气溶胶、几何错位）就系统性伤害。华能北方润达的 −8 分就是这个机理。

### 7.2 水平面 GHI 对跟踪与双面阵列不是上限

PVAnalytics 的文档默认把日晴空比上限设为 1.25，并明确「若传入跟踪系统的 POA 辐照，建议把日上限提高到 1.35」[49]；GMD 2026 的
PINN 流水线报告 GHI 与 POA 之差「可达约 200 W/m²」[48]；M8 的 1.05·p_clear 上限按水平 GHI 算，对跟踪、双面、冬季高纬的站结构性偏低。没有任何研究量化过晴空上限在双面跟踪电站上的
失效 [unconfirmed]，站内数字（上限 0.51·cap，观测峰 0.57–0.80·cap；放宽到 3.0 即 +1.7 分）是目前唯一的实测。

### 7.3 无元数据的晴空包络与方位推断

Statistical Clear Sky Fitting「不需要天气、辐照或系统配置元数据」，只用功率拟合日晴空包络 [50]，Solar Data Tools 把它做成了
舰队级的自动化流水线（数据质量、晴空拟合等）[51]。这是 M8 新加的「近正午 0.98 分位 y/p_clear」的原则化版本。方位推断对
固定倾角阵列可靠：44 个系统上 SDT 与 PVWatts 拟合的中位误差约 1°（倾角）与 5°（方位）[52]，但**数据集里没有跟踪系统**，没有
任何方法从功率拟合跟踪参数 [unconfirmed]。

### 7.4 缺坐标站：用历史日形代替几何

440 站的冷启动研究发现「精度主要取决于是否有合理的时间上下文，而不是由哪种生成器产生」，带协变量的 TabPFN-TS 日均 MAE 0.514
对比协变量盲的 TiRex 1.061 kWh/kWp [54]；ERCOT 评测里基础模型在未见站点上退化最少（TimesFM 4.73 vs TFT 6.11 vs LSTM 12.35 nMAE）[63]；两阶段方法直接把
电站特性（方位、遮挡）从历史里学出来 [55]。SolNet 则从反面说明「源位置误设」对迁移有重大影响 [56]。M5 在国能和煦站比 M8 高
12 分，是同一机理：TimesFM 不用几何，M8 用了错的几何。处方是用站内历史估计日出日落与逐时段包络，替换该站的 p_clear 与 elev
通道；冷启动时几何只能用来生成合成历史 [54]，不能当硬上限。

### 7.5 NWP 偏差在功率侧修

Mayer & Yang 在 3 站 4 年上比较 4 个后处理位置：「只对 GHI 后处理带来的收益是边际的，如果可量化的话」，结论是「后处理最好在
功率上做」[53]；单系统 20 kW 的随机森林研究报告输入侧 MOS 让功率误差降 70–79% [P4-14]，但没有功率侧对照。M8 的 k_nwp 与
nwp_resid 通道已经是隐式的功率侧 MOS，与证据一致；集合后处理研究也支持非参数分位网络在功率侧校准 [W05]。NWP 有误差时序列
模型会把依赖转向「历史观测与确定性物理先验」[57]，这解释了为什么先验错位的代价在 NWP 弱时更大。

**分支结论**：容量归一化保留；晴空功率从「硬上限」降级为「软先验 + 按站包络校准」；对无跟踪参数、无坐标的站，历史包络优先于几何。

## 8. 分支 D：把基础模型的预测当先验

M4 把 Chronos 零样本预测拼进 PV 通道、再用南网损失学修正，同事把 TimesFM-3 插值喂给模型；M8 的 d3/d4 实现了同样的思路但
从未跑过。文献里最接近的是 PARA-PV（2 个国网电站、15 分钟）：冻结 Chronos 中位数作为适配器输入并作加性基准，适配器只学残差，
消融去掉 Chronos 后 MAE 0.1352 → 0.1461、RMSE 0.3052 → 0.3314（约 −7.5% / −7.9%）[58]。TimesFM 官方的 XReg 示例在电价上把 MAE 从
6.73 降到 5.34（−21%）[59]，AutoGluon 把「表格模型吃协变量、TSFM 预测残差」做成了内置选项 [60]，干旱预报上残差细化把 TimesFM
的 MSE 降 18.9% [61]，LightGTS-Cov 用 1M 冻结骨干 + 0.1M 协变量 MLP 胜过 100 倍大的零样本模型 [62]。

上限同样清楚：零样本基础模型在光伏上不具竞争力（Chronos-2 零样本 7.70 vs 微调 4.59 vs LightGBM 4.50 nMAE [64]；TimesFM 零样本
9.25 vs 微调 3.71 [63]），而微调后的 Chronos-2 仍输给带 NWP 与几何特征的 LightGBM（nMAE 4.59 vs 4.50）与 8 专家融合（4.32）[64]；8 专家融合里加入微调
Chronos-2 只比 LightGBM 多 0.19 个百分点 [64]。因此残差学习器必须持有 TSFM 没有的信息（NWP、几何），收益在历史主导、数据稀缺、
未见站点时最大，在 NWP 主导时最小。M8 的 d4 变体（提示通道 + 恒等跳连 + hint_dropout）正是 PARA-PV 的形式；期望是单位数百分比
的误差改善，折成南网分约 0.5–1.5 分，且主要落在白马湖、国能和煦这类站。没有任何同行评审的光伏研究把零样本 TSFM 作为输入通道
喂给从零训练的序列模型 [unconfirmed]，M4 的做法在文献里没有先例可引。

## 9. 分支 E：替代模型类

| 模型类 | 对照证据（输入类型） | 结果 | 对 M8 的含义 |
|---|---|---|---|
| GBDT + 工程化 NWP/几何特征 | HEFTCom2024 实战（业务 ICON-EU/GFS，半小时）[65][66]；100 站纽约（小时）[67]；PVDAQ 15 分钟（存档 GFS）[64][70]；中国 5 城 15 分钟业务 NWP 部署 [71] | CatBoost 以 22.18 vs 24.64 pinball 胜最佳神经网络队；LightGBM 6.07 vs 线性回归 7.27 RMSE（DNN 数字未读到）；LightGBM 4.50 < 微调 Chronos-2 4.59 < N-HiTS 4.76；4.54 < iTransformer 5.03；SolarBoost 比 CNN-LSTM 好 3–14% | 唯一有「持平或更好」一致证据的替代；M8 缺这个基线 |
| 从零训练深度模型 | 7 站匈牙利 HRES 12 模型 [68]；NVIDIA Cross-Unet（PVOD 河北 4 站，15 分钟 WRF）[69] | 12 模型技巧分 46.7–47.7%（差 ≤1 分）；Cross-Unet 比 XGBoost/GBDT MAE 低 30–40%（自报告，树特征集未述） | 模型类差 ≈1 分；Cross-Unet 是唯一的大幅反例 |
| 微调 TSFM | ERCOT [63]；GIFT-Eval 能源 [73]；FETS [74]；楼宇 [75]；负荷 [76] | 微调 TimesFM 3.71 ≈ TimeXer 3.85 ≈ TFT 3.91；分位损失上 FT 反而变差（Chronos-2 MWQL 0.520 → 0.617）；LoRA 只在 16/54 序列有益；需 ≥80% 数据才拉开 | 期望 ≈0 ± 噪声，分位校准风险，V100 fp32 微调成本高 |
| 协变量对 TSFM 的风险 | 噪声协变量 [77]；真实 HRES 温度 [78] | 完美协变量下 Chronos-2 FT 胜 TimesNet 13%，噪声 λ=1 时反输 12%，λ=5 时反输 39%；真实预报让 Chronos-2 +10% CVRMSE | 业务 NWP 只有 GHI/T2m，TSFM 会过度信任 |
| TabPFN-TS | 日尺度 440 站 [54]；电价 [79]；协变量简单关系 [80]；低压负荷 [81] | 日尺度光伏 MAE 0.514 vs Chronos-2 0.537（再分析天气）；电价胜市场专用 DNN 14%；协变量利用远好于 Chronos-2；但 15 分钟光伏无证据，每次预报 1–2 s | 值得一次实验；期望 ±1 分 |
| iTransformer / 变量 token | [69][70] | 有 NWP 时通道相关设计优于通道独立，但 iTransformer 输 LightGBM 0.5 pp、输更小的编码-解码器至多 10% | 不优先 |
| TSFM + 小修正器 | [62][64] | 冻结小骨干 + 协变量 MLP 胜零样本大模型；融合里 +0.19 pp | 作集成成员可，不指望 >1 分 |
| 部署 | [82][83] | Chronos-2 120M 一张 A10G 300 序列/秒；TimesFM-2.0 500M 在 V100 上 P95 200 ms | 推理不是瓶颈，微调才是 |

表注：模型类之间的差距在业务 NWP 条件下普遍 ≤1 分，唯一超过 1 分且方向一致的是 GBDT 相对零样本基础模型。

一个必须记住的混杂因素：用再分析或实测天气的研究系统性高估协变量价值——换成业务 24 小时预报 RMSE +13.1% [W18]，HRRR 代替
卫星观测 MAE +11–68% [55]，Chronos-2 在真实 HRES 下 +10% [78]。这与站内「整条 NWP 通路只值 2.6 分」相容：不是通路接得不好，
是输入本身只有两个变量且带误差。HEFTCom 第四名的光伏模型只用了短波辐射、云量与小时三个特征 [66]，云量是 M8 没有的变量。

## 10. 跨分支讨论

三条独立线索汇到同一个判断。第一，**骨干在三个层面都不可分辨**：基础模型消融在短地平线上差 ≤0.006 CRPS（A），光伏基准里
12 个模型类差 ≤1 技巧分（E），站内 3 种骨干差 ≤0.4 分（§2.2）。第二，**可复现的大效应都在骨干之外**：多站 NWP 降尺度 −44% RMSE
[72]，物理特征 −5% 到 −10% MAE [47]，容量漂移在线修正 −9.9% pinball [66]，多站联合训练 −15%（LightGBM）到 −40%（LSTM）[W24]，
集成 −3% 到 −4% [64][41]；站内对应的是 NWP 通路 2.6 分、决策层 1 分、集成 0.3–0.5 分、先验错位单站 6–12 分。第三，
**训练协议文献与站内曲线互相印证**：ETTh1 量级的数据上模型「收敛极快」[28]、架构差异 p = 0.95 [24]、集成是最强正则化器 [30]，
与 M8 的半 epoch 见顶、变体全在噪声内、3 种子 +0.4 逐一对应。

两处张力值得保留而不是抹平。其一，Cross-Unet 在 PVOD 4 个河北站、15 分钟、WRF NWP 上比 GBDT 低 30–40% MAE [69]，与 PVDAQ、HEFTCom
的 GBDT 占优相反；差异的条件是树模型的特征工程（[64][70] 用了 40 个 GFS 网格与几何特征，[69] 未描述）和站数，无法裁决，只能
用站内 GBDT 基线来定。其二，TiRex 认为递推混合器的状态跟踪对长地平线有用（长地平线 CRPS 0.325 vs 0.342）[6]，而 M8 的 192 步
（2 天）地平线确实属于「长」；但 Hartl 等在同一协议下把标准 GDN 排在最后 [7]，所以即便要保留递推，也不该是现在这个变体。

一个被各分支共同忽略的前提：文献用 nMAE/MSE/CRPS，南网 D+1 用带 0.2·cap 地板的相对 RMSE。地板会压低低功率时段误差的权重，
文献里 RF 与 GBDT 在低功率段的差异 [W07] 在这个指标下可能消失，而夜间与边界日的处理（12-31 全站 0 分）会以文献里不存在的方式
进分。所有跨模型比较都必须在同一指标、同一容量口径、同一训练区间下重做——M4 的数字目前都不满足这三条。

## 11. 开放问题

1. **≤100 token、小数据、从零训练下混合器的直接消融不存在。** TiRex（64 token、35M、大语料）是最近的替身。站内 a6_no_mixer
   与 a3_attention（多种子）就是这个空格的实验，20 分钟一张卡。
2. **双面/跟踪电站上晴空上限失效没有实测文献**，跟踪参数无法从功率拟合 [52]。站内 h 系列是目前唯一的数据点。
3. **「零样本 TSFM 作输入通道 / 残差基准」在光伏上没有同行评审研究**；PARA-PV [58] 最近，单位数百分比收益。d3/d4 跑完即可
   填这个格。
4. **多站 15 分钟业务 NWP 下 FT-TSFM vs GBDT vs 从零混合模型的同协议对比不存在**；[63] 是 5 分钟 ERCOT、[64] 单站、[69] 无 TSFM。
5. **全局模型需要多少站**没有缩放研究 [unconfirmed]；FusionSF 的 20 站拐点（前一份综述）是唯一数字。
6. **EMA/SWA 在时序预报上的有/无对照表不存在**；TSPP [26] 自相矛盾。
7. **南网/国网考核口径下的模型类比较没有同行评审文献**；SolarBoost [71] 报告的 95.7% → 96.3% 是唯一相近的数字。

## 12. 结论

**RQ1。** DeltaNet 在 54 个 token 上没有效率意义：其全部加速论证建立在 ≥2K token 与 Triton 核上 [1][2][3]，M8 的 Python 循环
实现比 attention 慢 2.4 倍。精度上，两项含 attention 对照的同管线消融在短地平线上差 ≤0.006 [5][6]，与站内 ±0.6 内的结果一致；
在 5M 参数量级，唯一的受控比较把标准 Gated DeltaNet 排在 xLSTM 与 Mamba-2 之后 [7]。「上下文内在线回归」的归纳偏置在短上下文
上没有任何时序证据。因此：**用 DeltaNet 是多此一举，但换掉它不会涨分**；若换，换成 patched attention 以省训练时间，并补测
a6_no_mixer 下界。证据等级：效率结论充分，精度结论中等。

**RQ2。** 有 >1 分证据的杠杆按强度排序：（i）物理先验鲁棒化——按站历史包络替代水平 GHI 硬上限、无坐标站用历史日形
[44][47][49][50][54]，站内两站各值 6–12 分；（ii）训练协议——小模型、lr ≤1e-4、EMA、5–10 种子集成、固定评估协议
[24][27][28][30][33]，期望 0.5–1.5 分；（iii）TSFM 残差/提示 [58][59][61]，期望 0.5–1.5 分且集中在数据稀缺站；（iv）带工程特征
的全局 LightGBM/CatBoost 基线 [64][65][70][71]，期望 ±1 分、方向未知，但它是判断 M8 是否值得继续的参照。能稳定胜过调好的
5M 混合模型的替代模型类**不存在**：微调 TSFM 期望 ≈0 且分位损失有恶化风险 [63][73][74]，TabPFN-TS 在 15 分钟光伏上无证据 [54][80]，
iTransformer 类不优先 [69][70]。证据等级：中等（多数为 2026 年预印本，单站或再分析天气占比高）。

**RQ3。** 先验错位：把 p_clear 从硬上限降为软先验，用 SCSF 式包络 [50][51] 按站校准并门控（包络/p_clear 近正午比值出 [0.8, 1.3]
或日照窗口与几何差 >1 h 即判元数据不可信），跟踪站上限至少 1.35 [49]；缺坐标站用历史估计的日出日落与逐时段包络替代几何通道
[54][55]。训练震荡：这是 ETTh1 量级数据的常态 [28]，处方是 d_model 减半、lr 1e-4、更长 cosine、权重 EMA、按「最近 K 次验证
均值的种子中位数」选点，并把 dropout/wd 按集成分数调 [30]。超短期：文献支持的是持久化/最近观测直连与指标形状的损失项
（前一轮 g 系列的方向），不是换骨干；频域增强对短地平线有回退记录 [34]，超短期模块要单独验。

**本报告的贡献**是把「架构要不要改」这个问题从骨干转到先验与协议：五个独立视角（基础模型消融、光伏竞赛与基准、训练协议、
物理先验、替代模型类）没有一个把骨干列为一阶因素，而站内三轮消融恰好是这个结论在 54 token、17 站规模上的复现。

### 对 M8 的实验清单（按证据强度与成本）

| 优先级 | 实验 | 依据 | 成本 | 期望 |
|---|---|---|---|---|
| 1 | h1/h2 晴空出力比 + 国能和煦历史日形通道（新开关 `features.derived.history_envelope`） | §7 | 1 轮 | 两站各 +3 到 +8，舰队 +0.5 到 +1 |
| 2 | d3/d4 TimesFM 提示与残差（已实现未跑） | §8 | 1 轮 | +0.5 到 +1.5，白马湖/国能最大 |
| 3 | 协议改动：d_model 128、lr 1e-4、cosine 8k、权重 EMA 0.999、5 种子集成、按最近 3 次验证均值选点 | §6 | 1 轮 | +0.5 到 +1.5，震荡消失 |
| 4 | 全局 LightGBM/CatBoost 基线：逐步直接多步、9 分位、特征 = 滞后/差分 GHI 与 T2m、晴空指数、天顶角、多尺度历史统计、站 id | §9 | 半天 | 定 M8 的相对位置 |
| 5 | a6_no_mixer、a3_attention ×3 种子、GDN 负特征值变体（β ∈ [0,2]，一行改动） | §5 | 1 轮 | 回答 RQ1 的最后一个空格；若 a6 ≈ a1，骨干可整体简化 |
| 6 | TabPFN-TS 单站零样本（带站元数据） | §9 | 半天 | 探索性，±1 |

不建议：微调 TimesFM-3/Chronos-2（期望 ≈0，分位风险，V100 fp32 成本）；频域增强与同站自监督（回退风险，无光伏证据）；
再加任何融合位置 / 双流 / 更大模型的变体（三轮已证明在噪声内）。

## 参考文献

[1] S. Yang, B. Wang, Y. Zhang, Y. Shen, Y. Kim, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length," NeurIPS, 2024.
[2] S. Yang, J. Kautz, A. Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule," ICLR, 2025.
[3] Cerruti, Rieder, Rowlands, et al., "Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing," arXiv:2607.07953, 2026.
[4] S. Arora, S. Eyuboglu, M. Zhang, et al., "Simple linear attention language models balance the recall-throughput tradeoff," arXiv:2402.18668, 2024.
[5] Fu, Li, Papaioannou, Kim, "Reverso: Efficient Time Series Foundation Models for Zero-shot Forecasting," arXiv:2602.17634, 2026.
[6] A. Auer, P. Podest, D. Klotz, et al., "TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning," NeurIPS, 2025.
[7] Hartl, Zólyomi, Stap, et al., "On Subquadratic Architectures: From Applications to Principles," arXiv:2606.12364, 2026.
[8] Moroshan, Siems, Zela, et al., "TempoPFN: Synthetic Pre-training of Linear RNNs for Zero-shot Time Series Forecasting," arXiv:2510.25502, 2025.
[9] J. Siems, T. Carstensen, A. Zela, et al., "DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products," NeurIPS, 2025.
[10] Wen, Gifford, Reddy, et al., "Revisiting the Generic Transformer: Deconstructing a Strong Baseline for Time Series Foundation Models," arXiv:2602.06909, 2026.
[11] Liu, Aksu, Liu, et al., "Moirai 2.0: When Less Is More for Time Series Forecasting," arXiv:2511.11698, 2025.
[12] Lai, Deng, Yu, et al., "Rethinking Recurrent Neural Networks for Time Series Forecasting: A Reinforced Recurrent Encoder with Prediction-Oriented PPO," arXiv:2601.03683, 2026.
[13] Xu, Chen, Liang, et al., "SST: Multi-Scale Hybrid Mamba-Transformer Experts for Time Series Forecasting," CIKM, 2025.
[14] Wahl, Bayer, Rausch, Pruckner, "Benchmarking Transformer and xLSTM for Time-Series Forecasting of Heat Consumption," arXiv:2605.09722, 2026.
[15] Huang, Xu, Darlow, "How Good Can Linear Models Be for Time-Series Forecasting?," arXiv:2606.27282, 2026.
[16] X. Qiu, J. Hu, L. Zhou, et al., "TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods," PVLDB, 2024.
[17] A. Zeng, M. Chen, L. Zhang, Q. Xu, "Are Transformers Effective for Time Series Forecasting?," AAAI, 2023.
[18] Y. Nie, N. H. Nguyen, P. Sinthong, J. Kalagnanam, "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers," ICLR, 2023.
[19] Karadag, Talaz, Dino, Kalkan, "ms-Mamba: Multi-scale Mamba for Time-Series Forecasting," arXiv:2504.07654, 2025.
[20] D. Markovics, M. J. Mayer, "Comparison of machine learning methods for photovoltaic power forecasting based on numerical weather prediction," Renewable and Sustainable Energy Reviews, 2022.
[21] S. B. Hoo, S. Müller, D. Salinas, F. Hutter, "From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting," arXiv:2501.02945, 2025.
[22] S. Arora, S. Eyuboglu, A. Timalsina, et al., "Zoology: Measuring and Improving Recall in Efficient Language Models," arXiv:2312.04927, 2023.
[23] A. Bick, E. Xing, A. Gu, "Understanding the Skill Gap in Recurrent Language Models: The Role of the Gather-and-Aggregate Mechanism," ICML, 2025.
[24] Brigato, et al., "There are no Champions in Supervised Long-Term Time Series Forecasting," TMLR, 2025.
[25] Z. Shao, et al., "Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis," IEEE TKDE, 2024.
[26] Bączek, et al., "TSPP: A Unified Benchmarking Tool for Time-series Forecasting," arXiv:2312.17100, 2023.
[27] Shi, et al., "Scaling Law for Time Series Forecasting," NeurIPS, 2024.
[28] Kiefer, et al., "A Comparative Study of Pruning Methods in Transformer-based Time Series Forecasting," arXiv:2412.12883, 2024.
[29] Xu, et al., "A Unified Hyperparameter Optimization Pipeline for Transformer-Based Time Series Forecasting Models," arXiv:2501.01394, 2025.
[30] B. N. Oreshkin, D. Carpov, N. Chapados, Y. Bengio, "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting," ICLR, 2020.
[31] S. Makridakis, E. Spiliotis, V. Assimakopoulos, "The M4 Competition: Results, findings, conclusion and way forward," International Journal of Forecasting, 2018.
[32] K. Bogner, R. Padrón Flasher, "Improving sub-seasonal hydrological forecasts utilizing the randomness in Deep Learning models," Stochastic Environmental Research and Risk Assessment, 2026.
[33] D. Morales-Brotons, T. Vogels, H. Hendrikx, "Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits," TMLR, 2024.
[34] Zhao, et al., "Dominant Shuffle: A Simple Yet Powerful Data Augmentation for Time-series Prediction," arXiv:2405.16456, 2024.
[35] M. Chen, Z. Xu, A. Zeng, Q. Xu, "FrAug: Frequency Domain Augmentation for Time Series Forecasting," arXiv:2302.09292, 2023.
[36] Bakhshaliyev, et al., "Sliding-Window Reordering with Overlap Averaging (Temporal Patch Shuffle)," CIKM, 2026 (arXiv:2604.09067).
[37] J. Dong, H. Wu, H. Zhang, et al., "SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling," NeurIPS, 2023.
[38] J. Dong, H. Wu, Y. Wang, et al., "TimeSiam: A Pre-Training Framework for Siamese Time-Series Modeling," ICML, 2024.
[39] Fu, et al., "Selective Learning for Deep Time Series Forecasting," NeurIPS, 2025.
[40] Nie, et al., "SolarBench: A global solar energy nowcasting benchmark," arXiv:2609.06187, 2026.
[41] Sun, et al., "Ensemble Complexity in Photovoltaic Forecasting," arXiv:2609.15049, 2026.
[42] Shamim, et al., "Leakage-Robust Evaluation and Data-Scale Sensitivity of Attention-Enhanced Multi-Task Learning," arXiv:2607.16493, 2026.
[43] L. Nochumsohn, O. Azencot, "Data Augmentation Policy Search for Long-Term Forecasting," TMLR, 2025.
[44] S. Lin, et al., "CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns," NeurIPS, 2024.
[45] Y. Sun, V. Venugopal, A. R. Brandt, "Short-term solar power forecast with deep learning: exploring optimal input and output configuration," Solar Energy, 2019.
[46] C. Voyant, et al., "On the importance of clearsky model in short-term solar radiation forecasting," Solar Energy, 2025.
[47] M. J. Mayer, D. Yang, "Benefits of physical and machine learning hybridization for photovoltaic power forecasting," Renewable and Sustainable Energy Reviews, 2022.
[48] Zhang, et al., "MIPV-NWP-PINNs V1.0: a physics-informed framework for medium-range PV power forecasting," Geoscientific Model Development, 2026.
[49] pvlib community, "PVAnalytics documentation: system orientation inference, tracker detection, daily insolation limits," 2025.
[50] B. Meyers, M. Tabone, E. C. Kara, "Statistical Clear Sky Fitting algorithm," IEEE PVSC, 2019.
[51] S. Miskovich, B. Meyers, et al., "Solar Data Tools: a Python library for automated analysis of unlabeled PV data," Journal of Open Source Software, 2025.
[52] K. Perry, B. Meyers, K. Anderson, M. Muller, "A reproducible validation of algorithms for estimating array tilt and azimuth from PV power time series," IEEE PVSC, 2023.
[53] M. J. Mayer, D. Yang, "Optimal place to apply post-processing in the deterministic photovoltaic power forecasting workflow," Applied Energy, 2024.
[54] Longarini, Rongoni, Silenzi, et al., "Time-series foundation models based on physics-informed synthetic histories for cold-start PV forecasting," arXiv:2606.07457, 2026.
[55] D. Danner, H. de Meer, "Two-stage photovoltaic forecasting: separating weather prediction from plant characteristics," arXiv:2603.04132, 2026.
[56] J. Depoortere, J. Driesen, J. Suykens, H. Kazmi, "SolNet: Open-source deep learning models for photovoltaic power forecasting across the globe," International Journal of Forecasting, 2025.
[57] Chen, Zhao, Chen, "Robustness of deep learning models for PV power forecasting under NWP forecast errors," arXiv:2607.12954, 2026.
[58] Fan, et al., "PARA-PV: physics-aware retrieval-augmented PV prediction based on frozen foundation model and distribution-shift correction," arXiv:2607.08079, 2026.
[59] Google Research, "TimesFM covariates (XReg) example notebook," google-research/timesfm repository, 2024.
[60] AutoGluon developers, "AutoGluon 1.2 release notes: covariate_regressor," 2024.
[61] Gao, et al., "Residual-guided multi-resolution refinement of foundation models: drought forecasting," ICML, 2026 (arXiv:2607.17507).
[62] Shang, et al., "LightGTS-Cov: covariate-enhanced time series forecasting," arXiv:2602.10412, 2026.
[63] M. Za'ter, B.-M. Hodge, "Empirical assessment of time-series foundation models for power-system forecasting," arXiv:2604.22077, 2026.
[64] Xu, Zhou, Sun, Yu, Jiang, "Horizon-specific expert fusion for PV power forecasting," arXiv:2609.15035, 2026.
[65] J. Browell, D. van der Meer, et al., "The Hybrid Renewable Energy Forecasting and Trading Competition 2024," International Journal of Forecasting, 2025 (arXiv:2507.01579).
[66] Pu, Fan, Tai, Liu, Yu, "A hybrid strategy for probabilistic forecasting and trading of aggregated wind–solar power: design and analysis in HEFTCom2024," arXiv:2505.10367, 2025.
[67] Bao, Sun, Jiang, et al., "Toward better integration of solar energy in transportation systems: ML benchmarks for day-ahead PV power forecasting," Transportation Research Part A, 2026.
[68] D. Markovics, G. Békési, M. J. Mayer, "A unified benchmark of deep learning and classical ML models for weather-based solar power forecasting: balancing complexity and skill," Energy and AI, 2026.
[69] Zhang, Zhang, Dai, Zhang, "Rethinking the use of deep learning methods for PV power forecasting," Nature Communications, 2026.
[70] Xu, et al., "State transport routing for short-horizon adaptation in multi-horizon PV forecasting," arXiv:2609.36926, 2026.
[71] Geng, Yang, Gu, Sun, "SolarBoost: distributed PV power forecasting amid time-varying grid capacity," arXiv:2510.21129, 2025.
[72] Peng, Tang, Xiao, "Attention-enhanced CNN-LSTM with spatial downscaling for day-ahead PV power forecasting," Sensors, 2026.
[73] Laglil, et al., "Foundation Models and Fine-Tuning: toward a new generation of time series models," arXiv:2607.23146, 2026.
[74] Obermeier, et al., "FETS: a benchmark of time-series foundation models for energy," Energy and AI, 2026 (arXiv:2604.22328).
[75] Park, et al., "Probabilistic forecasting for building energy systems using time-series foundation models," arXiv:2506.00630, 2025.
[76] Pendyala, et al., "Assessing covariate-informed grid load forecasting with a time-series foundation model," arXiv:2609.06656, 2026.
[77] Kaljevic, et al., "Benchmarking time-series foundation models for load forecasting under covariate uncertainty," arXiv:2610.07232, 2026.
[78] Spoek, et al., "Systematic evaluation of TabPFN-TS and Chronos-2 for zero-shot heat-load forecasting," arXiv:2608.20024, 2026.
[79] A. Lipiecki, R. Weron, "Foundation models for electricity price forecasting and battery arbitrage," arXiv:2609.00089, 2026.
[80] Berthelier, et al., "Investigating simple target–covariate relationships for Chronos-2 and TabPFN-TS," arXiv:2605.12200, 2026.
[81] Kaas, et al., "Probabilistic LV peak-load forecasting with time-series foundation models on application metrics," arXiv:2607.01966, 2026.
[82] A. F. Ansari, et al., "Chronos-2: From Univariate to Universal Forecasting," arXiv:2510.15821, 2025.
[83] Soni, et al., "Assessing the operational viability of foundation models for time series forecasting," arXiv:2605.24381, 2026.
[84] C. Bergmeir, R. J. Hyndman, B. Koo, "A note on the validity of cross-validation for evaluating autoregressive time series prediction," Computational Statistics & Data Analysis, 2018.

辅证（正文以 W/P4 编号引用、未进主列表）：W05 Mayer 等集合后处理（Solar Energy 2026，arXiv:2508.15508）；W07 站点相对 NWP 偏差特征
（J. Renewable and Sustainable Energy 2026，摘要）；W18 Dehghan 等（arXiv:2608.02088）；W24 PVOD 10 站联合训练（GitHub 项目页，
灰色文献）；P4-14 Energies 19(2):486（2026）。前一份综述的 FusionSF、ChronosX、CITRAS 等见 `docs/survey_covariate_pv_models.md`。
