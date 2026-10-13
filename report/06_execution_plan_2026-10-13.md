# 最终执行方案：Physiology-Guided EEG-to-Speech Reconstruction via Conditional Flow Matching

日期：2026-10-13 · 版本：execution_plan_v1

## Material Passport

- Origin Skill / Mode：academic-research-suite / experiment-agent，实验规划。
- 依据：用户最终确定的研究路线、重构前完成的现状诊断与代码审计、项目文档与当时保存的结果，以及下文链接的原始数据、模型配置和源代码。用户已授权清理旧文件，旧产物与分散路线文档不再保留于工作目录。
- 重构前代码快照：a7edbc55；新数据已下载并按官方快照校验，预训练模型与环境版本已写入本地 manifest。实际运行的源代码、训练统计与配置由 provenance 记录；正式测试冻结仍须在训练矩阵完成后执行。
- 现状诊断：**ANALYZED**；代码按本方案实现，运行方式见 [项目说明](../README.md)。正式 oracle、TRF 放行与 EEG 重建效果仍为 **UNVERIFIED**，代码检查结果单独记录。
- 优先级：本文件作为唯一研究协议；旧扩展路线已清理。

## 1. 决策与论文主线

**只执行这一条路线：**

```text
真实 perceived-speech EEG
  → 轻量共享 TCN：预测逐帧 WavLM 表征
  → Conditional Flow Matching：生成 100-bin Mel
  → 冻结 Vocos：输出 24 kHz waveform

生理约束一：TCN 中的响应延迟、头皮空间分组、envelope / onset 监督
生理约束二：生成 waveform → 冻结 Audio-to-EEG TRF → Neural Consistency Loss
```

核心问题：**可验证的听觉神经响应先验，能否使生成语音包含更多与目标语音一致的内容？**

| 项目 | 固定选择 |
|---|---|
| 数据与任务 | OpenNeuro ds004408；真实听觉感知；连续语音 |
| EEG encoder | 小型 TCN；共享参数；主实验为新被试 zero-shot |
| Audio teacher | 冻结 microsoft/wavlm-base；固定第 6 个 Transformer block 的逐帧输出 |
| 对齐 | Temporal Feature Regression + 局部 Contrastive Loss |
| 生成 | 单一 Conditional Flow Matching 网络；直接生成 Mel |
| Mel / vocoder | Vocos 原生 100-bin log-Mel；冻结 Vocos 24 kHz |
| 生理约束 | 正向响应延迟、头皮电极分组、envelope / acoustic onset、TRF 一致性 |
| 泛化补充 | 同一个主模型加小型 adapter；明确计入 60 秒有标签校准 |

研究投入集中在时序监督和两个生理约束。REVE、CBraMod、LaBraM 的历史结果用于说明选择依据；其他数据集、生成架构、可靠度网络和 Simulation 暂不进入本轮执行范围。

旧实验已经暴露两个瓶颈：EEG 条件弱；Chisco 的正确语音条件 oracle 也不能可靠重建完整句子。因此，**新路线必须先通过语音侧 oracle，再接 EEG 条件**。现有 imagined-speech 结果用于动机，不能与新 perceived-speech 结果直接组成性能提升百分比。

### 三个交付节点

| 节点 | 交付物 | 放行条件 |
|---|---|---|
| A：数据与语音链路 | 数据 manifest、冻结 split、原生 Mel / Vocos 验证、WavLM→CFM oracle、TRF 诊断 | 时间配对可信；copy-synthesis 与 oracle 通过；TRF 在未见音频上可预测 |
| B：基础时序重建 | 基础 TCN→WavLM→CFM；EEG 条件与 wrong / null 对照 | 预测条件有独立信息；生成器确实利用 EEG |
| C：生理约束与论文证据 | 三组固定系统、跨被试 adapter、统计表、音频与图 | 内容指标改善且条件对照支持；完整报告负结果 |

**A 不通过，先修复或终止对应环节；不自动启动新模型或新数据集搜索。**

## 2. 数据协议：同时留出音频与被试

ds004408 公开描述包括 19 名被试、128 通道 BioSemi EEG、512 Hz 采样、WAV 刺激及词 / 音素 TextGrid；原始 EEG 未滤波、未参考。公开 stimuli 目录列出 audio01 至 audio20。下载后仍须逐 run 核对事件、文件对应关系与实际偏移，不能仅凭文件编号配对。[数据源与说明](https://github.com/OpenNeuroDatasets/ds004408)、[刺激文件目录](https://github.com/OpenNeuroDatasets/ds004408/tree/master/stimuli)

### 2.1 全局音频划分

| 用途 | 刺激 ID |
|---|---|
| 训练 | audio01–audio14 |
| 验证 | audio15–audio17 |
| 最终测试 | audio18–audio20 |

划分以真实刺激内容为单位，跨所有被试共享。同一测试音频不能通过另一名被试进入训练；重采样、重复文件和相邻重叠窗口不能产生新的内容组。首先核对 checksum、TextGrid 文本与音频内容是否重复；发现跨组重复必须在训练前修正并锁定 split v2，记录原因。

所有拟合统计、Mel / WavLM 标准化和 TRF 拟合只使用训练音频。冻结的外部 teacher 可以在评估时提取测试特征；这些特征只供评分使用。

### 2.2 被试划分

固定 sub-001–sub-003 为开发验证被试，始终不进入正式训练或最终测试汇总。其余 16 人采用五个预先确定的测试组：

| Fold | 最终测试被试 | 训练被试 |
|---|---|---|
| F0 | sub-004–sub-006 | sub-007–sub-019 |
| F1 | sub-007–sub-009 | sub-004–sub-006、sub-010–sub-019 |
| F2 | sub-010–sub-012 | sub-004–sub-009、sub-013–sub-019 |
| F3 | sub-013–sub-015 | sub-004–sub-012、sub-016–sub-019 |
| F4 | sub-016–sub-019 | sub-004–sub-015 |

- 正式训练：该 fold 的训练被试 × audio01–14。
- checkpoint 选择：开发被试 × audio15–17。
- 最终测试：该 fold 的测试被试 × audio18–20。
- TRF 放行诊断：该 fold 的训练被试 × audio15–17，只有评估，不重新拟合。
- 最终测试前锁定模型配置、比较项、统计规则和 checkpoint。测试结果不用于修改这些选择。

每名正式测试被试只贡献一次该人的 held-out 结果。这是五组跨被试交叉评估，**不是 19 人 LOSO**。若文件缺失或 QC 导致人数减少，按原组报告有效人数，不按效果重新分组。

该协议检验同一故事内的未见音频片段与新被试。全体 EEG 时长不能当作独立音频时长；三段测试刺激也不能支持跨故事、跨语言或新 narrator 的结论。

### 2.3 预处理与样本合同

| 对象 | 固定处理 |
|---|---|
| EEG 主输入 | 正确恢复 µV 单位；仅 EEG 通道；坏导 mask；有效导联 CAR；0.5–40 Hz；重采样至 128 Hz |
| EEG 生理目标 | 主输入另作 15 Hz 低通，重采样至 64 Hz；用于 TRF / neural loss |
| 滤波 | 离线 zero-phase FIR；Hamming / firwin；主带通长度约 8 秒；明确记录实际 kernel 与边界 |
| 通道统计 | 每 fold 只在训练 EEG 上拟合均值 / 标准差；测试沿用；使用 LayerNorm，避免测试 batch 统计 |
| 非 EEG 通道 | MIC / EOG / EMG / STIM 不进入 encoder；事件与 EOG 仅作同步 / QC |
| 声音 | 真实 WAV；单声道；保留时间轴；Vocos 路径 24 kHz，WavLM 路径 16 kHz |
| 核心窗口 | 固定 4 秒；训练步长 2 秒；验证 / 测试步长 4 秒；同内容窗口不跨 split |
| EEG 上下文 | 核心区间 [a,a+4] 对应 EEG [a−1,a+5.5]，共 6.5 秒 / 832 点 |
| Teacher 上下文 | WavLM 读取 [a−1,a+5] 的真实音频，只保留核心 4 秒的特征帧 |
| 边界 | 每个刺激首尾至少排除 10 秒；上下文、滤波支持区和标注必须位于同一内容组 |

先在完整且已核对的连续刺激区间处理，再切窗。滤波跨原始 run 进行时，进入窗口的全部有效滤波支持区也必须属于同一个刺激 split。实际 kernel 比上述边界更长时扩大 guard，写入 manifest。

QC 采用统一规则：既有 bad-channel 标注直接 mask；4 秒窗口内平坦导联标准差 <0.1 µV 或 peak-to-peak >400 µV 记为异常，异常 EEG 导联超过 10% 则拒绝该窗口；剩余坏导标准化后置零并携带 mask。以上为本轮预注册阈值。若训练数据拒绝率超过 20%，先核对单位、饱和与记录格式，锁定协议修订后再开始正式训练。测试 QC 只使用已锁定规则，不按模型效果调整。

每个样本记录 dataset snapshot、subject / recording / stimulus ID、原始采样范围、音频起点 / 偏移、有效时长、通道类型与位置、坏导 / QC、内容 checksum、split 和 preprocessing version。数据版本、独立音频时长、各 split 有效被试数与窗口数必须进入报告。

实施补充：该固定 BrainVision 快照存在标称单位与原作者数值尺度不一致的问题。源码按 [单位审计](dataset_unit_audit.json) 固定恢复原作者数值尺度；µV 的绝对单位解释基于设备和来源证据推断，尚不是作者正式勘误。幅值阈值未放宽，所有缩放与统计进入版本记录，不能对测试记录单独猜测单位。

## 3. 固定模型与特征接口

### 3.1 WavLM：保留序列

使用 microsoft/wavlm-base，eval 模式，全部参数冻结，沿用其 processor 的输入标准化。选定第 6 个 Transformer block 的 768 维输出；不做层搜索或全句平均。其特征卷积总 stride 为 320 个 16 kHz 样本，即名义 20 ms；感受野为 400 个样本，即 25 ms。帧中心依据实际卷积长度与 stride 计算，核心 4 秒约 200 帧，禁止仅按数组长度强行对齐。[模型](https://huggingface.co/microsoft/wavlm-base)、[配置](https://huggingface.co/microsoft/wavlm-base/blob/main/config.json)

训练音频的逐维均值 / 标准差用于 teacher feature regression。时间插值以时间戳和 valid mask 为准。Teacher 在 6 秒 crop 内可能使用双向上下文，因此本轮是离线重建；上下文不得跨内容 split。

本地旧 WavLM base-plus-sv 是说话人模型，不能替代这里的 WavLM-Base。

### 3.2 TCN 与生理编码

- 空间输入映射宽度 128；TCN 为 6 个残差 block，每 block 两层 kernel=3 卷积，dilation=1、2、4、8、16、32；dropout=0.1；LayerNorm。主干时序感受野 253 个 EEG 点，约 1.98 秒。
- 生理空间分组：根据真实 montage，将电极分配至 Fz、Cz、T7、T8、POz 五个位置的最近组；每组投影为 32 维，再合并至 128 维。使用统一头坐标并保存映射。这是**头皮位置分组**，不能当作已定位的听觉皮层源。
- 生理延迟池化：对于音频时刻 t，从 TCN 序列采样 t+δ；δ 固定为 {0、50、100、200、300、500} ms，采用共享 softmax 权重。权重只从训练数据学习；正号表示神经响应晚于刺激。
- 线性 head 输出逐帧 768 维 WavLM 预测；另一 head 输出标准化 envelope 与 onset。后两者只用于辅助监督，不另建生成分支。
- EEG 右侧 1.5 秒上下文包含响应延迟与 TCN 右侧感受野；不能声称实时或零延迟重建。延迟权重受离线滤波与卷积影响，也不能单独解释成真实神经潜伏期测量。

### 3.3 CFM 与 Vocos

CFM 使用 6 层 Transformer velocity network，hidden=256、4 个 attention heads，Mel self-attention + 对预测 WavLM 序列的 cross-attention；flow time 用 AdaLN 注入；输出 100 维逐帧 velocity。TCN 目标 <1M 可训练参数，CFM 目标 <10M；以实际参数统计为准，不增加 speaker / 文本条件。

实现细节：AdaLN 时间调制采用 128 维中间层，使六层 CFM 保持在 10M 参数预算内；这是固定模型实现，不进行结构搜索。

Vocos 固定使用作者发布的 charactr/vocos-mel-24khz 与 vocos 包。用户提供的 hf-audio/vocos-mel-24khz 是该模型的 Transformers 转换版；本轮采用作者接口以统一前端与梯度路径。首次下载记录完整模型 revision 和依赖版本，之后不从浮动 main 读取。[作者用法](https://github.com/gemelo-ai/vocos)、[转换版说明](https://huggingface.co/hf-audio/vocos-mel-24khz)

**目标 Mel 必须调用该模型的 feature_extractor：**

| 参数 | 值 |
|---|---|
| sample rate / n_mels | 24,000 Hz / 100 |
| n_fft / win_length / hop_length | 1024 / 1024 / 256 |
| padding | center |
| Mel power | 1，幅度谱 |
| log | natural log；最小值 clip=1e−7 |

其余 Mel filter、窗口、padding 和软件版本沿用原生前端。4 秒音频通常对应 376 帧；运行时以原生输出形状为准。CFM 使用训练音频拟合的逐 Mel-bin 标准化目标，送 Vocos 前恢复原始 log-Mel。[模型配置](https://huggingface.co/charactr/vocos-mel-24khz/blob/main/config.yaml)、[特征源码](https://github.com/gemelo-ai/vocos/blob/main/vocos/feature_extractors.py)、[safe_log 源码](https://github.com/gemelo-ai/vocos/blob/main/vocos/modules.py)

当前 eegspeech/audio.py 的 16 kHz、80-bin SpeechT5 log10-Mel 与新接口不同；不能简单改 Mel-bin 数复用。新前端与旧音频代码分别记录版本。

## 4. 损失定义与训练边界

### 4.1 时序对齐与生理辅助监督

```text
L_encoder = L_reg + 0.10 L_ctr + 0.10 L_env + 0.05 L_onset

L_reg：预测与冻结 WavLM teacher 的逐帧 MSE
L_ctr：1 秒局部片段平均特征的 InfoNCE，temperature=0.07
L_env / L_onset：标准化生理声学目标的逐帧 MSE
```

所有损失按 valid frame / feature 数取平均。B0 的 env / onset 权重为零，B1 / B2 使用上式。多名被试听同一刺激位置时，音频是同一正样本组；同组以及重叠邻窗不能作为彼此的负样本。batch 按被试和不同刺激位置均衡采样，避免模型只学习 narrator 或录制 ID。

声学生理特征 φ(y) 固定为 25 ms RMS envelope，以及 envelope 相邻帧差分的正部 acoustic onset，时间网格为 64 Hz；两维统计只由独立训练音频拟合。辅助 head 的监督再插值到 WavLM 帧中心。TextGrid 可用于评分与同步审计，不作为生成条件。

### 4.2 Conditional Flow Matching

采用噪声到数据的线性条件路径；不引入其他 diffusion schedule：[Flow Matching 原始论文](https://arxiv.org/abs/2210.02747)

```text
z ~ N(0,I), τ ~ Uniform(0,1)
Mτ = (1−τ)z + τM
target velocity = M−z
L_FM = mean_valid ||vθ(Mτ, τ, predicted_WavLM) − (M−z)||²

推理：M0 = z；积分 dMτ/dτ = vθ(Mτ,τ,predicted_WavLM)，直到 τ=1
```

训练中的真实 Mel 只定义监督路径。推理函数仅接收 EEG、固定窗口长度和随机种子，不能接收真实 Mel、真实 WavLM、目标文本或目标词数。10% 训练条件置空以支持 null control；不使用 classifier-free guidance 放大，固定 guidance=1。

正式采样为 32 步 Heun；每个 EEG 窗口固定生成种子 0、1、2 的三个样本，先在窗口内平均指标，不按 oracle 分数挑选最好样本。

### 4.3 先验证 Audio-to-EEG TRF

每个 fold 的**训练被试**分别拟合线性 ridge TRF：

```text
F_s(φ(y))(t) = b_s + Σδ W_s,δ φ(y)(t−δ)
δ = 0 ... 500 ms，以 64 Hz 网格离散
```

输入为 envelope / onset，输出为该人的低频 EEG。ridge λ 仅在 audio01–14 内按刺激分组三折选择，候选 {0.01、0.1、1、10、100}。随后在全部训练刺激重拟合并冻结；不能使用测试被试或测试音频建立 forward model。[TRF 方法依据](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2016.00604/full)

通道集合 P_s 由训练侧 cross-fitting 确定：预测 r>0，且超过 500 次、位移至少 2 秒的循环移位对照，移位拼接边界 mask 后评分，通道检验 BH-FDR q<0.05；最多保留 32 个通道。残差方差也来自 cross-fitting，在训练 EEG 标准化空间中冻结为对角权重，下限固定为 0.1。主 EEG 与生理 EEG 各自的标准化统计均只在本 fold 训练数据拟合。

**放行目标：**至少 80% 的训练被试具有不少于 8 个合格通道；在 audio15–17 上，合格通道的被试中位预测 r≥0.05，正确时间对齐相对移位对照的被试配对平均优势，其 95% bootstrap CI 下界>0。这些数值是预注册门槛，不是已获得结果。

不合格被试仍参与 EEG 对齐和 FM 训练，其神经项 mask 为零，报告覆盖率。若整体验证不通过，停止 B2 的 Neural Consistency 主张，保留 B0 / B1 的真实结果；不能用训练拟合好替代前向模型放行。

### 4.4 Neural Consistency：必须约束真实生成轨迹

```text
ŷ = Vocos(denormalize(CFM_rollout(z, predicted_WavLM)))
L_NC = mean_valid [P_s F_s(φ(ŷ)) − P_s x_lowfreq]² / frozen_residual_variance
```

这里比较与核心窗口对齐的真实 EEG；仅评分核心音频的 0.6–3.9 秒，避开 TRF、RMS / onset 和生成端点的边界。方差与 EEG 单位采用同一标准化空间。每 4 个更新，从合格训练被试中取 microbatch=1，保留非空正确 EEG 条件，用独立纯噪声、8 步 Euler 做一次可微生成轨迹；使用 activation checkpointing 控制显存。普通 FM 更新之外，加 λ_NC·L_NC；λ_NC 在第 5,000–10,000 步从 0 增至 0.10，之后固定。该权重只在神经项触发的更新上应用。

不能把含真实 Mel 的 Mτ 或从 Mτ 一步推回的估计当成完整 EEG-only 生成结果计算主要神经项，否则损失可能利用真实目标绕过 EEG 条件。

**实现必须保留 Mel→waveform→φ→TRF 的输入梯度。**Vocos 原生 forward / decode 带 torch.inference_mode()；训练时用 grad-enabled wrapper 直接调用冻结的 backbone 与 head，全部参数 requires_grad=False、eval，但生成 Mel 不 detach，也不包 no_grad。RMS、onset 和 TRF 均采用可微运算。[Vocos 解码源码](https://github.com/gemelo-ai/vocos/blob/main/vocos/pretrained.py)

神经项只更新 CFM；该分支的 predicted_WavLM detach。TCN 由对齐、辅助监督以及普通 FM 路径更新，防止神经项通过改写条件表示获得捷径。计算图必须验证 L_NC 对生成 Mel 和 CFM 参数具有非零、有限梯度，冻结参数没有梯度。

8 步训练轨迹与 32 步正式采样存在差异。开发验证记录两者表现；若短轨迹严重失真或梯度异常，先修正数值和实现问题，不能把 neural loss 下降当作正式采样改善。

**边界：**envelope / onset TRF 主要约束节律和低频听觉响应，不能验证词汇或完整语义。生成声音更符合 TRF 仍可能内容错误，最终结论必须依赖 ASR 与 EEG 条件对照。

## 5. 固定训练顺序与预算

先完成 A 的接口与数据检查，再进行以下训练。训练种子固定为 11、22、33；生成种子固定为 0、1、2，分别记录。

| 顺序 | 训练内容 | 预算与固定设置 |
|---|---|---|
| 1 | 原生 Mel→Vocos copy-synthesis | 无训练；只用训练 / 验证音频核对 |
| 2 | 真 WavLM→CFM→Vocos oracle | 每训练种子最多 10k updates；有效 batch=32；AdamW lr=2e−4 |
| 3 | 训练侧 TRF 与跨音频验证 | 上述固定 λ / 三折；输出通道 mask 和残差统计 |
| 4 | TCN 时序对齐预训练 | 每 fold / seed / encoder 设置 20k updates；batch=64；lr=3e−4 |
| 5 | 预测 WavLM→CFM | 继承同种子的 oracle CFM；30k updates；有效 batch=32；CFM lr=1e−4 |
| 6 | 同一路径联合优化 | 第 5k update 后解冻 TCN，lr=1e−5；总损失 L_FM+0.1L_encoder；B2 另加触发式神经项 |

以上 updates 5 / 6 是同一段 30k 训练，**不能累加成两段**。所有 AdamW weight_decay=0.01，gradient clip=1；前 1k steps warmup，之后 cosine decay；混合精度用于普通网络，TRF / RMS / neural loss 使用 FP32。CFM 显存不足时降低 microbatch 并累积梯度，保持有效 batch 与更新次数。

Oracle 只依赖固定训练音频，因此每个 seed 的 checkpoint 可供所有 fold 和系统共同初始化，避免重复语音预训练。每组比较使用同样的 oracle checkpoint、数据、种子、FM 更新次数和 checkpoint 规则。

### 放行与 checkpoint

- **Copy-synthesis：**相同 ASR 设置下，验证音频的 WER 相对原音频恶化≤5 个百分点；输出长度误差≤一个 Mel hop，无系统时间漂移。
- **Oracle：**验证音频 WER≤40%，且相对 copy-synthesis 恶化≤15 个百分点。未通过时先检查特征、时间、优化与采样；不开始完整 EEG 生成实验。该门槛意在排除旧 Chisco 那种语音侧自身失效。
- **Encoder：**在开发被试 / 验证音频上，正确局部 WavLM 对齐优于固定 wrong-time 对照；以被试为单位的配对优势 CI 下界>0。这里仅有 3 名开发被试，CI 只能用于开发放行，不能当论文的正式泛化证据；同时列出每人结果。
- **基础生成：**开发集的正确 EEG 比 wrong-time / null 条件具有更低 WER，且多数开发被试同向。若语音很自然但控制之间没有差异，先停止正式扩展，审计 conditioning 是否被使用。
- 每 2k updates 验证一次：audio15–17 各预先抽取 10 个固定非重叠窗口，EEG 模型再覆盖 3 名开发被试；只用生成种子 0 选 checkpoint。按被试宏平均 WER 选择；相同 WER 时用 Mel MAE 决胜。最终开发总结与测试用三个生成种子。

开发阶段先执行 F0 / seed=11，全部选择只看固定开发数据。达到放行条件后冻结配置，再启动正式矩阵。所有门槛均是拟执行的决策规则，没有保证当前数据必然通过。

## 6. 实验只保留三组系统

| 系统 | EEG encoder / alignment | 生成约束 | 对应问题 |
|---|---|---|---|
| B0：时序基线 | 通用空间投影 + 同一 TCN；regression + contrastive；无显式生理分组 / 延迟池化 / 辅助损失 | 同一 CFM | 连续语音条件本身能否产生有效重建？ |
| B1：生理编码 | 头皮分组 + 正延迟池化 + envelope / onset 辅助监督 | 同一 CFM | 编码侧生理先验是否改善可用内容？ |
| B2：完整主模型 | 与 B1 相同 | 同一 CFM + Neural Consistency | 神经一致性是否进一步改善生成内容？ |

B0 / B1 空间映射的宽度按实际参数统计匹配，总可训练参数差异≤5%；TCN 主干与 CFM 完全相同。B1 / B2 从同一个已训练生理 encoder checkpoint 启动。该对照检验编码侧先验的**组合效应**，不把 B1 的提升归因于其中某一个单独组件。

正式矩阵为 **3 系统 × 5 被试组 × 3 训练种子 = 45 次 CFM 训练**；oracle 只训练 3 次，B1 / B2 共用 encoder 预训练。先完成一个 fold / seed 的三组开发运行；正式矩阵复用符合冻结配置的运行。45 次属于同一问题的重复验证，不再增加 architecture / dataset grid。

B2 的神经项有额外计算开销；记录实际训练时间、GPU 峰值显存、更新数、采样步数与参数量，不声称三组 wall-clock 成本相同。现阶段没有吞吐量测量，不能给出可靠 GPU 小时总额；用开发运行测出的小时数推算正式预算。

### 必要条件对照：复用训练好的模型

每个系统以相同噪声种子评估：

1. 正确 EEG。
2. Wrong-time EEG：同一被试 / 刺激内配对另一有效窗口，核心时间相隔≥10 秒，确定性生成对应表；不能仅换为听同一内容的另一被试。
3. Null condition：使用训练过的空条件。
4. Oracle：真实时序 WavLM 条件的独立语音侧 checkpoint，仅作上界诊断。

这些是 inference controls，不另训练四套模型。Wrong / null 只用于离线评分；测试波形、文本、目标 embedding 不进入实际 EEG 推理接口。

## 7. 跨被试适配：只做一个清楚的补充实验

对每个测试被试，仅在 B2 加一个残差 adapter：128→16→128，输出投影零初始化，约 4.2k 参数；置于 TCN 隐藏表示和 WavLM head 之间。主干、head、CFM、WavLM、Vocos 均冻结。

- **Zero-shot：**使用 identity adapter，作为正式主结果。
- **60 秒有标签校准：**使用该被试 audio01 的一个预先指定连续 60 秒 EEG 区间及对应已知刺激；60 秒包括滤波与模型上下文，不能借用区间外 EEG。仅使用内部有效核心窗口。
- adapter 固定训练 100 updates，batch=16，lr=1e−3；损失为 feature regression + 0.1 envelope MSE + 0.05 onset MSE，并加 0.001 的 adapter 输出平方惩罚；不在校准块选择最好 checkpoint。
- 继续评估该人的 audio18–20，不能校准测试语句；没有可用 audio01 的被试只报告 zero-shot，说明缺失。
- 报告 zero-shot→60s 的配对改善、额外参数和真实校准时长；不把使用刺激标签的校准叫无监督，也不把校准结果并入 zero-shot。

这项实验不重训 TRF、不建立新被试生成器，只有主模型的一种轻量适配。

## 8. 评估与论文结论规则

### 8.1 主要指标

**唯一主要终点：B2 相对 B1 的自由 ASR WER 改善，方向为 WER(B1)−WER(B2)>0。**每个被试先将窗口和采样重复汇总，再将训练种子平均，最后被试宏平均。

- 使用固定的冻结 Whisper small.en；评分输入重采样至 16 kHz；greedy decoding，temperature 固定为单值 0，关闭温度回退和 previous-text conditioning，不提供目标文本 prompt。模型 revision、软件版本、英文规范化规则写入 model manifest。[ASR 实现与模型说明](https://github.com/openai/whisper)
- reference 来自 TextGrid；固定 4 秒评分窗口，按单词中心时间归入窗口。所有模型与原音频使用相同边界和规范化规则；保留边界词误差，报告原音频与 copy-synthesis 的 WER。
- WER 可超过 100%，直接报告误差；不能转换成截断的“准确率”，不能使用候选强制选择替代自由转写。
- B1−B0 WER 为编码侧关键次要对比；正确 EEG−wrong / null 为条件有效性证据。

### 8.2 次要诊断

| 指标 | 用途与限制 |
|---|---|
| 局部 teacher 特征误差 / 正负配对差 | 说明时序 encoder 是否有信息；不单独证明语音可懂 |
| 20 候选 WavLM retrieval Top-1 | 固定测试音频候选池与随机种子，排除重叠 / 重复目标；chance=5%；少于 20 个有效候选时报告实际规模。使用同一 teacher，作为辅助诊断 |
| Envelope r / onset correspondence | 检查节律追踪；不能替代词汇恢复 |
| Native Mel MAE / spectral convergence | 说明声学接近程度；自然度不作为主结论 |
| TRF neural consistency | 训练被试的未见验证音频上评估；不借用测试被试 EEG 拟合评分器 |
| 参数、显存、训练 / 推理耗时 | 量化生理项与适配的成本 |

主测试被试没有其训练 TRF，主表不报告使用测试被试数据拟合评分器得到的“神经一致性改善”。该指标来自单独标明的训练被试 / 留出音频诊断。

### 8.3 统计与展示

- 主要比较采用被试配对均值差、95% CI 与双侧被试配对置换检验；16 名被试可枚举符号翻转。三个 seed、三个生成样本、多个窗口都不是独立被试。
- 同时报告按“被试与测试音频”两维聚类的 bootstrap 敏感性结果。只有 3 段测试刺激，内容维度不确定性必须注明，不能因窗口数多声称广泛语料泛化。
- 主要比较只有 B2−B1；B1−B0 及两个条件对照的次要检验采用 Holm 校正。逐被试、逐 seed 结果公开，报告有效 n 与缺失。
- 音频展示固定预先选定的样本 ID，播放原始 / oracle / B0 / B1 / B2 / wrong；正文不只展示最好的成功例子。

**论文目标效应：**B2 相对 B1 的 WER 改善≥5 个百分点，主要配对 CI 下界>0；B2 同时优于 wrong / null，并且大多数被试同向。5pp 是本项目的实用目标，不是通用领域标准。

若只改善声学指标或 TRF 分数，词错误率和条件对照不改善，结论限于节律 / 声学生成，不能声称恢复了更多语义。如果 EEG 重建整体 WER 仍很高，即使相对改善显著，也必须明确尚未实现可靠的完整语句重建。结论始终限定于 perceived speech 与当前语料。

## 9. 项目落地：直接在当前根目录实现

项目根目录固定为 /Users/samxie/Research/eeg-voice/speech_decoding/。新路线放入现有代码、配置与报告目录：

```text
eegspeech/
  perceived_data.py       # ds004408 配对、QC、时间戳、严格 split
  perceived_audio.py      # WavLM / 原生 Vocos 前端及可微声码器 wrapper
  temporal.py             # 固定 TCN、lag pooling、adapter、对齐损失
  flow.py                 # 唯一 CFM、训练路径、ODE 采样
  neural_forward.py       # TRF、cross-fitting、可微声学特征 / 一致性
scripts/
  perceived_flow.py       # 单一入口：准备、训练、适配、评估、报告
configs/
  perceived_flow.yaml     # 本文件固定的全部参数
  perceived_flow_splits.json
  perceived_flow_models.json  # 完整 revision / 特征配置 / 软件版本
data/raw/ds004408/         # 原始文件，不可变，本地忽略
artifacts/perceived_flow/ # 配对缓存、teacher 特征、TRF / checkpoints，本地忽略
models/                   # 冻结预训练权重，本地忽略
outputs/perceived_flow/   # 音频、逐样本指标、评估汇总，本地忽略
report/                   # 轻量协议、最终结果表与论文图说明，纳入版本管理
```

复用通用日志、下载与设备工具；新任务的目标 Mel、split、训练统计和时间接口独立实现。旧全句 anchor / imagined renderer 保留历史可复现性，不作为新任务的条件接口。

现有根 .gitignore 已覆盖 data/、artifacts/、models/、outputs/、logs/ 和本地环境；新方案沿用这些位置。configs/ 与 report/ 中的小型协议、模型清单和结果摘要可跟踪。生成音频、数组和权重不放到 report/；不能为了提交汇总而解除整个 outputs/ 的忽略。无需重新创建 bundle 或 eeg-recon-0809 子目录。

### 已实现命令合同

下面的接口已实现；首次运行先按项目说明下载 / 初始化数据与模型。最终测试须先完成矩阵并执行 freeze，完整顺序见项目说明：

```bash
python scripts/perceived_flow.py prepare --config configs/perceived_flow.yaml
python scripts/perceived_flow.py oracle --config configs/perceived_flow.yaml
python scripts/perceived_flow.py fit-trf --config configs/perceived_flow.yaml --fold F0
python scripts/perceived_flow.py train-encoder --config configs/perceived_flow.yaml --fold F0 --seed 11
python scripts/perceived_flow.py train-flow --config configs/perceived_flow.yaml --fold F0 --seed 11 --system B0
python scripts/perceived_flow.py train-flow --config configs/perceived_flow.yaml --fold F0 --seed 11 --system B1
python scripts/perceived_flow.py train-flow --config configs/perceived_flow.yaml --fold F0 --seed 11 --system B2
python scripts/perceived_flow.py evaluate --config configs/perceived_flow.yaml --split validation
# 开发放行、配置冻结、45 次矩阵完成后：
python scripts/perceived_flow.py freeze --config configs/perceived_flow.yaml
python scripts/perceived_flow.py evaluate --config configs/perceived_flow.yaml --split test
python scripts/perceived_flow.py adapt --config configs/perceived_flow.yaml --seconds 60
python scripts/perceived_flow.py report --config configs/perceived_flow.yaml
```

入口负责遍历配置中声明的 seed / fold / system；具体单次命令须显式记录实验 ID。train-encoder 生成 B0 与生理版本两个 checkpoint，B1 / B2 共用后者；oracle 遍历三个 seed；评估与适配只读取已冻结的完整模型矩阵。

### 实现时必须通过的检查

1. 同一内容组在 train / validation / test 间零交叉；本 fold 的 held-out EEG 不参与训练统计。
2. 模型前端 Mel 与原生 Vocos feature_extractor 数值一致；反标准化与 waveform 长度正确。
3. EEG、WavLM、Mel 和 TRF 帧中心使用同一时间原点；真实偏移进入 manifest；不能通过测试 performance 反推偏移。
4. 相同刺激位置的多被试样本正确处理为同组正样本；相邻重叠窗不会进入负样本集合。
5. 神经项的梯度能穿过冻结 Vocos 到达 CFM；冻结 teacher / TRF / Vocos 参数保持不变。
6. 从纯噪声开始的生成函数不需要真实目标；wrong / null 共享生成种子；长度不是目标句子 duration。
7. 用少量训练窗口检查损失下降与 oracle 可懂度；任何检查结果都明确标注开发集 / 训练集，不能提前查看最终测试效果。

## 10. 交付顺序与论文结构

**下一步先实现节点 A。**交付可审计的数据配对表、全局 split、时间 / 特征验证、语音 oracle 音频及误差、TRF 的 held-out-audio 预测结果。只有这些证据通过，再完成 B0，然后 B1 / B2 的固定正式比较与 adapter。

论文正文围绕一个问题组织：

1. 现有 EEG→全句压缩表示与生成的双重瓶颈，作为研究动机。
2. 单一时序 CFM 框架、编码侧生理先验、可微 TRF 一致性。
3. 同时留出刺激与被试的三组比较，主要报告内容恢复与条件有效性。
4. 一个 60 秒 adapter 实验，以及内容、样本量、离线延迟和感知 / 想象任务差异的限制。

EEG-conditioned Flow Matching 本身已有相关工作，不能仅凭采用 CFM 声称首次。论文贡献应落在**可验证的生理约束是否改善未见内容恢复，以及严谨的条件依赖证据**；全部新效果目前仍需实验验证。[EEG-conditioned flow matching 相关工作](https://arxiv.org/abs/2606.24087)
