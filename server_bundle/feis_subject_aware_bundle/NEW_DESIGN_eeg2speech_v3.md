# EEG → Speech 重建：全新方案设计（v3）

> 目标定位（按你的选择）：**输出尽可能自然、像真实语音**；约束**不设限**（可用跨被试、冻结 EnCodec/HuBERT/HiFi-GAN、EEG 基础模型）。
> 本文是设计文档，不含实现代码。落地分阶段计划见第 7 节。

---

## 0. 先看清现状：为什么现在"重建很差"

这一节先把诊断讲透，因为新方案的每一个决策都是针对这些病根来的。

### 0.1 不是质量差，是**坍缩到均值（mode collapse）**

实测 `artifacts/outputs_waveform_protocol/` 三个协议：

| 协议 | 测试集 | Waveform L1 | STFT dist | 分类 acc | NTA（最近模板）|
|------|--------|-------------|-----------|----------|----------------|
| G（pooled 全被试）| 320 | 0.053 | 1.600 | 0.0625（=chance）| **0.053（< chance 0.0625）** |
| S（单被试 01）| 16 | 0.069 | 1.805 | 0.1875 | 0.125 |
| U（holdout 21）| 160 | 0.054 | 1.415 | 0.075 | 0.0625（=chance）|

在 G 协议 320 条测试里，模型把 `m` 选了 112 次、`fleece` 选了 93 次——**无论输入什么 EEG，输出都是那几条"平均波形"**。
NTA 甚至低于随机（0.053 < 0.0625）。Waveform L1 看着小（0.053），只是因为"预测全体平均"本身就能最小化 L1。

**结论：当前 EEG→waveform 的映射几乎没学到，decoder 在输出一个跟 EEG 无关的通用平均音。** 听起来"糊、闷、像一团噪声/哼鸣"正是因为它是 16 条 prompt 的平均。

### 0.2 这是**目标函数错了**，不是调参能救的

FEIS 的音频目标是 **canonical（非 trial-synchronous）**：同一个 label 的所有 trial 共享**同一条** 1.0s 波形，全数据集只有 **16 条不同的目标波形**。

这意味着：
- "重建波形"本质是 **16 选 1 的识别 + 播放干净片段**，不是回归。
- 直接对 24000 个原始采样点做 L1+STFT 回归，存在一个平凡最优解：**输出均值**。当 EEG 里类别信号很弱时，梯度必然滑向这个解 → 坍缩。
- 单层 VQ bottleneck + ConvTranspose 直接上采样，即使不坍缩也无法生成可懂、自然的语音。

> 一句话：**只要还在"EEG 直接回归原始波形"这条路上，再怎么换 decoder、调 loss 权重都会坍缩。** 必须换框架。

### 0.3 FEIS 数据的硬约束（决定能做到什么程度）

- 21 被试（05 异常，clean=20），每被试 160 trial，16 label，每 label 10 trial。
- EEG：Emotiv EPOC 14 通道（F3 FC5 AF3 F7 T7 P7 O1 O2 P8 T8 F8 AF4 FC6 F4），256 Hz，thinking 窗 5s → `[14,1280]`。
- 14 通道 imagined-speech 的**可解码音素信息极弱**，这是领域共识（FEIS 自身基线在 imagined 阶段接近 chance）。
- **没有 per-trial 声学变化可还原**：canonical 目标里不存在 trial 级别的韵律/音色，"还原得像"只能落在"选对 16 选 1"+"合成得自然"。

**因此本方案对"自然度"和"正确性"分别处理**：自然度可以做到很高（用冻结高质量声码器合成），正确性受神经科学上限约束（这是真正难的部分，靠跨被试 + 跨阶段 + 对比学习去逼近上限）。

---

## 1. 核心思想：**解耦"识别"与"合成"**

把一个被坍缩诅咒的回归问题，拆成两个各自可解的问题：

```
         ┌─────────────────────────┐      ┌──────────────────────────┐
 EEG ──► │  A. 神经识别/内容解码     │ ──► │  B. 冻结神经声码器合成     │ ──► 自然语音波形
[14,1280]│  EEG → 语音表征/codec token│ token│  EnCodec/HiFi-GAN decoder  │     [1,16000]
         └─────────────────────────┘      └──────────────────────────┘
              ↑ 难点（神经科学上限）            ↑ 已解决（音频侧 SOTA，冻结）
```

**关键洞察：永远不要让网络回归原始采样点。** 让网络去预测一个"语音表征"（EnCodec 离散 token 或 HuBERT 单元），再交给一个**冻结的、已经会发音的**声码器去合成。这样：

1. 输出**天生自然**——因为它是干净 codec 的解码结果，不是糊在一起的均值。（直接满足你"自然度优先"的目标）
2. 识别和合成**解耦**，可以分别评估、分别改进。
3. 彻底避开"回归均值"坍缩——目标空间是结构化、可判别的 token，而非 24000 维连续回归。

这与你 Phase-2 报告里已经推荐的 **路线 C（EEG → codec latent → frozen decoder）** 一致；本方案补上它缺的那块：**正确的训练目标（对比 + 分类 + 序列），以及跨被试/跨阶段的数据杠杆。**

---

## 2. 模块 B（先定下来）：冻结声码器 = 自然度的保证

先把"合成"这端钉死，因为它决定了输出的自然度天花板，而且**完全不依赖 EEG**。

**两种目标表征二选一（建议都试，A1 为主）：**

- **B1（推荐）EnCodec 24 kHz 离散 token。** 把 16 条 canonical wav 用冻结 EnCodec（你 bundle 里已有 `models/encodec_24khz/`）编码成 RVQ token 序列（约 75 帧/秒 × 8 codebook）。模块 A 预测这些 token，EnCodec decoder 还原波形。自然度 = EnCodec 重建质量（非常高）。
- **B2 HuBERT 单元 + HiFi-GAN unit-vocoder。** 把 wav 转成 HuBERT 离散单元，用 unit-HiFi-GAN 合成。内容性更强、对音素更友好，但要额外引入 unit vocoder。

**为什么这一步保证自然：** 因为目标是真实语音经过 SOTA codec 的编码，模块 A 哪怕只预测对"大致内容 + 韵律轮廓"，解码出来也是清晰人声，而不是均值噪声。

> 注意：因为 FEIS 只有 16 条目标，模块 A 在极限情况下退化为"16 选 1 后合成对应 token 序列"。这没关系——**对你的研究而言，能稳定、自然地合成出"正确那一条"已经是质的飞跃**（从当前的均值哼鸣 → 干净人声）。

---

## 3. 模块 A：EEG 内容解码器（真正的难点）

### 3.1 EEG 前端（encoder）

抛弃"纯 1D-CNN + 单层 VQ"。换成**空间-时间分解 + 可选基础模型**：

```
EEG [B,14,1280]
  → 空间滤波: Conv1d(14→Csp, kernel=1)              # 学跨电极空间模式（类 EEGNet depthwise）
  → 时间前端: 4× 时序卷积块 (TCN/Conformer-lite)     # 多尺度时间感受野，含 BN+GELU+dropout
  → 时间池化到 T_lat（与目标 token 帧率对齐，如 75 或 50）
  → 投影到 d_model (256)
  → [B, T_lat, 256]  EEG 表征序列
```

可选增强（"anything goes"下推荐试）：用 **EEG 基础模型**（LaBraM / NeuroLM / BrainOmni）作为前端，先在大规模 EEG 上预训练，再在 FEIS 上微调。14 通道、低数据量场景下，基础模型迁移通常显著优于从零训练。

### 3.2 被试条件化（解锁跨被试数据，×20 数据量）

当前是 subject-specific（每人 160 trial，杯水车薪）。改为 **pooled + subject-conditioning**：

```
subject_id → subject_embedding (e_s, dim=64)
            → FiLM 调制每个时间块: h = γ(e_s)·h + β(e_s)
```

- 训练数据从 160 → **20×160 = 3200 trial**，这是脱离过拟合/坍缩最直接的杠杆。
- 推理未见被试（协议 U）时，用零向量或最近邻被试 embedding。
- 这正是 Défossez 2023 跨被试 + subject layer 的做法。

### 3.3 输出头（三个并行 head，对应三类信息）

```
EEG 表征 [B,T_lat,256]
  ├─► Content head:  → codec/unit token logits  [B,T_lat,n_codebook,vocab]   # 给模块 B 合成
  ├─► Class head:    → 16-way label logits        [B,16]                      # 辅助识别
  └─► Embed head:    → pooled embedding z_eeg [B,512]                          # 对比对齐
```

---

## 4. 训练目标：用**对比 + 分类 + 序列**取代裸回归（核心修复）

这是整套方案最关键的改动。三个 loss 协同，从根上杜绝坍缩：

### 4.1 对比检索 loss（InfoNCE）—— 反坍缩主力

把 `z_eeg` 与目标语音 embedding `z_audio`（= 冻结 HuBERT/EnCodec 对 canonical wav 的池化表征）做 CLIP 式对齐：

```
L_contrastive = InfoNCE(z_eeg, z_audio; batch 内同 label 为正，其余为负)
```

- 直接优化"检索到正确那条片段"，这就是你最终评测的 NTA/top-k，**目标与评测一致**。
- 对比目标天然抗坍缩：把所有样本映射到同一点会被负样本项惩罚。
- 参考 Défossez 2023（非侵入语音感知解码的金标准做法）。

### 4.2 Token 序列 loss —— 驱动可合成内容

```
L_token = CrossEntropy(content_logits, encodec_tokens_of_canonical_wav)
# 若用连续 latent 路线：L_latent = MSE(predicted_latent, encodec_latent)
```

### 4.3 分类 loss —— 稳定 + 可解释

```
L_class = CrossEntropy(class_logits, label)   # 16 类
```

### 4.4 总目标

```
L = λ_c · L_contrastive + λ_t · L_token + λ_l · L_class
# 建议起点 λ_c=1.0, λ_t=1.0, λ_l=0.5
# 明确不要：原始波形 L1。STFT 仅作为离线评测指标，不进训练。
```

> **为什么这样就不会坍缩：** 没有任何一项的最优解是"输出均值"。对比项要求可判别、token 项要求结构化、分类项要求类间分离——三者都惩罚"对所有输入给同一输出"。

---

## 5. 数据杠杆：把"能用的信号"全用上

FEIS 有 5 个阶段，当前只用了 thinking（信号最弱的那个）。新方案用**跨阶段课程 + 跨被试池化**榨取信号：

### 5.1 跨阶段课程学习（cross-stage curriculum）

| 阶段 | 信号强度 | 用法 |
|------|----------|------|
| `speaking`（overt 实说）| **最强**（有真实发音运动）| 先在此预训练 encoder，作为**能力上限/teacher** |
| `stimuli`（hearing 听到）| 强（听觉诱发清晰）| 辅助预训练，提供听觉-语音对齐 |
| `thinking`（imagined 想象）| **最弱**（最终目标）| 课程最后阶段，蒸馏/迁移自上面 |
| `articulators` / `resting` | 噪声/基线 | 仅用于增强与基线归一 |

**做法：** 三阶段共享 encoder + 阶段 embedding。先 speaking→stimuli 学到"EEG→语音表征"的强映射，再用 **teacher-student 蒸馏**（speaking 的表征/logits 作软标签）把能力迁到 thinking。
这给你一个极重要的副产品：**speaking 阶段的结果就是流程正确性的"天花板探针"**——如果 speaking 都解不出来，说明是 pipeline bug；如果 speaking 好、thinking 差，那才是真正的神经科学难度。

### 5.2 跨被试池化 + subject embedding

如 3.2，数据量 ×20，是脱离坍缩最有效的单一手段。

### 5.3 数据增强

- 通道 dropout（模拟坏导，正则）
- 时间 jitter / 裁剪（thinking 窗内滑动）
- 同类 mixup（同 label trial 间插值）
- baseline（resting）重归一，跨被试统一尺度

---

## 6. 评估方案（与"自然 + 正确"双目标对齐）

**主指标（识别正确性）：**
- 16 类 top-1 / top-5 检索准确率，按协议 G/S/U 分别报。
- NTA（最近模板，沿用你现有口径，便于对照）。
- **置换检验**：打乱标签重训/重测，给出经验 chance 与显著性，避免被随机波动误导。

**上限探针（pipeline 健康度）：**
- speaking 阶段同套指标——这是"流程能不能work"的体检。

**自然度（你这次的首要目标）：**
- 因为合成走冻结 EnCodec/HiFi-GAN，输出客观上是干净人声；报告 EnCodec 重建的 STFT/Mel 距离（合成保真）与人耳抽样即可。
- 听感评估的语义从"像不像噪声"变成"选对了哪一条 + 合成是否清晰"。

**分层报告：** 按被试、按 label、按阶段分层，定位是哪些音素/被试可解。

---

## 7. 分阶段落地计划

| 阶段 | 目标 | 关键交付 | 成功判据 |
|------|------|----------|----------|
| **P0** 合成端打通 | 冻结 EnCodec 把 16 条 canonical wav 编码→解码 | token 提取脚本 + 重建 wav | 重建 wav 人耳清晰、Mel 距离低 |
| **P1** speaking 上限探针 | pooled + subject-emb，仅 speaking，对比+分类+token | speaking 检索曲线 | top-1 显著 > chance（证明 pipeline 对）|
| **P2** thinking 主线 | 跨阶段蒸馏 + 跨被试，目标 thinking | 三协议 G/S/U 指标 | thinking top-1 稳定 > chance，输出自然 |
| **P3** 自然度打磨 | 切 HuBERT-unit + HiFi-GAN 对比 EnCodec | A/B 合成质量 | 选更自然的合成后端固定 |
| **P4** 消融与汇报 | EEG 基础模型 vs 从零、各 loss 消融 | 报告 + 图 | 明确每个组件贡献 |

建议先做 **P0 + P1**：P0 让你立刻听到"干净人声"（自然度问题当场解决），P1 用 speaking 验证整条识别链路能 work——这两步做完，方向对不对就有客观答案了，再投入 thinking 主线。

---

## 8. 与现状的关键差异速查

| 维度 | 现状（坍缩）| 新方案（v3）|
|------|------------|-------------|
| 框架 | EEG→原始波形回归 | **识别 + 冻结声码器合成（解耦）** |
| 目标表征 | 24000 原始采样 | **EnCodec token / HuBERT 单元** |
| 主 loss | L1 + STFT（→均值）| **InfoNCE + token CE + 分类 CE** |
| 合成 | ConvTranspose 上采样 | **冻结 EnCodec/HiFi-GAN（天生自然）** |
| 训练数据 | 单被试 160 | **池化 20×160 + subject-emb** |
| 信号来源 | 仅 thinking | **speaking/stimuli 课程蒸馏 → thinking** |
| EEG 前端 | 1D-CNN + 单层 VQ | 空间-时间分解 +（可选）EEG 基础模型 |
| 抗坍缩 | 无 | 对比 + 负样本 + 结构化 token |
| 自然度 | 均值哼鸣 | 冻结 SOTA 声码器，干净人声 |

---

## 9. 诚实的预期管理

- **自然度**：基本必达。只要合成走冻结 EnCodec/HiFi-GAN，输出就是干净人声——这条你这次最看重的目标，P0 当天就能听到效果。
- **正确性（选对 16 选 1）**：受 14 通道 imagined-speech 的神经科学上限约束。跨被试 + 对比 + 跨阶段蒸馏后，thinking 阶段 top-1 现实区间大概在 chance（6.25%）到 ~20-30% 之间；speaking 阶段应明显更高，作为流程正确性的证明。
- **如果要"per-trial 声学还原"**：FEIS 的 canonical 目标里物理上不存在这种信号，需换数据集（如 KaraOne overt，有 trial-synchronous 录音）。本方案在第 5.1 已为多数据集留好阶段接口。

---

## 参考（均在 `paper-ref/`）

- Lee 等 2023, *Towards Voice Reconstruction from EEG during Imagined Speech* — 同任务直接参考。
- Lee 等 2025, *Enhancing Listened Speech Decoding from EEG via Parallel Phoneme Sequence Prediction* — 并行音素序列、内容头设计。
- Défossez 等 2023, *Decoding speech perception from non-invasive brain recordings* — 跨被试 subject layer + InfoNCE 检索（4.1/3.2 主要依据）。
- Duan 等 2024, *DeWave* — EEG 离散编码（codec/VQ 目标思路）。
- Défossez 等 2022, *EnCodec: High Fidelity Neural Audio Compression* — 模块 B1。
- Kong 等 2020, *HiFi-GAN*；Hsu 等 2021, *HuBERT* — 模块 B2。
- Jiang 等 2024 *LaBraM* / 2025 *NeuroLM*；Xiao 等 2025 *BrainOmni* — 可选 EEG 基础模型前端（3.1）。
