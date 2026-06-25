# FEIS EEG→语音 模型优化与升级路线（CCF-A 视角）

> 角色定位：以 EEG decoding / Speech Foundation Model / Neural Codec / Diffusion 审稿人的标准，
> **针对 `feis_subject_aware_bundle` 当前真实代码、真实实验输出、`paper-ref/eeg-to-speech-ccf-a` 论文库**做的优化方案。
> 配套：KaraOne 版见 [`../karaone_overt_recon_bundle/OPTIMIZATION.md`](../karaone_overt_recon_bundle/OPTIMIZATION.md)；
> 自查 prompt 见 [`../SELF_CHECK_PROMPT.md`](../SELF_CHECK_PROMPT.md)。
> 本文所有"当前事实"均来自 `app/src/`、`app/configs/`、`artifacts/outputs_mel/` 与论文库，可逐条对照。

---

## 0. 本文档依据（读了什么）

- 代码：`src/feis_mel/{model,losses,targets,audio,data,diffusion,gan}.py`、`src/direct_eeg2speech/*`、
  `configs/{feis_mel_align,direct_eeg2speech}.yaml`、`MODEL_TECH.md`。
- 结果：`artifacts/outputs_mel/four_stage_summary_20260625_001404.json`、各 stage 的
  `metrics/test_holdout_mel_eval.json`、`final_wavs_test_holdout/waveform_compare/waveform_compare_manifest.csv`、
  `original_vs_pred_scaled_contact_sheet.png`。
- 论文库：`paper-ref/eeg-to-speech-ccf-a/{README.md,papers.csv}` + 25 篇 PDF 的分组与定位。

> 凡是无法从文件确认的，文中标注 **【需进一步验证】**，不臆造。

---

## 1. 当前管线（从代码确认，不预设）

FEIS bundle **不是** 单一 "EEG→mel→HiFiGAN"。代码里有两条并行路线，**这次四阶段 headline 结果跑的是 B 路**：

| 路线 | 包 | 目标 | 解码 | 声码器 | 这次结果是否用 |
|---|---|---|---|---|---|
| A. direct | `src/direct_eeg2speech` | EnCodec 连续潜在 75×128 | Transformer + **潜在扩散(DDPM+DDIM)** | 冻结 EnCodec 解码器 | ❌ 未用于四阶段 |
| **B. mel（headline）** | `src/feis_mel` | **label mel 库** 64×80 | 可学习 query + cross-attn 回归头 | **Griffin-Lim(48 iter)** | ✅ `four_stage_summary` 全是 `target_kind=mel, decoder=regression, diffusion=false, gan=false` |

关键代码事实：

1. **B 路目标是"按 label 聚合的 mel 参考库"**（`targets.py:MelLabelTargets`，`feis_label_mel_banks.npz`，
   `banks[num_labels, K≤20, 64, 80]`）。损失是 `softmin_dtw_mel_loss`（`losses.py:42`）：对该 label 库里
   **朴素 L1 最近的 top-3 条**真实朗读 mel 各做带状 DTW 对齐再 L1，按 DTW 代价 softmin 加权。
2. **B 路完全不看 stage**：`configs/feis_mel_align.yaml: input.use_stage_idx: false`。四阶段是
   **对每个 stage 的 EEG 各训一个独立模型**，但模型输入里没有 stage。
3. B 路对比损失是 `label_contrastive_loss`（`losses.py:84`）——EEG 句向量 vs **label 原型**，
   **不是** EEG↔真实音频的对称 InfoNCE。
4. 扩散 / GAN / EnCodec 目标在代码里都存在（`diffusion.py`/`gan.py`/`target.kind=encodec_latent`），
   但 **headline 四阶段没开**。

---

## 2. 实证：当前结果与波形证据（必须先看清这个）

### 2.1 四阶段 mel 级指标（`four_stage_summary_20260625_001404.json`，test_holdout n=320）

| stage | mel_PCC | content_top1 (chance 0.0625) | retrieval_top1 | pred→bank DTW | mean-baseline DTW |
|---|---|---|---|---|---|
| stimuli | 0.9293 | 0.0594 | 0.0625 | 0.0236 | 0.0962 |
| thinking | 0.9293 | 0.0625 | 0.0625 | 0.0235 | 0.0962 |
| speaking | 0.9294 | 0.0750 | 0.0625 | 0.0237 | 0.0962 |
| resting | 0.9297 | 0.0656 | 0.0625 | 0.0248 | 0.0962 |

### 2.2 波形级证据（`waveform_compare_manifest.csv`）

同一条 `feis_f_000011`，四个 stage 的波形 `pearson_r`：
`stimuli 0.0007728 / thinking 0.0007255 / speaking 0.0007699 / resting 0.0007682`。
全部样本 `pearson_r ∈ [−0.010, +0.0065]`，即 **波形级与原始几乎零相关**。
`original_rms` 因 label 而异（"f" 0.0134、"fleece"/"goose"/"k"/"m" ~0.080），但 `reconstruction_rms`
基本恒定在 ~0.058–0.071，**与原始幅度脱钩**。

### 2.3 接触图（`original_vs_pred_scaled_contact_sheet.png`）

每对：上(红)=原始参考波形（宽带、密集），下(青)=预测（**单峰、纺锤形、过度平滑的包络**，无精细时序/高频）。
不同 label 的预测包络彼此相似。

### 2.4 这三组证据合起来说明什么（决定性）

> **`resting`（无任何语音）的重建质量 ≈ `speaking`/`thinking`/`stimuli`；content/retrieval 全在随机线；
> 每条样本的波形 `pearson_r` 跨四阶段逐位相同。**

结论：**当前 B 路并没有从 EEG 解码出语音内容**。0.929 的 mel_PCC 来自
**label mel 库这个目标本身**——网络学到的是"给定 label 输出该类的平均 mel 包络"，再被 DTW-to-bank +
平滑性把 PCC 抬高；它既不携带逐 trial 信息，连 stage 也区分不了（B 路压根没输入 stage，且换哪个 stage 的 EEG
都塌到同一类均值）。`pred→bank DTW (0.024) ≪ mean-baseline DTW (0.096)` 只证明"预测比全局均值更接近**类**均值"，
**不证明 EEG 有用**。

这正是 reviewer 会一击致命的点，下一节展开。

---

## 3. 像 CCF-A reviewer 一样评价

### 3.1 逐维度

| 维度 | 现状 | 评价 |
|---|---|---|
| EEG encoder | `SpatialTemporalEEGEncoder` + 通道 MoE + FiLM | 设计合理但**不是瓶颈**；14ch 低 SNR 下再强的 encoder 也救不了无对齐 GT |
| Target | **label mel 库**（类均值参考） | **最大问题**：目标只含"类级参考发音"，逐 trial 内容信息为零，指标可被 label prior 刷出来 |
| Decoder | query+cross-attn 回归头 | 回归→均值塌缩/纺锤包络（接触图实锤） |
| Codec 使用 | 代码有 EnCodec 路，但 headline 未用；声码器是 Griffin-Lim | GL 丢相位→任何波形级指标无意义；落后 EnCodec/RVQGAN/DAC 一代 |
| Diffusion/Flow | 有 DDPM+DDIM，未用于 headline；无 flow matching | 落后 NeuroSonic(2026, **conditional flow matching**)/Voicebox 的范式 |
| Subject 建模 | 彻底身份无关（assert 拦截） | 科学上干净，但**没做被试域自适应**→跨被试天花板低 |
| Voice 条件 | 无 | 想象语音本就无说话人，可接受；但缺一个固定 voice prior 渲染器 |
| Loss | DTW(类库) + label 原型对比 + content CE + log-RMS + MoE 正则 | **缺真正的 EEG↔音频对称 InfoNCE**（Défossez 2022 的核心）；缺零-EEG 对照 |
| Training | 每 stage 独训；无 overt→imagined 迁移 | **没用上 FEIS 自带的 speaking(发声) 段**做 NeuroTalk 式域自适应（巨大浪费） |
| Generalization | 跨被试未单独报告 | 【需进一步验证】FEIS 是否有 subject-holdout split 报告 |
| Evaluation | mel_PCC / content / retrieval / pred-vs-mean DTW | **缺 zero-EEG / shuffled-EEG / label-prior 上界对照**（KaraOne 有，FEIS 没有）→ 现指标可被刷 |
| Ablation | 四个 stage 但同配置 | 缺 target/loss/encoder 消融；"四阶段"不是消融 |
| Inference 质量 | 纺锤包络、波形 r≈0 | 只能算 demo |

### 3.2 三个必须直说的判断

- **最大短板**：**目标表示 = label mel 库**。它让"高 PCC"与"是否真的解码到 EEG"解耦。这是方法学硬伤，不是调参能补的。
- **最易被攻击点**：`resting ≈ speaking` 且 content/retrieval=chance。审稿人一句话："你的重建在没有语音的静息 EEG 上一样好，
  且不能识别说了哪个词——这说明你重建的是 label 先验，不是 EEG。" 你**现在没有 zero-EEG 对照去自证清白**。
- **哪些只能算 demo**：当前全部 mel_PCC / 波形图。**没有 strong evidence**，因为缺关键阴性对照。
- **不改就基本无望 CCF-A**：①目标必须从"类均值 mel 库"换成**逐 trial 可对齐**的内容表示或**只做识别/检索任务**；
  ②必须补 zero-EEG/shuffled-EEG/label-prior 对照；③声码器必须换掉 GL。

---

## 4. Target 重新设计（核心）

为什么现代语音生成不再只用 mel：mel→声码器有损（GL 还丢相位），且 **mel 回归在弱条件下必然过平滑**（接触图实锤）。
EnCodec / RVQGAN / SpeechTokenizer / X-Codec / NaturalSpeech3-FACodec 之所以转向 codec latent/token，是因为
**learned decoder 端到端把相位与细节补回**，你只需预测紧凑潜在。但对 14ch 低 SNR 的 FEIS，瓶颈不是声码器而是
**EEG 几乎不携带内容**——所以"直接预测 128 维 EnCodec 连续潜在 / 多码本 RVQ token"反而过载（MODEL_TECH §5 也记录了
EnCodec 连续 latent + cosine 回归会塌缩）。

| Target | 优点 | 缺点 | 适合 FEIS | 适合 KaraOne | 推荐 | 理由 |
|---|---|---|---|---|---|---|
| Mel(现状) | 简单、可视 | 过平滑、GL 丢相位、可被 label prior 刷 | △ 仅作 sanity | △ | ⛔ 不再当主目标 | 当前问题的根源之一 |
| HuBERT hidden | 内容性强、低维平滑、好学 | 需预训练模型 | ✅ | ✅ | ✅✅ 主推 | 与 Défossez 对比解码同源；弱 EEG 也能回归/检索 |
| HuBERT cluster(离散单元) | 可做检索/CTC、抗噪 | 量化损失 | ✅(做识别) | ✅ | ✅ | AudioLM 语义单元；天然支持 top-k 检索 |
| wav2vec2 feature | 同 HuBERT | 同上 | ✅ | ✅ | ✅ | wav2vec2.0(2020 NeurIPS) 正是 README 指定语义对齐目标 |
| Whisper embedding | 鲁棒、内容强、可挂 ASR 自检 | 帧率/维度大 | ✅ | ✅ | ✅ | Whisper(2023 ICML) 同时给可懂度自检 |
| EnCodec 连续 latent | 高保真渲染 | 高熵、弱 EEG 下塌缩 | ⛔ | △ | ⛔(当主目标) | 你们已实测塌缩 |
| EnCodec / RVQGAN token | 高保真、可 LM 化 | 多码本、极数据饥渴 | ⛔ | △ | ⛔ | 数据规模下不现实 |
| SpeechTokenizer / X-Codec | 语义+声学分层 | 仍数据饥渴 | △ | △ | △(二阶段渲染) | 作"语义→声学"第二阶段渲染器，不作 EEG 回归目标 |
| FACodec(NaturalSpeech3) | 内容/韵律/音色解耦 | 复杂 | △ | △ | △(Plan C) | 用于把 EEG 能/不能携带的因素显式分离 |

> **FEIS 最终目标推荐**：**主目标 = HuBERT/wav2vec2 隐藏特征（连续）+ 其 k-means 离散单元（做检索/CTC）**；
> mel 仅留作便宜 sanity 声码器；**音频渲染交给冻结的"语义→波形"渲染器**（HuBERT-unit vocoder 或 SpeechTokenizer/MaskGCT 声学阶段）。
> 这条路同时满足：①更可学（弱 EEG 友好）；②给出**可信的检索/可懂度**评估；③绕开 GL。

---

## 5. 重新设计模型（主方案 + 两备选）

### 方案 A（**FEIS 主推**）：EEG → 语义特征 → 冻结语义声码器（+可选 flow 精修）

```
EEG[14×1280]
  └─ Channel-MoE 前端(保留) ─ FiLM ─ Conformer/Temporal Trunk ─► EEG memory[T,d]
        ├─(对齐头) clip_head ──► 句向量 ──[对称 InfoNCE]── Whisper/HuBERT 音频句向量(冻结)
        ├─(内容头) CTC/分类头 ──► phoneme/word logits
        └─(回归头) cross-attn query ─► pred_hubert[T',768]
                                          │
                  ┌── 评估: 对 GT HuBERT 的特征距离 / top-k 检索 ──┐
                  └── 渲染: 冻结 HuBERT-unit vocoder ─► waveform ──┘
```
- 输入/输出：EEG→(句向量, 内容 logits, HuBERT 特征序列)→波形。
- 为什么更强：把"重建保真"与"EEG 解码"分离——EEG 只需命中**低维内容表示**，渲染由冻结预训练模型负责。
- 参考：wav2vec2.0、Whisper、Défossez 2022（对比检索）、AudioLM（语义单元）。
- 适合：FEIS（尤其 imagined/thinking）。数据需求：中（用现有规模可训对比/检索头）。难度：中。推理：低（无 AR）。
- 风险：HuBERT-unit vocoder 需离线权重【需进一步验证是否可在服务器离线获取】。

### 方案 B：EEG → 语义单元 → MaskGCT/SoundStorm 声学补全 → codec 解码

```
EEG ─► EEG Encoder ─► HuBERT 语义 token ─► MaskGCT/SoundStorm 掩码生成声学 token ─► DAC/EnCodec ─► wav
```
- 参考：MaskGCT、SoundStorm、AudioLM、SpeechTokenizer、VALL-E。
- 适合：KaraOne 多于 FEIS。数据需求：**高**（声学补全阶段数据饥渴）。难度：高。风险：FEIS 规模下声学阶段难训→**不现实**，列为远期。

### 方案 C：因子化（content / prosody / voice）→ FACodec 解码

```
EEG ─► 共享 Encoder ─┬─ content: HuBERT/CTC
                     ├─ prosody: F0 / energy / duration 头
                     └─ voice : 固定 voice prior(数据级常量, 非身份)
                                      └─► NaturalSpeech3 FACodec 解码 ─► wav
```
- 参考：NaturalSpeech 3 / FACodec、StyleTTS2（韵律扩散）。
- 适合：想显式回答"EEG 到底携带 content / prosody / voice 哪一部分"——**很适合写成 FEIS 的科学贡献**。难度：中高。

> **推荐主路线：方案 A**。它是 NeuroSonic/Défossez 同源、最不数据饥渴、且天然带可信评估。
> Plan C 作为"机制分析/消融"补充，能把论文的科学性拉满。

---

## 6. 重新设计 Loss

| Loss | 保留/删除/新增 | 初始权重 | 仅 KaraOne? | FEIS 适用? | 备注 |
|---|---|---|---|---|---|
| softmin-DTW(label 库) mel L1 | **降级为辅助** | 0.3 | 否 | 是(辅助) | 仍可作 mel sanity，但不再是主目标，且必须配 zero-EEG 对照 |
| **EEG↔音频对称 InfoNCE** | **新增(主)** | 0.5 | 否 | **是** | Défossez 2022 核心；把"label 原型对比"升级为真跨模态；FEIS 当前缺 |
| HuBERT/wav2vec2 特征回归(SmoothL1+cos) | **新增(主)** | 1.0 | 否 | 是 | 新主目标 |
| 多分辨率 STFT loss | 新增 | 0.3 | 否 | 是(渲染端) | 治过平滑，比单一 mel L1 强 |
| SSL 特征匹配(HuBERT/WavLM)感知损失 | 新增 | 0.2 | 否 | 是 | 渲染音频 vs GT 在 SSL 空间比 | 
| content CE / CTC | 保留(FEIS 用 CE) | 0.3 | CTC 仅 KaraOne | CE 适用 | FEIS 无逐 trial 序列对齐，用句级 CE |
| **conditional flow matching** | 新增(替代 DDPM) | 1.0(若开生成) | 否 | 是 | NeuroSonic/Voicebox；比 DDPM 步少更稳 |
| log-RMS | 保留 | 0.1 | 否 | 是 | |
| MoE 4 正则 | 保留 | 现值 | 否 | 是 | |
| std/diversity/mean_margin(A 路抗塌缩) | **删除/弃用** | 0 | — | — | 换生成式解码后不需要这些 band-aid |
| label 原型 contrastive | 被 InfoNCE 取代 | 0 | — | — | |

- 可能造成塌缩的：EnCodec 连续 latent 上的纯 recon_cos（你们已实测）、过大的 mean_margin。
- 如何监控每个 loss 有效：沿用 `train` 里已有的逐项 detach 记录 + 监控 **std_ratio→1**、**pred_pairwise_corr↓**、
  **retrieval_top1 显著高于 chance** 三个"真有用"判据。

---

## 7. 训练策略（FEIS 重点）

**必须加：**
- **zero-EEG / shuffled-EEG / label-prior 三对照**（把 KaraOne `eval.py` 的 `pred_over_mean/zero_cos_gain` 思路移植过来）。
  这是当前最廉价、最能自证清白的一步。
- **NeuroTalk 式 overt→imagined 域自适应**：FEIS 自带 `speaking`(发声) 段——用它当"spoken 域"先训，再把
  `thinking/imagined` 当目标域做对抗/特征对齐迁移。**这是 FEIS 现在完全没用上的最大杠杆**（参考 NeuroTalk）。
- **被试域自适应**（不泄露身份）：每被试 z-score / CORAL 对齐，缓解跨被试塌缩。

**建议加：** 自监督 EEG 预训练（mask-reconstruct）、数据增强（时间抖动/同 label mixup，channel dropout 已有）、
课程学习（speaking→thinking）。

**暂时别加：** 大型 codec LM / VALL-E 式 AR（数据饥渴，会训不动）；叠加第二层 MoE；重 GAN（不稳）。

---

## 8. 评估指标重新设计

- **FEIS 想象/感知无逐 trial GT → 禁用 STOI/PESQ/MCD/WER 当主指标**（没有逐 trial 真值，这些不成立）。
- **主指标（诚实）**：within-subject **top-k 检索准确率**、content/identification 准确率、
  **pred_over_mean / pred_over_zero gain**（移植）、HuBERT/WavLM 特征距离、Whisper-embedding 距离、CLAP 相似度。
- **关键对照**：**resting 作阴性对照**——若 resting 的检索/gain ≈ speaking，则证明无效；论文里把这条当作"诚实性检验"反而是亮点。
- **上界**：oracle-codec/oracle-HuBERT 上界（把 GT 目标过同一渲染器，给出天花板）。
- **辅助**：UTMOS/DNSMOS（渲染自然度，仅在换神经声码器后有意义）、F0/energy/envelope 相关。
- **会误导的指标**：对 label 库的 mel_PCC、EnCodec latent 上的 cosine、任何 GL 波形级指标——**当前 headline 全踩中**。

---

## 9. 升级路线（按优先级）

### 第一优先级（不改基本无望）

| 改动 | 收益 | 难度 | 依据/参考 | 代码范围 |
|---|---|---|---|---|
| 移植 zero-EEG/shuffled-EEG/label-prior 对照 + gain 指标到 B 路 | ★★★★★ | 低 | KaraOne `eval.py` 已有范式 | `src/feis_mel/eval.py`、`scripts/feis_mel_eval.py` |
| 主目标换 HuBERT/wav2vec2 特征(+离散单元) | ★★★★★ | 中 | wav2vec2.0、Défossez 2022、AudioLM | `targets.py`、`model.py`(回归头维度)、`losses.py` |
| 新增 EEG↔音频对称 InfoNCE，retrieval 设为主指标 | ★★★★★ | 低 | Défossez 2022 | `losses.py`(仿 KaraOne `clip_alignment`)、`model.py`(clip_head) |
| 换掉 Griffin-Lim：冻结 HuBERT-unit / 神经声码器渲染 | ★★★★☆ | 中 | EnCodec、RVQGAN、SpeechTokenizer | `audio.py`、`synth.py` |

### 第二优先级（建议改）

| 改动 | 收益 | 难度 | 依据 | 代码范围 |
|---|---|---|---|---|
| NeuroTalk 式 speaking→thinking 域自适应 | ★★★★☆ | 中 | NeuroTalk | `data.py`(双域 split)、训练脚本 |
| DDPM→conditional flow matching 生成解码 | ★★★★ | 中 | NeuroSonic 2026、Voicebox、CoMoSpeech | `diffusion.py`→新 `flow.py` |
| 被试域自适应(每被试归一/CORAL，不泄露 ID) | ★★★☆ | 中 | 域适应通用 | `data.py`、`encoder.py` |
| 加 oracle 上界 + SSL/Whisper 距离指标 | ★★★☆ | 低 | Whisper、WavLM | `eval.py` |

### 第三优先级（锦上添花）

- 多分辨率 STFT + SSL 特征匹配损失（治平滑）；自监督 EEG 预训练；FACodec 因子化分析(Plan C)；UTMOS/DNSMOS 报告。

### ROI Top-10（FEIS，按性价比排序）

| Rank | 改动 | ROI | 依据论文 | 预计提升 | 代码范围 | 难度 | 风险 |
|---|---|---|---|---|---|---|---|
| 1 | zero/shuffled-EEG + label-prior 对照与 gain 指标 | ★★★★★ | (KaraOne 范式) | 让结论可信/可发表 | `eval.py` | 低 | 低 |
| 2 | EEG↔音频对称 InfoNCE + 检索主指标 | ★★★★★ | Défossez 2022 | 出现真信号即可量化 | `losses.py`/`model.py` | 低 | 低 |
| 3 | 主目标→HuBERT/wav2vec2 特征 | ★★★★★ | wav2vec2、AudioLM | 可学性大升 | `targets.py`/`model.py` | 中 | 中(需权重) |
| 4 | 神经声码器替换 GL | ★★★★ | EnCodec/RVQGAN | 音质+UTMOS 可报 | `audio.py`/`synth.py` | 中 | 中 |
| 5 | speaking→thinking 域自适应 | ★★★★ | NeuroTalk | imagined 段提升 | `data.py`/训练 | 中 | 中 |
| 6 | flow matching 生成解码 | ★★★★ | NeuroSonic/Voicebox | 抗塌缩+保真 | `flow.py` | 中 | 中 |
| 7 | 被试域自适应 | ★★★ | 域适应 | 跨被试不再塌 | `data.py`/`encoder.py` | 中 | 中 |
| 8 | oracle 上界 + SSL/Whisper 距离 | ★★★ | Whisper/WavLM | 评估完整 | `eval.py` | 低 | 低 |
| 9 | 多分辨率 STFT + SSL 特征匹配损失 | ★★★ | Voicebox/StyleTTS2 | 治平滑 | `losses.py` | 中 | 中 |
| 10 | FACodec 因子化分析(Plan C) | ★★☆ | NaturalSpeech 3 | 机制贡献 | 新模块 | 高 | 中 |

---

## 10. 数据规模下哪些不现实（FEIS）

- VALL-E/UniAudio 式 codec **AR LM**、从 EEG 直接预测多码本 RVQ token：**不现实**（数据量差几个量级）。
- 期望"逐词可懂的想象语音重建"：**不现实**。14ch + 1s 窗 + 无逐 trial GT → 天花板在**类级检索/包络**，不在逐词还原。
- 因此论文叙事应定位为：**"诚实的非侵入 EEG↔语音对齐 + 可信检索 + 神经声码器渲染 + 机制分析"**，而不是"我们重建了可懂语音"。

---

## 附录 A：本 bundle 的 WAV 对比图路径

每个目录含 `*_waveform.png`、`waveform_compare_manifest.csv`、`original_vs_pred_scaled_contact_sheet.png`：

```
artifacts/outputs_mel/feis_mel_stimuli_mel_align_20260625_001404/final_wavs_test_holdout/waveform_compare
artifacts/outputs_mel/feis_mel_thinking_mel_align_20260625_001404/final_wavs_test_holdout/waveform_compare
artifacts/outputs_mel/feis_mel_speaking_mel_align_20260625_001404/final_wavs_test_holdout/waveform_compare
artifacts/outputs_mel/feis_mel_resting_mel_align_20260625_001404/final_wavs_test_holdout/waveform_compare
artifacts/outputs_mel/feis_mel_stimuli_mel_smoke/final_wavs_test_holdout/waveform_compare
artifacts/outputs_mel/feis_mel_stimuli_mel_impl_smoke/final_wavs_test_holdout/waveform_compare
artifacts/outputs_mel/feis_mel_stimuli_diffusion_impl_smoke/final_wavs_test_holdout_20260625_000345/waveform_compare
artifacts/outputs_mel/feis_mel_stimuli_encodec_impl_smoke/final_wavs_test_holdout_20260625_000543/waveform_compare
```
绘图 notebook：`plot_waveform_comparisons_feis.ipynb`。

## 附录 B：自查 prompt

见 [`../SELF_CHECK_PROMPT.md`](../SELF_CHECK_PROMPT.md)（FEIS 版自查清单在文末"FEIS 专用"一节）。
