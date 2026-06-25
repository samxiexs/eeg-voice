# KaraOne EEG→语音 模型优化与升级路线（CCF-A 视角）

> 角色定位：以 EEG decoding / Speech Foundation Model / Neural Codec / Diffusion 审稿人的标准，
> **针对 `karaone_overt_recon_bundle` 当前真实代码、真实实验输出、`paper-ref/eeg-to-speech-ccf-a` 论文库**做的优化方案。
> 配套：FEIS 版见 [`../feis_subject_aware_bundle/OPTIMIZATION.md`](../feis_subject_aware_bundle/OPTIMIZATION.md)；
> 自查 prompt 见 [`../SELF_CHECK_PROMPT.md`](../SELF_CHECK_PROMPT.md)。
> 本文所有"当前事实"均来自 `app/src/`、`app/configs/karaone.yaml`、`artifacts/outputs_karaone/` 与论文库，可逐条对照。

---

## 0. 本文档依据（读了什么）

- 代码：`src/karaone_recon/{model,encoder,losses,targets,eval,diffusion,discriminator,synth,audio_features}.py`、
  `configs/karaone.yaml`、`MODEL_TECH.md`。
- 结果：`artifacts/outputs_karaone/karaone_moe_overt_like_20260625_001741/metrics/test_metrics.json`、
  `wav_test_20260625_051705/waveform_compare/{waveform_compare_manifest.csv, original_vs_pred_scaled_contact_sheet.png}`。
- 论文库：`paper-ref/eeg-to-speech-ccf-a/{README.md,papers.csv}` + 25 篇 PDF 分组。

> 无法从文件确认的标注 **【需进一步验证】**。

---

## 1. 当前管线（从代码确认）

默认：**`EEG(62ch) → 编码器(可选通道MoE) → 回归头 → mel(80) → Fast Griffin-Lim → 波形`**，全开关化。

| 开关 | 默认 | 备选 |
|---|---|---|
| `target.kind` | **mel** | encodec_latent |
| `vocoder.kind` | **griffinlim**(momentum 0.99, 100 iter) | encodec |
| `model.decoder` | **regression** | diffusion(cosine schedule, ε-pred, DDIM 50) |
| `--model` | — | baseline / **moe**(通道选择/聚类) |
| `lambda_dtw` | **1.0** | 0 |
| `lambda_clip` | **0.5** | 0 |
| `lambda_gan` | **0** | >0(LSGAN+feat match) |

代码事实（与 FEIS 的关键差异）：

1. **音频是 trial-synchronous 的真实 overt 录音**（同 trial、2.0s@16k、RMS 归一）。这意味着
   **target 是这一 trial 真实说的话，不是类均值**——这是 KaraOne 比 FEIS"更真"的根本原因。
2. 损失里**已有真正的 EEG↔音频对称 InfoNCE**（`losses.py:58 clip_alignment`，Défossez 2022）+
   **DTW 对齐重建**（`losses.py:36 dtw_recon_loss`，NeuroTalk 式，治跨 trial 起始/语速抖动）。
3. **eval 是诚实的**（`eval.py`）：同时报 `zeroeeg`(EEG 置零) 与 `mean_latent`(全局均值) 两个基线，
   主选择指标是 `pred_over_mean_cos_gain`，并报 std_ratio / pairwise_corr / 检索 top-1。
4. 扩散头是真扩散（cosine schedule + ε 预测 + DDIM + x0 钳制），但 headline 跑的是 **mel+MoE+回归+DTW**。

---

## 2. 实证：当前结果与波形证据

### 2.1 mel/latent 级指标（`test_metrics.json`，mel+MoE+DTW）

| 指标 | test(同被试 n=132) | subject_test(跨被试 P02/MM21 n=297) |
|---|---|---|
| **pred_over_mean_cos_gain** | **+0.0695** | **−0.0144（≈0）** |
| pred_over_zero_cos_gain | +0.1543 | +0.1103 |
| pred_pcc | 0.6801 | 0.6922 |
| pred_std_ratio_median | 0.654 | 0.530 |
| pred_pairwise_corr_median | 0.701 | 0.798 |
| content_acc (chance 0.0909) | **0.1364** | 0.0976(≈随机) |
| within_subject_label_top1 (vs zeroeeg 0.0909) | **0.1212** | 0.1178 |
| within_subject_trial_top1 (chance≈0.0076) | **0.0227** | 0.0168 |

### 2.2 波形级证据（`waveform_compare_manifest.csv`，n=132）

- `pearson_r`：mean **0.00051**、median **−0.00008**、范围 [−0.035, +0.109] → **波形级近零相关**。
- `original_rms`：mean 0.0161、median 0.0134（**很轻，~12% 有声**）；`reconstruction_rms`：mean **0.0084**、median 0.0082。
  → **能量塌缩到原始约一半**，且与原始幅度脱钩：响词 `pot`(orig 0.046)→recon 0.011、`pat`(0.044)→0.009、`gnaw`(0.041)→0.0085。

### 2.3 接触图（`original_vs_pred_scaled_contact_sheet.png`）

每对：上=原始 overt（短促、稀疏有声段），下=预测。**部分 panel 预测的能量峰与原始有声段大致同位**（比 FEIS 强），
但幅度明显更小、形状更平滑。即：**有微弱时间包络耦合，但能量被压缩、细节被平滑**。

### 2.4 合起来说明什么（诚实读数）

> KaraOne **有"微弱但真实"的 within-subject 信号**：pred 同时**超过** zero-EEG 与 mean 两个基线
> （gain +0.07）、content_acc 比随机高约 50% 相对、label 检索 0.121>0.091、std_ratio 0.65（不塌缩）。
> 但**跨被试 ≈ 0**（gain −0.01）。波形 r≈0 主要是 **Griffin-Lim 丢相位**所致（波形级 PCC 在 GL 下本就无意义），
> 真正的硬伤是 **①能量塌缩 ②跨被试不泛化 ③可懂度天花板低**。

**为什么 KaraOne 显著优于 FEIS** → 是**数据**不是模型：两者共享同一 encoder/MoE/DTW 设计，
唯一结构差别是 **KaraOne 有 trial-synchronous 真实音频**（target 是真话而非类均值）。这点要在论文里明确写。

---

## 3. 像 CCF-A reviewer 一样评价

### 3.1 逐维度

| 维度 | 现状 | 评价 |
|---|---|---|
| EEG encoder | SpatialTemporal + 通道 MoE(62ch 选择/聚类) | 合理；62ch 比 FEIS 14ch 有优势；非主瓶颈 |
| Target | mel(默认) / encodec latent(实测塌缩) | mel 过平滑；codec latent 弱信号塌缩 → **应转语义/codec-token 二阶段** |
| Decoder | 回归(默认) / 真扩散(未用于 headline) | 回归→能量塌缩(实锤)；扩散"多样但不准" |
| Codec 使用 | 有 EnCodec 路；headline 用 GL | GL 丢相位 → 波形指标失真；落后一代 |
| Diffusion/Flow | DDPM+DDIM，**无 flow matching** | 落后 NeuroSonic(2026 conditional flow matching) |
| Subject 建模 | 身份无关 + 真 subject-holdout split | **方法学干净，是亮点**；但**没做域自适应**→跨被试 0 |
| Voice 条件 | EEG 推断 global head(非查表) | 合理；可加固定 voice prior 渲染 |
| Loss | DTW + 真 InfoNCE + content CE/supcon/proto + log-RMS + MoE | **设计已相当现代**；缺 CTC、缺感知/SSL 特征损失、缺 flow |
| Training | 120ep, 但 MODEL_TECH 记录 20ep 最干净(治过拟合) | 小数据**过拟合**是真问题；需正则/增强/预训练 |
| Generalization | 跨被试 gain≈0 | **核心短板**；域自适应是唯一出路 |
| Evaluation | zeroeeg/mean 双基线 + gain + std_ratio + 检索 | **诚实，做得好**；可再加 oracle 上界 + WER/Whisper 距离(有真音频) |
| Ablation | 开关齐全(mel/encodec, reg/diff, moe, dtw, gan) | **消融基础设施好**；需补系统消融表 |
| Inference 质量 | 能量塌缩、平滑、GL 机械音 | demo+ |

### 3.2 三个必须直说的判断

- **最大短板**：**跨被试不泛化**（subject_test gain −0.014）。within-subject 的 +0.07 若不能跨被试，
  审稿人会问"是否只是记住了被试特定噪声"。**域自适应 + 跨被试协议是 KaraOne 能不能上 CCF-A 的关键。**
- **最易被攻击点**：①能量塌缩（recon RMS≈orig 一半，且与内容脱钩）；②content_acc 仅比随机高一点点，
  绝对可懂度极低；③用 GL 时波形 r≈0，任何"波形重建"措辞都会被攻击。
- **哪些只能算 demo**：当前波形音频、GL 渲染、单次 mel+MoE 结果。
- **strong evidence 已有的**：within-subject **同时超过 zero-EEG 与 mean 两基线的 gain** + 高于随机的 label 检索——
  这是**真信号**，是论文可立的地基。**保住它、把它跨被试化、把渲染换好**，就有故事。

---

## 4. Target 重新设计

为什么不再只用 mel：GL 丢相位 + mel 回归过平滑（接触图 + 能量塌缩实锤）。但**直接预测 EnCodec 连续 latent / 多码本 RVQ token
对 KaraOne 这种又轻又短的弱信号同样会塌缩**（MODEL_TECH §7 实测：latent+cosine → std 0.15、pairwise 0.94）。

| Target | 优点 | 缺点 | 适合 KaraOne | 推荐 | 理由 |
|---|---|---|---|---|---|
| Mel(现状) | 简单 | 平滑、GL 丢相位 | △(sanity) | ⛔ 主目标 | 能量塌缩根源之一 |
| HuBERT hidden / cluster | 内容强、低维、抗噪、可 CTC/检索 | 需权重 | ✅✅ | ✅✅ 主推 | 有真音频→可直接对齐+评 WER |
| wav2vec2 feature | 同上 | 同上 | ✅ | ✅ | README 指定语义对齐目标 |
| Whisper embedding | 鲁棒、可挂 ASR 自检 | 维度大 | ✅ | ✅ | 直接给可懂度自检(WER/CER) |
| EnCodec 连续 latent | 高保真渲染 | 弱信号塌缩(已实测) | △ | ⛔ 主目标 | 仅作渲染后端，不作回归目标 |
| EnCodec/RVQGAN token | 高保真、可 LM | 多码本、数据饥渴 | △ | △(二阶段) | 作"语义→声学"第二阶段 |
| SpeechTokenizer/X-Codec | 语义+声学分层 | 数据饥渴 | ✅(渲染器) | ✅(二阶段) | 语义层正好接 EEG 预测 |
| FACodec(NaturalSpeech3) | content/prosody/timbre 解耦 | 复杂 | ✅(分析) | △(Plan C) | 显式分离 EEG 携带的因素 |

> **KaraOne 最终目标推荐**：**主目标 = HuBERT/wav2vec2 隐藏特征 + 离散单元（做 CTC/检索）**，
> 因为**有 trial-synchronous 真音频**，可以直接对齐到这一 trial 的真实语义、并用 Whisper 跑 WER/CER 自检；
> **渲染交给 SpeechTokenizer/MaskGCT 声学阶段或冻结 EnCodec/DAC 解码器**，mel/GL 仅留 sanity。

---

## 5. 重新设计模型（主方案 + 两备选）

### 方案 A（**KaraOne 主推**）：EEG → 语义特征 + 条件 flow matching → codec 渲染

```
EEG[62×1280]
 └─ Channel-MoE(保留) ─ FiLM ─ Conformer Trunk ─► EEG memory[T,d]
      ├─ clip_head ──[对称 InfoNCE]── Whisper/HuBERT 句向量(冻结)        (已有，保留)
      ├─ CTC 头 ──► phoneme 序列    (新增：KaraOne 有音素标签+真音频)
      ├─ 回归头 ─► pred_hubert[T',768]
      └─ Conditional Flow-Matching 解码器(以 EEG+pred_hubert 为条件)
                 ─► codec latent ─► 冻结 DAC/EnCodec 解码 ─► waveform
 评估: 对 GT HuBERT 距离 / Whisper-WER / 检索 / oracle 上界
```
- 为什么更强：①语义目标可学性 ≫ mel；②flow matching 抗塌缩且步少更稳（NeuroSonic 在**同任务**上正是此法）；
  ③codec 渲染替掉 GL，给出可信 UTMOS/MCD。
- 参考：NeuroSonic 2026（conditional flow matching for EEG→speech）、Voicebox（flow matching）、wav2vec2、Whisper、Défossez 2022。
- 数据需求：中（语义头/对齐头现规模可训；flow 解码器可冻结渲染端）。难度：中。推理：低-中（flow 少步）。
- 风险：跨被试仍需配合 §7 域自适应才不为 0。

### 方案 B：EEG → 语义 token → MaskGCT/SoundStorm 声学补全 → codec

```
EEG ─► Encoder ─► HuBERT 语义 token ─► MaskGCT/SoundStorm 掩码并行声学 token ─► DAC ─► wav
```
- 参考：MaskGCT、SoundStorm、AudioLM、SpeechTokenizer、VALL-E/VALL-E2。
- 适合：KaraOne 比 FEIS 更可行（有真音频对齐声学阶段），但**声学阶段仍数据饥渴**→建议先用大语料预训练声学阶段再接 EEG。难度：高。

### 方案 C：因子化 content/prosody/voice → FACodec

```
EEG ─► 共享 Encoder ─┬─ content: HuBERT/CTC
                     ├─ prosody: F0/energy/duration(KaraOne 有真音频可监督)
                     └─ voice  : 固定 voice prior(数据级常量)
                                    └─► NaturalSpeech3 FACodec ─► wav
```
- 参考：NaturalSpeech 3 / FACodec、StyleTTS 2。
- 适合：把"EEG 携带 content 还是只携带 prosody/energy 包络"做成消融贡献（结合你们已观察到的"包络层面有信号"）。

> **推荐主路线：方案 A**（NeuroSonic 同源、最不数据饥渴、保住现有真信号）。Plan C 作机制消融。

---

## 6. 重新设计 Loss

| Loss | 保留/删除/新增 | 初始权重 | 仅 KaraOne? | 备注 |
|---|---|---|---|---|
| DTW 对齐重建(`dtw_recon_loss`) | **保留(主)** | 1.0 | 适合 KaraOne | 治跨 trial 起始/语速抖动；目标换语义后继续用 |
| EEG↔音频对称 InfoNCE(`clip_alignment`) | **保留(主)** | 0.5 | — | Défossez 2022；KaraOne 已有，FEIS 缺 |
| HuBERT/wav2vec2 特征回归(SmoothL1+cos) | **新增(主)** | 1.0 | — | 替 mel/latent 当主目标 |
| **CTC loss** | **新增** | 0.3 | **仅 KaraOne** | 有音素标签+真音频→对齐无关的内容监督(ASR 式) |
| Whisper/HuBERT 特征匹配(感知损失) | 新增 | 0.2 | — | 渲染音频 vs 真音频在 SSL 空间比，治平滑 |
| 多分辨率 STFT loss | 新增 | 0.3 | — | 治过平滑/能量塌缩 |
| **conditional flow matching** | 新增(替 DDPM) | 1.0(开生成时) | — | NeuroSonic/Voicebox/CoMoSpeech |
| content CE / supcon / proto | 保留 | 现值 | — | 音素内容监督 |
| log-RMS + **能量/包络相关损失** | 保留+强化 | 0.2→0.3 | — | 直接对治"能量塌缩" |
| std_match / GAN | 保留为可选 | gan 0.1(可选) | — | 换 flow 后 GAN 非必须 |
| recon_cos on EnCodec latent | **删/弃** | 0 | — | 你们实测会塌缩 |

- 易塌缩：EnCodec latent 纯 cosine 回归（实测）；监控 `pred_std_ratio_median→1`、`pred_pairwise_corr↓`、
  `pred_over_mean_gain>0` 三判据（eval.py 已有）。

---

## 7. 训练策略（KaraOne 重点）

**必须加：**
- **跨被试域自适应**（修 subject_test gain≈0）：每被试统计对齐(z-score/CORAL/AdaBN)、或对抗式被试不变特征
  （domain-adversarial，**不引入身份监督**，只做不变性）。这是 KaraOne 上 CCF-A 的**头号杠杆**。
- **抗过拟合**（MODEL_TECH 记录 20ep 比 120ep 干净）：early stop、更强 dropout/weight decay、数据增强
  （时间抖动、通道 dropout 已有、同 label mixup）、自监督 EEG 预训练(mask-reconstruct)。
- **CTC 头**（有音素标签 + 真音频）。

**建议加：** 课程学习（先 overt_like 强信号，再 thinking 弱信号）；contrastive EEG-audio 预训练再微调；
轻量 per-subject adapter / LoRA（若可接受少量被试特异性）。

**暂时别加：** 大型 codec AR LM / VALL-E 式（数据饥渴）；叠多层 MoE；强 GAN（小数据不稳）。

---

## 8. 评估指标重新设计

KaraOne **有 trial-synchronous 真音频**，因此比 FEIS 能用更多客观指标：

- **主指标（诚实，保留并强化）**：`pred_over_mean_gain` 与 `pred_over_zero_gain`（双基线缺一不可）、
  within-subject **检索 top-k**、content/CTC/**音素准确率**。
- **有真音频可加**：**Whisper-WER/CER**（可懂度）、HuBERT/WavLM 特征距离、Whisper-embedding 距离、
  **STOI/MCD**（仅在换神经声码器后才有意义；GL 下别报）、F0/energy/**envelope 相关**（直接量化你们已观察到的包络耦合）。
- **必报上界**：oracle-codec / oracle-HuBERT（GT 过同一渲染器的天花板）。
- **必报跨被试**：subject_holdout 全指标（你们已有 split，要把它当主表，而不是附注）。
- **会误导**：GL 下的波形 PCC、EnCodec latent cosine（塌缩时仍高）。

---

## 9. 升级路线（按优先级）

### 第一优先级（不改难达 SOTA）

| 改动 | 收益 | 难度 | 依据/参考 | 代码范围 |
|---|---|---|---|---|
| 跨被试域自适应(AdaBN/CORAL/域对抗) | ★★★★★ | 中 | 域适应；Défossez 跨被试 | `encoder.py`、`data.py`、训练脚本 |
| 主目标换 HuBERT/wav2vec2 特征(+单元) | ★★★★★ | 中 | wav2vec2、AudioLM、Défossez | `targets.py`、`model.py`、`losses.py` |
| 神经声码器替换 GL(冻结 DAC/EnCodec 或语义 vocoder) | ★★★★☆ | 中 | EnCodec、RVQGAN、SpeechTokenizer | `audio_features.py`、`synth.py` |
| 加 Whisper-WER + oracle 上界 + 跨被试主表 | ★★★★ | 低 | Whisper | `eval.py` |

### 第二优先级（建议改）

| 改动 | 收益 | 难度 | 依据 | 代码范围 |
|---|---|---|---|---|
| DDPM→conditional flow matching 解码 | ★★★★ | 中 | NeuroSonic 2026、Voicebox、CoMoSpeech | `diffusion.py`→`flow.py`、`train_karaone_diffusion.py` |
| CTC 音素头 | ★★★★ | 低 | Whisper/ASR | `model.py`、`losses.py` |
| 抗过拟合(增强+自监督 EEG 预训练) | ★★★☆ | 中 | 通用 | `data.py`、新预训练脚本 |
| 能量/包络相关损失 + 多分辨率 STFT | ★★★☆ | 低 | Voicebox/StyleTTS2 | `losses.py` |

### 第三优先级（锦上添花）

- FACodec 因子化分析(Plan C)；UTMOS/DNSMOS；per-subject LoRA；系统化消融大表（mel vs HuBERT、reg vs flow、moe vs baseline、dtw on/off、域自适应 on/off）。

### ROI Top-10（KaraOne，按性价比排序）

| Rank | 改动 | ROI | 依据论文 | 预计提升 | 代码范围 | 难度 | 风险 |
|---|---|---|---|---|---|---|---|
| 1 | 跨被试域自适应 | ★★★★★ | Défossez 2022/域适应 | subject_test 由 0 转正 | `encoder.py`/`data.py` | 中 | 中 |
| 2 | 主目标→HuBERT/wav2vec2 特征 | ★★★★★ | wav2vec2、AudioLM | 可学性+检索大升 | `targets.py`/`model.py` | 中 | 中(权重) |
| 3 | Whisper-WER + oracle 上界 + 跨被试主表 | ★★★★★ | Whisper | 评估可信、可对标 SOTA | `eval.py` | 低 | 低 |
| 4 | 神经声码器替 GL | ★★★★ | EnCodec/RVQGAN | 音质+可报 STOI/MCD/UTMOS | `audio_features.py`/`synth.py` | 中 | 中 |
| 5 | conditional flow matching 解码 | ★★★★ | NeuroSonic/Voicebox | 抗塌缩+保真 | `flow.py` | 中 | 中 |
| 6 | CTC 音素头 | ★★★☆ | ASR/Whisper | 内容准确率提升 | `model.py`/`losses.py` | 低 | 低 |
| 7 | 抗过拟合(增强+自监督预训练) | ★★★☆ | 通用 | 减小 train/test 差 | `data.py`/预训练 | 中 | 低 |
| 8 | 能量/包络相关 + 多分辨率 STFT 损失 | ★★★ | Voicebox/StyleTTS2 | 治能量塌缩 | `losses.py` | 低 | 低 |
| 9 | 系统化消融大表 | ★★★ | — | 投稿必需 | 脚本/`eval.py` | 低 | 低 |
| 10 | FACodec 因子化分析 | ★★☆ | NaturalSpeech 3 | 机制贡献 | 新模块 | 高 | 中 |

---

## 10. 数据规模下哪些不现实（KaraOne）

- 从 EEG 直接预测多码本 RVQ token 的 **codec AR LM(VALL-E/UniAudio)**：**不现实**（数据量差几个量级）。
- "逐词高可懂度重建"：**短期不现实**。音频又轻又短(~12% 有声)、同音素跨 trial 不对齐、为分类而非重建采集 →
  天花板在**包络/音素级 + within-subject 检索**。
- 因此论文叙事应是："**在为分类设计的 KaraOne 上，建立诚实(双基线+gain)、可跨被试(域自适应)的 EEG→语义→神经渲染管线**，
  并量化 EEG 能携带的 content/prosody 成分"——而非"我们重建了可懂语音"。
- **真正的天花板杠杆在数据**（为重建设计、说满整段、有对齐音频）；模型侧的 ROI 在上表前 5 项。

---

## 附录 A：本 bundle 的 WAV 对比图路径

含 `*_waveform.png`、`waveform_compare_manifest.csv`、`original_vs_pred_scaled_contact_sheet.png`：

```
artifacts/outputs_karaone/karaone_moe_overt_like_20260625_001741/wav_test_20260625_051705/waveform_compare
```
绘图 notebook：`plot_waveform_comparisons_karaone.ipynb`。

## 附录 B：自查 prompt

见 [`../SELF_CHECK_PROMPT.md`](../SELF_CHECK_PROMPT.md)（KaraOne 版自查清单在文末"KaraOne 专用"一节）。
