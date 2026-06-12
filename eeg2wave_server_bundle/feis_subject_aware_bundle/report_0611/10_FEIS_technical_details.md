# FEIS EEG-to-Speech 技术细节说明

日期：2026-06-11  
对应主文档：`report_0611/demo_FEIS_summary.md`  
用途：补充 demo 背后的实现细节、脚本分工、关键参数、设计依据和当前结论。

---

## 1. 目标与路线

当前 demo 的核心路线是：

```text
EEG
-> speech/content representation
-> EnCodec latent
-> frozen EnCodec decoder
-> wav
```

这条路线替代了最早的直接波形回归：

```text
EEG -> waveform
```

替代原因很直接：早期 `EEG -> waveform` 使用 VQ-VAE 风格模型，直接输出 1 秒 wav。模型结构是 1D CNN EEG encoder、EMA-VQ codebook、ConvTranspose waveform decoder，训练损失包含 L1、multi-resolution STFT、log-STFT、RMS、envelope、VQ commitment 和 prompt classification CE。结果输出低频、平均化、不可辨语音，说明在 FEIS 的 EEG content signal 很弱时，直接回归 16000 个 waveform 点容易收敛到平均波形。

所以当前 demo 不让 EEG 直接生成原始采样点，而是预测更稳定的 decoder-compatible 表征：EnCodec latent。这样模型只需要学：

```text
EEG -> latent
```

而声音合成由预训练声码器负责：

```text
latent -> wav
```

好处是把问题拆开：如果 `target_oracle` 能还原出接近原始 wav 的声音，就说明 codec path 是健康的；如果 predicted wav 不好，问题主要在 `EEG -> latent` 预测，而不是 decoder。

---

## 2. FEIS 数据结构

FEIS 在当前 bundle 里被整理成一个 subject-aware 网格：

```text
subject x label x stage x repetition
```

关键事实：

| 项目 | 当前设定 |
|---|---|
| EEG 通道 | 14 |
| EEG 采样率 | 256 Hz |
| 阶段窗口 | 5 秒 |
| EEG 输入长度 | `1280 = 256 x 5` |
| prompt 数量 | 16 |
| 主实验 subject | 20 个，排除 anomalous subject 05 |
| 每个 prompt repetition | 约 10 次 |
| 音频目标 | 每个 `(subject, label)` 一条 subject-specific wav |

16 个 label 是：

```text
p, t, k, f, s, sh, v, z, zh, m, n, ng, fleece, goose, trap, thought
```

也就是 12 个辅音音素和 4 个代表性元音词。

FEIS 的关键限制是：同一个 `(subject, label)` 下，多个 EEG trial 共享同一条音频模板。因此模型不能学习 trial-level pronunciation variation，只能学习 subject-content cell 的 canonical target。

---

## 3. Target Cache 是什么

当前主线使用：

```text
artifacts/audio_targets/feis_subject_templates_encodec_latents.npz
```

这个 cache 由 `scripts/extract_audio_targets.py` 生成。它把每个 subject-specific wav 通过 EnCodec 编码成 latent。

主要字段：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `template_ids` | `[336]` | 每个模板的 id，通常对应 `subject:label` |
| `subject_ids` | `[336]` | subject id |
| `labels` | `[336]` | 16 类 prompt label |
| `audio_paths` | `[336]` | 原始 wav 路径 |
| `target_sequences` | `[336, 75, 128]` | EnCodec latent sequence |
| `target_mean` | `[128]` | latent 标准化均值 |
| `target_std` | `[128]` | latent 标准化标准差 |
| `decoder_scales` | `[336, 1]` | EnCodec decode 时的 scale |

这里的 `336` 来自原始 FEIS 的 21 个 subject 和 16 个 label：

```text
21 x 16 = 336
```

但 factored 主实验会排除 subject 05，所以训练/测试通常使用 20 个 subject。

为什么要保存 `target_mean/std`：模型训练时预测的是 normalized latent，数值范围更稳定；解码前再 denormalize 回 EnCodec 原始 latent 空间。

---

## 4. EnCodec Latent 与 Frozen Decoder

当前使用的是 Meta/Facebook 的 EnCodec 24 kHz 模型：

```text
../models/encodec_24khz
```

代码里通过 `transformers.EncodecModel.from_pretrained(..., local_files_only=True)` 加载。

提取 target latent 时：

```text
原始 wav
-> resample 到 EnCodec 24 kHz
-> EnCodec encode
-> audio_codes
-> quantizer.decode(audio_codes)
-> continuous latent [75,128]
```

注意这里 cache 的不是离散 token id，而是 decoder-ready continuous quantized latent。这样 v1/v2 不需要做 codebook 分类，直接预测连续 latent。

解码时：

```text
latent [75,128]
-> transpose 成 [1,128,75]
-> EnCodec decoder
-> wav [24000]
```

因为 EnCodec 是 24 kHz，1 秒输出是 24000 个采样点。为了和 FEIS 16 kHz 的参考音频及旧指标对齐，某些评估会把 24 kHz 输出 resample 到 16 kHz。

为什么 decoder 要 frozen：EnCodec decoder 已经在大量音频上预训练过，知道如何从 latent 合成自然声音。FEIS 很小，如果一起训练 decoder，容易把声码器能力训练坏。因此训练只更新 EEG encoder、content head、speaker embedding、generator 和 RMS head，decoder 只负责固定解码。

---

## 5. 当前模型结构

当前核心模型在：

```text
app/src/feis_factored/model.py
app/src/feis_factored/encoder.py
```

整体结构：

```text
EEG [B,14,1280]
-> SpatialTemporalEEGEncoder
-> seq [B,75,256]
-> content_seq / content_embed / content_logits

subject_idx [B]
-> speaker_embedding [B,64]

content_seq [B,75,128] + speaker_embedding [B,75,64]
-> generator
-> predicted EnCodec latent [B,75,128]
-> frozen EnCodec decoder
-> wav
```

其中当前关键超参数来自 `configs/factored.yaml`：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `n_channels_eeg` | 14 | FEIS EEG 通道数 |
| `d_model` | 256 | EEG encoder 内部通道维度 |
| `cond_dim` | 32 | stage FiLM 条件维度 |
| `content_dim` | 128 | content 表征维度 |
| `speaker_dim` | 64 | 每个 subject 的 speaker embedding 维度 |
| `num_blocks` | 5 | temporal trunk 的 residual block 数 |
| `kernel_size` | 5 | temporal conv 卷积核 |
| `channel_dropout` | 0.2 | 训练时随机丢 EEG 通道，增强鲁棒性 |
| `dropout` | 0.2 | 常规 dropout |
| `adv_lambda` | 1.0 | gradient reversal 强度 |

---

## 6. SpatialTemporalEEGEncoder

`SpatialTemporalEEGEncoder` 是自定义 CNN/TCN 风格 EEG encoder，不是 Transformer。

结构：

```text
EEG [B,14,1280]
-> SpatialAdapter
-> TemporalTrunk
-> [B,256,75]
```

### 6.1 SpatialAdapter

代码位置：

```text
app/src/feis_factored/encoder.py
```

做法：

```text
Conv1d(14 -> 256, kernel_size=1)
-> GroupNorm
-> GELU
-> FiLM
```

输入输出：

```text
[B,14,1280] -> [B,256,1280]
```

为什么这样设：EEG 是多通道信号，14 个电极不是孤立的。`1x1 Conv1d` 在时间长度不变的情况下学习通道组合，相当于轻量空间滤波器。相比直接 flatten，它保留时间结构；相比复杂图网络，它更简单、数据需求更低。

`channel_dropout=0.2` 的作用是训练时随机丢整条电极通道，防止模型过度依赖某一个通道。FEIS 通道少、噪声大，这个正则很重要。

### 6.2 TemporalTrunk

做法：

```text
5 个 ResidualTemporalBlock
每个 block: dilated Conv1d + GroupNorm + GELU + Dropout + Conv1d + FiLM + residual
最后 adaptive_avg_pool1d 到 75 帧
```

输入输出：

```text
[B,256,1280] -> [B,256,75]
```

为什么用 dilated conv：EEG 内容相关信号可能分布在几百毫秒到几秒范围内。普通小卷积只能看局部，dilated conv 可以扩大时间感受野，同时参数量不大。

为什么用 residual：网络更深时训练更稳定，避免梯度退化。

为什么最后是 75 帧：EnCodec 1 秒 latent 是 `[75,128]`，所以 EEG temporal feature 也对齐到 75 个时间步，方便后续 frame-wise 生成 latent。

### 6.3 FiLM 条件调制

FiLM 形式是：

```text
h <- gamma(cond) * h + beta(cond)
```

当前 factored 版本里，cond 来自 stage embedding：

```text
stage_idx -> stage_embedding [B,32]
```

stage 包括：

```text
stimuli, thinking
```

为什么用 stage 条件：听到声音和想象声音不是同一种脑状态，EEG 分布不同。FiLM 让同一个 encoder 在不同阶段下用不同方式解释 EEG。

---

## 7. Content Branch

EEG encoder 输出：

```text
seq [B,75,256]
pooled = mean(seq, time) -> [B,256]
```

然后分成三条 content 输出。

### 7.1 content_seq

代码：

```text
LayerNorm(256)
-> Linear(256 -> 256)
-> GELU
-> Dropout(0.2)
-> Linear(256 -> 128)
```

输入输出：

```text
[B,75,256] -> [B,75,128]
```

作用：保留时间帧结构，作为 generator 的主要输入。它表示 EEG 解出来的“语音内容序列”。

### 7.2 content_embed

代码：

```text
pooled [B,256]
-> LayerNorm
-> Linear(256 -> 128)
```

输出：

```text
[B,128]
```

作用：作为整体内容向量，用于 supervised contrastive loss、content prototype matching 和 evaluation 中的 within-subject retrieval。

### 7.3 content_logits

代码：

```text
pooled [B,256]
-> LayerNorm
-> Linear(256 -> 256)
-> GELU
-> Linear(256 -> 16)
```

输出：

```text
[B,16]
```

作用：辅助预测 16 个 FEIS prompt。它不是最终生成路径，但可以约束 content 表征显式包含“说什么”。

---

## 8. Speaker Branch

输入是 subject index：

```text
subject_idx [B]
```

当前主实验 20 个真实 subject，加 1 个 unknown subject：

```text
speaker_embedding_table [21,64]
```

查表后：

```text
speaker [B,64]
```

为什么用查表：FEIS 的目标是 subject-specific wav。也就是说，重建目标不仅包括“说什么”，还包括“谁在说”。如果要求 EEG 同时解出 content 和 speaker，任务会混在一起。当前 demo 把 speaker 作为已知条件输入，EEG 只负责 content。

好处是明确分工：

```text
EEG -> content
subject_id -> speaker/voice
content + speaker -> subject-specific speech latent
```

这也让评估更诚实：如果生成声音像某个 subject，不代表 EEG 解出了内容，因为 speaker 是条件输入给的。

speaker 还会经过：

```text
speaker_to_proto: Linear(64 -> 128)
```

用于和 audio-derived speaker prototype 对齐，帮助 speaker embedding 学到更接近目标音频空间的说话人特征。

---

## 9. Generator

generator 接收：

```text
content_seq [B,75,128]
speaker [B,64]
```

speaker 会复制到每个时间帧：

```text
[B,64] -> [B,75,64]
```

拼接后：

```text
[B,75,128] + [B,75,64] -> [B,75,192]
```

generator 是 3 层 Linear MLP：

```text
Linear(192 -> 256)
-> GELU
-> Dropout(0.2)
-> Linear(256 -> 256)
-> GELU
-> Linear(256 -> 128)
```

输出：

```text
pred_latent [B,75,128]
```

为什么用 MLP：到这一步，时间帧已经由 EEG encoder 对齐到 75 帧；每一帧只需要把“内容特征 + 说话人特征”映射到 EnCodec latent frame。MLP 简单、稳定，参数量可控，适合小数据 demo。

局限是它没有额外建模 latent 帧之间的生成依赖。当前更重视验证 pipeline 和信号可解性，不优先堆复杂 decoder。

---

## 10. RMS Head

v2 增加了 loudness / RMS head：

```text
content_pooled [B,128] + speaker [B,64]
-> [B,192]
-> MLP
-> pred_log_rms [B]
```

为什么加：早期 EnCodec latent prediction 即使能 decode，也可能输出音量很小的 wav。RMS head 用模型预测的 log-RMS 对 decoded wav 做缩放，不使用目标音频的真实 RMS，避免 target leak。

好处是把“有没有声音/多大声”与“内容结构是否正确”区分开。v2 结果里 `pred_scaled/ref RMS ratio≈0.97`，说明响度问题基本可修；但谱图仍不对，说明瓶颈不是单纯小声，而是 content latent 预测不准。

---

## 11. 训练目标

训练目标在：

```text
app/src/feis_factored/losses.py
```

整体 loss：

```text
total =
  lambda_recon_cos * recon_cos
+ lambda_recon_mse * recon_mse
+ lambda_supcon * supervised_contrastive
+ lambda_content_ce * content_ce
+ lambda_proto * proto_cos
+ lambda_speaker * speaker_loss
+ lambda_adv * adversarial_subject_ce
+ lambda_log_rms * log_rms_loss
+ lambda_std * std_match
```

当前配置：

| loss | 权重 | 作用 |
|---|---:|---|
| `lambda_recon_cos` | 1.0 | 让 predicted latent 和 target latent 方向一致 |
| `lambda_recon_mse` | 0.25 | 控制数值接近，但下调以减少均值化 |
| `lambda_supcon` | 1.0 | 同 label 样本拉近，不同 label 推远 |
| `lambda_content_ce` | 0.5 | 16 类内容分类约束 |
| `lambda_proto` | 0.5 | content_embed 对齐 speaker-independent content prototype |
| `lambda_speaker` | 0.5 | speaker embedding 对齐 audio speaker prototype |
| `lambda_adv` | 0.3 | content 表征不要编码 subject identity |
| `lambda_log_rms` | 0.2 | 预测响度 |
| `lambda_std` | 0.0 | 可选反塌缩，默认关闭 |

### 11.1 Latent reconstruction

包括 cosine 和 MSE：

```text
pred_latent [B,75,128]
target_seq [B,75,128]
```

cosine 关注方向/结构，MSE 关注数值。MSE 从 1.0 降到 0.25，是因为强 MSE 在弱信号下容易鼓励模型输出均值 latent。

### 11.2 Supervised contrastive

同一个 label 的样本作为 positive：

```text
same label -> pull together
different label -> push apart
```

依据是 FEIS 中多个 subject 的同一 label 共享 content 语义，但音频模板不同。SupCon 可以帮助 content_embed 更关注“说什么”，而不是“谁在说”。

### 11.3 Content CE

16-way classification：

```text
content_logits [B,16] vs label_idx [B]
```

好处是有一个可读指标：content_acc。它不是最终成功标准，但能辅助训练和诊断。

### 11.4 Prototype matching

`FactoredTargets` 会从 cache 里构造：

```text
content prototype = 同 label 跨 subject 的平均 summary
speaker prototype = 同 subject 跨 label 的平均 summary
```

content_proto 让 content_embed 靠近“去说话人化”的 label 原型；speaker_proto 让 speaker embedding 对齐“去内容化”的 subject 声音原型。

### 11.5 Adversarial subject loss

模型有一个 subject adversary：

```text
content_embed -> Gradient Reversal -> subject classifier
```

前向看起来是预测 subject，反向时梯度反转，逼 content_embed 去掉 subject 信息。

为什么要这样：早期 subject-aware 成绩容易被 subject identity 抬高。这个对抗项是为了减少 content 分支偷学“谁在说”。

### 11.6 log-RMS loss

监督：

```text
pred_log_rms [B] vs target_log_rms [B]
```

作用是修复 predicted wav 太小声的问题。

---

## 12. Dataset Split 与 Hold-out Cell

代码：

```text
app/src/feis_factored/data.py
```

FEIS 被看成：

```text
subject x label x stage x repetition
```

一个 cell 是：

```text
(subject, label)
```

当前 split：

| split | 含义 |
|---|---|
| `train` | seen cells 的前面 repetition |
| `val_seen` | seen cells 的倒数第 2 个 repetition |
| `test_seen` | seen cells 的最后 1 个 repetition |
| `test_holdout` | held-out cells 的所有 repetition |

Hold-out cell 默认用 Latin-square：

```text
subject i -> label (i + offset) % 16
```

为什么这样：每个 subject 都留出一个 label 组合，同时每个 subject 和每个 label 仍然会在训练集中以其他组合出现。这样测试的是：

```text
已知 subject voice + 已知 label content
-> 未见过的 subject-label 组合
```

这比普通随机划分更严格，因为它避免模型直接记住某个 `(subject,label)` 模板。

`holdout_random=True` 是进一步防止 deterministic Latin-square 被 zero-EEG 或先验策略利用。

---

## 13. Evaluation 逻辑

评估代码：

```text
app/src/feis_factored/eval.py
```

核心指标不是普通 top1，而是：

```text
content_gain = within_subject_content_top1 - within_subject_content_top1_zeroeeg
```

### 13.1 within_subject_content_top1

对于每个样本，只在该 subject 自己的 16 个 label target summary 里检索：

```text
pred content_embed
vs
该 subject 的 16 个 label prototype
```

chance 是：

```text
1 / 16 = 0.0625
```

为什么用 within-subject：跨 subject 的音频差异很大，如果不限制 subject，模型可能靠“是谁”而不是“说什么”拿分。

### 13.2 zero-EEG control

把 EEG 输入置零：

```text
model(torch.zeros_like(eeg), subject_idx, stage_idx)
```

如果真实 EEG 没有超过 zero-EEG，说明成绩不是来自 EEG 内容信息，而是来自 subject/stage/template 先验。

所以最终看：

```text
content_gain > 0
```

而不是只看 top1。

### 13.3 Coarse metrics

除了 16 类内容，还看三种粗分类：

| 指标 | 含义 |
|---|---|
| `manner` | 发音方式：plosive/fricative/nasal/vowel |
| `voicing` | 清浊：voiced/voiceless |
| `vowel/consonant` | 元音/辅音 |

如果 16 类解不出，但 coarse 能解出，说明 EEG 里可能仍有粗粒度语音信息。当前结果里 coarse 也低于 majority baseline，因此负结果更强。

### 13.4 Collapse diagnostics

评估还记录：

| 指标 | 作用 |
|---|---|
| `pred_std_ratio_median` | predicted latent 的方差是否接近 target |
| `pred_pairwise_corr_median` | 不同样本预测是否过于相似 |
| `recon_cos_to_cell` | predicted latent 与 cell target 的 cosine |

这些是诊断指标，不是最终成功标准。因为 latent 看起来接近，不一定代表内容正确。

---

## 14. Audio QC 五路 wav

脚本：

```text
scripts/factored_recon_eval.py
scripts/factored_synthesize.py
```

每个样本生成五路音频：

| wav | 含义 | 作用 |
|---|---|---|
| `original_ref` | 原始参考音频 | 目标声音 |
| `target_oracle` | 真实 target latent 解码 | codec 上限对照 |
| `mean_latent` | 全局平均 latent 解码 | 均值塌缩下限 |
| `pred_unscaled` | predicted latent 直接解码 | 原始模型输出 |
| `pred_scaled` | predicted wav 按模型预测 RMS 缩放 | 检查响度修复后是否内容正确 |

`target_oracle` 的具体过程：

```text
original_ref wav
-> EnCodec encoder
-> target latent [75,128]
-> EnCodec decoder
-> target_oracle wav
```

如果 `target_oracle` 接近 `original_ref`，说明 EnCodec path 健康。如果 `pred_scaled` 仍不像目标，说明问题在 EEG-to-latent prediction。

---

## 15. Content Probe

脚本：

```text
scripts/content_probe.py
```

目的：不依赖生成模型，直接问 EEG 里有没有 content signal。

特征构造：

```text
EEG [14,1280]
-> 每通道 log variance
-> 每通道 log mean absolute diff
-> 每通道 FFT band power，默认 5 bands
-> 特征维度 = 14 x (2 + 5) = 98
```

分类器：

```text
closed-form ridge one-vs-all classifier
5-fold cross-validation
200 permutation test
```

为什么用简单 probe：如果一个干净、透明的线性 probe 都不能从 EEG 解出 content，复杂生成模型也很难凭空制造出内容信息。

`n=3152` 的来源：

```text
20 subject x 16 label x 10 repetition = 3200
subject 12 少 48 trial
3200 - 48 = 3152
```

因此 `stimuli` 和 `thinking` 每个阶段都有 3152 个可用 EEG trial。

当前结果：

```text
content probe:
stimuli top1=0.0457 < chance=0.0625, p=0.920
thinking top1=0.0466 < chance=0.0625, p=0.896
```

说明 16 类 content 没有显著可解。

positive control：

```text
subject probe:
stimuli top1=0.7846, chance=0.0500, p=0.005
thinking top1=0.8115, chance=0.0500, p=0.005
```

说明同一套 EEG 特征和 probe 管线能解出 subject identity。也就是说 content null 更像是真负结果，不是代码或评估坏了。

---

## 16. 主线脚本说明

### 16.1 `scripts/extract_audio_targets.py`

作用：从 FEIS subject-specific wav 提取音频 target cache。

典型输入：

```text
configs/alignment_encodec_local.yaml
```

典型输出：

```text
artifacts/audio_targets/feis_subject_templates_encodec_latents.npz
```

为什么需要：训练不能每个 epoch 都跑 EnCodec encoder，太慢；提前缓存 latent 可以保证训练稳定可复现。

### 16.2 `scripts/factored_train.py`

作用：训练当前 factored model。

流程：

```text
加载 factored.yaml
-> 加载 FactoredTargets
-> 构造 train/val_seen/test_seen/test_holdout
-> 建立 FactoredEEG2Speech
-> AdamW + CosineAnnealingLR
-> 每 epoch 用 val content_gain 选 best checkpoint
-> 输出 test metrics
```

为什么用 AdamW：带 weight decay 的 Adam 对小数据神经网络更稳。  
为什么用 cosine scheduler：让学习率平滑下降，减少后期震荡。  
为什么用 gradient clip=1.0：EEG 小数据、多损失项训练容易梯度不稳，clip 提高稳定性。  
为什么用 `val content_gain` 选模型：避免选择一个只靠 subject/stage 先验表现好的 checkpoint。

### 16.3 `scripts/factored_recon_eval.py`

作用：做 reconstruction QC 和 collapse diagnostics。

输出包括：

```text
audio_qc.json
collapse_diagnostics.json
recon_eval.json
recon_pairs.csv
listening_manifest.csv
saved wavs
```

为什么要做：生成 wav 可能“听起来有声音”，但不代表内容正确。五路 wav 对照可以拆开判断 codec 健康、音量问题、均值塌缩和 EEG 预测质量。

### 16.4 `scripts/factored_synthesize.py`

作用：批量合成 wav，便于听感检查。

它保存同样的五路 wav：

```text
original_ref / target_oracle / mean_latent / pred_unscaled / pred_scaled
```

适合汇报和人工听感筛查。

### 16.5 `scripts/content_probe.py`

作用：Stage-1 decodability gate。它回答“EEG 里到底有没有 content signal”。

为什么重要：这是独立于生成模型的 sanity check。当前结果显示 content 不显著，但 subject positive control 显著，因此结论是 FEIS content signal 不稳定，而不是 pipeline 无效。

### 16.6 `scripts/factored_interpolate.py`

作用：做 speaker interpolation demo。

用途：检查 speaker embedding 是否形成可控的 voice axis。它不是主指标，但可以展示 subject embedding 的连续性。

---

## 17. Legacy / Phase 2 脚本说明

这些脚本不是当前 factored demo 主线，但保留为对照和阶段性实验记录。

### 17.1 `scripts/train_waveform_protocol.py`

作用：训练早期 raw waveform baseline。

模型：

```text
EEG2WaveVQModel 或 SubjectConditionedEEG2WaveVQModel
```

结构：

```text
1D CNN EEG encoder
-> VectorQuantizerEMA
-> ConvTranspose waveform decoder
-> wav
```

损失：

```text
L1 + STFT + log-STFT + RMS + envelope + VQ + class CE
```

当前地位：失败比较基线。它说明直接 `EEG -> waveform` 在 FEIS 上不稳定。

### 17.2 `scripts/eval_waveform_protocol.py`

作用：评估 raw waveform baseline，保存重建 wav，并计算 waveform-side 指标，如 STFT distance、nearest-template accuracy 等。

当前地位：只用于说明旧路线失败，不再优化。

### 17.3 `scripts/train_alignment.py`

作用：训练 Phase 2 的 EEG-to-speech-representation alignment 模型。

支持 target：

```text
pooled HuBERT
sequence HuBERT
EnCodec latent
```

训练目标包括：

```text
sequence cosine / MSE
summary contrastive InfoNCE
label CE
phoneme auxiliary
codec scale loss
```

当前地位：Phase 2/3 alignment 框架，帮助比较 pooled HuBERT、sequence HuBERT、codec latent。最终主线转向 factored EnCodec latent reconstruction。

### 17.4 `scripts/eval_alignment.py`

作用：评估 alignment checkpoint。

如果是 HuBERT sequence target，主要做 retrieval：

```text
EEG -> predicted sequence -> nearest template retrieval -> wav
```

如果是 EnCodec target，做：

```text
EEG -> predicted codec latent -> frozen decoder -> wav
```

输出 retrieval metrics、waveform NTA、recon wav 等。

### 17.5 `scripts/eval_alignment_retrieval.py`

作用：兼容入口，实际调用 `eval_alignment.py`。保留是为了旧命令不失效。

### 17.6 `scripts/analyze_alignment_space.py`

作用：分析 template embedding space。

输出：

```text
PCA by subject
PCA by label
within/cross subject-label distance
centroid probe
```

为什么重要：它回答 speech representation space 主要编码 subject、label 还是两者。这个分析支持“FEIS 里 subject 特征强，content 特征弱”的判断。

### 17.7 `scripts/audit_recon_audio.py`

作用：对 recon wav 做批量音频 QC。

指标包括：

```text
RMS
peak
silence/clipping
spectral centroid
low-frequency energy
template entropy / unique top1
mel/spectral summary
```

为什么重要：防止只看 top1 或 cosine，忽略声音实际不可听、过小声或模板塌缩。

### 17.8 `scripts/audit_pipeline.py`

作用：审计早期 waveform/alignment pipeline 的输入输出形状、参数量、checkpoint 加载和基本前向传播。

当前地位：工程检查脚本，用于确认 baseline pipeline 是否能跑通。

### 17.9 `scripts/report_phase2.py`

作用：把 waveform baseline、sequence HuBERT retrieval、EnCodec latent reconstruction 和 space analysis 汇总成 Phase 2 report。

当前地位：阶段报告生成器。当前 demo 主文档已转向 factored v2，但它仍是 Phase 2 实验记录的一部分。

---

## 18. 核心模块说明

### 18.1 `app/src/audio_features.py`

负责音频 target extraction。

主要功能：

```text
HuBERT pooled target
HuBERT sequence target
EnCodec latent target
EnCodec decode
resample/pad/crop
```

当前 demo 主要使用 `_EncodecLatentBackend`。

### 18.2 `app/src/feis_factored/targets.py`

负责从 target cache 构造训练用目标。

关键产物：

```text
normalized target_seq
content prototype
speaker prototype
coarse phonological maps
target RMS / decoder scales
```

为什么需要 prototype：FEIS 目标是 subject-specific，如果只训练 cell latent，模型可能过度记模板；prototype 把 content 和 speaker 两个因素拆出来。

### 18.3 `app/src/feis_factored/data.py`

负责 factored dataset 和 hold-out-cell split。

关键是把每个样本返回为：

```text
eeg [14,1280]
subject_idx
label_idx
stage_idx
target_seq [75,128]
content_proto [128]
speaker_proto [128]
target_log_rms
coarse labels
metadata
```

### 18.4 `app/src/feis_factored/model.py`

负责模型主体：

```text
stage embedding
SpatialTemporalEEGEncoder
content heads
speaker embedding
subject adversary
generator
log RMS head
```

### 18.5 `app/src/feis_factored/losses.py`

负责 factored training objective。核心目标是同时做到：

```text
latent 接近 target
content 可分
speaker 可控
content 不泄漏 subject
响度可预测
避免均值塌缩
```

### 18.6 `app/src/feis_factored/eval.py`

负责 honest evaluation：

```text
within-subject content top1
zero-EEG baseline
content gain
coarse gains
latent collapse diagnostics
```

### 18.7 `app/src/feis_factored/synth.py`

负责：

```text
normalized latent -> denormalized raw EnCodec latent -> frozen EnCodec decoder -> wav
```

---

## 19. 当前结果如何解释

当前结果有三层。

第一层：codec path 健康。

```text
target_oracle ≈ original_ref
```

说明 EnCodec latent 和 frozen decoder 可以还原出合理语音。

第二层：响度问题可修。

```text
pred_scaled/ref RMS ratio ≈ 0.97
```

说明 v2 的 log-RMS head 解决了“太小声”的主要问题。

第三层：content signal 不稳定。

```text
content probe stimuli/thinking < chance
factored content_gain = 0 或负
coarse phonological metrics < majority baseline
subject positive control 显著
```

这说明当前 FEIS EEG 特征中确实有可解的 subject-level signal，但没有稳定可泛化的 16 类 speech content signal。

---

## 20. 为什么当前负结果可信

它不是简单“模型没训好”，原因有四个：

1. **旧 waveform baseline 失败**  
   直接生成波形塌缩，说明原始 waveform target 不适合当前 FEIS 信号强度。

2. **codec oracle 通过**  
   真实 target latent 解码正常，说明 decoder 不是主要问题。

3. **subject positive control 通过**  
   同一套 EEG 特征和 probe 能显著解出 subject identity，说明 EEG pipeline 和 probe 没坏。

4. **zero-EEG 对照没有输给真实 EEG**  
   如果真实 EEG 不能超过 zero-EEG，说明生成模型没有从 EEG 中拿到额外 content 信息。

因此当前更合理的结论是：

```text
FEIS 可以作为方法诊断和诚实负结果基线；
但不适合作为 EEG-to-speech content reconstruction 的主线长训数据集。
```

---

## 21. 推荐后续方向

下一步不建议继续在 FEIS-only 上堆更复杂的 generator。更合理的路线是：

```text
auditory perception EEG / MEG
-> 先验证 EEG -> speech representation
-> 再迁移到 auditory imagery
-> 复用 factored + codec latent + frozen decoder + honest evaluation
```

FEIS 后续适合保留为：

```text
sanity check
negative baseline
subject identity confound analysis
codec path smoke test
```

而不是主要追求更高重建质量的数据集。

---

## 22. 最短技术口径

当前 demo 的技术贡献不是“已经从 FEIS EEG 重建出清晰语音”，而是建立了一条可诊断的路线：

```text
EEG content branch
+ subject speaker branch
-> EnCodec latent
-> frozen EnCodec decoder
-> wav
```

并通过 content probe、subject positive control、zero-EEG baseline 和五路 wav QC 证明：

```text
codec 和工程路径基本可用；
subject 信号可解；
但 FEIS 的 16 类 speech content signal 当前不可稳定解码。
```

