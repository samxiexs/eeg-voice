# 方法说明：声音怎么重建、EEG 和音频怎么对齐、做了哪些优化

本文档面向研究/复现，讲清三件事：
1. **声音是怎么重建的**（pipeline 全链路）
2. **EEG 和音频是怎么对齐的**（这是最关键、也最容易被误解的部分）
3. **参考论文做了哪些优化**，以及它们解决什么问题

> 一句话总览：本方案 = **EEG → EnCodec 连续潜在（回归 + 跨模态对比对齐）→ 冻结的 EnCodec 解码器 → 波形**。模型**完全不使用被试 ID**。

---

## 1. 声音是怎么重建的

分**离线建目标**和**在线生成**两段。

### 1.1 离线：把每段语音压成"目标潜在"
代码：[`scripts/extract_karaone_targets.py`](app/scripts/extract_karaone_targets.py) +
[`src/audio_features.py`](app/src/audio_features.py) 的 `_EncodecLatentBackend`。

1. 读该 trial 的 overt 语音 wav，定长 **2.0s @ 16kHz**，按 RMS 归一化到 0.08。
2. 重采样 16k→24k，pad/crop 到 48000 点（EnCodec 24kHz 输入）。
3. `EncodecModel.encode(...)` 得到离散 RVQ 码 `audio_codes [1,B,Q,T]`
   （bandwidth=6kbps → 约 8 个码本，`T = 75 帧/秒 × 2s = 150` 帧）。
4. **关键一步**：`quantizer.decode(codes)` 把离散码还原成**连续量化潜在**
   （各码本向量求和），得到 `sequence [150, 128]`。
   → 所以**目标不是离散 token，而是 EnCodec 解码器前的连续 latent（150×128）**。
5. 另存响度 `target_log_rms`、EnCodec 的分块归一化系数 `decoder_scales`。
6. [`targets.py`](app/src/karaone_recon/targets.py) 对 latent 按维度做 **z-score 归一化**
   （减 `target_mean`、除 `target_std`）——这才是模型真正回归的目标。

### 1.2 在线：EEG → latent → 波形
代码：[`model.py`](app/src/karaone_recon/model.py) + [`scripts/synthesize_karaone.py`](app/scripts/synthesize_karaone.py)。

```text
EEG → 编码器 → 归一化 latent pred_latent[150,128] + 响度 pred_log_rms
   → 反归一化 (×std + mean)
   → EnCodec decoder（跳过 encoder/quantizer，直接喂 latent）× decoder_scales
   → 24kHz 波形
   → 按 exp(pred_log_rms) 调整音量
```

合成时会同时导出对照音便于"诚实评估"：`original`、`oracle_codec`（目标 latent 直接解码 = 质量上限）、
`mean_latent`（全局均值 = 模糊基线）、`zeroeeg`（EEG 置零）、`pred`、`pred_scaled`。

> 注意：模型**不直接生成波形**，波形保真度上限由 EnCodec 决定（`oracle_codec` 即上限）。

---

## 2. EEG 和音频是怎么对齐的

这是核心。分两层：**配对**和**时间**。

### 2.1 配对层：trial 级（同一次发声）
- 每个 trial 的 EEG 存为 `[62 通道, ~1280 点]`（256Hz 5s 窗，实际发声 ~2.2s，**其余补零**）。
- 音频目标是**同一个 trial 的 overt wav**（2.0s）。配对键 = `subject:trial_index`。
- 因此是**整段 EEG ↔ 整段语音**，粒度是"一次发声"，不是音素、不是采样点。

### 2.2 时间层：靠"自适应池化"对长度 + 有效长度掩码
EEG 是 62×1280（≈5s@256Hz），音频 latent 是 150×128（2s@75Hz），速率/时长都不同。怎么对上：

- 编码器用 strided 卷积下采样时间，最后 `F.adaptive_avg_pool1d(x, 150)` **把 EEG 特征序列重采样到 150 帧**，正好等于音频 latent 帧数；loss 逐帧比对 `pred[150,128]` vs `target[150,128]`。
- **这是一个隐式、均匀的时间对应**（第 i 帧 EEG ↔ 第 i 帧音频），不是 DTW / CTC / cross-attention 的显式对齐。

### 2.3 诚实的局限（务必在论文里写清楚）
1. **没有显式时间对齐**：纯靠均匀池化把时间轴对上。
2. **逐帧 MSE/cos 求条件均值** → 倾向输出"闷糊的平均声"（参见 `mean_latent` 基线）。
3. （已部分修复，见 §3.2）EEG 末尾补零段曾被平均进句向量。

---

## 3. 参考论文做的优化（保持简单）

参考 `paper-ref/`，尤其 **Défossez et al. 2022, "Decoding speech perception from
non-invasive brain recordings"**，以及视觉解码主线（NICE/ATM/UBP，见 `deep-research-report.md`）。
两条贯穿性结论：**(a) 跨模态对比对齐优于纯回归；(b) 只用有效时间窗**。据此做了两个**改动小、收益明确**的优化。

### 3.1 跨模态对比对齐损失（CLIP / InfoNCE）—— 治"求均值变模糊"
- **依据**：Défossez 2022 用"脑特征 ↔ 冻结语音特征"的对比损失，并指出**直接回归 mel 效果差、对比对齐显著更好**；NICE/ATM/UBP 同样以对比对齐为核心。
- **做法**（[`losses.py::clip_alignment`](app/src/karaone_recon/losses.py)）：模型新增 `clip_head` 把 EEG 句向量投到音频潜在空间（[`model.py`](app/src/karaone_recon/model.py)），与该 trial 的**音频 latent 摘要**（`target_seq.mean(time)`，音频侧冻结）做**对称 InfoNCE**：同 trial 为正样本，batch 内其它 trial 为负样本。
- **作用**：在逐帧回归之外，逼模型学"EEG ↔ 语音"的对应关系，缓解 §2.3 的均值化。
- **开关**：`lambda_clip`（默认 0.5，设 0 关闭）、`clip_temperature`（0.07）。

### 3.2 有效长度掩码池化 —— 只用"有效时间窗"
- **依据**：视觉解码反复指出 EEG 信息集中在有效时窗；补零段不该当真实信号。
- **做法**（[`model.py::_masked_time_mean`](app/src/karaone_recon/model.py)）：数据集本就算了 `eeg_valid_len` 但此前**未被使用**。现在把有效比例映射到编码器输出帧（补零在末尾），**句向量只对有效帧求平均**。无有效长度时退化为普通均值（向后兼容）。
- **作用**：句向量（喂给 content/voice/CLIP 头）不再被补零稀释。

### 3.3 同时还做了（前序工作）
- **EEG encoder 内的通道选择/聚类 MoE**（`ChannelMoEFrontend`，`--model moe`）：每通道门控 + 把相似通道软聚类到专家。对应"通道不是每个都有用"。
- **去除被试 ID**：删掉 `subject_condition`/`speaker_embedding`/`subject_classifier` 及所有 ID 监督；voice 改由 EEG（`global_head(pooled)`）推断。输出对 subject_idx **零依赖**（已验证 max_diff=0）。
- **澄清 refiner 不是 diffusion**：它是单步残差后处理，文档已写明真正 diffusion 需要什么。

---

## 4. 没有做什么（避免过度复杂）
- 没有引入真正的 diffusion 采样器（见 `refiner.py` 注释中关于"为什么"和"怎么做"）。
- 没有做显式时间对齐（DTW/CTC/cross-attention）——留作后续；当前用"均匀池化 + 有效长度掩码"。
- 没有引入额外的冻结语音大模型（wav2vec2/HuBERT）；对比对齐直接用现成的 EnCodec 目标潜在作为音频侧表征。

## 5. 怎么验证优化有效
对比 `--model baseline` 与 `--model moe`、以及 `lambda_clip=0` 与 `lambda_clip=0.5`，
**只看 `pred_over_zero_cos_gain`**（不是原始 cosine）。具体命令见 [RUN_SERVER.md](RUN_SERVER.md)。

## 参考文献（见 `paper-ref/`）
- Défossez et al. 2022, *Decoding speech perception from non-invasive brain recordings* — 跨模态对比对齐。
- Lee et al. 2023, *Towards Voice Reconstruction from EEG during Imagined Speech* — 任务设定参考。
- `paper-ref/deep-research-report.md` — EEG 解码/重建主线综述（对比对齐、有效时窗、跨被试难点）。
