# KaraOne v1-v8 EEG-to-Speech 系统性重构方案



> 日期：2026-06-30
> 范围：`karaone_overt_recon_bundle`、历史输出目录、`karaone_semantic_first_model_tech_zh.md`、本地 `paper-ref/eeg-to-speech-ccf-a` 文献库、arXiv 检索。
> 立场：v1-v8 的问题不是某个 loss 权重或 decoder 不够强，而是 EEG 表示、语音 latent、跨模态对齐和生成目标之间的结构性错配。后续应重建 pipeline，而不是继续在 v8 上局部修补。

## 1. 关键结论

当前 KaraOne 数据提供了 14 名被试、1913 trials、11 个 prompt、62 通道 EEG、四个 stage。`overt_like` EEG 的有效长度中位数约 575 samples，`thinking` 约 1280 samples；音频时长中位数 1.44s，但训练目标被规整为 2s。这个设置天然带来三个冲突：

1. `overt_like` 与音频不是稳定逐帧同构映射，存在 onset、速度、发音长度和有声段稀疏性差异。
2. `thinking` 与真实 overt audio 只有语义或意图层关联，不能作为 frame-level acoustic reconstruction 任务处理。
3. 跨被试 EEG 分布差异大于当前模型能学到的 speech-bearing invariant signal。

历史指标支持这个判断。早期 mel/semantic-token/codec 版本经常出现 `pred_std_ratio_median` 接近 0、`pred_pairwise_corr_median` 接近 1 的均值或模板塌缩；v5-v7 缓解了塌缩，但 subject-holdout 上仍接近或低于 zero/mean/shuffled baseline。v8 目前是 v7 架构的训练目标修正：复用 v7 model class、v7 feature cache、v7 synthesis，只把 strict InfoNCE 改成 soft-positive 并加强 leakage 抑制；本地技术说明明确写着未执行 smoke run。因此 v8 不是系统重构。

## 2. v1-v8 失败链条

| 版本       | 主思路                                                                      | 本地证据                                                                                                                                  | 核心失败                                                                                                                     |
| ---------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| v1         | EEG -> full Mel/EnCodec residual/global-mean 回归                           | `melres_speaking_v1` subject_test gain 约 +0.009，但 pairwise corr 0.929、std ratio 0.377                                               | 表面超过 mean，实质仍靠均值残差与低方差模板，无法形成 EEG-specific acoustic trajectory。                                     |
| v2         | Mel envelope / loudness repair                                              | `melenv_overt_v2` subject_test gain 约 -0.004、pairwise corr 0.893                                                                      | 只修能量包络，未解决内容空间与跨被试表示；能量监督不能替代语音 latent 对齐。                                                 |
| v3         | alignment-aware Mel residual + HuBERT semantic-token diagnostic             | v3 semantic token val_gain 仍为负；melalign subject_test gain 约 -0.034                                                                   | DTW/lag 能缓解局部错位，但 frame-level Mel 仍是错误主目标；semantic token 小样本训练不稳定。                                 |
| v4         | semantic prototype / label-prototype residual                               | v4 melproto subject_test gain 约 -0.041，content 近随机                                                                                   | prototype 引入 label/template shortcut，改善可视化不等于 EEG-to-speech；未知 EEG 仍无法选择 trial-specific acoustic detail。 |
| v4.2/4.2.1 | speech-core、shift supervision、anti-collapse                               | v421 subject_test gain 约 -0.075、pairwise corr 0.943                                                                                     | active-core 缩小目标后仍是 Mel 形状回归；shift/head 监督不能产生跨被试可迁移语音表示。                                       |
| v5         | temporal-elastic active speech-core                                         | speaking subject_test gain 约 -0.034，active-shape gain 约 -0.068；thinking active-shape gain 约 -0.088                                   | 局部形状和响度更可控，但生成仍靠 active-core prior，EEG 对 subject_test 没有正增益。                                         |
| v6.1       | retrieval-first：EEG embedding -> HuBERT retrieval -> active-core prior     | thinking trial/test HuBERT gain 可正，但 subject_val HuBERT gain 约 -0.047、active gain 约 -0.241；overt smoke 近塌缩                     | 检索把生成转成 train-bank 选择，降低了声学难度，也把成功标准退化为是否找近邻；跨被试依旧不成立。                             |
| v7         | raw EEG + feature + envelope -> subject-robust HuBERT embedding             | overt subject_test HuBERT gain 约 -0.012、same-label cross-subject gain -0.129、leakage 0.406；thinking HuBERT gain -0.053、leakage 0.459 | subject-balanced sampler 和 GRL 不足以消除 subject/session shortcut；句级 HuBERT summary 丢掉时序对齐信息。                  |
| v8         | v7 + soft-positive speech-SSL contrastive + semantic-neighborhood selection | 技术说明写明未执行 smoke run；runner 仍调用 v7 training/synthesis                                                                         | 软正样本修正了“同 label 全当负样本”的问题，但仍没有重定义 EEG latent、speech latent、sequence alignment 和生成结构。       |

## 3. 结构性失败原因

### 3.1 EEG 与语音对齐不是同一时间轴上的回归问题

KaraOne 的 overt audio 是 trial-synchronous，但不是 frame-synchronous。`overt_like` EEG 窗口有效长度差异大，音频有声段短且稀疏；thinking stage 与 overt audio 更只有意图/语义层对应。v3 的 lag/DTW、v5 的 temporal-elastic core 说明团队已经发现错位问题，但旧 pipeline 仍把目标写成“EEG frame -> Mel/active-core frame”。这会把不可辨识的时间自由度推给模型，模型最优解自然是均值、模板或 retrieval prior。

### 3.2 表示空间不匹配

EEG 是低 SNR、空间混合、跨被试非平稳的神经测量；Mel/EnCodec 连续 latent 是高维高频 acoustic trajectory。用 SmoothL1/cosine 直接回归 Mel/codec latent，会鼓励条件均值。HuBERT summary 比 Mel 更可学，但 v7/v8 使用的是句级 summary/retrieval，不是 sequence-level speech representation；它能判断粗 semantic neighborhood，却不足以约束语音时序、duration、prosody 和 acoustic detail。

### 3.3 语义引导被误用为生成入口

v3-v4 的 semantic tokens/prototypes 把语义辅助变成了 prototype/template 路径；v6.1-v8 把生成转成 train-bank retrieval。这样能得到看似合理的 active-core prior，但它无法证明 EEG 携带了 trial-specific speech information。v8 的 soft-positive 方向比 strict InfoNCE 合理，但如果目标只是“进入相似 HuBERT neighborhood”，模型仍可能只学到 prompt/subject/session 相关统计。

### 3.4 跨被试不变性没有被建成核心目标

v7/v8 有 RevIN、feature dropout/noise、subject adversarial head，但本地结果仍显示 subject leakage 高，same-label cross-subject gain 长期为负。手工 feature branch 包含 covariance/logvar/bandpower，天然容易携带 subject identity；把这些 feature 与 raw EEG 句向量拼接后再做 fusion，模型会优先利用 subject-stable shortcut，而不是微弱的 speech-bearing EEG signal。

### 3.5 训练目标彼此拉扯

旧系统同时使用 recon、DTW、content CE、supcon、proto、CTC、HuBERT aux、InfoNCE、VICReg、hard negatives、subject adversarial、energy/loudness、anti-collapse 等目标，但没有明确“哪些变量可由 EEG 预测，哪些变量只能由语音生成先验补全”。在小样本 KaraOne 上，多目标堆叠会产生选择指标漂移：trial split 上变好，subject-holdout 不动；best-shift envelope 变好，但 active-shape gain 低于 mean；content CE 降低，但 waveform/latent 无增益。

## 4. 相关 SOTA 启示

### 4.1 直接 EEG/MEG 到语音或语言解码

- Défossez et al. 的 non-invasive speech perception decoding 证明：对比学习、预训练 speech representations、多被试共同卷积架构比直接声学回归更关键。链接：https://arxiv.org/abs/2208.12266
- NeuroTalk 使用 spoken EEG 训练并迁移到 imagined speech，强调 spoken-to-imagined domain adaptation 和 ASR/phoneme 约束。链接：https://arxiv.org/abs/2301.07173
- EEG-to-Voice 2025 采用 subject-specific generator、pretrained vocoder、ASR/WER/CER 评价，说明 direct EEG-to-Mel 可以做 demo，但它依赖 subject-specific path，不适合作为跨被试 KaraOne 主结论。链接：https://arxiv.org/abs/2512.22146
- NeuroSonic 2026 是当前最直接的 EEG-to-speech 新参考：conditional flow matching、shared EEG/audio token space、time-conditioned gated Transformer、cross-subject evaluation，核心观点是避免 waveform regression 与 stochastic diffusion 对 artifact/subject variability 的敏感性。链接：https://arxiv.org/abs/2606.24087

### 4.2 自监督语音表征与 neural codec

本地 `paper-ref/eeg-to-speech-ccf-a/papers.csv` 的 CCF-A 主线与当前任务高度一致：

- wav2vec 2.0 / HuBERT：适合作为 EEG 对齐的中间语音表征，而不是把 Mel/codec latent 作为第一目标。链接：https://arxiv.org/abs/2006.11477、https://arxiv.org/abs/2106.07447
- Whisper：不宜作为训练唯一目标，但适合作为 reconstructed wav 的 ASR sanity check、CER/WER 与 embedding-level intelligibility metric。链接：https://arxiv.org/abs/2212.04356
- EnCodec / RVQGAN / X-Codec / SpeechTokenizer：适合作为 neural codec backend 或 factorized token target；不应在 KaraOne 小数据上直接回归完整连续 latent。链接：https://arxiv.org/abs/2210.13438、https://arxiv.org/abs/2306.06546、https://arxiv.org/abs/2408.17175
- Voicebox / NaturalSpeech 3 / CoMoSpeech / P-Flow：说明现代 speech generation 已从 Mel+Griffin-Lim 转向 flow/diffusion/consistency 与 factorized codec。KaraOne 应借鉴“content/prosody/timbre/acoustic detail 分解”，而不是让 EEG 预测所有声学细节。链接：https://arxiv.org/abs/2306.15687、https://arxiv.org/abs/2403.03100

### 4.3 EEG foundation model 启示

LaBraM、NeuroLM 等 EEG foundation model 把 EEG 表示学习改成 channel-time patch tokenization、masked neural/spectrum prediction、VQ tokenizer 和大规模预训练。它们不直接解决 speech generation，但提供了关键原则：先学习跨数据集/跨通道/跨任务的 EEG latent，再做下游语音对齐。链接：https://arxiv.org/abs/2405.18765、https://arxiv.org/abs/2409.00101

## 5. 新架构：KaraOne Neural Semantic Transport

新 pipeline 不再定义为 `EEG -> Mel/active-core -> Griffin-Lim`，而定义为：

```text
EEG sequence
  -> subject-robust neural EEG token sequence
  -> speech semantic/prosody latent sequence
  -> conditional transport in factorized codec space
  -> frozen neural codec decoder
  -> waveform
```

### 5.1 EEG 表示空间

建立 `Z_e = {z_t}`，它不是 pooled sentence embedding，也不是 logvar/bandpower/covariance 拼接向量。

1. 输入：62 通道 raw EEG、valid mask、stage embedding、electrode topology。
2. 前端：per-trial robust normalization + channel-time patch embedding + spatial graph/attention encoder。
3. 自监督预训练：masked channel-time reconstruction、masked spectral-token prediction、跨增强一致性、channel dropout、time jitter。
4. 输出：25-50 Hz neural token sequence，包含 content-sensitive stream、prosody/envelope stream、uncertainty stream。
5. 不变性：subject adversarial 只作为辅助；主约束应加入 group-DRO、CORAL/MMD 分布对齐、leave-one-subject validation、feature leakage audit。

### 5.2 语音 latent 目标

语音目标分四层，不再把单一 Mel/codec latent 当主目标：

```text
C: content/phonetic semantic sequence
   HuBERT/wav2vec2 middle-layer features + k-means semantic units + prompt-token CTC

P: prosody and event sequence
   active mask, onset/offset distribution, duration, energy envelope, optional F0 proxy

V: voice/timbre prior
   dataset-level or externally provided neutral voice prior; not from subject id

A: acoustic detail / codec latent
   EnCodec/DAC/SpeechTokenizer/FACodec token or latent, generated by conditional flow/codec decoder
```

EEG 主要预测 `C` 和 `P`，对 `V` 只给弱条件或固定先验，对 `A` 通过 generative prior 补全。这样符合 EEG 的信息带宽，也避免把不可辨识声学细节强行归因给 EEG。

### 5.3 对齐机制

旧的 same-trial InfoNCE 和 v8 soft-positive 只能做句级邻域约束。新方案使用三层对齐：

1. 序列级对齐：monotonic cross-attention / differentiable OT / soft-DTW between `Z_e` and speech SSL sequence，带 learnable lag prior。
2. 语义级对齐：semantic-unit CTC + supervised prompt CE 作为弱辅助；soft positives 来自 speech SSL 相似度，而不是 label prototype。
3. 跨被试对齐：same-label different-subject positives 只用于 invariance；hard negatives 应区分 same-subject/different-label 与 different-subject/same-label，防止模型把 subject 当 content。

thinking stage 只能训练 `EEG_thinking -> C/P`，不能训练 frame-accurate acoustic loss。overt_like 作为正控制学习 `C/P/A` 的完整对齐；thinking 通过 overt-pretrained encoder 和 stage adaptation 迁移。

### 5.4 解码结构

解码器采用 conditional flow matching，而不是 Griffin-Lim、直接 regression 或 train-bank retrieval：

```text
predicted C/P + uncertainty + optional retrieved prior
  -> gated Transformer / Conformer conditioner
  -> conditional flow matching in factorized codec latent/token space
  -> frozen codec decoder
```

retrieval 只允许作为 initialization 或 diagnostic prior，不作为主生成路径。生成时如果 `C/P` uncertainty 高，decoder 应退回保守的 neutral prior，并在报告中标记 gate failure，不能宣称 EEG-to-speech 成功。

## 6. 训练方案

### Stage 0: 数据审计与 canonical cache

- 固定 subject-holdout：`subject_val=P02`、`subject_test=MM21` 或进行 leave-one-subject cross-validation。
- 对 overt_like、thinking、stimulus_like 分别生成 EEG QC：bad channels、valid length、speech-artifact proxy、trial energy、stage consistency。
- 音频只生成一次 canonical targets：HuBERT/wav2vec2 sequence、Whisper embedding、semantic tokens、codec tokens/latents、active mask、duration、energy envelope。

### Stage 1: EEG self-supervised pretraining

目标是在 KaraOne 全 stage、全 train subject 上学 `Z_e`：

```text
L_eeg = L_masked_time_channel + L_masked_spectrum_vq + L_aug_consistency
      + L_channel_dropout_consistency + L_subject_invariance
```

subject_val/test 不参与训练；所有 normalization 参数必须只从 train subjects 估计。

### Stage 2: speech latent teacher

冻结 speech SSL 和 codec teacher。对每个 trial 建立：

```text
S_sem(t): HuBERT/wav2vec2/Whisper hidden sequence
U_sem(t): k-means semantic unit
P(t): active, duration, energy, onset/offset
Q(t): codec token/latent
```

语音 target 使用 train-only standardization；test audio 只用于评价，不进入 retrieval/generation bank。

### Stage 3: EEG-to-semantic/prosody alignment

训练 encoder + semantic/prosody heads：

```text
L_align =
  λ_seq * OT/soft-DTW(Z_e, S_sem)
+ λ_nce * sequence/global contrastive
+ λ_soft * speech-SSL soft-positive CE
+ λ_ctc * semantic-unit CTC
+ λ_prompt * weak prompt CE
+ λ_prosody * active/duration/energy loss
+ λ_inv * subject-invariance/group-DRO
+ λ_var * variance/covariance anti-collapse
```

主选择指标必须来自 `subject_val`，且要同时超过 zero-EEG、mean-query、shuffled-query。

### Stage 4: conditional transport decoder

先用 teacher-forced `C/P` 训练 codec-space flow，再用 EEG-predicted `C/P` 做 scheduled sampling：

```text
L_flow = E_t || v_θ(x_t, t, C, P, U) - (x_1 - x_0) ||^2
L_render = MR-STFT + SSL perceptual + energy/envelope + codec reconstruction
```

decoder 不应反向把所有 acoustic loss 压回 EEG encoder；先冻结 encoder，确认 semantic/prosody subject-holdout 过关后再小学习率联合微调。

### Stage 5: thinking adaptation

thinking 不做 waveform-level 主训练。推荐：

1. overt_like 训练出 EEG-to-`C/P`。
2. thinking 只对齐到 same-trial/same-label overt 的 semantic neighborhood 和 prompt CTC。
3. 只在 thinking 的 `subject_val` semantic gate 通过后输出 waveform demo；否则只报告 semantic decoding。

## 7. 评价协议

必须保留并强化当前诚实基线：

- zero-EEG、mean latent/query、shuffled EEG、train-bank retrieval prior。
- subject_val selection，subject_test final report。
- collapse：std ratio、pairwise corr、variance floor。
- leakage：subject classifier accuracy、same-label cross-subject gain、feature-branch leakage。
- semantic：HuBERT/wav2vec2 cosine gain、semantic top-k/mrr、semantic-unit edit distance、prompt CTC accuracy。
- generation：oracle codec ceiling、MR-STFT/MCD/STOI（仅神经声码器后）、Whisper CER/WER、active envelope、RMS/peak bounds。
- 禁止单独用 best-shift envelope 或 label_top1 宣称 EEG-to-speech 成功。

最低通过标准：

```text
subject_val:
  semantic_top3_gain > 0
  HuBERT/wav2vec2 cosine gain > 0
  same-label cross-subject gain >= 0
  subject leakage 显著低于 v7/v8

subject_test:
  上述指标仍为正，且 generation 超过 retrieval prior/mean prior。
```

## 8. 相对 v8 的本质变化

| 维度         | v8                                                             | 新方案                                                                                        |
| ------------ | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| EEG 表示     | raw pooled + feature vector + envelope fusion                  | channel-time neural token sequence + topology-aware encoder + self-supervised pretraining     |
| 语音目标     | HuBERT summary + active-core Mel retrieval                     | HuBERT/wav2vec2 sequence + semantic units + prosody/event + factorized codec target           |
| 对齐         | soft-positive sentence-level InfoNCE                           | sequence OT/soft-DTW/CTC + global contrastive + subject-invariant alignment                   |
| 生成         | train-bank active-core prior + optional residual + Griffin-Lim | conditional flow matching in neural codec/factorized speech space                             |
| thinking     | 仍可走 wav synthesis runner                                    | 先做 semantic/prosody decoding，未过 gate 不宣称 waveform generation                          |
| subject 泛化 | GRL、feature dropout/noise                                     | pretrain + group-DRO/CORAL/MMD + leakage audit + leave-subject validation                     |
| 研究叙事     | v7 的 better loss/selection                                    | 从“回归声学模板”重定义为“EEG -> speech semantic/prosody latent -> speech prior transport” |

## 9. 实施优先级

1. 冻结旧 v1-v8 结论，建立 `v9_rebuild` 新目录，不复用 v7 class 作为主模型。
2. 先实现 speech target cache：HuBERT/wav2vec2 sequence、semantic units、codec latent/token、prosody/event。
3. 实现 EEG tokenizer pretraining 与 subject-holdout audit。
4. 训练 EEG-to-semantic/prosody；只有 subject_val/test 同时正增益后进入生成。
5. 实现 conditional flow decoder；先 teacher-forced，再 EEG-conditioned。
6. 最后才做 waveform synthesis、Whisper CER/WER 和 subjective demo。

## 10. 参考来源

本地来源：

- `karaone_overt_recon_bundle/reports/karaone_semantic_first_model_tech_zh.md`
- `karaone_overt_recon_bundle/MODEL_TECH.md`
- `karaone_overt_recon_bundle/OPTIMIZATION.md`
- `karaone_overt_recon_bundle/reports/karaone_data_summary.json`
- `karaone_overt_recon_bundle/artifacts/outputs_karaone/*/metrics/test_metrics.json`
- `/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/paper-ref/eeg-to-speech-ccf-a/README.md`
- `/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/paper-ref/eeg-to-speech-ccf-a/papers.csv`

外部检索来源：

- NeuroSonic: https://arxiv.org/abs/2606.24087
- Défossez et al. non-invasive speech decoding: https://arxiv.org/abs/2208.12266
- NeuroTalk: https://arxiv.org/abs/2301.07173
- EEG-to-Voice 2025: https://arxiv.org/abs/2512.22146
- wav2vec 2.0: https://arxiv.org/abs/2006.11477
- HuBERT: https://arxiv.org/abs/2106.07447
- Whisper: https://arxiv.org/abs/2212.04356
- EnCodec: https://arxiv.org/abs/2210.13438
- Voicebox: https://arxiv.org/abs/2306.15687
- NaturalSpeech 3: https://arxiv.org/abs/2403.03100
- LaBraM: https://arxiv.org/abs/2405.18765
- NeuroLM: https://arxiv.org/abs/2409.00101

