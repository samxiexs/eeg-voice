# KaraOne 语义优先 EEG-to-Speech 当前模型技术说明

> 版本：2026-06-28i
> 范围：只针对 `karaone_overt_recon_bundle`。本轮没有实际跑训练/验证，只做代码级复核、运行入口整理、结构文档化。
> 核心目标：EEG -> Speech Generation，而不是 EEG 分类、通道选择或单纯 speech recognition。

---

## 1. 总体结论

当前实现已经较好地落地了“语义优先、增量式”的优化方向：

1. **Phase 0 基础协议已明显加强**：`eeg_valid_len` mask 进入 diffusion/flow encoder；生成式 trainer 有 `subject_test`；评估输出 `zeroeeg`、`mean_target`、retrieval top-k 等可比较基线。
2. **Phase 1 已经有独立语义主训练入口**：新增 `scripts/train_karaone_semantic.py`，主目标从 mel/EnCodec 连续 latent 转为 HuBERT sequence prediction。
3. **MoE 没有被设为主路径**：semantic trainer 默认 `--model baseline`；`--model moe` 只作为 ablation，这符合“小数据先改目标，不盲目堆结构”的要求。
4. **ASR/Whisper 指标已作为渲染后评估钩子加入**：不会默认下载模型，不会影响训练目标；适合在生成 wav 后做 label-constrained ASR/CER 辅助评价。

仍需明确的边界：

1. **当前还不是完整 Phase 3 的 semantic-conditioned acoustic rendering**。现有 diffusion/flow 仍是 EEG -> acoustic latent/mel，不是 `(EEG memory + predicted HuBERT) -> codec latent`。
2. **当前 `--stages` 是 stage-aware 多阶段训练接口，不是同时融合多个 stage 的 cross-attention memory**。也就是说，它可以训练 `clearing/stimulus_like/thinking/overt_like`，但每个样本仍是单 stage 输入；真正的多 stage 同 trial 融合还没实现。
3. **semantic checkpoint 只能评估语义特征，不直接渲染 wav**。要生成语音，还需要 Phase 3 的语义条件声学渲染器，或把 semantic output 接到冻结/条件 codec renderer。

---

## 2. 当前推荐信息流

当前新主线应理解为：

```text
KaraOne NPZ stages + eeg_valid_len
    -> KaraOneTrialDataset
    -> EEG encoder (baseline Conformer by default)
    -> HuBERT semantic sequence prediction
    -> semantic metrics / retrieval / subject_test
    -> later: semantic-conditioned flow or codec renderer
```

而旧声学路径仍保留为：

```text
EEG
    -> EEG encoder
    -> mel or EnCodec continuous latent
    -> regression or diffusion/flow
    -> Griffin-Lim or EnCodec decoder
    -> wav + rendered-audio metrics
```

关键原则：**先确认 EEG 能否稳定预测 speech semantic representation，再进入高保真声学渲染**。

---

## 3. 数据与输入结构

### 3.1 数据源

代码入口：`app/src/karaone_recon/data.py`

主要文件：

- `data/karaone/segments.csv`
- `data/karaone/subjects/<subject>.npz`
- `artifacts/audio_targets/karaone_trial_hubert.npz`
- `artifacts/audio_targets/karaone_trial_encodec_latents.npz`
- `artifacts/audio_targets/karaone_trial_mel.npz`

`KaraOneTrialDataset` 从 `segments.csv` 读取每个 `(subject, label, stage, trial_index)`，再从每个 subject 的 NPZ 中读取：

- `stage__clearing`
- `stage__stimulus_like`
- `stage__thinking`
- `stage__overt_like`
- 对应的 `stage__<stage>__valid_lengths`

当前默认 stage 是：

```yaml
data:
  stages: overt_like
```

可以通过命令行覆盖：

```bash
--stages clearing,stimulus_like,thinking,overt_like
```

注意：这会把多个 stage 都加入训练集，但当前不是同一 trial 的多 stage 同时输入融合。

### 3.2 EEG 形状

默认 EEG：

```text
eeg: [B, 62, 1280]
eeg_valid_len: [B]
```

`eeg_valid_len` 的作用：

- instance normalization 只统计有效 EEG 区间；
- Transformer/Conformer key padding mask 忽略补零尾部；
- pooled utterance embedding 只平均有效 frame；
- diffusion/flow encoder path 现在也会接收 valid length。

---

## 4. Phase 1 语义主模型结构

训练入口：`app/scripts/train_karaone_semantic.py`

该脚本复用 `KaraOneEEG2Codec`，但把它配置成语义预测模型：

```python
target_steps = targets.T      # HuBERT: 通常 50
target_dim = targets.D        # HuBERT: 通常 768
content_dim = targets.D       # content prototype / contrastive 维度对齐 HuBERT
hubert_dim = 0                # 不再使用 aux HuBERT head，因为 pred_latent 本身就是 HuBERT
```

所以在 semantic trainer 中：

```text
out["pred_latent"] 实际代表 pred_hubert_sequence
```

这点命名上沿用了旧代码，但语义上已经从 acoustic latent 改成 HuBERT semantic feature。

### 4.1 Encoder

默认配置来自 `configs/karaone.yaml`：

```yaml
model:
  encoder_kind: conformer
  d_model: 256
  transformer_layers: 4
  transformer_heads: 4
  patch_stride: 4
  channel_dropout: 0.15
  dropout: 0.15
```

默认 semantic 运行使用：

```bash
--model baseline
```

也就是：

```text
EEG [B,62,1280]
  -> per-trial instance norm (optional, config 默认 true)
  -> channel dropout
  -> Conv1d(62 -> d_model)
  -> FiLM(stage embedding)
  -> temporal patch conv
  -> adaptive pooling to target_steps
  -> positional embedding
  -> 4-layer Conformer
  -> key padding mask by eeg_valid_len
  -> encoded memory [B, d_model, T]
```

如果使用 `--model moe`，空间前端改为 `ChannelMoEFrontend`：

```text
per-channel temporal stats
  -> channel gate
  -> soft expert assignment
  -> expert-specific channel mixing
  -> channel_balance regularization
```

但这只建议作为 ablation，不建议作为 Phase 1 默认。

### 4.2 Stage 条件

`stage_idx` 进入：

```python
stage_embedding(stage_idx) -> FiLM condition
```

作用是告诉 encoder 当前 EEG 来自哪个 cognitive/speech stage。
当前没有 subject embedding；`subject_idx` 只为接口兼容与 DANN train-only head 服务。

### 4.3 输出头

`KaraOneEEG2Codec` 当前输出：

- `pred_latent`: 在 semantic trainer 中是 `[B, 50, 768]` HuBERT sequence prediction；
- `content_embed`: utterance-level content embedding；
- `content_logits`: 11 个 KaraOne label 的辅助分类；
- `ctc_logits`: prompt-token CTC；
- `clip_embed`: EEG side embedding，用于 EEG-HuBERT summary InfoNCE；
- `pred_log_rms` / `pred_frame_log_energy` / `pred_log_decoder_scale`: acoustic 旧头仍在模型里，但 semantic trainer 对相关 loss 置零；
- `subject_logits`: 仅在 `domain_adapt.adversarial=true` 时出现，用于 gradient reversal subject-DANN。

---

## 5. Phase 1 Loss 设计

semantic trainer 中的主 loss：

```text
total =
  lambda_recon_cos * (1 - cosine(pred_hubert, target_hubert))
  + lambda_recon_mse * MSE(pred_hubert, target_hubert)
  + lambda_clip * InfoNCE(EEG embedding, HuBERT summary)
  + lambda_ctc * prompt-token CTC
  + lambda_content_ce * label CE
  + lambda_supcon * supervised contrastive
  + lambda_proto * prototype cosine
  + lambda_std * feature std matching
  + optional domain adversarial CE
```

被关闭的 acoustic loss：

```text
lambda_log_rms = 0
lambda_energy_env = 0
lambda_multiscale_mel = 0
lambda_frame_energy = 0
lambda_voiced_rms = 0
lambda_decoder_scale = 0
lambda_hubert_aux = 0
lambda_hubert_clip = 0
```

这个设计符合当前路线：**先让 EEG 预测 semantic speech representation，而不是先优化 waveform/mel/codec latent 回归**。

---

## 6. 评估结构

核心评估函数：`app/src/karaone_recon/eval.py`

semantic trainer 最终写出：

```json
{
  "target_kind": "hubert_sequence",
  "selection": {
    "criterion": "val semantic pred_over_mean_cos_gain"
  },
  "test": {...},
  "subject_test": {...}
}
```

每个 split 主要包含：

- `pred_recon_cos`
- `zeroeeg_recon_cos`
- `mean_recon_cos`
- `pred_over_zero_cos_gain`
- `pred_over_mean_cos_gain`
- `pred_recon_mse`
- `zeroeeg_recon_mse`
- `mean_recon_mse`
- `pred_std_ratio_median`
- `pred_pairwise_corr_median`
- `pred_pcc`
- `content_acc`
- `pred_within_subject_trial_top1/top3/top5`
- `pred_within_subject_label_top1/top3/top5`
- `zeroeeg_within_subject_*`
- `mean_within_subject_*`
- `by_stage`

静态复核后已补齐普通 `evaluate()` 的 `mean_*` retrieval top-k，使 regression/semantic eval 和 diffusion eval 的 baseline 更一致。

---

## 7. 生成式声学路径现状

训练入口：`app/scripts/train_karaone_diffusion.py`
模型：`app/src/karaone_recon/diffusion.py`

当前支持：

```text
EEG -> EEG encoder -> conditioning sequence
    -> diffusion or conditional flow over acoustic target
    -> mel / EnCodec latent
```

其中：

- `mode=diffusion`: DDPM/DDIM epsilon prediction；
- `mode=flow`: rectified conditional flow matching；
- `target=encodec_latent`: 输出 EnCodec continuous latent；
- `target=mel`: 输出 mel；
- `eeg_valid_len` 已进入 encoder；
- final metrics 包含 `test` 和 `subject_test`；
- diffusion eval 已包含 `pred/zeroeeg/mean` retrieval top-k。

限制：

```text
当前 flow/diffusion 条件 = EEG encoder memory
还没有接入 predicted HuBERT semantic sequence
```

因此它是 Phase 3 的声学路径基础，不是完整的 semantic-conditioned renderer。

---

## 8. 渲染与 ASR 指标

渲染入口：

- regression checkpoint: `app/scripts/synthesize_karaone.py`
- diffusion/flow checkpoint: `app/scripts/synthesize_karaone_diffusion.py`

已输出：

- `original`
- `oracle_encodec` 或 `oracle_griffinlim`
- `mean_latent`
- `zeroeeg`
- `pred` / `pred_scaled`
- diffusion 的 `sample1/sample2/...`
- `synth_metrics.json`
- `listening_manifest.csv`

waveform-level metrics：

- envelope correlation；
- active-region envelope correlation；
- RMS over original；
- active voiced RMS over original；
- oracle codec/vocoder ceiling。

可选 ASR：

```bash
--asr-model tiny.en
--asr-allow-download   # 只有首次下载时才需要；默认不会下载
```

ASR 指标包括：

- `*_asr_label_acc`
- `*_asr_cer_mean`
- 每 trial transcript / candidate / CER / WER。

注意：KaraOne 是短音素/短词，ASR 只适合作为辅助指标，不能作为唯一模型选择依据。

---

## 9. 推荐运行命令

项目根目录：

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/karaone_overt_recon_bundle
```

### 9.1 准备 targets

```bash
bash run_semantic_first.sh targets
```

等价手工命令：

```bash
cd app
python scripts/extract_karaone_targets.py --target hubert_sequence
python scripts/extract_karaone_targets.py --target encodec_latent
```

### 9.2 Phase 1：语义主干，推荐第一优先级

```bash
bash run_semantic_first.sh semantic 30 baseline sem_overt_v1
```

手工命令：

```bash
cd app
python scripts/train_karaone_semantic.py \
  --model baseline \
  --stages overt_like \
  --epochs 30 \
  --run-suffix sem_overt_v1 \
  --device cpu
```

### 9.3 Phase 1 ablation：Channel MoE

```bash
bash run_semantic_first.sh semantic 30 moe sem_overt_moe_ablation
```

只接受同 split/seed 下优于 baseline 的结果，否则不应进入默认路线。

### 9.4 Phase 2 预备：stage-aware 多阶段训练

```bash
STAGES=clearing,stimulus_like,thinking,overt_like \
bash run_semantic_first.sh semantic 30 baseline sem_allstages_v1
```

解释：这会训练 stage-aware semantic model，但还不是同 trial 多 stage fusion。

### 9.5 评估 semantic checkpoint

```bash
CKPT=artifacts/outputs_karaone/karaone_semantic_baseline_overt_like_sem_overt_v1/checkpoints/best.pt \
bash run_semantic_first.sh semantic_eval
```

手工命令：

```bash
cd app
python scripts/eval_karaone_recon.py \
  --checkpoint ../artifacts/outputs_karaone/karaone_semantic_baseline_overt_like_sem_overt_v1/checkpoints/best.pt \
  --split test \
  --device cpu

python scripts/eval_karaone_recon.py \
  --checkpoint ../artifacts/outputs_karaone/karaone_semantic_baseline_overt_like_sem_overt_v1/checkpoints/best.pt \
  --split subject_test \
  --device cpu
```

### 9.6 声学 flow baseline

```bash
bash run_semantic_first.sh acoustic_flow 30 baseline flow_overt_v1
```

手工命令：

```bash
cd app
python scripts/train_karaone_diffusion.py \
  --model baseline \
  --mode flow \
  --target encodec_latent \
  --stages overt_like \
  --epochs 30 \
  --run-suffix flow_overt_v1 \
  --device cpu
```

### 9.7 渲染 wav + 可选 ASR

diffusion/flow checkpoint：

```bash
CKPT=artifacts/outputs_karaone/karaone_diffusion_baseline_overt_like_flow_overt_v1/checkpoints/best.pt \
LIMIT=24 \
bash run_semantic_first.sh synth_diffusion
```

带 ASR，默认要求模型已缓存：

```bash
CKPT=artifacts/outputs_karaone/karaone_diffusion_baseline_overt_like_flow_overt_v1/checkpoints/best.pt \
LIMIT=24 \
ASR_MODEL=tiny.en \
bash run_semantic_first.sh synth_diffusion
```

如果允许首次下载 Whisper：

```bash
CKPT=artifacts/outputs_karaone/karaone_diffusion_baseline_overt_like_flow_overt_v1/checkpoints/best.pt \
LIMIT=24 \
ASR_MODEL=tiny.en \
ASR_ALLOW_DOWNLOAD=1 \
bash run_semantic_first.sh synth_diffusion
```

### 9.8 一次性跑 thinking / speaking / thinking+speaking 并合成全部 wav

语义实验三组批量运行：

```bash
bash run_thinking_speaking_semantic.sh
```

默认等价于：

```bash
bash run_thinking_speaking_semantic.sh 30 baseline thinking_speaking_<timestamp> cpu 1
```

如果要一次性得到三组全部重建好的 wav，需要使用 acoustic flow/diffusion 路径，因为 semantic checkpoint 只输出 HuBERT 特征，不能直接声码器渲染：

```bash
bash run_thinking_speaking_wav.sh
```

默认含义：

```bash
bash run_thinking_speaking_wav.sh 30 baseline wav_<timestamp> cpu subject_test -1 flow
```

它会依次训练并合成：

```text
thinking only
speaking only = overt_like
thinking + speaking = thinking,overt_like
```

并且每一组都使用 `checkpoints/best.pt` 合成，不使用 `last.pt`。`LIMIT=-1` 表示合成该 split 的全部样本。

---

## 10. 当前实现质量复核

| 项目                      | 当前状态                                           | 判断                                                |
| ------------------------- | -------------------------------------------------- | --------------------------------------------------- |
| `eeg_valid_len` mask    | regression/conformer/diffusion path 均有使用       | 基本到位                                            |
| `subject_test`          | semantic 和 diffusion trainer 均输出               | 到位                                                |
| zeroeeg baseline          | semantic/regression/diffusion eval 均有            | 到位                                                |
| mean_target baseline      | cosine/MSE 到位；retrieval 已补齐                  | 到位                                                |
| oracle_codec ceiling      | synthesis scripts 输出                             | 到位，但只对 renderable acoustic checkpoints 有意义 |
| HuBERT semantic target    | `resolve_target_cache()` + semantic trainer 支持 | 到位                                                |
| semantic-first objective  | acoustic loss 置零，HuBERT 为主目标                | 到位                                                |
| MoE priority              | 默认 baseline，MoE ablation                        | 符合路线                                            |
| multi-stage               | `--stages` 支持 stage-aware 训练                 | 部分到位，尚非 fusion                               |
| semantic-conditioned flow | 尚未实现                                           | 后续 Phase 3                                        |
| ASR metrics               | 可选 Whisper hook                                  | 到位，但短音素需谨慎解释                            |

总体判断：**优化方向实现得比较干净，已经适合作为下一轮训练实验的主入口**。下一步不应该继续堆 encoder，而应该先用 Phase 1 的 semantic metrics 判断 EEG 是否真的超过 zero/mean，再决定是否进入 semantic-conditioned acoustic rendering。

---

## 11. 下一步最小闭环

建议按以下顺序推进：

1. 跑 `semantic baseline overt_like`，只看 `pred_over_mean_cos_gain`、HuBERT retrieval、`subject_test`。
2. 如果 Phase 1 不超过 zero/mean，不进入声学渲染，先修数据/目标/正则。
3. 如果 Phase 1 稳定超过 baseline，再跑 `STAGES=clearing,stimulus_like,thinking,overt_like`。
4. 只有 multi-stage 语义指标优于 overt_like，才实现真正 multi-stage fusion。
5. 最后实现 Phase 3：`EEG memory + predicted HuBERT -> conditional flow -> EnCodec latent -> frozen EnCodec decoder`。
