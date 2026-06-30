# KaraOne 语义辅助 EEG-to-Speech 当前模型技术说明

> 版本：2026-06-30
> 范围：`karaone_overt_recon_bundle` 当前 v9 Neural Semantic Transport；v8 Soft-Positive Cross-Subject 作为上一代 baseline 保留。
> 当前主目标：未知 EEG -> 对应语音生成。推理时默认只输入 EEG，不使用真实 prompt label、真实 onset、真实 insert frame、真实 target audio。
> 当前状态：v9 已完成独立代码骨架、canonical cache/audit、synthetic smoke、1-epoch CPU align/transport diagnostic；尚未跑通可宣称 EEG-to-speech 成功的完整训练或 waveform 解码。

---

## 1. 当前定位

目标没有改变，仍然是：

```text
未知 EEG -> 对应语音 / wav
```

但 v9 不再把当前主线定义h v8 的：

```text
EEG embedding -> train-bank retrieval -> active-core Mel prior -> Griffin-Lim wav
```

v9 的主线改为：

```text
raw 62ch EEG sequence
  -> subject-robust channel-time EEG token sequence
  -> speech semantic/prosody latent sequence
  -> conditional flow matching in codec latent space
  -> frozen/external neural codec decoder
```

当前实现必须按这个状态理解：

- 已完成：v9 独立 package、数据/target bank、模型、loss、eval、transport 模块、runner、protocol audit、smoke/epoch-1 diagnostic 输出。
- 未完成：长程 Stage 1/3/4 训练、subject_val semantic gate 通过、neural codec waveform rendering、Whisper CER/WER 或听感有效性报告。
- 因此当前不能宣称 v9 已生成可理解 speech，只能说已搭好 v9 训练/评估骨架，并有 early diagnostic 指标。

label 只允许作为：

```text
弱辅助 prompt CE / CTC
分组评估
oracle/diagnostic 解释
```

不允许作为：

```text
生成入口
prototype selector
checkpoint 主指标
未知 EEG 合成时的输入
```

---

## 2. v8 与 v9 的本质区别

v8 是 v7 的训练目标修正：复用 v7 cross-subject model、v7 feature cache 和 v7 synthesis，把 strict same-trial InfoNCE 改为 speech-SSL soft-positive，并加强 subject leakage 抑制。v8 的最终生成仍依赖 train-bank retrieval 和 Griffin-Lim。

v9 是重建，不是 v8 patch：

| 维度          | v8                                                                                  | v9                                                                                                                |
| ------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| EEG 表示      | raw EEG pooled embedding + 手工 feature vector + EEG envelope fusion                | raw 62ch EEG 的 channel-time patch token sequence，Transformer encoder，content/prosody/uncertainty streams       |
| speech latent | HuBERT summary / active-core Mel prior                                              | HuBERT/wav2vec-style sequence、semantic tokens、prosody/event targets、codec latent                               |
| 对齐          | sentence-level soft-positive InfoNCE + semantic-neighborhood retrieval metric       | monotonic soft-OT、framewise sequence cosine、global/soft InfoNCE、semantic-token CE、prompt CTC/CE、prosody loss |
| subject 泛化  | GRL、feature dropout/noise、subject-holdout selection                               | forward 不接收 subject/speaker input；subject adversarial、CORAL、group-DRO、subject leakage audit                |
| 解码          | train-bank active-core prior + optional residual + Griffin-Lim                      | conditional flow matching in codec latent space；waveform 需外接/frozen neural codec decoder                      |
| 成功标准      | subject-holdout semantic-neighborhood/HuBERT gain 与 retrieval-based wav diagnostic | 先过 semantic/prosody gate，再谈 codec/wav generation；当前 gate 未通过                                           |

关键变化是：v9 把问题从“EEG 回归或检索声学模板”改成“EEG 先进入 speech semantic/prosody latent，再由生成先验补全 codec/acoustic detail”。这更符合 KaraOne 小样本、跨被试、overt_like 非严格逐帧同步、thinking 只有语义/意图关联的约束。

---

## 3. 当前代码与产物审计

### 3.1 代码入口

v9 root runner：

```text
karaone_overt_recon_bundle/run_karaone_v9_rebuild.sh
```

注意：该 root-level runner 被仓库 ignore 规则覆盖，不会出现在 `git status` 的 untracked 列表里。

v9 untracked 代码/配置：

```text
karaone_overt_recon_bundle/app/configs/karaone_v9.yaml
karaone_overt_recon_bundle/app/scripts/build_karaone_v9_canonical_cache.py
karaone_overt_recon_bundle/app/scripts/audit_karaone_v9_protocol.py
karaone_overt_recon_bundle/app/scripts/train_karaone_v9.py
karaone_overt_recon_bundle/app/src/karaone_v9/
karaone_overt_recon_bundle/app/tests/test_karaone_v9_smoke.py
karaone_overt_recon_bundle/reports/karaone_v9_rebuild_implementation_20260630.md
```

Git 状态审计结论：

```text
tracked diff before this note:
  none

untracked v9 files:
  app/configs/karaone_v9.yaml
  app/scripts/audit_karaone_v9_protocol.py
  app/scripts/build_karaone_v9_canonical_cache.py
  app/scripts/train_karaone_v9.py
  app/src/karaone_v9/*.py
  app/tests/test_karaone_v9_smoke.py
  reports/karaone_v9_rebuild_implementation_20260630.md

ignored but relevant local files:
  run_karaone_v9_rebuild.sh
  artifacts/audio_targets/karaone_v9_*.json
  artifacts/outputs_karaone/karaone_v9_*/
```

### 3.2 Canonical cache / audit output

已生成：

```text
artifacts/audio_targets/karaone_v9_canonical_manifest_overt_like.json
artifacts/audio_targets/karaone_v9_protocol_audit.json
```

审计状态：

```text
status: pass
stage: overt_like
segments: 1913
subject_val: P02
subject_test: MM21
subject_train: 12 train subjects, 1616 samples
subject_val: 165 samples
subject_test: 132 samples
semantic_missing: 0
codec_missing: 0
prosody_missing: 0
semantic_token_missing: 0
split overlaps: all 0
```

target shapes：

```text
semantic sequence:       [50, 768]
semantic tokens:         [50, 64]
codec latent sequence:   [150, 128]
prosody steps:           64
EEG input length:        1280
EEG channels:            62
```

### 3.3 当前训练/实验输出

本地 v9 output 只有三组：

```text
artifacts/outputs_karaone/karaone_v9_neural_semantic_transport_align_overt_like_v9_align_smoke_final
artifacts/outputs_karaone/karaone_v9_neural_semantic_transport_align_overt_like_v9_20260630_053520
artifacts/outputs_karaone/karaone_v9_neural_semantic_transport_transport_overt_like_v9_transport_smoke
```

三组都只有 `history_len=1`、`epoch=1`，device 为 CPU。它们是 smoke/early diagnostic，不是完整训练。

当前较新的 align diagnostic：

```text
subject_val_semantic_over_mean_gain:       +0.2268
subject_val_semantic_top3_gain_over_mean:  -0.0182
subject_val_same_label_cross_subject_gain: -0.0525
subject_val_semantic_gate_pass:            false
subject_test_semantic_top3_gain_over_mean: -0.0758
subject_test_same_label_cross_subject_gain:-0.0533
subject_test_semantic_gate_pass:           false
pred_pairwise_corr_median:
  subject_val 0.9641
  subject_test 0.9697
```

当前 transport smoke：

```text
train_flow:                              2.0323
train_condition_semantic:                1.0034
subject_val_semantic_gate_pass:          false
subject_test_semantic_gate_pass:         false
```

解释：

- semantic cosine 相对 mean query 为正，但相对 zero-EEG 不是正增益。
- semantic top-3 gain over mean 为负。
- same-label cross-subject gain 为负。
- semantic gate 未通过。
- pairwise correlation 仍高，说明 early output 仍有 collapse/template 风险。
- 没有 waveform decoding 指标，不能说 v9 已完成 EEG-to-speech generation。

---

## 4. 数据与输入空间

v9 使用 `segments.csv` 和 subject NPZ 作为 canonical EEG source。默认只跑：

```text
stage: overt_like
subject_val: P02
subject_test: MM21
train split: subject_train
```

每个样本包含：

```text
eeg:                     [62, 1280]
eeg_valid_len:            scalar
stage_idx:                scalar
subject_idx:              scalar, 仅用于 loss/eval
label_idx:                scalar, 仅用于 weak prompt supervision/eval
semantic_seq:             [50, 768]
semantic_summary:         [768]
semantic_token_targets:   [50]
semantic_token_mask:      [50]
codec_seq:                [150, 128]
prosody_active:           [64]
prosody_energy:           [64]
prosody_duration:         scalar
prosody_onset:            scalar
```

模型 forward path 不接收 `subject_idx` 或 `speaker_id`。这点由 smoke test 明确检查：

```text
KaraOneV9NeuralSemanticTransport.forward has no subject_idx / speaker_id parameter
```

subject 信息只用于：

```text
subject adversarial loss
CORAL/group-DRO
subject leakage metric
split protocol audit
```

---

## 5. v9 EEG 表示空间

v9 的 EEG 表示不是 v8 的 pooled branch + handcrafted feature fusion，而是：

```text
raw EEG [B, 62, T]
  -> per-trial valid-length normalization
  -> channel log-variance reliability gate
  -> Conv1d channel-time patch tokenization
  -> positional embedding + stage embedding
  -> Transformer token encoder
  -> EEG token sequence
```

模型内部输出四类 stream：

```text
eeg_tokens:        subject-robust token representation
content_tokens:    speech semantic/content prediction stream
prosody_tokens:    active/event/energy/duration/onset stream
uncertainty:       token-level uncertainty condition
condition_seq:     transport decoder condition
```

Stage 1 pretraining 路径已实现：

```text
mask EEG patch tokens
  -> reconstruct patch token target
  -> VICReg-style variance regularization
```

当前限制：

- 没有接入真实 electrode topology 坐标。
- 没有大规模 EEG foundation model 初始化。
- 还没有跑完整 Stage 1 pretraining。

---

## 6. Speech Semantic / Prosody / Codec Latent

v9 把 speech target 分成三层：

### 6.1 Semantic sequence

来源：

```text
artifacts/audio_targets/karaone_trial_hubert.npz
```

用途：

```text
sequence-level speech SSL target
semantic summary target
retrieval/evaluation bank
```

### 6.2 Semantic tokens

来源：

```text
artifacts/audio_targets/karaone_trial_hubert_tokens_k64_trainonly.npz
```

用途：

```text
semantic_token_ce
prompt CTC auxiliary supervision
train-only semantic unit diagnostic
```

### 6.3 Prosody / event targets

来源：

```text
artifacts/audio_targets/karaone_temporal_elastic_core_v5.npz
```

用途：

```text
active mask
energy envelope
duration
onset
```

这些 target 不是最终声学模板，而是 EEG->C/P 对齐时的 prosody/event supervision。

### 6.4 Codec latent

来源：

```text
artifacts/audio_targets/karaone_trial_encodec_latents.npz
```

用途：

```text
conditional flow matching target
codec-space transport smoke
future neural codec decoder input
```

当前没有把 codec latent 解码成 wav 的完成路径；报告中的 transport 只说明 flow loss 可以反向传播并生成 latent sample，不代表已完成 waveform synthesis。

---

## 7. Sequence Alignment 与 Loss

v9 alignment loss 已实现为组合目标：

```text
L_align =
  lambda_seq_ot        * monotonic_soft_ot(pred_sem_seq, speech_ssl_seq)
+ lambda_seq_cos       * framewise semantic cosine
+ lambda_global_nce    * symmetric EEG/speech InfoNCE
+ lambda_soft_nce      * speech-SSL soft-positive InfoNCE
+ lambda_semantic_token* semantic-token CE
+ lambda_ctc           * prompt CTC
+ lambda_prompt        * weak prompt CE
+ lambda_prosody       * active/energy/duration/onset loss
+ lambda_subject_adv   * subject adversarial CE
+ lambda_coral         * subject distribution alignment
+ lambda_group_dro     * worst-subject emphasis
+ lambda_variance      * anti-collapse variance
```

sequence alignment 的核心是 `monotonic_soft_ot_loss`：

```text
pred EEG semantic tokens <-> speech SSL sequence
cost = 1 - cosine
加 monotonic position penalty
双向 soft assignment 后取平均 cost
```

这比 v8 的 sentence-level soft-positive 更强，因为它显式保留时间序列结构、duration/prosody 信息和 token-level C/P 条件。

---

## 8. Conditional Transport / Codec 解码

v9 transport 模块：

```text
ConditionalTransportDecoder
```

已实现：

```text
Gaussian noise x0
codec latent x1
t ~ Uniform(0,1)
x_t = (1 - t) * x0 + t * x1
condition = EEG-derived semantic/prosody tokens
Transformer velocity field predicts x1 - x0
```

训练 loss：

```text
L_transport =
  lambda_flow * MSE(pred_velocity, target_velocity)
+ lambda_condition_semantic * semantic_guard
```

runner 支持 transport 阶段：

```text
CKPT=<align checkpoint> DEVICE=mps bash run_karaone_v9_rebuild.sh transport 20
```

并支持：

```text
FREEZE_ENCODER=1
```

当前限制：

- transport smoke 可以训练 codec latent velocity field，但没有完成 frozen codec decoder 的 wav rendering。
- 还没有 teacher-forced speech C/P -> codec 的稳定训练计划结果。
- 还没有 scheduled sampling 从 EEG-predicted C/P 过渡到 codec generation。
- 还没有 oracle-codec ceiling、MR-STFT、MCD、STOI、Whisper CER/WER。

---

## 9. 训练流程

### Stage 0: canonical cache / protocol audit

```text
build_karaone_v9_canonical_cache.py
audit_karaone_v9_protocol.py
```

目标：

```text
确认 target coverage
确认 subject_train/subject_val/subject_test 不重叠
记录 split sample count 和 EEG valid length
记录 semantic/prosody/codec target shape
```

当前状态：已通过 overt_like audit。

### Stage 1: EEG masked-token pretraining

命令：

```bash
DEVICE=mps bash run_karaone_v9_rebuild.sh pretrain 20
```

目标：

```text
学 subject-robust EEG token sequence
masked token reconstruction
variance anti-collapse
```

当前状态：代码已实现，未见完整训练产物。

### Stage 3: EEG-to-semantic/prosody alignment

命令：

```bash
DEVICE=mps bash run_karaone_v9_rebuild.sh align 50
```

目标：

```text
EEG tokens -> speech semantic sequence / semantic summary / semantic tokens / prosody
```

当前状态：有 1-epoch CPU diagnostic；semantic gate 未通过。

### Stage 4: codec-space conditional transport

命令：

```bash
CKPT=artifacts/outputs_karaone/<v9_align_run>/checkpoints/best.pt \
DEVICE=mps bash run_karaone_v9_rebuild.sh transport 20
```

目标：

```text
condition_seq -> codec latent flow matching
```

当前状态：有 1-epoch CPU transport smoke；未完成 waveform decode。

---

## 10. Evaluation Protocol 与 Gate

v9 训练每个 epoch 都在：

```text
subject_val = P02
subject_test = MM21
```

上收集指标。train-bank 只来自 `subject_train`。

主指标：

```text
semantic_cos
zero_semantic_cos
mean_semantic_cos
semantic_over_zero_gain
semantic_over_mean_gain
semantic_label_top1/top3/mrr
semantic_top3_gain_over_mean
same_label_cross_subject_gain
subject_leakage_acc
prompt_acc
pred_std_ratio_median
pred_pairwise_corr_median
```

v9 semantic gate：

```text
subject_val:
  semantic_over_mean_gain > 0
  semantic_top3_gain_over_mean > 0
  same_label_cross_subject_gain >= 0
```

只有当 subject_val gate 通过，并且 subject_test 仍保持正向趋势时，waveform generation 才能作为 EEG-to-speech 结果讨论。当前 gate 是：

```text
subject_val_v9_semantic_gate_pass = false
subject_test_v9_semantic_gate_pass = false
```

因此当前 v9 输出只能作为：

```text
code path smoke
protocol/canonical cache audit
early diagnostic metrics
transport latent smoke
```

不能作为：

```text
successful EEG-to-speech generation
intelligible speech reconstruction
cross-subject semantic decoding success
```

---

## 11. 如何运行

进入 bundle：

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/karaone_overt_recon_bundle
```

### 11.1 Audit

```bash
bash run_karaone_v9_rebuild.sh audit
```

生成：

```text
artifacts/audio_targets/karaone_v9_canonical_manifest_overt_like.json
artifacts/audio_targets/karaone_v9_protocol_audit.json
```

### 11.2 Smoke

```bash
DEVICE=cpu bash run_karaone_v9_rebuild.sh smoke
```

runner 会执行：

```text
python tests/test_karaone_v9_smoke.py
MAX_STEPS=2 train_karaone_v9.py --phase align --epochs 1
```

### 11.3 Full-ish align 训练

```bash
DEVICE=mps bash run_karaone_v9_rebuild.sh align 50 v9_align_$(date +%Y%m%d_%H%M%S)
```

输出：

```text
artifacts/outputs_karaone/karaone_v9_neural_semantic_transport_align_overt_like_<suffix>/
```

### 11.4 Transport 训练

```bash
CKPT=artifacts/outputs_karaone/karaone_v9_neural_semantic_transport_align_overt_like_<suffix>/checkpoints/best.pt \
DEVICE=mps \
FREEZE_ENCODER=1 \
bash run_karaone_v9_rebuild.sh transport 20 v9_transport_$(date +%Y%m%d_%H%M%S)
```

### 11.5 可调环境变量

```text
CONFIG=configs/karaone_v9.yaml
STAGES=overt_like
DEVICE=cpu|mps|cuda
MAX_STEPS=<int>
CKPT=<path>
FREEZE_ENCODER=1|0
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
PYTORCH_ENABLE_MPS_FALLBACK=1
```

---

## 12. 已完成项

已完成并审计到本地文件：

```text
1. v9 独立 package: app/src/karaone_v9
2. v9 config: app/configs/karaone_v9.yaml
3. canonical target bank:
   semantic / semantic tokens / prosody / codec
4. subject_train / subject_val / subject_test split
5. canonical manifest + protocol audit
6. EEG token encoder + content/prosody/uncertainty/condition streams
7. masked EEG token pretrain loss
8. sequence alignment losses
9. subject adversarial / CORAL / group-DRO / VICReg anti-collapse losses
10. conditional flow-matching transport decoder
11. v9 metrics and semantic gate
12. synthetic smoke test
13. real-data 1-epoch CPU align diagnostic
14. real-data 1-epoch CPU transport smoke
```

---

## 13. 未完成项与当前风险

### 13.1 未完成

```text
1. 完整 Stage 1 EEG masked-token pretraining
2. 完整 Stage 3 EEG-to-semantic/prosody alignment
3. subject_val semantic gate 通过
4. subject_test 正向 generalization
5. teacher-forced speech C/P -> codec transport baseline
6. EEG-predicted C/P -> codec scheduled sampling
7. frozen neural codec decoder wav rendering
8. oracle-codec ceiling
9. Whisper CER/WER、SSL perceptual、MR-STFT/MCD/STOI
10. leave-one-subject cross-validation
11. reliable electrode topology / spatial graph encoder
12. thinking stage 的单独 adaptation 协议
```

### 13.2 当前风险

1. **semantic gate 未过**
   当前 top-3 gain over mean 和 same-label cross-subject gain 为负，不能宣称跨被试语义 decoding 成功。
2. **collapse 风险仍在**
   1-epoch align diagnostic 的 `pred_pairwise_corr_median` 约 0.96/0.97，transport smoke 约 0.90/0.91，仍需完整训练和 anti-collapse 观察。
3. **zero-EEG baseline 仍强**
   当前 semantic cosine 相对 zero-EEG 不是正增益。说明模型还没有证明 EEG-specific signal 优于零输入诊断。
4. **codec transport 尚未等于 waveform generation**
   latent flow smoke 只证明 transport loss/backward path 可运行，未证明音频可听、可识别或优于 prior。
5. **root runner 被 ignore**
   `run_karaone_v9_rebuild.sh` 当前被 `.gitignore` 的 `eeg2wave_server_bundle/**` 规则覆盖。如果需要提交 v9 runner，需要 force-add 或调整 ignore 规则。

---

## 14. 当前项目定位

当前系统不是：

```text
EEG classification
label decoding
channel selection benchmark
speech recognition
subject-specific generator
```

当前 v9 系统是：

```text
EEG-only input
  -> subject-robust neural token representation
  -> speech semantic/prosody latent prediction
  -> conditional codec-space transport
  -> future neural codec waveform rendering
```

最重要的判断顺序：

```text
1. canonical cache / split protocol 是否干净
2. subject_val semantic gate 是否通过
3. subject_test 是否仍为正向
4. collapse/leakage 是否受控
5. codec transport 是否超过 teacher/oracle/mean prior
6. waveform rendering 是否通过 ASR/perceptual/acoustic metrics
```

截至当前审计，v9 完成了“正确问题定义 + 可运行骨架 + protocol audit + smoke/early diagnostic”，但还没有完成“可宣称的 EEG-to-Speech 结果”。