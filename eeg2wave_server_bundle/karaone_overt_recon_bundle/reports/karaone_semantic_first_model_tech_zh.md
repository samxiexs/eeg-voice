# KaraOne 语义辅助 EEG-to-Speech 当前模型技术说明

> 版本：2026-06-30 v10 final pipeline  
> 范围：`karaone_overt_recon_bundle` 当前 KaraOne v10 Final；v9.1 Clustered Channel-MoE Semantic Flow 作为上一版 baseline；v9 Neural Semantic Transport 与 v8 Soft-Positive Cross-Subject 作为历史对照。  
> 当前主目标：未知 EEG -> speech semantic/prosody latent -> diagnostic reconstructed wav。推理时默认只输入 EEG，不使用真实 prompt label、真实 onset、真实 target audio。  
> 当前状态：v10 已完成代码骨架、训练脚本、一键 runner、train-only cluster/audit、Channel-MoE、v10 EEG-specific alignment losses、subject-holdout gate、训练图、diagnostic wav、wav 对比图与 smoke 验证。当前仍不能宣称 EEG-to-Speech 已成功，只能说 v10 已具备可系统训练和审计的最终版实验管线。

---

## 1. 当前定位

KaraOne 的目标没有改变：

```text
未知被试 EEG -> 对应语音内容 / speech latent / waveform
```

但当前不能把任务理解为“直接从 EEG 生成好听 wav”。v10 的核心判断是：

```text
先证明 EEG -> speech semantic/prosody decoding
再讨论 codec latent / waveform generation
```

原因是 EEG-to-Speech 很容易出现伪成功：

```text
训练 loss 下降，但模型只学到 speech prior
retrieval wav 听起来像语音，但不是 EEG-specific signal
semantic cosine 有改善，但没有超过 zero/mean/label prior
subject-holdout 上正向信号不稳定
```

因此 v10 的成功标准不是“生成了 wav”，而是：

```text
EEG prediction 在 subject_val / subject_test 上稳定优于 zero-EEG、mean prior 和 label/speech prior
```

---

## 2. 从 v8 到 v10 的变化

| 维度 | v8 | v9 | v9.1 | v10 final |
|---|---|---|---|---|
| 核心路线 | EEG embedding -> train-bank retrieval -> Mel/wav diagnostic | EEG -> semantic/prosody latent -> codec flow | train-only cluster + Channel-MoE + cluster-aware flow | v9.1 架构 + EEG-specific semantic margin + gate-aware selection + one-click artifact pipeline |
| EEG 表示 | pooled embedding + hand-crafted features | raw 62ch channel-time patch tokens | sparse Channel-MoE + Transformer | 沿用 Channel-MoE，并把 gate entropy/collapse 纳入 selection |
| 数据组织 | subject-holdout + retrieval bank | canonical target bank | train-only EEG/speech/cross-modal cluster bank | 继续 train-only cluster/audit，smoke 与正式 run 独立 cluster 文件 |
| 语义对齐 | soft-positive InfoNCE | soft-OT、sequence cosine、InfoNCE、token CE、prompt CTC/CE | cluster positives、hard negatives、gate consistency | zero/mean prior margin、cross-subject semantic NCE、same-label prototype pull、balanced prompt CE |
| 音频生成 | Griffin-Lim / retrieval diagnostic | codec latent flow, waveform gated | NeuroSonic-style codec-space flow | 当前先输出 diagnostic semantic-retrieval wav；只有 gate 过后才允许 waveform claim |
| 成功证据 | retrieval/听感辅助 | semantic/prosody gate | v9.1 research gate | v10 research gate + subject_test signs + anti-collapse + artifact report |

v9.1 已经证明工程骨架可运行，但你已跑完的 thinking run 没通过 gate：`semantic_top3_gain_over_mean`、`same_label_cross_subject_gain`、`prompt_acc` 和 subject_test 泛化仍不足。v10 不是继续修补 v9.1，而是把这些失败模式直接写进训练目标、模型选择和输出报告。

---

## 3. v10 当前代码入口

核心 runner：

```text
karaone_overt_recon_bundle/run_karaone_v10.sh
```

配置、训练和评估：

```text
app/configs/karaone_v10.yaml
app/scripts/train_karaone_v10.py
app/scripts/audit_karaone_v10_protocol.py
app/scripts/plot_karaone_v10_training.py
app/scripts/synthesize_karaone_v10.py
app/scripts/summarize_karaone_v10_run.py
app/src/karaone_v10/
app/tests/test_karaone_v10_smoke.py
```

注意：root-level `run_karaone_v10.sh` 会被当前上层 `.gitignore` 忽略，但文件已存在且可执行。本地运行不受影响；如果之后要提交，需要 `git add -f` 或调整 ignore 规则。

---

## 4. v10 Canonical Pipeline

v10 的一键流程是：

```text
Stage 0: train-only EEG/speech/cross-modal cluster bank + leakage audit
Stage 1: EEG semantic/prosody alignment training
Stage 2: subject_val / subject_test research gate evaluation
Stage 3: training curves + gate/collapse/channel figures
Stage 4: diagnostic reconstructed wav
Stage 5: wav comparison figures + final summary report
```

端到端数据流：

```text
raw 62ch EEG
  -> per-channel descriptor
  -> sparse Channel-MoE top-k gate
  -> EEG patch Transformer
  -> content stream C: HuBERT semantic latent / semantic tokens
  -> prosody stream P: active / duration / energy / onset
  -> v10 cluster-aware + prior-aware alignment losses
  -> subject-holdout semantic/prosody gate
  -> diagnostic semantic retrieval wav
  -> waveform comparison report
```

---

## 5. 数据与 Cluster Bank

v10 继续使用 v9.1 的 train-only cluster 约束：

```text
subject_train:
  fit EEG cluster centroids
  fit speech semantic/prosody cluster centroids
  fit cross-modal cluster centroids

subject_val / subject_test:
  only assign to existing clusters
  never fit centroids
```

cluster 用途：

```text
same speech cluster + different subject -> soft positives
same EEG cluster + different label -> hard negatives
same label + different EEG cluster -> hard negatives
cluster leakage / stability -> audit metrics
```

v10 runner 中 smoke cluster 文件会带 run suffix，避免覆盖正式 cluster bank。

---

## 6. Channel-MoE EEG Encoder

v10 沿用 v9.1 的 62 通道 Channel-MoE：

```text
每个通道 -> descriptor(mean/std/logvar/abs/envelope proxy/diff-energy)
descriptor + channel embedding -> gate logits
sparse top-k gate(default top_k=16)
channel expert assignment
expert outputs -> EEG patch tokenizer / Transformer
```

正则与解释输出：

```text
channel load balance
gate sparsity
gate entropy floor
cross-subject gate consistency
channel dropout
```

每个 run 保存：

```text
channel_gate_summary.csv
channel_importance_by_stage.csv
channel_importance_by_label.csv
channel_importance_by_cluster.csv
top_channels_report.md
figures/channel_gate_top_channels.png
```

解释原则：Channel-MoE 不在训练前硬删通道。通道重要性只能先作为 gate-based diagnostic；要做神经科学结论，还需要 permutation 或 leave-channel-out 验证。

---

## 7. Speech Latent 与音频生成约束

v10 仍然采用 factorized speech latent：

```text
C: semantic content, HuBERT/wav2vec-style latent, semantic tokens
P: prosody/event, active mask, duration, energy, onset
A: codec/acoustic latent, future neural codec / flow target
```

音频生成与图像生成不同。音频不是静态二维像素采样，关键约束是：

```text
time continuity
duration / onset
harmonic and envelope structure
phase or codec consistency
active speech sparsity
chunk boundary continuity
```

因此当前 v10 不把 waveform 作为主训练目标。wav 输出分两级：

```text
gate fail:
  EEG semantic summary -> train-only semantic retrieval wav
  output = diagnostic artifact

gate pass:
  才允许进一步训练 / 报告 codec-space flow 或 neural codec waveform
```

---

## 8. v10 Loss 设计

v10 保留 v9.1 alignment loss：

```text
monotonic soft-OT
sequence semantic cosine
global InfoNCE
soft-positive InfoNCE
semantic-token CE
prompt CTC
prompt CE
prosody/event loss
subject adversarial
CORAL / group-DRO
cluster NCE
hard-negative margin
Channel-MoE regularizers
```

v10 新增对 v9.1 失败模式的直接约束：

```text
zero_prior_margin:
  要求 EEG prediction 优于 zero-EEG baseline

mean_prior_margin:
  要求 EEG prediction 优于 train-bank mean speech prior

cross_subject_semantic_nce:
  same label / same speech cluster, different subject 作为正样本

same_label_prototype_pull:
  预测向跨被试同标签 prototype 靠近

balanced_prompt_ce:
  缓解 prompt label 不均衡，提升 prompt_acc

pairwise_decorrelation:
  惩罚 prediction collapse，降低 pairwise_corr
```

v10 selection score 也改成 gate-aware：

```text
奖励 semantic_over_zero_gain / semantic_over_mean_gain / top3_gain / cross-subject gain / prompt_acc
惩罚 pairwise_corr 过高、std_ratio 越界、channel gate entropy collapse、subject leakage
如果 subject_test signs 反向，额外扣分
```

---

## 9. v10 Research Gate

subject_val 必须满足：

```text
semantic_over_zero_gain > 0.01
semantic_over_mean_gain > 0
semantic_top3_gain_over_mean > 0.02
same_label_cross_subject_gain >= 0
prompt_acc >= 0.13
pred_std_ratio_median in [0.7, 1.5]
pred_pairwise_corr_median < 0.75
channel_gate_entropy_mean > 0.20
```

subject_test 不强行用同一个阈值做 checkpoint selection，但必须报告同向性：

```text
semantic_over_zero_gain 不应明显反向
semantic_top3_gain_over_mean 不应明显反向
same_label_cross_subject_gain 不应明显反向
collapse 指标不能失控
```

只有 semantic/prosody gate 过了，才允许把 codec/wav 结果作为主结果；否则所有 wav 都必须写成 diagnostic。

---

## 10. 当前验证状态

v9.1 已跑过 thinking 训练，但没有通过 research gate；因此当前结论是：

```text
v9.1 有弱语义信号迹象，但不能证明跨被试 EEG-to-speech semantic decoding 成功。
```

v10 已完成工程验证：

```text
python3 app/tests/test_karaone_v10_smoke.py
```

通过内容：

```text
v10 model forward/backward
pretrain / alignment / transport loss smoke
subject_idx 不进入 inference forward
v10 metrics / selection score
training plot generation
dummy wav comparison generation
```

完整 runner smoke 也已通过：

```text
cluster/audit -> align smoke -> training figures -> diagnostic wav -> wav comparison -> summary report
```

smoke 输出包含：

```text
metrics/history.json
metrics/latest_metrics.json
figures/training_curves.png
figures/gate_metrics.png
figures/collapse_metrics.png
figures/channel_gate_top_channels.png
wavs/listening_manifest.csv
wavs/recon_*.wav
wavs/reference_*.wav
wavs/waveform_compare/*.png
reports/v10_run_summary.md
```

---

## 11. 推荐运行命令

### 11.1 本地 MPS 跑 thinking

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/karaone_overt_recon_bundle

mkdir -p artifacts/outputs_karaone/logs

RUN_TAG=v10_thinking_50ep_mps_$(date +%Y%m%d_%H%M%S)

nohup env \
DISABLE_TQDM=1 \
VERBOSE=1 \
LOG_INTERVAL=10 \
DEVICE=mps \
LOG_FILE=artifacts/outputs_karaone/logs/${RUN_TAG}.log \
./run_karaone_v10.sh full thinking 50 ${RUN_TAG} \
> artifacts/outputs_karaone/logs/${RUN_TAG}.nohup.log 2>&1 &
```

### 11.2 服务器 CUDA 跑 thinking

```bash
cd ~/aicloud-data/eeg2wave_server_bundle/eeg2wave_server_bundle/karaone_overt_recon_bundle

mkdir -p artifacts/outputs_karaone/logs

RUN_TAG=v10_thinking_50ep_cuda_$(date +%Y%m%d_%H%M%S)

nohup env \
DISABLE_TQDM=1 \
VERBOSE=1 \
LOG_INTERVAL=10 \
DEVICE=cuda \
LOG_FILE=artifacts/outputs_karaone/logs/${RUN_TAG}.log \
./run_karaone_v10.sh full thinking 50 ${RUN_TAG} \
> artifacts/outputs_karaone/logs/${RUN_TAG}.nohup.log 2>&1 &
```

### 11.3 服务器 CUDA 跑 stimulate / overt_like

```bash
RUN_TAG=v10_stimulate_50ep_cuda_$(date +%Y%m%d_%H%M%S)

nohup env \
DISABLE_TQDM=1 \
VERBOSE=1 \
LOG_INTERVAL=10 \
DEVICE=cuda \
LOG_FILE=artifacts/outputs_karaone/logs/${RUN_TAG}.log \
./run_karaone_v10.sh full stimulate 50 ${RUN_TAG} \
> artifacts/outputs_karaone/logs/${RUN_TAG}.nohup.log 2>&1 &
```

查看日志：

```bash
tail -f artifacts/outputs_karaone/logs/${RUN_TAG}.log
```

---

## 12. 输出解释规则

v10 每次 run 的输出目录：

```text
artifacts/outputs_karaone/karaone_v10_final_align_<thinking|overt_like>_<RUN_TAG>/
```

核心文件：

```text
metrics/latest_metrics.json
metrics/history.json
figures/training_curves.png
figures/gate_metrics.png
figures/collapse_metrics.png
figures/channel_gate_top_channels.png
wavs/listening_manifest.csv
wavs/waveform_compare/original_vs_pred_env_scaled_contact_sheet.html
reports/v10_run_summary.md
```

论文/汇报中的表述边界：

```text
可以说：
  已建立 v10 semantic-first EEG-to-Speech pipeline
  已实现 train-only cluster audit、Channel-MoE、v10 prior-aware alignment 和一键 artifact 输出
  已能生成 diagnostic wav 和 waveform comparison figures

不能说：
  已成功实现跨被试 EEG-to-Speech
  已生成由 EEG 驱动的可理解语音
  Channel-MoE 已证明哪些脑区/通道一定负责语音
```

当前最重要的实验问题：

```text
v10 50 epoch thinking / stimulate 后：
1. subject_val v10 research gate 是否通过
2. subject_test signs 是否同向
3. collapse 指标是否恢复到合理范围
4. prompt_acc 是否高于 0.13
5. same_label_cross_subject_gain 是否不再为负
```

如果这些不成立，wav 仍只能是 diagnostic；如果成立，下一步才是训练/验证 codec-space flow 与真正 waveform decoder。

