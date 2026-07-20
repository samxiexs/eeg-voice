# Combined 0715 V1：FEIS、KaraOne 与 ds004306 联合训练方案

## 1. 目标与边界

第一版复用 KaraOne `0715` 的两阶段 codec-token 方法，但输入改为三个数据集统一后的 imagined EEG。目标分为两层：

1. **内容层**：受试者、数据集和 trial 未见过的 EEG 能否预测语音类别及共享音频条件。
2. **样本级声学层**：KaraOne 的 EEG-conditioned EnCodec code 是否优于 label-only prior。

FEIS 和 ds004306 不能被解释为 KaraOne 式同 trial 波形监督：FEIS 是 subject-label canonical audio，ds004306 是类别级候选音频。因此 ds004306 默认只提供 label supervision；只有 `app/configs/ds_audio_audit.yaml` 改为 `enabled: true` 且人工完成音频语义确认后，才启用弱 prototype alignment。

当前 `karaone_0715` 只有架构和 smoke-test 基线，并没有完整成功训练结果。Combined V1 是“0715-compatible imagined-EEG adaptation”，不是原 KaraOne overt 实验的数值复现。

### 当前实现状态与正式训练边界

本版已经补齐 signal probe、真实 EnCodec round-trip、六类 synthesis control、FEIS multi-positive、预处理 QC 和 v2 checkpoint lineage。它们解决的是可验证性、对照组和可复现性问题，**不等于此前发现的所有正式训练阻断项都已关闭**。

在解释 60/80 epoch 正式结果之前，仍须单独修复并验证：dataset-slice distillation 的 mask/重新归一化、ds004306 audio-audit 开关对所有 loss 的实际约束、基于 validation 的 EEG checkpoint 选择和自动 gate，以及 AudioConditionEncoder 内对有效音频长度的处理。在这些 P0 项关闭前，长训练只能标记为 exploratory，不能声称已经重建了经受试确认的幻听或想象语音。

## 2. 必须先完成的数据修正

### 2.1 KaraOne valid-length baseline

联合预处理之前会把 `stage__clearing__valid_lengths` 忽略，导致 padding 尾部进入每个 trial 的 median/MAD。`scripts/preprocess_combined_eeg.py` 已改为只使用每条 clearing 的有效样本。

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle
bash run_preprocess.sh --datasets karaone --overwrite
```

如果需要三数据集全部重建：

```bash
bash run_preprocess.sh --datasets feis karaone ds004306 --overwrite
```

选择性重建会保留未选数据集的 manifest/QC 行。

### 2.2 sample key 与 split

全局 trial key 使用 `subject_recording_id + ":" + trial_index`。不能使用 `subject_group_id + trial_index`，因为 ds004306 一个 subject 有多个 session。

正式训练使用 `app/configs/split/combined_0715_v1_split.yaml`：

- KaraOne：12 train、P02 validation、MM21 locked test。
- FEIS：除 05、20、21 外 train，20 validation，21 locked test。
- ds004306：8 train、2 validation、2 locked test subject groups。

训练脚本不使用 `eeg_output/manifests/subject_holdout_splits.json` 作为正式 split；该文件只用于预处理随机 QC，并明确写入 `purpose=preprocessing_qc_only`、`authoritative_for_training=false`。所有 probe、cache、训练、validation 和 locked test 只能以 locked YAML 为准。

## 3. 统一数据契约

`unified_trials.csv` 至少包含：

```text
sample_key, dataset, subject_group_id, subject_recording_id, trial_index
label, eeg_relpath, eeg_row, eeg_valid_samples
audio_key, audio_relpath, audio_valid_samples, audio_pairing, pairing_confidence
```

训练输入为：

- EEG：14 个公共通道，256 Hz。
- V1 取前 768 点（3 秒），保留 `eeg_valid_len` mask；完整 1280 点作为后续消融。
- 音频：16 kHz、最多 2 秒、单声道。
- EnCodec 目标：8 个 codebooks × 150 steps，vocabulary 1024。
- `audio_valid_samples` 转换为 `code_valid_steps`；所有 code、包络和 timing loss 都使用有效 mask。

`bash run_preprocess.sh --verify-only` 不重新生成数据，而是扫描现有 manifest 和 EEG NPZ。它验证 `[N,14,1280]`、通道顺序、256 Hz、valid length、trial/label/manifest 映射、NaN/Inf、零 padding、sample key 唯一性，以及 locked split 的互斥和完整覆盖。每个 NPZ/通道的 clip fraction 达到 1% 记 warning，达到 5% 记 error。真实结果先写入 `eeg_output/qc/eeg_verification.jsonl`；critical error 会使命令以非零状态结束。该报告同时绑定 locked split、manifest 和 preprocessing SHA256；验证后若 NPZ、QC、manifest 或 split 改变，训练入口会把报告判为 stale，并要求重新运行 `--verify-only`。

三个数据集标签使用 namespaced global index：

```text
FEIS       0:16
KaraOne   16:27
ds004306  27:30
```

balanced accuracy 和 chance level 始终按数据集计算：FEIS 1/16、KaraOne 1/11、ds004306 1/3。

## 4. 音频 cache 与音频模型

```bash
bash app/run_combined_0715_v1.sh cache --rebuild
```

cache 按唯一 `audio_key` 训练，而不是按 manifest trial 行训练，避免 FEIS canonical audio 被重复 7–10 次以及 ds004306 的三类音频被重复 2078 次。

cache schema 为 `combined-0715-cache-v2`，除 codes、label 和声学辅助目标外，必须包含：

```text
audio_relpaths, audio_valid_samples
encodec_scale, encodec_scale_valid
cache_schema_version
```

所有 per-audio 数组首维必须一致，codes 必须是 `[N,8,150]` 且取值位于 `[0,1024)`。旧 cache 不会静默复用；必须加 `--rebuild` 重建。由于 checkpoint lineage 绑定 cache SHA256，重建 cache 后旧 audio/EEG checkpoint 也必须废弃重训。

ds004306 只使用 canonical `audio_relpath`，不随机使用候选列表。已知 `flower/1.ogg`、`flower/2.ogg` 和 `hammer/2.ogg` byte-identical，因此 candidate list 不可作为 trial target。

保持 KaraOne 0715 的音频主体：

- frozen local EnCodec 24 kHz、6 kbps；
- 3 层 AudioConditionEncoder，`d_model=192`，压缩为 50 个 condition tokens；
- 4 层 MaskGIT decoder；
- 随机 mask ratio 50%–95%，25% full mask，推理 12 iterations；
- label embedding 使用 30 维 namespaced global condition。

默认音频训练参数：60 epochs、batch 8、lr `3e-4`、weight decay `1e-4`、MPS/FP32。

## 5. EEG encoder 与损失

`app/src/combined_0715/model.py` 保持 0715 的多尺度 stem、Transformer、50 learned queries 和 cross-attention，但输入为 `14×768`，不是原 KaraOne 的 `62×768`。当前 EEG 已完成 CAR，因此 V1 使用单流 multiscale temporal stem，kernel `[15,31,63]`、stride 4，不重复拼接 raw+CAR。

subject ID 只用于 adversarial loss，不进入 forward；dataset ID 只用于选择分类头、label slice 和 loss mask。

| 损失 | KaraOne | FEIS | ds004306 |
|---|---:|---:|---:|
| dataset-specific label CE | 1.00 | 1.00 | 1.00 |
| EEG–audio condition alignment | 1.00 | 0.50 | 0，审计通过后 0.20 |
| paired/multi-positive contrastive | 0.25 | 0.25 | 0 |
| exact/coarse code CE | 0.50，warmup 20 epochs | 0.15，仅 q0/q1 | 0 |
| envelope | 0.30 | 0 | 0 |
| onset/duration | 0.15 | 0 | 0 |
| subject adversary | 0.10 | 0.10 | 0.10 |
| audio-label distillation | 0.15 | 0.15 | 0 |
| variance regularizer | 0.05 | 0.05 | 0.05 |

每个 optimizer step 依次取 FEIS 4、KaraOne 4、ds004306 4 个样本，按 dataset 权重 `0.40/0.40/0.20` 累积梯度。默认 EEG 参数：80 epochs、batch 4/dataset、lr `2e-4`、weight decay `1e-3`、gradient clip 1.0、MPS/FP32。

FEIS contrastive 使用 symmetric weighted multi-positive：相同 `audio_key/audio_idx` 为权重 1 的强正样本；同标签、不同受试者、不同 audio 为权重 0.25 的软正样本；同标签同受试者但不同 audio 为 neutral；不同标签才是 negative。KaraOne 只保留 exact pair，同标签非配对项为 neutral；ds004306 不计算 contrastive。训练日志同时记录双向 loss、`extra_positive_fraction` 和 `mean_positive_count`，用于确认软多正样本确实进入优化。

## 6. 运行顺序

推荐且唯一的正式顺序如下：

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle

bash run_preprocess.sh --verify-only
bash app/run_combined_0715_v1.sh probe
bash app/run_combined_0715_v1.sh cache --rebuild
bash app/run_combined_0715_v1.sh audit-audio
bash app/run_combined_0715_v1.sh train-audio
bash app/run_combined_0715_v1.sh train-eeg
bash app/run_combined_0715_v1.sh validate
```

### Phase 0：预处理验证与信号探针

```bash
bash run_preprocess.sh --verify-only
bash app/run_combined_0715_v1.sh probe
```

探针与 audio cache 完全解耦。它只使用 locked YAML 中的 train subject 拟合 `StandardScaler + class-balanced LogisticRegression`，在 validation subject 上报告三数据集结果，不读取 locked test。

每条 EEG 使用 141 维、保留空间信息的 KaraOne 0715 风格特征：14 通道 ×（6 个相对 log band-power：1–4、4–8、8–13、13–20、20–30、30–40 Hz + 4 个时间块 RMS）+ 1 个有效时长（秒）。只读取有效前缀，padding 不进入特征。每个数据集报告 train/validation BA、chance、length-only BA 和相对 length-only 增益。

### Phase 1：cache 与审计

```bash
bash app/run_combined_0715_v1.sh cache --rebuild
bash app/run_combined_0715_v1.sh audit-audio
```

round-trip audit 使用固定 seed，从 locked validation 按“数据集×标签”分层抽样，每组最多 4 个唯一 audio key；不会用 locked-test manifest 行做抽样，也不会 decode test waveform。它使用 cache 中的 exact code 和 scale 执行真实 EnCodec decode，在有效音频区间计算 waveform correlation、SI-SDR 和 RMS-normalized log-spectrogram MAE，并输出逐样本及 dataset-level mean/median/p05/min。

### Phase 2：音频模型

```bash
bash app/run_combined_0715_v1.sh train-audio
```

开发 smoke test 可使用 `--epochs 1`。

### Phase 3：EEG alignment

```bash
bash app/run_combined_0715_v1.sh train-eeg
```

探索性 smoke run：

```bash
bash app/run_combined_0715_v1.sh train-eeg --epochs 1 --allow-failed-gate
```

`--allow-failed-gate` 不能用于正式结果。

### Phase 4：验证与测试

```bash
bash app/run_combined_0715_v1.sh validate
bash app/run_combined_0715_v1.sh test --allow-final-test
```

测试默认锁定；只有 validation gate 通过且其 lineage、audio checkpoint SHA 和 EEG checkpoint SHA 与当前输入完全一致时才允许访问。`--allow-failed-gate` 只允许 exploratory validation，禁止绕过 locked test。

## 7. 验证门槛

Round-trip gate 首先要求所有 cache 结构/manifest 一致性检查为真。然后每个数据集都必须满足：median waveform correlation ≥ 0.65、median SI-SDR ≥ 0 dB、median RMS-normalized log-spectrogram MAE ≤ 12 dB。报告必须记录 config/cache SHA256、固定 seed、抽样 key、codec 参数，以及 `test_audio_waveforms_decoded=false`。

Audio gate：KaraOne audio content BA ≥ 0.50，FEIS ≥ 0.60，label-assisted BA ≥ 0.70，KaraOne q0/q1 相对 label-only 有正增益，且 round-trip gate 全部通过。

EEG gate 对每个数据集要求：

```text
validation BA >= max(chance + 0.03, signal_probe_BA + 0.02)
bootstrap 95% CI lower bound > chance
```

同时要求真实 EEG 优于 zero EEG、shuffled EEG、dataset-only 和 length-only control。只有 KaraOne q0/q1 的 `EEG-conditioned - label-only` 增益为正时，才可报告 trial-level acoustic gain。

FEIS 只报告 canonical-audio 对齐和粗粒度 code 指标；ds004306 只报告三分类和 prototype retrieval，不报告逐 trial 波形重建。

## 8. 生成与评估

每次 synthesis 在 `<output>/<dataset>/<split>/` 下生成一个 reference 和六种模式：

```text
reference/
codec_oracle/
eeg_conditioned/
label_only/
zero_eeg/
shuffled_eeg/
dataset_only/
synthesis_manifest.json
```

- `codec_oracle`：cached true codes + cached exact scale。
- `eeg_conditioned`：真实 EEG condition + EEG 预测标签概率。
- `label_only`：零 condition + 同一 trial 的 EEG 预测标签概率，不使用 true label。
- `zero_eeg`：零 EEG 经完整 EEG encoder，保留 valid length 和 dataset head。
- `shuffled_eeg`：固定 seed、同数据集同标签、无自环置换；condition 和标签概率均来自错误 EEG。
- `dataset_only`：零 condition + train manifest 的该数据集经验类别先验。

主结果同时报告六种生成模式；跨数据集汇总使用 macro mean，不能用 trial 数量较多的数据集掩盖其他数据集表现。`synthesis_manifest.json` 保存 sample/audio key、subject、label、pairing、shuffle 来源、checkpoint/hash、lineage，以及每种模式的 q0/q1 和波形指标。

生成 validation 样例：

```bash
/opt/anaconda3/bin/python app/scripts/synthesize_combined_0715.py \
  --cache artifacts/combined_0715_v1/cache/combined_0715_encodec_codes.npz \
  --audio-checkpoint artifacts/combined_0715_v1/audio/checkpoints/best.pt \
  --eeg-checkpoint artifacts/combined_0715_v1/eeg/checkpoints/best.pt \
  --dataset karaone --split validation \
  --output artifacts/combined_0715_v1/samples
```

validation synthesis 要求 round-trip gate 通过；若只做探索，可显式加入 `--allow-failed-gate`，输出 manifest 会记录 `exploratory=true` 和原因。test synthesis 必须同时提供 `--allow-final-test`、通过的 validation gate 和完全匹配的 checkpoint/lineage，且禁止 `--allow-failed-gate`。程序先只用 config、checkpoint/gate metadata 和 validation report SHA 完成预授权；通过后才打开 manifest/cache/EEG，并再次核验完整 current lineage。因此所有 gate 校验均在实例化/读取 test Dataset 之前完成。

FEIS 只能报告 canonical-audio 对齐和 coarse-code 指标。ds004306 manifest 固定写入 `ds004306_trial_level_claim_allowed=false`，只报告类别/prototype 层面，不把类别候选音频解释成 trial target。只有 KaraOne 在 validation controls 全部通过后才可能支持 trial-level acoustic claim。

正式实验使用 seeds 15、31、47；测试集只能在配置冻结后访问一次。

## 9. 代码、版本和解释限制

关键实现位于 `app/configs/`、`app/src/combined_0715/` 和 `app/scripts/`。checkpoint schema v2 的统一 lineage 包含：config SHA256、locked split SHA256/version、unified manifest SHA256、所有 manifest-referenced EEG NPZ 加 `channel_qc.csv` 的 preprocessing SHA256，以及 cache SHA256/schema version。Audio checkpoint 保存完整 lineage；EEG checkpoint 还绑定实际 audio checkpoint SHA256。validation report/gate 进一步绑定 audio/EEG checkpoint SHA 和 report SHA。

resume、evaluate 和 synthesis 在加载模型/optimizer 或读取 locked test 之前逐字段校验 lineage 与 dependency。旧 checkpoint 没有 schema/lineage 时严格拒绝，不提供 bypass；split、manifest、EEG NPZ、QC 或 cache 任何变化都要求重新生成相应 artifact，并通常需要重新训练。

`data/`、`eeg_output/`、音频、EnCodec cache、checkpoint 和生成 wav 均由 bundle-local `.gitignore` 与仓库根规则忽略；只提交代码、配置和文档。

如果 KaraOne EEG gate 未通过，Combined V1 仍可作为 imagined-speech content decoding 实验报告，但不能声称完成了样本级脑内声音波形重建。ds004306 的类别候选音频存在重复和语义不确定性，也不能作为确认过的 trial-level acoustic target。
