# FEIS + KARA ONE + ds004306 preprocessing bundle

This bundle creates a unified imagined-speech EEG dataset without modifying
anything under `data/`.

The output is written to `eeg_output/` and contains one compressed subject
bundle per recording, a trial manifest, audio cache, subject-disjoint split
manifest, and quality-control reports.

## What is harmonised

- EEG channel order: `F3 FC5 AF3 F7 T7 P7 O1 O2 P8 T8 F8 AF4 FC6 F4`
- EEG shape: 14 channels x 1280 samples (5 seconds at 256 Hz)
- FEIS: existing `thinking` stage, normalized against its `resting` stage
- KARA ONE: existing `thinking` stage, normalized against its `clearing` stage
- ds004306: raw 1024-Hz EEGLAB data, temporary `.set/.fdt` staging, 50-Hz
  notch, 1--40-Hz bandpass, average reference, 256-Hz resampling, then
  `Imagination_*` event epoching
- Audio: lossless mono 16-kHz WAV cache; variable durations are retained and
  recorded in `audio_valid_samples`

The default uses ds004306 **auditory-cued imagination** only.  Its published
audio files are stored by category rather than unambiguously by trial.  The
manifest therefore marks them `weak_category_level`; do not evaluate direct
waveform reconstruction on ds004306 as though every trial had a unique,
confirmed waveform target.

## Run

First validate paths without writing output:

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle
/opt/anaconda3/bin/python scripts/preprocess_combined_eeg.py --dry-run
```

Then launch preprocessing with visible per-subject progress:

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle
bash run_preprocess.sh
```

The default excludes FEIS subject `05`, which the source manifest marks as
anomalous.  To include it, or to also preprocess ds004306 text/image prompts:

```bash
/opt/anaconda3/bin/python scripts/preprocess_combined_eeg.py \
  --data-root data \
  --output-root eeg_output \
  --ds-modalities auditory text image \
  --include-feis-subject-05
```

The run processes ds004306 one continuous recording at a time.  It normally
needs several hours, roughly 4--8 GB peak RAM, and a few GB for outputs.  If a
run is interrupted, simply run the same command again: already completed
subject files are reused and only missing recordings are processed.  Use
`--overwrite` only when deliberately regenerating every output NPZ/CSV.

## Training invariants

- Mask all samples at or after `eeg_valid_samples`.
- Split only on `subject_group_id`; never randomly split trials.
- The **only authoritative training split** is
  `app/configs/split/combined_0715_v1_split.yaml`.
- `eeg_output/manifests/subject_holdout_splits.json` is generated only for
  preprocessing QC (`purpose=preprocessing_qc_only`,
  `authoritative_for_training=false`) and must never select training,
  validation, or locked-test rows.
- Preserve `dataset`, `modality`, `audio_pairing`, and `pairing_confidence` as
  model covariates/weights.  In particular, ds004306 is weaker audio
  supervision than KARA ONE.

## Combined 0715 V1 training

The first combined training implementation is documented in
[`reports/combined_0715_v1_training_plan.md`](reports/combined_0715_v1_training_plan.md).
It adapts the latest KaraOne 0715 codec-token method to 14-channel imagined EEG,
uses a locked cross-dataset subject split, and treats FEIS/ds004306 pairing as
weaker than KaraOne trial-level pairing.

Before training for the first time, rebuild the KaraOne normalized output with
valid clearing lengths:

```bash
bash run_preprocess.sh --datasets karaone --overwrite
```

After preprocessing exists, use the following order.  `--verify-only` scans
the existing manifest/NPZ files, validates the locked YAML split and writes a
real pass/fail QC record bound to the exact split, manifest, EEG NPZ and QC
hashes; it does not rebuild EEG data.  Any later change to those inputs makes
that verification stale and training stops until `--verify-only` is rerun.
The signal probe reads only train and validation EEG and no longer requires an
audio cache.

```bash
bash run_preprocess.sh --verify-only
bash app/run_combined_0715_v1.sh probe
bash app/run_combined_0715_v1.sh cache --rebuild
bash app/run_combined_0715_v1.sh audit-audio
bash app/run_combined_0715_v1.sh train-audio
bash app/run_combined_0715_v1.sh train-eeg
bash app/run_combined_0715_v1.sh validate
```

音频阶段现在是“监督音频初始化 → 三数据集音频微调”，不是随机初始化：
`train-audio` 会检查
`../karaone_overt_recon_bundle/artifacts/outputs_karaone_0715/karaone_0715_audio_codec_s15/checkpoints/best.pt`；
如果不存在，会先自动运行 KaraOne 0715 `prepare` 和 `audio`，再将其 EnCodec
代码生成器/条件编码器权重迁移到 30-label combined 模型并继续微调。KaraOne
的 11 个 label head 行会复制到 combined 的 KaraOne label slice，其余 FEIS 和
ds004306 行保留为新初始化并在 combined 音频监督上训练。初始化 checkpoint 的
SHA256 和迁移报告会写入 combined audio checkpoint；微调默认使用比 scratch 更
小的 `audio_model.finetune_lr=1e-4`；resume 时必须再次提供同一初始化 checkpoint。

如需显式指定已训练好的 KaraOne 音频 checkpoint：

```bash
KARAONE_AUDIO_CHECKPOINT=/absolute/path/to/karaone_0715_audio_codec_s15/checkpoints/best.pt \
bash app/run_combined_0715_v1.sh train-audio
```

`--allow-scratch-audio` 仅用于轻量诊断 smoke；它会明确标记为
`scratch_diagnostic`，不应作为最终 label-to-audio 结果。

当前 wrapper 默认将 `ALLOW_FAILED_GATE=1` 传给 combined EEG/validation
流程，因此音频 gate 未通过时会继续生成 exploratory EEG checkpoint；gate
仍然写为 `passed=false`，不会解锁 locked test。若要恢复严格阻断：

```bash
ALLOW_FAILED_GATE=0 bash run_combined_0715_full.sh
```

本轮音频优化包括按 dataset/label 加权采样和将 combined `lambda_label`
从 `0.25` 提高到 `1.0`。若明确需要重新训练 KaraOne 音频初始化模型及
combined 音频模型，必须同时打开 `RUN_AUDIO=1`：

```bash
RUN_AUDIO=1 RETRAIN_KARAONE_AUDIO=1 ALLOW_FAILED_GATE=1 \
bash run_combined_0715_full.sh
```

已有 cache v2 和 combined audio checkpoint 后，推荐的一键重跑会**复用音频
产物**，只执行 combined EEG 40 epochs、validation、FEIS/KaraOne/ds004306
全量 validation synthesis、分层 reconstruction gate 和
reference-vs-reconstruction 对比图。新训练和重建产物统一写到
`artifacts/0721v1/`，既有 cache 与已微调音频 checkpoint 仍从
`artifacts/combined_0715_v1/` 只读复用：

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle
RUN_NAME=0721v1 COMBINED_DEVICE=mps EEG_EPOCHS=40 bash run_combined_0715_full.sh
```

等价的显式写法是：

```bash
RUN_NAME=0721v1 RUN_AUDIO=0 REBUILD_CACHE=0 RUN_PRECHECKS=0 \
EEG_EPOCHS=40 SYNTHESIS_LIMIT=-1 PLOT_COMPARISONS=1 \
ALLOW_FAILED_GATE=1 COMBINED_DEVICE=mps bash run_combined_0715_full.sh
```

`run_combined_0715_full.sh` 的默认值即为 `RUN_NAME=0721v1`、`RUN_AUDIO=0`、
`REBUILD_CACHE=0`、`RUN_PRECHECKS=0`、combined EEG `40` epochs。启动时会先
检查已有 cache/audio checkpoint 是否存在且非空；缺失时立即报出具体路径，
不会悄悄从头微调音频。随后在全量 validation synthesis 后自动执行
reconstruction audit，并生成 reference-vs-reconstruction pair 图。
WAV 输出位于 `artifacts/0721v1/samples/<dataset>/validation/`，
对比图和 pair CSV 位于对应的 `comparison_pairs/` 子目录。可用
`SYNTHESIS_LIMIT=12` 做快速初始检查，或用 `PLOT_COMPARISONS=0` 跳过绘图。
reconstruction audit 至少需要每个数据集 12 条 validation 样本。

新的重建判据不再把 raw waveform correlation 当作唯一 gate。每条样本同时
报告严格波形相关/SI-SDR，以及 25-ms RMS envelope correlation、envelope
overlap、activity IoU、onset/offset error、20/50/100-ms RMS correlation 和
multi-scale log-spectrogram MAE。对比图中的虚线是 25-ms RMS envelope，横轴
保持真实 2 秒音频时长。

最终 validation gate 分成两层：

- `structure_reconstruction_passed`：EEG-conditioned 的 median envelope
  correlation 与 activity IoU 均至少为 `0.30`；
- `eeg_specific_reconstruction_passed`：同一 trial 下，EEG-conditioned 相对
  shuffled EEG、zero EEG 和 dataset-only 的 paired median gain 均至少为
  `0.03`，且逐 trial 胜率均至少为 `0.55`。

`label_only` 使用同一 trial EEG 预测的标签概率，因此单独报告、但不作为纯负
对照。top-level `passed` 只允许 KaraOne 的 same-trial overt 配对贡献；FEIS
只允许 subject-label canonical 声明，ds004306 只允许 category-candidate
声明。审计失败默认仍写完整报告并继续绘图，只有显式 `--strict` 才以非零状态
退出；locked test 仍要求 top-level gate 真正通过。报告位于
`artifacts/0721v1/eeg/metrics/reconstruction_validation_report.json`。

EEG 训练阶段的 `best.pt` 也改为按 KaraOne validation 的 envelope correlation、
label balanced accuracy 与 onset/duration error 联合选择，不再用最低 training
loss 选择 checkpoint。该改动没有修改 combined YAML，因此现有 audio checkpoint
的 lineage 仍可直接复用。

### 0721v1 EEG loss recipe

0721v1 不再只靠 code CE 或逐点 waveform 指标学习。有效训练目标包括：

- dataset-specific label CE；
- EEG/audio condition alignment；
- FEIS 软多正样本与 KaraOne exact-pair contrastive；
- KaraOne 全 codebook 与 FEIS q0/q1 EnCodec code CE；
- KaraOne envelope MSE，以及 1/5/9 code-step 三尺度 envelope correlation；
- differentiable activity Dice、onset/duration Smooth-L1；
- KaraOne 同标签、不同音频 morphology ranking；
- subject-adversarial、dataset-sliced audio-label distillation 与 variance regularization。

KaraOne train loader 会显式生成同标签、不同 `audio_key` 的成对 batch，避免
morphology ranking 在 batch size 4 时长期失活。每项 loss、三个 envelope scale、
ranking active fraction 和 correct-vs-shuffled correlation 都写入逐 epoch 日志；
完整权重与 recipe version 也写入 `best.pt`/`last.pt` metadata。FEIS 没有可靠的
trial-level envelope target，ds004306 只有 category candidate audio，所以两者
不会伪装成 KaraOne 式 trial-level morphology supervision。

该 wrapper 会显示阶段级总进度条；cache、audit、训练和 synthesis 阶段还会
显示各自的 tqdm 进度。默认不会重复覆盖 KaraOne EEG 输出。如需先重建
KaraOne valid-length 预处理：

```bash
REBUILD_KARAONE=1 bash run_combined_0715_full.sh
```

常用选项包括 `RUN_SYNTHESIS=0`（只运行到 validation）、
`SYNTHESIS_LIMIT=12`（每个数据集只生成 12 条 validation 样本）和
`COMBINED_DEVICE=mps|cuda|cpu`。只有需要重新核验输入时才设置
`RUN_PRECHECKS=1`；只有需要重建 codec cache 时才设置 `REBUILD_CACHE=1`；
只有明确要重新微调音频时才设置 `RUN_AUDIO=1`。从原始预处理检查到音频
微调的完整显式运行方式为：

```bash
RUN_PRECHECKS=1 REBUILD_CACHE=1 RUN_AUDIO=1 \
COMBINED_DEVICE=mps EEG_EPOCHS=40 \
bash run_combined_0715_full.sh
```

locked test 不由该 wrapper 自动执行，必须先人工审查 validation gate。

The cache command now writes `combined-0715-cache-v2`, including source audio
paths, valid sample counts and exact EnCodec scale metadata.  A legacy cache
must be rebuilt with `--rebuild`.  Checkpoints use the v2 checkpoint/lineage
contract and bind the config, locked split, unified manifest, all referenced
preprocessed EEG payloads, QC table and cache.  Legacy smoke checkpoints are
strictly rejected and must be retrained.

`audit-audio` performs a real EnCodec decode on a deterministic stratified
validation sample (at most four unique audio keys per dataset and label).  It
does not sample locked-test waveforms.  Each dataset passes only when median
waveform correlation is at least `0.65`, median SI-SDR is at least `0 dB`, and
median RMS-normalized log-spectrogram MAE is at most `12 dB`, in addition to
all cache structure checks passing.

The locked test phase requires explicit `--allow-final-test`, a passed
validation gate, the exact validation-report SHA, and exact checkpoint/lineage
hashes.  A metadata-only preauthorization happens before the manifest, cache,
or test EEG is opened; full current-lineage validation then runs again.
`--allow-failed-gate` is forbidden for test.  The default configuration uses the first 768 EEG
samples (3 seconds) to match KaraOne 0715; the original 1280-sample output
remains available for later ablations.

## Synthesis controls

The synthesis script exports one reference directory plus six generated
controls under `<output>/<dataset>/<split>/`:

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

`label_only` uses the same-trial EEG label-head probabilities, not the true
label. `shuffled_eeg` is a deterministic, same-dataset/same-label derangement
without self-loops. `dataset_only` uses the empirical training-label prior.
Validation export requires a passed round-trip audit unless
`--allow-failed-gate` is supplied, in which case the manifest is explicitly
marked exploratory.  Locked-test export cannot bypass either gate.

Example validation export:

```bash
/opt/anaconda3/bin/python app/scripts/synthesize_combined_0715.py \
  --cache artifacts/combined_0715_v1/cache/combined_0715_encodec_codes.npz \
  --audio-checkpoint artifacts/combined_0715_v1/audio/checkpoints/best.pt \
  --eeg-checkpoint artifacts/0721v1/eeg/checkpoints/best.pt \
  --dataset karaone --split validation \
  --validation-gate artifacts/0721v1/eeg/metrics/validation_gate.json \
  --output artifacts/0721v1/samples
```

ds004306 audio remains category-level candidate supervision; every synthesis
manifest therefore records `ds004306_trial_level_claim_allowed=false`.
FEIS permits only canonical-audio/coarse-code claims.  Only KaraOne can support
a trial-level acoustic claim, and only after its validation controls pass.

## Remaining formal-study limitations

Dataset-sliced distillation, validation-based EEG checkpoint selection and
automatic reconstruction auditing are now implemented. The remaining limits
are scientific rather than silent code fallbacks: FEIS audio is canonical/coarse
supervision; ds004306 audio is category-candidate supervision; only KaraOne has
same-trial overt audio. Valid-audio-length handling inside the shared audio
condition encoder remains an ablation item. Runs should therefore be described
as exploratory cross-dataset EEG-to-audio reconstruction, not confirmed
reconstruction of hallucinated speech.
