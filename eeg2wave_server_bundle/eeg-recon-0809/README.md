# eeg-recon-0809

新增独立实验 **aligned_speech_v1**：保留真实时间的 EEG → HuBERT 序列 → 原生 mel → HiFi-GAN，
包含公开语音预训练、已知被试的新句子分组、分阶段训练、反事实评估和匿名听音。
运行说明见 [docs/aligned_speech_v1.md](docs/aligned_speech_v1.md)。旧版 MFCC 实验保留用于复现；
新增代码的工程验证不等于已完成真实 EEG 训练或可懂度验证。

EEG-to-audio reconstruction experiment bundle. 两个数据集统一放在
`data/` 下；原始 EEG、WAV 和内部音频均保留在本地，不进入 Git。

## 数据布局

```text
eeg-recon-0809/
├── data/
│   ├── ds004940/              # 英语句子听觉 EEG；默认 Active 试跑
│   └── ds006104/              # 2021 / ses-02；S01-S16
│       └── audio_internal/    # 用户已有的 DS006104 内部音频和映射表
├── scripts/
├── app/
├── reports/
└── bundle_manifest.json
```

## 数据集入口

### DS004940

首轮建议下载 N400Active 的 4 位受试者，验证 EEG、事件和 WAV 对齐；确认流程后
再扩展到全部 Active。Passive 是可选控制条件。

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/eeg-recon-0809

SUBJECTS="001 002 003 004" ./scripts/download_ds004940.sh
SUBJECTS="001 002 003 004" ./scripts/verify_ds004940.sh

# 最终跨受试者模型
SUBJECTS=all ./scripts/download_ds004940.sh
SUBJECTS=all ./scripts/verify_ds004940.sh

# 可选：追加 Passive 控制任务
SUBJECTS=all MODE=active_passive ./scripts/download_ds004940.sh
```

### DS006104（仅 2021 部分）

下载范围固定为 `sub-S01`–`sub-S16` 的 `ses-02`，不包含 2019 年的 P01–P08。
公开部分下载的是 EEG/events；内部 trial-level 音频应放在
`data/ds006104/audio_internal/`，并保留音频映射 manifest。

```bash
./scripts/download_ds006104_2021.sh
./scripts/verify_ds006104_2021.sh
```

两个下载脚本都使用 OpenNeuro 公共 S3 的 include/exclude 过滤方式并支持断点续传。
下载前请确认 AWS CLI 可用；脚本不会自动开始下载。

## Harmonized v3 / joint content pilot

当前模型是 content-first EEG→speech pilot，不是已经验证的 waveform
reconstruction 系统。DS004940 为精确 presented-WAV 配对；DS006104 的
Words/phonemes 只按 `candidate_filename_timing` 使用。S15 官方 auxiliary 已
从固定 commit 获取并通过预登记 SHA256，single-phoneme 仅提供 label supervision。

### 推荐运行入口

运行脚本沿用参考 bundle 的阶段化、日志化和 fail-closed 约定。它们自动选择
`PYTHON_BIN`（优先项目 venv，其次 `/opt/anaconda3/envs/eegvoice`），日志写入
`outputs/joint_pilot_v1/logs/`，并默认复用参考 bundle 中锁定的本地 HuBERT
snapshot。可用同名环境变量覆盖这些路径。

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/eeg-recon-0809

# 只读查看当前 gate / M0 / renderer / Stage-2 状态
./app/run_joint_status.sh

# 已构建 artifact 上的 forward/backward、renderer dry-run 和 5-step smoke
# 不会计入正式 M0
./app/run_joint_smoke.sh

# 从审计、split、M0 artifact 开始；人工听音未通过时会安全地 exit 3
./app/run_joint_pilot_all.sh

# 人工逐条听音、核对转录并真实填写下列 CSV 后，再继续正式流程
# artifacts/training_data/v3/qc/ds004940_pair_review_20.csv
./app/run_joint_after_review.sh
```

也可以单独运行某一阶段：

```bash
./app/run_joint_stage0.sh
./app/run_joint_audio_oracle.sh
./app/run_joint_m0.sh ds006104   # DS006 content-only M0 不受 DS004 人工 gate 阻断
./app/run_joint_m0.sh all        # 3 modes × 3 registered seeds
./app/run_joint_stage2.sh        # 仅在全部 12 个 M0 evaluation gate 通过后执行
```

### Explore mode（不通过 scientific gate 也完整运行）

`explore` 用于检查端到端可运行性、损失曲线和潜在负迁移，不替代 preregistered
pilot。它仍保留数据完整性、source lock、target/normalizer provenance、NaN 和
shape 检查；它只绕过人工 pair-review、M0 成功标准及 Stage-2 的 M0 先决条件。

```bash
# 完整探索流程：audio oracle、M0 三模式、独立 explore Stage 2、single-vs-joint 评估
./app/run_joint_explore.sh

# 先用最小预算验证完整 runner；仍会构建 explore Stage-2 EEG artifacts
EXPLORE_SEEDS="31" EXPLORE_MAX_STEPS=20 ./app/run_joint_explore.sh

# 需要重建既有 exploration artifacts（而非 resume）时使用
REBUILD_M0=1 REBUILD_EXPLORE=1 ./app/run_joint_explore.sh

# audit/split 已完成但 M0 artifact 构建中断时，从 M0 继续
EXPLORE_FROM=m0 REBUILD_M0=1 ./app/run_joint_explore.sh

# M0 artifact、normalizer 和 speech target 已完成，但模型训练中断时继续
EXPLORE_FROM=overfit ./app/run_joint_explore.sh

# 训练每 10 个完整 optimizer step 保存一次；Ctrl-C 后重跑相同命令会自动续跑。
EXPLORE_FROM=overfit CHECKPOINT_EVERY=10 ./app/run_joint_explore.sh

# 只限制尚未开始的 Stage 2 run；M0 和当前已完成/续跑的 Stage 2 checkpoint 不受影响。
EXPLORE_FROM=overfit EXPLORE_STAGE2_MAX_STEPS=2700 CHECKPOINT_EVERY=50 ./app/run_joint_explore.sh
```

如果需要在普通 Mac 上用一个晚上完成一套新的、相互隔离的 Stage-2 比较实验，使用
8 小时 runner。它只运行本问题真正需要比较的 3 种模式
（DS004940-only、DS006104-only、joint）× 3 seeds，不重复 M0；默认每个 run
2,400 optimizer steps。预估训练约 5.5 小时，并为预处理、评估和制图保留约
2 小时。实际速度取决于机器与当前负载。

```bash
# 首次运行；Ctrl-C 后原样重跑即从最近 checkpoint 继续
./app/run_joint_explore_8h.sh

# 明确写出默认预算；可在更慢机器上继续调低 EXPLORE_8H_MAX_STEPS
EXPLORE_8H_MAX_STEPS=2400 CHECKPOINT_EVERY=50 ./app/run_joint_explore_8h.sh
```

这套实验使用独立的 `explore_stage2_8h_v1` 数据 artifacts 和
`outputs/joint_pilot_v1/explore_8h_v1_corrected/` 结果目录，不读取旧实验的完成 checkpoint。
所有训练、validation/test 评估完成后，runner 自动生成以下可复现对比图的 PNG 与
PDF 版本：MFCC single-vs-joint、EEG counterfactual controls、content retrieval
以及训练检索曲线。它还会从第一个 seed 导出每个 dataset/role 最多 3 个**逐 trial
音频对照包**：源 WAV、target-log-mel oracle、single/joint/zero/time/channel-control
的诊断 WAV，以及 `energy_envelope_comparison`、`mel_comparison`、`mfcc_comparison`
图。图、作图源 JSON 的 SHA256 和 subject-bootstrap 汇总保存在：

```text
outputs/joint_pilot_v1/explore_8h_v1_corrected/generalization/figures/
```

只需基于既有评估结果重新生成图片时运行：

```bash
/opt/anaconda3/envs/eegvoice/bin/python app/plot_joint_comparison.py \
  --input-root outputs/joint_pilot_v1/explore_8h_v1_corrected \
  --seeds 31,47,73 --formats png,pdf --dpi 300
```

已有 checkpoint 时，只生成参考项目同语义的逐 trial 音频/能量图，不重新训练：

```bash
# 用户已完成的第一套 8h exploratory 输出
./app/run_joint_audio_pair_comparisons.sh \
  outputs/joint_pilot_v1/explore_8h_v1

# 指定新 run 的 root；默认 seed-31、两个 dataset、validation/test 各 3 个 pair。
./app/run_joint_audio_pair_comparisons.sh \
  outputs/joint_pilot_v1/explore_8h_v1_corrected
```

产物在：

```text
outputs/joint_pilot_v1/<experiment>/generalization/audio_pair_comparisons/
  seed-31/<dataset>/<role>/<trial-id>/
    00_source_reference_16k.wav
    01_target_logmel_griffinlim_oracle.wav
    02_single_eeg_mfcc_griffinlim.wav
    03_joint_eeg_mfcc_griffinlim.wav
    04_joint_zero_eeg_griffinlim.wav
    05_joint_time_shuffled_eeg_griffinlim.wav
    06_joint_channel_shuffled_eeg_griffinlim.wav
    energy_envelope_comparison.png
    mel_comparison.png
    mfcc_comparison.png
    metadata.json
```

当前项目还没有可验证的 neural vocoder，因此 `01`--`06` 是 renderer 预测的
Slaney log-mel 经固定随机种子 Griffin--Lim 得到的**诊断 listening WAV**；它们不
能作为 waveform 重建质量或主实验指标。DS006104 的 `candidate_filename_timing`
pair 也在 metadata 中明确标为 candidate reference，而不是 verified acoustic pair。

每个 mode/stage/seed 都写入一个原子 `training_state.pt`，包含 model、optimizer、
step 和随机状态。中断最多损失最近一个 checkpoint interval 内的工作；已完成的
checkpoint 会自动跳过。若有意重新训练某个已完成 run，直接调用
`app/train_joint.py` 并加 `--restart`。

### DS004940-only 10-hour exploratory run

如果目标是提高 DS004940-only 的严格 held-out content 覆盖，而不是比较 joint
迁移，可使用独立的 10 小时 runner。它只构建 DS004940 的新 Stage-2 artifacts，
并将 train grid 从旧实验的 `4 subjects × 28 contents = 112` 扩展为
`4 × 128 = 512`；validation/test 各为 `1 × 16 = 16`。模型结构和 loss 保持不变，
仅为 9,000-step run 启用从 `3e-4` 衰减到 `3e-5` 的 cosine learning rate。

```bash
caffeinate -dimsu env \
  CHECKPOINT_EVERY=100 \
  ./app/run_ds004940_explore_10h.sh
```

默认运行 3 个 seed（31、47、73）各 9,000 step。按此前约 0.92 秒/step 的速度，
训练约 6.9 小时，并为 EEG shard、HuBERT target cache、evaluation 和机器波动预留
约 3.1 小时。中断后重跑**同一条命令**会从每个 seed 最近的原子 checkpoint 继续。
所有输出隔离于：

```text
outputs/joint_pilot_v1/ds004940_explore_10h_v1/
```

训练完成后，导出 DS004940-only 的原始/目标/EEG/control WAV 及能量、mel、MFCC
对照图（不要求 joint checkpoint）：

```bash
AUDIO_PAIR_MAX_PAIRS=3 GRIFFIN_LIM_ITERATIONS=32 \
  ./app/run_ds004940_audio_pair_comparisons.sh
```

这会导出三个 seed 的 validation/test 各最多三个 pair，结果位于：

```text
outputs/joint_pilot_v1/ds004940_explore_10h_v1/generalization/audio_pair_comparisons/
```

### DS004940 large-scale full-content exploratory run

该实验保留同一 N400Active-only EEG→MFCC/HuBERT 模型，并在锁定的15%坏通道
上限后选择14名完整受试者，将402个 unique speech contents 全部分配到严格双重
OOD split：`10 × 338 = 3,380` train pairs、
`2 × 32 = 64` validation pairs 和 `2 × 32 = 64` locked-test pairs。训练是三个
独立随机 seed 的最多 50 epoch 无放回 shuffle；每 2 epoch 在 validation fold 以
MFCC retrieval MRR 选择 best checkpoint，最少20 epoch、连续8次未改善则早停。

```bash
caffeinate -dimsu env \
  CHECKPOINT_EVERY=100 \
  ./app/run_ds004940_large_scale_v1.sh
```

中断后重复完全相同的命令即可从 `training_state.pt` 继续。每个 seed 同时保留
`best_checkpoint.pt`、`last_checkpoint.pt` 和下游兼容的 `checkpoint.pt`（best）。
输出完全隔离于：

```text
outputs/joint_pilot_v1/ds004940_large_scale_v1/
```

训练完成后，以单独、可续跑的阶段导出所有 held-out validation/test 对照 bundle，
并为 train 的338个内容各导出一个 seed-31 代表样本；总计722个音频/能量/mel/MFCC
bundle。生成 WAV 均为 Griffin–Lim diagnostic，不是经验证的 neural-vocoder output。

```bash
caffeinate -dimsu env \
  GRIFFIN_LIM_ITERATIONS=32 \
  ./app/run_ds004940_large_scale_audio_comparisons.sh
```

人工听音/转录审计仍未完成，因此本节所有模型、图和音频都只能解释为 exploratory
evidence，而不能替代注册结果。

Explore outputs 永远隔离在：

```text
outputs/joint_pilot_v1/explore/
artifacts/training_data/v3/manifests/manifest_explore_m0.csv
artifacts/training_data/v3/shards/explore_m0/
artifacts/training_data/v3/normalizers/explore_m0_joint_ood_fold-0.json
artifacts/training_data/v3/speech_targets/speech_targets_explore_m0.h5
artifacts/training_data/v3/manifests/manifest_explore_stage2.csv
artifacts/training_data/v3/shards/explore_stage2/
artifacts/training_data/v3/normalizers/explore_stage2_joint_ood_fold-0.json
artifacts/training_data/v3/speech_targets/speech_targets_explore_stage2.h5
```

因此 explore 的 checkpoint、metrics 和 comparison 文件不能被 M0 registry 或
`run_joint_stage2.sh` 当作正式实验结果使用。

常用安全覆盖参数：

```bash
PYTHON_BIN=/absolute/path/to/python \
HUBERT_LOCAL_PATH=/absolute/path/to/hubert-snapshot \
./app/run_joint_stage0.sh

# 仅用于重新构建已注册 M0 shards；默认是 provenance-compatible resume
REBUILD_M0=1 ./app/run_joint_stage0.sh

# 调试 runner 时可缩短步数，但此结果不能冒充预注册正式结果
SEEDS="31" MAX_STEPS=20 ./app/run_joint_m0.sh ds006104
```

`run_joint_stage2.sh` 会先物化隔离的 `manifest_stage2.csv`、`shards/stage2/`、
Stage-2 normalizer 和 `speech_targets_stage2.h5`，再运行 single-dataset/joint 的
validation/test 评估与 paired comparison。Stage 2 不会覆盖 M0 artifacts。

### 等价的底层命令（调试用）

```bash
# 1. 审计与冻结 subject × linguistic-content split
/opt/anaconda3/envs/eegvoice/bin/python scripts/prepare_training_data.py \
  --config configs/training_data_v3.yaml audit --fetch-aux
/opt/anaconda3/envs/eegvoice/bin/python scripts/prepare_training_data.py \
  --config configs/training_data_v3.yaml make-splits

# 2. tiny build；先指定 subjects/tasks，不要直接全量 build
/opt/anaconda3/envs/eegvoice/bin/python scripts/prepare_training_data.py \
  --config configs/training_data_v3.yaml build --dataset ds004940 \
  --subjects sub-001,sub-002 --tasks N400Active \
  --limit-trials-per-group 4 --allow-audit-warnings

# 3. normalizer 只拟合 frozen train fold
/opt/anaconda3/envs/eegvoice/bin/python scripts/prepare_training_data.py \
  --config configs/training_data_v3.yaml fit-normalizer \
  --split-csv artifacts/training_data/v3/splits/joint_ood_fold-0.csv --fold 0

# 3b. raw→harmonized PSD gate（只抽固定少量 trial，不改原始数据）
/opt/anaconda3/envs/eegvoice/bin/python scripts/eeg_preprocessing_qc.py \
  --config configs/training_data_v3.yaml

# 4. speech targets；HuBERT 可使用已锁定的本地 snapshot，禁止隐式下载
/opt/anaconda3/envs/eegvoice/bin/python scripts/cache_speech_targets.py \
  --config configs/training_data_v3.yaml --manifest built --include-hubert \
  --hubert-local-path /absolute/path/to/hubert-base-ls960-snapshot

# 5. 两种 channel space 的联合 forward/backward gate
/opt/anaconda3/envs/eegvoice/bin/python app/train_joint.py \
  --config configs/joint_pilot_v1.yaml --mode joint --dry-run

# 输出中必须同时出现 ds004940、ds006104 和 ds006104_label_only

# 6. 工程 smoke，仅验证优化器；不构成 Stage 1 科学结果
/opt/anaconda3/envs/eegvoice/bin/python app/train_joint.py \
  --config configs/joint_pilot_v1.yaml --mode joint \
  --smoke-model --max-steps 4

# 7. 冻结精确的 Stage-2 4/1/1 subject × 28/6/6 content split。
# 只生成 split；M0 全部通过前不要 build/train Stage 2。
/opt/anaconda3/envs/eegvoice/bin/python scripts/prepare_stage2_split.py \
  --data-config configs/training_data_v3.yaml \
  --pilot-config configs/joint_pilot_v1.yaml

# 查看 Stage-2 readiness；M0 或独立 artifact 未完成时返回非零状态。
/opt/anaconda3/envs/eegvoice/bin/python scripts/prepare_stage2_split.py \
  --data-config configs/training_data_v3.yaml \
  --pilot-config configs/joint_pilot_v1.yaml --check-readiness

# 仅在所有正式 M0 evaluation gate 通过后执行。Stage-2 使用独立的
# manifest_stage2、shards/stage2 和 speech_targets_stage2，不覆盖 M0 artifacts。
/opt/anaconda3/envs/eegvoice/bin/python scripts/prepare_stage2_split.py \
  --data-config configs/training_data_v3.yaml \
  --pilot-config configs/joint_pilot_v1.yaml --materialize \
  --hubert-local-path /absolute/path/to/hubert-base-ls960-snapshot

# 8. audio-only MFCC→log-mel/RMS/activity renderer 工程 gate
/opt/anaconda3/envs/eegvoice/bin/python app/train_audio_renderer.py \
  --config configs/joint_pilot_v1.yaml --dry-run
```

训练 checkpoint 固定 source lock、manifest、split、speech target、normalizer
和模型 runtime code 的 SHA256；评估时任一 artifact 改变都会被拒绝。
缺失 speech target 会直接报错，绝不回退为全零监督。

完整 gate、文件职责和当前限制见 [docs/joint_pilot_v1.md](docs/joint_pilot_v1.md)。

## Recovery v3 (speech-frame-masked EEG → speech)

`app/run_aligned_recovery.sh` drives the v3 recovery route (see the 2026-09-14
addendum in `reports/CURRENT_EEG_SPEECH_STATUS.md` for why v2 metrics were
duration-dominated):

```bash
bash app/run_aligned_recovery.sh linear   # G1 gate: linear envelope tracking vs. prior / wrong-trial / shift null
bash app/run_aligned_recovery.sh m0       # closed-loop fit on 50 train trials (gate only)
bash app/run_aligned_recovery.sh full     # fresh-init training; best_passed.pt only when validation controls pass
.venv-aligned-local/bin/python app/evaluate_aligned_recovery.py --checkpoint outputs/aligned_recovery_v3/full_seed322/best_metric.pt --role validation --output outputs/aligned_recovery_v3/eval_validation --export-wavs
```

All gate metrics (`native_mel_mae`, `template_mae` = median template, `*_gain`,
retrieval, `envelope_corr`) are computed on presented-speech frames only;
`full_*` keys report the whole four-second window for comparison with v2.

The v3 acoustic run that passed all validation controls is
`outputs/aligned_recovery_v3/full_seed322_positional/best_passed.pt` (update 1800); its
formal validation report and exported waveforms are in
`outputs/aligned_recovery_v3/eval_validation_positional/`. Reproduce with
`bash app/run_aligned_recovery.sh m0` then
`python app/aligned_recovery.py --mode full --updates 4000 --eval-every 200 --sequence-weight 0 --delta-weight 0 --contrastive-weight 0.5 --m0-checkpoint outputs/aligned_recovery_v3/m0_seed322/best_passed.pt --output <dir>`
(positional code, subject layer and augmentation are on by default).

### v2 data route (prepared, not yet run)

`app/run_aligned_v2_data.sh` audits and materializes both DS004940 tasks
(Active + Passive) for all 22 participants with content roles pinned to the v1
assignment, then rebuilds the target cache and acoustic decoder for the v2
manifest (`configs/aligned_speech_local_v2.yaml`). Afterwards drive the recovery
route with `ALIGNED_CONFIG=configs/aligned_speech_local_v2.yaml
ALIGNED_RUN_ROOT=aligned_recovery_v3_data_v2 bash app/run_aligned_recovery.sh
linear|m0|full|evaluate|sweep`. `ALIGNED_MIX=0.5` enables same-sentence EEG
averaging during training; `ALIGNED_SEEDS="322 323 324" ... sweep` replicates
over seeds and summarises with `app/aggregate_recovery_runs.py`.
