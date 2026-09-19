# eeg-recon-0809

新增独立实验 **aligned_speech_v1**：保留真实时间的 EEG → HuBERT 序列 → 原生 mel → HiFi-GAN，
包含公开语音预训练、已知被试的新句子分组、分阶段训练、反事实评估和匿名听音。
运行说明见下文 **Recovery v3** 与 **Generative recovery** 两节；工程验证不等于已完成可懂度验证。

EEG-to-audio reconstruction experiment bundle. 两个数据集统一放在
`data/` 下；原始 EEG、WAV 和内部音频均保留在本地，不进入 Git。

## 数据布局

```text
eeg-recon-0809/
├── data/
│   ├── ds004940/              # 英语句子听觉 EEG；Active（v1）/ Active+Passive（v2）
│   └── karaone/               # KaraOne 想象/发音语音 EEG（scripts/download_karaone.sh）
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

### KaraOne（想象 / 发音语音 EEG，Zhao & Rudzicz 2015）

14 名参与者（MM05–MM21、P02），每人一个 `tar.bz2`（1.3–2.4 GB，合计约 24.8 GB），
来源 `https://www.cs.toronto.edu/~complingweb/data/karaOne/`。脚本支持断点续传、
按服务器大小校验、`tar` 完整性检查，并把作者的绝对路径前缀去掉，解压到
`data/karaone/<参与者>/`；作者代码 `src.zip` 和论文放在 `data/karaone/_meta/`。

```bash
bash scripts/download_karaone.sh                       # 全部 14 人
SUBJECTS="MM05 P02" bash scripts/download_karaone.sh    # 指定参与者
bash scripts/download_karaone.sh verify                 # 只校验已下载内容
```

DS006104（TMS 音素/词感知）已于 2026-09-15 从本项目移除：数据、下载脚本以及
预处理里的全部 DS006104/TMS 代码路径都已删除。

## 已移除的 joint / `train_joint` 流水线（2026-09-18）

早期的 harmonized-v3 joint content pilot（`train_joint.py`、`evaluate_joint.py`、
`run_joint_*.sh`、`run_ds004940_*.sh`、MFCC renderer、Griffin–Lim 诊断导出等）及其
configs、tests 和 artifacts 已从本目录删除，以保持项目精简。完整的代码快照、
checkpoint 契约和 README 保留在只读备份
`../eeg-recon-0809_explore_8h_v1_backup/`（commit `f8c6f247`）；需要复现该路线时从那里复制到
新目录运行。该路线的结论（renderer 先验带来"像语音"的质感、零 EEG 与真实 EEG 无差别）见
`reports/DS004940_PROGRAMME_STATUS.md` §6 与 §16。

仍保留的共享基础：`scripts/prepare_training_data.py`、`scripts/prepare_m0_artifacts.py`、
`scripts/eeg_preprocessing_qc.py`、`scripts/cache_speech_targets.py`、
`app/src/eeg2speech/{data,model,losses}.py` 和 `configs/training_data_*.yaml`
（aligned 路线的数据配置继承自 v4 → v3）。

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

## Generative recovery (diffusion decoder on the frozen v3 encoder)

`app/generative_recovery.py` — added 2026-09-18 after the v3 WAVs were judged not to
sound like speech. Diagnosis: the v3 route regresses mel/HuBERT under L1, and with
weak single-trial EEG the loss-optimal output is the *conditional mean* — an
onset-locked blur of the 320 training sentences (the periodic "syllable blob" in
`outputs/aligned_recovery_v3/audio_comparison/audio_comparison.png`,
`prediction_variance_ratio` ≈ 0.09). The legacy `explore_8h_v1` route sounded like
speech only because its audio-trained renderer projected any input onto the speech
manifold (zero-EEG ≡ real EEG there). This module keeps the v3 encoder (the
CLIP-aligned EEG embedding) frozen and replaces the deterministic decoder with a
conditional diffusion model p(mel | EEG features): every sample lies on the speech
manifold, and the EEG enters through classifier-free guidance, so its influence is
explicit and is tested against the same counterfactuals as before.

Conditioning (`--conditioning compact`, 18 channels at mel frame rate): the encoder
head output projected on the top-16 PCs of the frozen teacher space, the predicted
duration, and the frozen envelope decoder's prediction. The raw 128-d trunk
(`--conditioning trunk`) was abandoned: the decoder memorises per-trial fingerprints
and the validation EEG-advantage goes negative. With `--crossfit`, the features come
from two encoders/envelope decoders trained on disjoint content halves
(`crossfit` stage), so every training trial is conditioned by models that never saw
its sentence — the same situation as validation, which removes the over-trust that
train-fold features otherwise teach the decoder.

```bash
PY=.venv-aligned-local/bin/python
$PY app/generative_recovery.py cache                         # EEG/mel/teacher -> outputs/generative_recovery/cache (≈4 min)
$PY app/generative_recovery.py crossfit                      # 2 fold encoders + 2 fold envelope decoders (≈25 min)
$PY app/generative_recovery.py train --run crossfit --crossfit \
    --updates 8000 --eval-every 500 --condition-noise 0.1 --condition-dropout 0.05   # ≈45 min, resumable
$PY app/generative_recovery.py export --crossfit --checkpoint outputs/generative_recovery/crossfit/best.pt \
    --export-output outputs/generative_recovery/crossfit/export_validation --limit 240 --guidance 2 --steps 50
$PY app/generative_recovery.py compare --export-output outputs/generative_recovery/crossfit/export_validation
$PY -m unittest tests/test_generative_recovery.py
```

Run these one at a time; each training process holds ~2 GB. The export writes, per
validation trial, `original.wav`, `native_mel_oracle.wav` (vocoder ceiling),
`teacher_oracle.wav` (audio → teacher → diffusion, the decoder's own ceiling),
`regression.wav` (the v3 route on the same trial), and diffusion samples for
`correct` / `zero` / `wrong_trial` / `time_block_shuffle` / `pooled` (EEG averaged over
all presentations of the sentence) — all six samples share one noise seed per trial,
so differences between them are due to the conditioning only, and nothing uses the
oracle duration. `compare` scores them with `app/audio_comparison.py` measures (STOI,
PESQ, MCD, envelope and modulation correlation, 2AFC against a duration-matched foil
sentence), adds three speech-likeness measures of the mel (spectral contrast, temporal
modulation depth, frame flux — real speech is high on all three, a conditional-mean
blur is low), and paired real-minus-control differences with bootstrap CIs. Results
and their interpretation: `reports/DS004940_PROGRAMME_STATUS.md` §16.

## KaraOne 想象语音路线

```bash
bash scripts/download_karaone.sh          # 14 人，约 24.8 GB
bash app/run_karaone.sh all               # prepare -> baselines -> transfer
```

`scripts/prepare_karaone.py` 把原始 `.cnt`（含 VEO/HEO/EKG/EMG）处理成 `artifacts/karaone/shards/`；
`app/karaone_baselines.py` 用分块 CV + 置换检验 + 伪迹对照做 11 类提示词解码；
`app/karaone_transfer.py` 检验 DS004940 预训练编码器在 KaraOne 上的迁移和听→想象跨阶段泛化。
结果与解读见 `reports/KARAONE_STATUS.md`。

### 报告入口

- `reports/DS004940_PROGRAMME_STATUS.md` — 感知语音重建的当前状态（复现、test 集、v2 数据、
  Broderick 预训练、群体解码、N400 探针、音频/听觉对比）。
- `reports/KARAONE_STATUS.md` — 想象语音路线（研究的真正目标）的现状与结论。
- `reports/CURRENT_EEG_SPEECH_STATUS.md` — 已被取代，保留早期 v2 指标问题的诊断。

`tests/test_analysis_traps.py` 固定了分析中踩过的坑（oracle 时长捷径、foil 必须是不同句子、
置换检验的层级、median 模板基线），改动分析代码后请先跑它。
