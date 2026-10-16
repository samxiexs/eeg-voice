# 复原的 9 月 recovery v3 代码（DS004940）

这个文件夹是从 git 历史里复原的 9 月份那套代码：DS004940 听觉句子 EEG 的"recovery v3"确定性重建，以及在它上面的条件扩散解码器。它和当前的 `eegspeech/neurospeech/` 完全独立，互不影响。原来的说明保留在 [README.md](README.md)，这份是复原说明。

## 复原了什么

- **来源：** 提交 `9f56b99b`（2026-09-23，"structure-adj"）下的 `eeg2wave_server_bundle/eeg-recon-0809/`，是 9 月 27 日删除各路线之前、v3 驱动脚本和扩散解码器都还在的最后一个提交。用 `git archive` 原样导出，没有改动任何一个文件（71 个文件，约 1.1 MB）。
- **不进 git：** 本文件夹已写入本地的 `.git/info/exclude`（不是 `.gitignore`），`git status` 里看不到它。要删掉整个复原，直接删文件夹和 `.git/info/exclude` 里的 `/recovery_v3_restored/` 一行即可。
- **环境：** 文件夹里的 `.venv-aligned-local` 是指向 `../.venv-aligned-local` 的符号链接，脚本默认用的 `PYTHON_BIN` 因此直接可用（Python 3.12.13、torch 2.11.0、transformers 4.57.6，和这套代码要求的版本一致）。

## 已经验证的

在这个文件夹里运行了全部单元测试：**95 个测试通过，3 个跳过，用时约 6 秒**。这些测试用的是合成数据，只证明代码在当前环境里能导入、能跑通，**不证明能复现 9 月的结果**。

```bash
cd recovery_v3_restored
PYTHONPATH=app:app/src NUMBA_DISABLE_JIT=1 ../.venv-aligned-local/bin/python -m unittest discover -s tests -p "test_*.py"
```

## 没有的东西（所以现在还不能直接训练）

我在整台机器上找过，下面这些都不在磁盘上：

| 缺什么 | 用来做什么 | 怎么补 |
|---|---|---|
| DS004940 的原始 EEG 和刺激 | 一切的起点 | `bash scripts/download_ds004940.sh`（走 OpenNeuro 的 S3，需要 AWS CLI），再 `bash app/run_aligned_local.sh stimuli` |
| 预处理后的 EEG（`artifacts/training_data/aligned_v1`，原来约 3.2 GB） | 训练和评估的输入 | 由原始数据经 `prepare` 重新生成 |
| 本地的 HuBERT 和 HiFi-GAN 权重（`models/aligned_local_base`） | 音频侧教师和声码器 | `bash app/run_aligned_local.sh download`（需要联网下载） |
| 在本项目音频上微调过的 HuBERT / HiFi-GAN、目标缓存 | 音频侧 | `bootstrap-cache`、`hubert`、`hifigan`、`cache`、`audio` 各阶段 |
| 9 月的检查点和输出（`outputs/aligned_recovery_v3/...`，包括通过验证的 `full_seed322_positional/best_passed.pt`） | 复现 0.384 / 0.66 的那组数字 | 只能重新训练；检查点没有进过 git |
| 当时的状态报告（`reports/DS004940_PROGRAMME_STATUS.md` 等） | 完整的结论和解读 | 没有进过 git，找不回来 |

所以这份复原是"代码可跑、数据和权重要重新准备"。下载和训练都需要你自己决定何时启动，我没有运行其中任何一步。

## 从头复现的顺序

以下顺序来自 [README.md](README.md) 和 `app/run_aligned_local.sh`、`app/run_aligned_recovery.sh`，我只核对了脚本里的步骤，没有执行。先用 4 个人试跑（`SUBJECTS="001 002 003 004"`）再扩到全体，是原 README 的建议。

```bash
cd recovery_v3_restored

# 1. 数据（需要联网）
SUBJECTS="001 002 003 004" bash scripts/download_ds004940.sh
SUBJECTS="001 002 003 004" bash scripts/verify_ds004940.sh
bash app/run_aligned_local.sh stimuli

# 2. 音频侧（各阶段依次，download 要联网；hubert / hifigan / audio 是训练）
bash app/run_aligned_local.sh download
bash app/run_aligned_local.sh prepare
bash app/run_aligned_local.sh references
bash app/run_aligned_local.sh bootstrap-cache
bash app/run_aligned_local.sh hubert
bash app/run_aligned_local.sh hifigan
bash app/run_aligned_local.sh cache
bash app/run_aligned_local.sh audio

# 3. EEG 编码器：recovery v3（确定性重建）
bash app/run_aligned_recovery.sh linear     # 先过 G1 门槛：线性包络跟踪是否高于先验/错试次/时间平移
bash app/run_aligned_recovery.sh m0         # 50 条训练 trial 上的闭环拟合，只是门槛
bash app/run_aligned_recovery.sh full       # 新初始化训练；只有验证对照全过才保存 best_passed.pt
bash app/run_aligned_recovery.sh evaluate   # 验证集报告（bootstrap 区间）并导出 wav

# 4. 扩散解码器（在冻结的 v3 编码器上）
PY=.venv-aligned-local/bin/python
$PY app/generative_recovery.py cache
$PY app/generative_recovery.py crossfit
$PY app/generative_recovery.py train --run crossfit --crossfit --updates 8000 --eval-every 500 --condition-noise 0.1 --condition-dropout 0.05
$PY app/generative_recovery.py export --crossfit --checkpoint outputs/generative_recovery/crossfit/best.pt --export-output outputs/generative_recovery/crossfit/export_validation --limit 240 --guidance 2 --steps 50
$PY app/generative_recovery.py compare --export-output outputs/generative_recovery/crossfit/export_validation
```

`full` 的配方是脚本里写明的"2026-09-15 通过全部验证对照的那一版"：声学为主的损失权重（序列权重 0、差分权重 0、对比权重 0.5）加位置编码，4000 次更新。

## 看懂这套代码的入口

| 想看什么 | 文件 |
|---|---|
| v3 的训练入口、门槛、检查点保存 | [app/aligned_recovery.py](app/aligned_recovery.py) |
| EEG 编码器、损失、"只在有语音的帧上算"的掩码 | [app/aligned_recovery_model.py](app/aligned_recovery_model.py) |
| 验证指标和对照（先验、错试次、时间平移、模板基线） | [app/aligned_recovery_eval.py](app/aligned_recovery_eval.py) |
| 正式验证报告与 bootstrap | [app/recovery_reports.py](app/recovery_reports.py) |
| 扩散解码器、交叉拟合、对照导出 | [app/generative_recovery.py](app/generative_recovery.py) |
| STOI、PESQ、MCD、包络相关、2AFC 等音频对比指标 | [app/audio_comparison.py](app/audio_comparison.py) |
| 分析中踩过的坑（时长捷径、foil 必须是不同句子、置换的层级、中位数模板） | [tests/test_analysis_traps.py](tests/test_analysis_traps.py) |
| 音频侧（HuBERT、HiFi-GAN、数据划分）的主体 | [app/aligned_speech.py](app/aligned_speech.py) |

## 和当前项目的口径差异（比较数字之前先看）

当时：只用 DS004940 一个数据集；按句子划分（320 / 41 / 41 个句子），所有参与者出现在每个划分里；EEG 编码器带参与者专属层；指标只在有语音的帧上算，窗口是 4 秒；最好的 0.341 / 2AFC 0.66 是把同一句 17 次呈现的 EEG 平均之后得到的。当前：四个数据集联合，按内容块和人两个方向划分，单试次。两边的数字不能直接比较。
