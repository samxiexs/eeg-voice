# 本机训练：所有生成链路模型使用项目训练折适配

本入口是独立实验 `aligned_speech_local_v1`，不会覆盖原有模型结果。不再要求 LibriSpeech 数据。沿用内容分组切分；只有 train 用于参数更新，validation 用于选 checkpoint，test 用于最终评价。旧测试数据已被检查，因此结果仍是探索性验证。

## 已准备的环境与初始化

项目内 `.venv-aligned-local` 继承 `/opt/anaconda3/envs/eegvoice` 的科学计算包，另安装 soundfile 0.13.1；已验证 torch 2.11.0、transformers 4.57.6。无需 conda activate。此环境依赖原 eegvoice 环境继续存在。

官方初始化来自 [HuBERT](https://huggingface.co/facebook/hubert-base-ls960) 与 [SpeechT5 HiFi-GAN](https://huggingface.co/microsoft/speecht5_hifigan)。下载版本固定记录在 `models/aligned_local_base/*_download.json`；模型文件保留本地，不上传数据。这里的项目适配不会抹去初始化模型曾接受外部语音预训练的事实。

本机已通过 MPS 合成 EEG 前后向以及真实 WAV 的 HuBERT/HiFi-GAN 梯度检查；探测报告在 `outputs/aligned_speech_local_v1/`。检测受运行环境限制，终端实际运行时自动优先 CUDA，再 MPS，再 CPU。批量默认 1，保持原模型容量。HiFi-GAN 的 STFT 损失在 MPS 模式下通过可微 CPU 运算计算。

## 每个模型如何使用我们的数据

| 模型 | 初始化和训练目标 | 后续状态 |
|---|---|---|
| HuBERT | 官方权重初始化；在训练折语音上用 35% 帧掩蔽、固定原始教师的干净第 9 层表示做蒸馏，另加干净目标回归约束；lr=1e-5，10 epochs | 选验证集最优，重新缓存目标后冻结 |
| HiFi-GAN | 官方权重初始化；训练折原生 mel→原音频，256/512/1024 多分辨率频谱损失；lr=1e-5，10 epochs | 选验证集最优，在 EEG 阶段冻结 |
| 序列→mel 解码器 | 随机初始化，仅用训练折适配后的 HuBERT 表示监督；mel 与时间差分损失 | 由验证集 mel 误差选取，供 EEG 训练 |
| MFCC 基线 renderer | 随机初始化，同一训练折训练 | 仅作为音频上限对照 |
| EEG 编码器和表示头 | 随机初始化；先固定内容 bank 对齐和序列回归，再加 mel 损失微调 | M0 后进入全部 EEG 训练 |

HuBERT 的低层卷积和不被第 9 层输出使用的上层保持冻结，所消费的编码器层参与适配。这是项目内掩蔽表示蒸馏，并非重新实现原始 HuBERT 聚类预训练。HiFi-GAN 采用保守的频谱监督微调，并未加入 GAN 判别器。使用官方初始化，是为了避免在约 18 分钟独立语音上从零训练大型语音模型。

训练音频按独立 audio_key 去重，不把不同被试重复听的录音当成新的语音。HuBERT 干净目标固定，避免与待学习模型一起收缩。声码器训练使用约一秒片段、左右各 16 mel 帧上下文，验证使用完整四秒。所有正式输出固定四秒。

## 直接运行

在 macOS 终端执行：

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/eeg-recon-0809
mkdir -p logs
caffeinate -i bash app/run_aligned_local.sh start 2>&1 | tee -a logs/aligned_local_start.log
```

`start` 顺序为：已有权重检查 → EEG 分片 → 原始教师缓存 → HuBERT 适配 → HiFi-GAN 适配 → 新教师缓存 → 声学解码器/MFCC 训练和三路音频导出 → 50 配对 M0。它不启动 15 组完整 EEG 实验。首次新环境才需要先运行 `bash app/run_aligned_local.sh setup`；当前机器已安装。

其中 `references` 会提前写出 `artifacts/aligned_speech_local_v1/official_reference_transcripts.csv`（402 条）及其 provenance 文件；这一步完全来自官方刺激参数表，不需要你人工创建参考文本。

如果终端没有检测到 MPS，可以显式检查：

```bash
ALIGNED_DEVICE=mps bash app/run_aligned_local.sh probe
```

CPU 可用 `ALIGNED_DEVICE=cpu`，速度可能较慢。`ALIGNED_BATCH=2` 可调整 EEG/声学解码器批量；HuBERT/声码器适配固定 batch=1。不要在恢复既有训练时更改批量、配置或模型目录；签名不匹配会拒绝恢复。

也可逐阶段运行，便于观察：

```bash
bash app/run_aligned_local.sh bootstrap-cache
bash app/run_aligned_local.sh hubert
bash app/run_aligned_local.sh hifigan
bash app/run_aligned_local.sh cache
bash app/run_aligned_local.sh audio
bash app/run_aligned_local.sh m0
```

不应同时运行同一输出目录的两个训练进程。结束一项再运行下一项。

## 盲听与正式 EEG 训练

`audio` 导出 `outputs/aligned_speech_local_v1/audio_listening/`。其中 `reference_transcripts.csv` 已由 DS004940 作者公开的 `N400PvsA_stimuli_parameters.tsv` 自动生成、逐句标记 `verified=true`，不需要人工抄写或核实。脚本会校验该表的 SHA-256，并且拒绝没有官方句子映射的导出。

仍需要的是听者的自由转写：至少 3 名独立听者填写 `blind/transcriptions.csv`，只看 `blind/*.wav` 和 sample_id，不看 private_key 或参考文本。若研究暂时无法招募听者，训练和所有客观 EEG 对照仍可继续；但不能声称通过了“可懂”的人工验收门槛。

```bash
bash app/run_aligned_local.sh review
caffeinate -i bash app/run_aligned_local.sh full 2>&1 | tee -a logs/aligned_local_full.log
```

`full` 要求真实音频上限盲听通过且 M0 报告通过，才会运行 seeds 31/47/73 × 延迟 0/100/200/300/400 ms。验证集选择延迟和 checkpoint，然后测试及导出。全部训练可能很长；目前每步约 4 秒的 EEG 探测不包含数据读取、验证和对照评估，不能据此承诺完成时间。

如果暂时没有听者转写，可运行 `bash app/run_aligned_local.sh full-objective`。它只要求 M0 客观门槛，完整运行会标记为 objective-only；它可以训练、评估和导出 EEG，但不能把结果报告为人工可懂度通过。

```bash
bash app/run_aligned_local.sh status
```

## 恢复和结果含义

声学解码器/EEG 按已有 checkpoint 恢复到批次位置；HuBERT/声码器适配从最后完成的 epoch 恢复，中断的 epoch 会重跑。不要在它们仍训练时生成下游缓存。完成后不要修改 best 模型目录；缓存会绑定模型内容哈希。

训练完成表示优化过程结束，不能等同于“模型已经训练好”。必须查看三路音频盲听、验证曲线、M0 控制增益和最后的未见内容测试。若上限音频不能听懂，先排查 HuBERT/声码器适配和解码器；不要绕过门槛运行全部 EEG 实验。当前未声称取得可懂的 EEG 重建。
