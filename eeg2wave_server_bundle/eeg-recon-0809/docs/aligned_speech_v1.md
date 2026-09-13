# Aligned speech v1：从时间序列 embedding 生成声音

本机、仅用项目训练折适配所有生成模型的新入口见 [本地训练说明](aligned_speech_local.md)。

此版本针对旧实验中“不同 EEG 输出几乎相同”的问题，实现独立的
`EEG → HuBERT 第 9 层序列 → SpeechT5 原生 mel → HiFi-GAN`。
代码可运行、工程测试通过，并不代表已达到未见句子的可懂重建。
真实 EEG 训练、声码器试听和人工评价必须由对应结果文件证明。

## 实际模型与时间接口

- `AlignedEEGModel.forward(eeg, channel_xyz, channel_mask, time_mask)` 不接受音频、句长、文字、类别或目标 mask。
- EEG 必须包含真实的 1178 点，256 Hz，起点 −0.25 秒，全部 time mask 为真。
- 输出 `aligned_sequence: [B,T_hubert,768]`、`sequence_times_s`、`native_mel: [B,80,T_mel]`、`mel_times_s` 和辅助 `global_embedding`。
- 教师输入是从刺激起点开始的 4 秒音频。短音频补静音，长于 4 秒的实验刺激报错；不按句长拉伸，不做逐句声学 CMVN。HuBERT 自身固定的 waveform frontend normalization 沿用预训练配置。
- HuBERT 时间轴根据卷积 kernel/stride 的感受野中心计算。标准 base 模型 4 秒产生 199 帧，步长 20 ms，而非将整个句子拉成 96 帧。
- SpeechT5 centered STFT 产生 251 帧，步长 16 ms。声码器输出仅作固定边界裁剪到 64000 点；所有样本都是 4 秒。
- 神经延迟候选为 0/100/200/300/400 ms，仅用 validation 选择。400 ms 时最后约 20 ms 超出 EEG token 边界，统一使用端点延拓；此规则不依赖目标句长。当前刺激最长 3.842 秒。
- EEG 空间和时序骨干复用旧结构；移除其未使用输出头的训练梯度。新增 speech head 回归固定标准化的教师序列，decoder 直接消费这条序列。
- 音频 decoder 为 256 通道、6 个残差时序卷积块。音频预训练时拟合教师维度均值/标准差，适配和 EEG 阶段冻结这些统计量。
- EEG 对齐损失：标准化序列 Smooth-L1 + 0.2 时间差分 L1 + 0.1 全内容 bank InfoNCE。声学微调另加 mel L1 + 0.2 mel 差分 L1。decoder 在所有 EEG 阶段冻结，但其输入梯度保留。
- MFCC 仅用于独立旧路线基线；旧模型和已有输出不改写。不使用 diffusion。

## 数据与可复现约定

`prepare` 从现有审计/预处理代码建立新的 `artifacts/training_data/aligned_v1`。
新的 source audit 仅锁定 DS004940 Active 所需源文件，不要求 DS006104 或其 auxiliary 文件。
迁移服务器时保留 DS004940 数据目录及旧 50 配对 M0 manifest；旧模型 checkpoint 不用于新训练。
先按既有 QC 筛选全部 DS004940 N400Active 精确配对，不设 512 配对上限。
用并查集合并同一 linguistic content 或相同音频 SHA256；同组的所有被试归入同一个角色。
按内容约 80/10/10 切分，保留原 50 配对 M0 的内容在 train。
M0 固定为原来那 50 配对，声学 decoder 仍只使用新 train fold 适配。

新实验的目标、manifest、训练统计放在 `artifacts/aligned_speech_v1`，checkpoint 和导出放在
`outputs/aligned_speech_v1`。不会将原始 EEG 数据写入 Git。
EEG robust median/MAD 只拟合 train fold，排除无效通道；读取 shard 时检查单位、固定窗、身份和 provenance。
物化前重新校验源文件哈希，dataset 初始化校验实际 EEG shard SHA256。
为兼容旧构建器，全部角色一次性构建；build-only 的 `materialize` 标记不用于训练，
训练与 normalizer 仅使用最终 manifest 中真正的 train/validation/test 角色。
完整训练 content bank 按内容聚合不同音频实现，同一内容的不同被试映射到同一个正例。
物理 batch 小时依然使用完整 bank，而不是把梯度累积当作对比负例。

缓存记录源音频 SHA256、教师目录 SHA256、代码版本、真实时间轴和 manifest SHA256。
checkpoint 记录配置、缓存、初始化权重、代码指纹、seed、优化器及随机数状态。
Ctrl-C 会等待当前 optimizer step 完成后保存，可从 minibatch 边界恢复；完成的 run 再执行会跳过。
修改输入、配置、批量或训练预算须使用新输出目录，不能静默续接。

旧测试结果已被查看，因此新分组仍标记为 exploratory；重新随机分组不能创造独立确认数据。
同时导出 `joint_ood_assignment.csv` 供第二阶段检查。实际进行第二阶段时切换到
`configs/aligned_speech_joint_ood_v1.yaml`：该配置会物化独立 shards、normalizer 和 targets，
并在 joint train fold 内选择新的 5×10 M0；不能混用第一阶段的适配 decoder/checkpoint。

## 环境和运行入口

在 GPU 服务器上安装项目的 `requirements-preprocess.txt`。建议 Python 3.12。
PyTorch 请使用与服务器驱动相容的 CUDA 构建。先用真实 encoder 配置探测显存；训练用 float32，
当前未启用 AMP，以避免在未验证精度时改变结果。物理 batch 默认 4，显存不足可在新 run 中减小。

```bash
cd /absolute/path/eeg-recon-0809
export PYTHON_BIN=/absolute/path/environment/bin/python
export HUBERT_LOCAL_PATH=/absolute/path/hubert-base-ls960
export HIFIGAN_LOCAL_PATH=/absolute/path/speecht5_hifigan
export ALIGNED_DEVICE=cuda
export ALIGNED_BATCH=4

bash app/run_aligned_speech_v1.sh readiness
bash app/run_aligned_speech_v1.sh probe
bash app/run_aligned_speech_v1.sh prepare
bash app/run_aligned_speech_v1.sh cache
```

权重使用本地 Hugging Face 模型目录。教师为 `facebook/hubert-base-ls960`，声码器为
`microsoft/speecht5_hifigan`；缓存和导出会固定实际文件哈希，不隐式下载权重。
不能复用旧 96-frame HuBERT cache，因为它已丢失本版本所需的物理时间坐标。

下载公开语音并预训练音频后端：

```bash
"$PYTHON_BIN" scripts/download_aligned_audio.py --output /absolute/path/public_audio
export LIBRISPEECH_ROOT=/absolute/path/public_audio/LibriSpeech
bash app/run_aligned_speech_v1.sh corpus
bash app/run_aligned_speech_v1.sh audio
```

下载脚本使用 OpenSLR 官方 `train-clean-100`、`dev-clean` 和校验清单，校验归档并拒绝路径穿越或链接成员。
FLAC 分成不重叠的 4 秒块，末尾不足 0.5 秒的块丢弃。train/dev 必须被试和文件分离。
100 小时语音的逐帧 float32 缓存可能需要几十 GB；先确认服务器空间。
解码器先用外部 train 预训练、dev 选模型，再仅在 EEG 新 train fold 音频上适配、validation 选模型。
MFCC renderer 在同一个新 train fold 重新拟合，避免旧 renderer 看过新 validation 内容。
该音频上限对照允许使用真实时长，生成后补静音到固定窗口；真实时长不会进入 EEG 模型。

`audio` 导出三条音频路径：真实 mel oracle、真实 HuBERT 序列 oracle、MFCC oracle。
所有全量音频保存在 `audio_listening/bundles`，匿名试听在 `audio_listening/blind`。

## 人工评估和 EEG 训练

向听者只分发 `blind/`，不要分发 `private_key.csv`、源 WAV 或参考转录。
固定抽取 40 个不同内容，至少 3 位独立听者转写；CSV 中 `listener_id` 可替换成实际匿名编号。
`reference_transcripts.csv` 从 DS004940 作者公开的 `N400PvsA_stimuli_parameters.tsv` 自动生成，并逐条标记 `verified=true`；该表按 `stim_file` 提供逐词原始句子。听者只填写匿名音频的自由转写。
听不懂但已完成的回答用 `[unintelligible]`，不能留空或自动生成“人工”转录。

```bash
"$PYTHON_BIN" app/aligned_speech.py score-review \
  --export-root outputs/aligned_speech_v1/audio_listening \
  --transcriptions outputs/aligned_speech_v1/audio_listening/blind/transcriptions.csv \
  --references outputs/aligned_speech_v1/audio_listening/reference_transcripts.csv \
  --output outputs/aligned_speech_v1/audio_review.json

bash app/run_aligned_speech_v1.sh m0
bash app/run_aligned_speech_v1.sh full
```

音频 oracle 必须达到 `max(0, 1−WER)` 平均值 ≥90%。代码验证表格完整性、40 个内容和 ≥3 个 listener，
实际听者是否独立仍由研究组织者核实。未通过不会启动完整 EEG 训练。
M0 可先做工程诊断，但必须同时达到：内容 R@1≥90%、mel 相对模板改善≥10%、预测方差比≥0.25，
并胜过零 EEG、同被试错试次和时间块打乱。通道打乱保留为诊断，不把微小差异当作空间解码成立。

`full` 对 seeds 31/47/73 分别训练 5 个延迟候选，每个先 align 再 finetune。
align 按验证集序列误差选模型；finetune 只有优于模板和错试次时才有合格 best checkpoint，
再按验证集 mel MAE 选择。最多 100 epoch，最少 20 epoch 后连续 10 次未改善停止。
不会因训练检索到达 90% 而停止。
每个 seed 的 5 个候选训练完成后仅在 validation 选延迟，生成选择清单后才允许 test 评估/导出。
实际时间以 `resource_probe.json` 和最初数个 epoch 为准，未测 GPU 时不宣称训练预算。

跨被试加跨句子阶段使用同一个 runner，但必须切换独立目录后重新执行 prepare/cache 和训练折适配：

```bash
export ALIGNED_CONFIG="$PWD/configs/aligned_speech_joint_ood_v1.yaml"
export ALIGNED_OUTPUT="$PWD/outputs/aligned_speech_joint_ood_v1"
bash app/run_aligned_speech_v1.sh prepare
bash app/run_aligned_speech_v1.sh cache
# 随后按相同 audio → 人工音频评价 → m0 → full 顺序运行。
```

EEG 测试包包含 source、mel oracle、正确 EEG、同被试错试次、zero、时间块和通道打乱 WAV。
声音写成 float32 WAV，不逐条响度归一化或隐藏截幅。每个样本固定 4 秒。
匿名听音只比较 correct/wrong_trial；使用同一个 `score-review` 命令，换成对应 `eeg_listening` 目录。
EEG 听音成功要求正确 EEG 词正确率≥70%，且比错试次高≥20 个百分点。

`test_evaluation.json` 报告全量 mel 误差、序列误差、检索、预测方差和反事实增益，
同时给出被试×内容的交叉 bootstrap 区间。`waveform_metrics.csv` 另报告真实波形 10 ms RMS 包络相关。
mel temporal-profile correlation 不是波形包络相关，报告中保留不同名称。

随时运行 `bash app/run_aligned_speech_v1.sh report` 生成 `stage_report.md/json` 阶段评估表；
缺失结果会显示未运行，不会被自动填成通过。

## 本地工程检查

```bash
"$PYTHON_BIN" -m unittest discover -s tests -p test_aligned_speech.py -v
bash -n app/run_aligned_speech_v1.sh
"$PYTHON_BIN" app/aligned_speech.py --help
```

测试覆盖：HuBERT 感受野时间轴、padding、延迟与边界延拓、内容/同 WAV 切分隔离、训练折统计、
完整内容 bank、同被试错试次、冻结 decoder、embedding 替换影响输出、checkpoint 恢复、
EEG 梯度和匿名 WAV 文件导出。合成测试使用假的声码器验证文件协议，不作为神经声码器质量证据。
