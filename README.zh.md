# Physiology-Guided EEG-to-Speech via Conditional Flow Matching

项目根目录：`speech_decoding/`。唯一主线是 **ds004408 真实听觉 EEG → TCN 预测 WavLM 时序表征 → CFM → Vocos 原生 Mel → 24 kHz waveform**。

[当前研究方案：NeuroSpeech v2.1](report/neurospeech_research_plan_2026-10-13.md) · [历史执行协议 v1](report/06_execution_plan_2026-10-13.md) · [报告索引](report/README.md)

研究方案已更新为四个英语听觉数据来源（Kymata仅英文）与KaraOne后续迁移、REVE适配和完整生理约束。v2.1规定逐批下载 / 处理、核验后删除raw，仅保留processed与来源记录，数据大小见方案2.4节。下面的命令和原始保留方式对应现有v1代码；新增数据接口、processed-only读取与自动清理尚待实现。

## 当前实现

- 真实 EEG / WAV / TextGrid 同时下载；断点续传；与固定 OpenNeuro Git-annex 快照比对 checksum。
- BrainVision 校准单位读取、128 通道、坏导 / CAR、FIR、时间上下文与严格双重划分。
- 冻结 WavLM-Base 第 6 层逐帧 teacher；原生 Vocos 100-bin natural-log magnitude Mel。
- TCN、头皮分组、正响应延迟池化、局部多正样本对比与 envelope / onset 监督。
- 唯一 CFM，oracle 预训练、预测条件训练、联合优化、32 步 Heun 推理。
- 训练侧 cross-fitted TRF、循环移位 / FDR 通道筛选、未见音频放行门槛和可微 Neural Consistency。
- 三系统 B0 / B1 / B2；wrong / null controls；自由 ASR WER；被试统计、结果表与论文图。
- 60 秒有标签 adapter，滤波和模型上下文全部计入校准预算。
- 可恢复 checkpoint、模型 / 数据 / 统计版本检查、正式测试冻结锁和 EEG-only 推理接口。

训练效果需要执行正式实验后判断。单元检查与真实文件接口检查不能替代 oracle、TRF 或 EEG 条件有效性门槛。

## 环境

Python 3.12。当前本地环境是 `.venv-aligned-local/`：

```bash
source .venv-aligned-local/bin/activate
python -m pip install -r requirements.txt
```

CUDA 服务器先安装匹配的 torch / torchaudio，再安装其余依赖。普通训练支持 BF16 与 microbatch 梯度累积；Vocos、声学生理特征及 TRF 使用 FP32。Apple 平台默认 CPU，保证 STFT / ISTFT 梯度路径可用。正式训练建议使用至少 24 GB 的 CUDA GPU；本地 CPU 用于代码检查和小规模接口核对。

## 数据与权重

官方数据源：[ds004408](https://github.com/OpenNeuroDatasets/ds004408)。固定快照 `b2d91784a90c18e1d76bd77dcb2b29b0ce27a0be`，下载规模约 20 GB。包含 19 名被试的 BrainVision EEG、20 段真实 WAV 和 20 个 TextGrid。

```bash
python scripts/perceived_flow.py download
```

重复执行可续传并校验。使用 `--data-only` 或 `--models-only` 分别下载；`--subjects sub-007` 只下载指定人的 EEG，但仍下载全部 20 个 audio / TextGrid 对。局部数据可以用于接口核对，正式矩阵要求全体 19 人。

`models/manifest.json` 保存 WavLM、Vocos 与 Whisper small.en 的完整 revision 和文件 SHA256。训练和推理只读取本地固定权重，不会在模型加载时隐式联网。

### 固定快照的单位修正

真实文件检查发现：直接把 BrainVision / MNE 的标称 V 数值乘 1e6，会使该快照的 EEG 幅值比原作者 CND 数据大百万倍。已用同一数据源的训练侧记录核对，动态轨迹的通道中位相关约 0.99995；配置固定乘 1e−6 恢复原作者数值尺度，不根据各条记录自动估计缩放。

[单位审计](report/dataset_unit_audit.json) 保存来源、文件 SHA256、样本对应和缩放证据。**数值尺度恢复已经直接核对；绝对 µV 的解释来自 BioSemi 设备与源数据幅值证据，属于推断，不是作者发布的正式单位勘误。**因此绝对幅值结论须保留这一限制。模型使用训练侧标准化，QC 的幅值阈值仍须按此单位解释。

原始数据未改写。`data/raw/ds004408/source_calibration/` 中的单个 CND 文件只用于同一语料的来源核对，不进入训练。`scripts/audit_source_units.py` 可利用该文件复现审计。

## 三个交付节点的运行顺序

### A：准备与语音侧验证

```bash
python scripts/perceived_flow.py prepare --device cuda
python scripts/perceived_flow.py oracle --seed 11 --device cuda
python scripts/perceived_flow.py fit-trf --fold F0
```

Prepare 会提取冻结语音特征并为五个 fold 拟合**训练侧** EEG 标准化。该步骤会处理全数据，不应当作快速单元测试。Oracle 的 copy-synthesis / WER 门槛与 TRF 的留出音频门槛失败时，后续训练会停止对应路径。

### B：F0 / seed 11 开发运行

```bash
python scripts/perceived_flow.py train-encoder --fold F0 --seed 11 --device cuda
python scripts/perceived_flow.py train-flow --fold F0 --seed 11 --system B0 --device cuda
python scripts/perceived_flow.py train-flow --fold F0 --seed 11 --system B1 --device cuda
python scripts/perceived_flow.py train-flow --fold F0 --seed 11 --system B2 --device cuda
```

TCN 预训练默认生成 baseline 与 physiology 两个 checkpoint；B1 / B2 使用相同的 physiology 初始化。开发模型只用 audio15–17 和 sub-001–003 选择 checkpoint。B0 的 EEG 条件未通过开发对照时，停止后续正式扩展。

### C：固定正式矩阵与测试

```bash
python scripts/perceived_flow.py matrix --device cuda --resume
python scripts/perceived_flow.py freeze
python scripts/perceived_flow.py evaluate --split test --device cuda
python scripts/perceived_flow.py neural-diagnostic --fold all --seed 11 --device cuda
python scripts/perceived_flow.py adapt --seconds 60 --device cuda
python scripts/perceived_flow.py evaluate --split test --system B2 --adapted --device cuda
python scripts/perceived_flow.py report --split test
```

矩阵是 3 系统 × 5 fold × 3 训练种子；不是架构搜索。`--resume` 恢复 optimizer、数据采样与随机状态。最终测试要求完整训练矩阵、放行记录和未被修改的协议锁。Neural diagnostic 只使用训练被试的留出音频，不能将它的结果当作新被试神经预测能力。

评估保存全部指标；试听音频按预先确定的样本 ID 保存，每被试 / 刺激一个窗口、训练 seed 11、生成 seed 0。避免保存数万份重复音频；任何窗口都可按记录的噪声种子复现。报告自动标注完整 / 部分结果。

## EEG-only 推理

```bash
python scripts/perceived_flow.py infer --fold F0 --seed 11 --system B2 \
  --input data/my_processed_eeg.npy --output outputs/my_reconstruction.wav
```

输入数组为 `[832,128]`：经过协议滤波、CAR 和重采样的 128 Hz EEG，单位 µV，通道顺序 A1–D32。包含 4 秒核心之前 1 秒与之后 1.5 秒。模型内部使用训练均值 / 标准差；可用 `--mask` 提供 128 个有效通道标记。输出固定 4 秒，接口不接收目标音频、文本、embedding 或真实句长。

## 全部实现完成后的测试

```bash
python -m pytest tests -q
python scripts/verify_pipeline.py --asr --device cpu
```

单元测试不下载数据。真实文件检查使用训练侧 sub-007 / audio01，检查 BrainVision、WavLM、原生 Vocos、冻结网络输入梯度，以及真实 CFM→Vocos→TRF 的计算图。该计算图检查的 TRF 权重是合成测试夹具，不是拟合后的神经前向模型，也不报告科学效果。

## 目录与 ignore

| 目录 | 内容 | Git |
|---|---|---|
| eegspeech / scripts / tests | 完整新流水线、统一入口与测试 | 跟踪 |
| configs | 固定参数、音频 / 被试划分、预训练模型规格 | 跟踪 |
| report | 执行协议、小型检查报告与最终表格 | 跟踪 |
| data/raw/ds004408 | 不可变 EEG + WAV + TextGrid；acquisition receipt | 忽略 |
| artifacts/perceived_flow | 预处理、teacher cache、TRF、stats、checkpoints、protocol lock | 忽略 |
| models / outputs / logs | 冻结权重、生成音频 / 图 / 逐样本指标、日志 | 忽略 |

旧 imagined-speech 代码、配置、停用数据与失败产出已经删除。清理路径见 [清理记录](report/cleanup_2026-10-13.json)。论文参考 PDF / BibTeX 保留于 Paper-ECoG-EEG。
