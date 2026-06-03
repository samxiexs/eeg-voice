# EEG Stage Segmentation Preprocessing

## 目标

这套脚本现在不再只导出 `thinking`。

它会把 `FEIS` 和 `KaraOne` 里**所有当前能可靠切出来的 EEG 阶段**统一处理、切割、标注，并配好 waveform target，方便你后面自由筛选：

- 只做 `thinking`
- 做 `thinking vs speaking`
- 做 `hearing/stimulus -> thinking -> overt`
- 做全阶段条件建模

脚本位置：

- [preprocess_thinking_waveform_pairs.py](../../scripts/preprocess_thinking_waveform_pairs.py)

## 导出的阶段

### FEIS

- `stimuli`
- `articulators`
- `thinking`
- `speaking`
- `resting`

这里：

- `stimuli` 可以理解为最接近 `hearing / prompt perception` 的阶段
- `thinking` 是 imagined speech
- `speaking` 是 overt speech

### KaraOne

- `clearing`
- `stimulus_like`
- `thinking`
- `overt_like`

这里：

- KaraOne 没有像 FEIS 那样直接写死一个叫 `hearing` 的阶段
- 当前最稳妥的 `hearing-like` / prompt-processing 对应段是 `stimulus_like`

## 处理原则

- 不保留新的解压中间产物
- `FEIS` 直接读取现有 phase 文件
- `KaraOne` 可优先从 `.tar.bz2` 临时抽取最少文件，处理完删除临时目录
- 最终只保留：
  - 处理后的 EEG 分段
  - 处理后的 waveform
  - trial 级与 segment 级元数据

## 当前配对策略

### FEIS

- EEG：全部五个阶段都导出
- baseline：同 trial 的 `resting`
- waveform target：同受试者、同 label 的 canonical `wav`

注意：

- `FEIS` 不是每个 trial 都有独立 overt 音频
- 所以它更适合 prompt/phoneme 级对齐和快速原型

### KaraOne

- EEG：全部四个阶段都导出
- baseline：同 trial 的 `clearing`
- waveform target：同一个 trial 的 overt `wav`

## 默认预处理

- EEG bandpass：`1-40 Hz`
- FEIS notch：`50 Hz`
- KaraOne notch：`60 Hz`
- 参考：common average reference
- 输出 EEG 采样率：`256 Hz`
- 输出 audio 采样率：`16 kHz`

## 输出结构

默认输出到：

```text
data/processed/thinking_waveform_pairs/
```

每个数据集下会有：

```text
{dataset}/
  audio/
  subjects/
  trials.csv
  segments.csv
  manifest.json
```

其中：

- `subjects/*.npz`
  - `trial_indices`
  - `labels`
  - `audio_relpaths`
  - `channel_names`
  - `stage_names`
  - `stage__thinking`
  - `stage__speaking` / `stage__overt_like`
  - 以及其他阶段数组
- `trials.csv`
  - 一行一个 trial
- `segments.csv`
  - 一行一个 `trial x stage`
  - 这是你后面筛选 `thinking`、`speaking`、`stimuli/hearing` 最直接的表

## 使用建议

如果你当前主要研究 imagined speech：

1. 先从 `segments.csv` 里筛 `segment_stage == thinking`
2. 之后再把 `stimuli` 或 `stimulus_like` 拉进来做对照
3. 再把 `speaking` 或 `overt_like` 拉进来做 transfer / teacher supervision

## 常用命令

只处理 FEIS：

```bash
python3 scripts/preprocess_thinking_waveform_pairs.py \
  --datasets feis
```

只处理 KaraOne，并优先用原始 `.tar.bz2`：

```bash
python3 scripts/preprocess_thinking_waveform_pairs.py \
  --datasets karaone \
  --prefer-karaone-archives
```

只处理几个受试者：

```bash
python3 scripts/preprocess_thinking_waveform_pairs.py \
  --datasets feis karaone \
  --feis-subjects 01 02 03 \
  --karaone-subjects MM05 MM08 \
  --prefer-karaone-archives
```

指定输出目录：

```bash
python3 scripts/preprocess_thinking_waveform_pairs.py \
  --datasets feis karaone \
  --output-root /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/data/processed/thinking_waveform_pairs \
  --prefer-karaone-archives
```
