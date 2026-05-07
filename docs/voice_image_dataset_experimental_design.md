# Voice Image EEG Dataset 简化实验方案

## 1. 研究目标

Voice Image EEG Dataset 用于建立语音声音形象的 EEG 表征、检索与重构数据基础。研究对象为具备正常听力和正常语言理解能力的成人受试者。核心目标不是疾病分型，也不是单纯 `EEG -> text`，而是学习：

```text
EEG -> discrete token -> speech content / pitch / timbre / speaker / style representation
```

声音形象定义为一组可建模的 voice profile：

| 维度 | 目标变量 |
| --- | --- |
| 内容 | phoneme、syllable、word、short phrase、speech rhythm |
| 音调 | F0、pitch contour、intonation、tone、voicing |
| 音色 | spectral envelope、formant、MFCC/mel、brightness、roughness、breathiness |
| 说话人 | speaker id、gender impression、age impression、familiarity |
| 风格 | neutral、happy、angry、commanding、whisper、soft |
| 空间感 | center、left/right、distance、externalization |

模型终点：

```text
1. EEG token reconstruction: EEG -> token -> EEG reconstruction
2. 语音内容对齐: EEG token <-> phoneme / word / speech content embedding
3. 声音属性对齐: EEG token <-> F0 / timbre / intensity / style
4. 声音形象检索: EEG token -> Top-K voice-bank candidates
5. 条件式重构接口: Top-K voice + attribute vector -> downstream vocoder / voice conversion
```

## 2. 科学问题

### Q1. EEG token 是否保留语音内容

同一说话人、不同内容的语音呈现时，EEG token 对 phoneme、syllable、word 和 sentence-level content embedding 的可分性。

### Q2. EEG token 是否保留声音属性

同一内容、不同 F0、formant、timbre、style 的语音呈现时，EEG token 对音调、音色和风格的预测能力。

### Q3. EEG token 是否能检索声音形象

给定一个 EEG segment，模型在 voice bank 中检索正确 voice item 或相近 voice item 的能力。

### Q4. 内容、说话人、音调、音色能否分离

通过同内容不同说话人、同说话人不同内容、同内容同说话人不同 F0/formant/style 的对照，检验 EEG token 的 disentanglement。

## 3. 受试者

| 阶段 | 样本量 | 目的 |
| --- | --- | --- |
| Pilot | 12-20 名成人 | trigger、音频同步、评分稳定性、EEG token dry-run |
| Main | 40-60 名成人 | 跨受试者 tokenizer、voice retrieval、attribute alignment |

纳入标准：

- 年龄 18-65 岁。
- 正常或矫正正常听力。
- 能完成普通语音听觉任务、按键任务和简短评分。
- 实验语言与受试者语言背景匹配。
- 完成书面知情同意。

排除标准：

- 严重听力障碍。
- EEG 采集禁忌。
- 无法稳定完成听觉判断或评分任务。
- 实验当天明显疲劳、睡眠不足或状态异常。

记录协变量：

| 协变量 | 用途 |
| --- | --- |
| language background | 内容和声调建模 |
| music/speech training | pitch/timbre 敏感性 |
| hearing threshold | 声压级校准 |
| sleep/caffeine/nicotine state | EEG 状态协变量 |
| voice familiarity rating | 说话人熟悉度 |

## 4. 整体采集结构

采集包含两个部分：

```text
Part A: 声音库校准与评分，约 25-35 分钟
Part B: EEG 主采集，约 55-70 分钟有效任务时间
```

两部分可安排在同一天完成。EEG 主采集中途包含短休息。全流程围绕外部语音感知展开；声音想象任务只作为可选短 block，不作为主数据闭环。

## 5. 设备与同步

| 模块 | 参数 |
| --- | --- |
| EEG | 64 或 128 channel |
| 采样率 | 1000 Hz；最低 500 Hz |
| 参考 | 在线 Cz 或 mastoid；离线 average/mastoids |
| 辅助通道 | EOG、ECG、jaw/neck EMG、trigger、audio loopback |
| 音频呈现 | 插入式耳机或封闭耳机 |
| 声压级 | 听阈校准后约 60-70 dB SPL |
| 同步 | TTL trigger + audio loopback |

EEG 预处理目标采样率为 250 Hz。原始 500/1000 Hz 数据保留，用于精确 trigger/audio-loopback 延迟校正。

## 6. 声音刺激设计

### 6.1 Voice bank

每名受试者使用一个统一 voice bank，覆盖真实说话人和参数化语音：

| 刺激类型 | 内容 |
| --- | --- |
| phoneme / CV / VC | 低层语音单位和发音特征 |
| short words | 内容识别和声调/韵律 |
| short phrases | 1-3 秒自然语音 |
| same-content multi-speaker | 同内容多说话人音色对照 |
| same-speaker multi-content | 同说话人多内容对照 |
| F0-shifted speech | 音调操控 |
| formant-shifted speech | 音色操控 |
| style speech | neutral、happy、angry、commanding、whisper |
| spatialized speech | center、left/right、front/externalized |

刺激规模：

| 阶段 | voice-bank items |
| --- | --- |
| Pilot | 180-240 |
| Main | 300-500 |

每条音频保存原始 wav、loudness-normalized wav、音频特征和 metadata。

### 6.2 参数操控

| 维度 | 水平 |
| --- | --- |
| Speaker | 8-12 名真实说话人或合成说话人 |
| F0 | original、-4 semitones、+4 semitones |
| Formant | original、0.9x、1.1x |
| Style | neutral、happy、angry、commanding、whisper |
| Spatialization | center、left 60°、right 60°、externalized front |

采用部分因子设计，避免全组合爆炸。主对照保持“单一维度变化，其余维度固定”。

### 6.3 每条语音 metadata

```text
stim_file
content_id
transcript
language
speaker_id
speaker_type
speaker_gender
speaker_age_bin
emotion_style
f0_shift_semitone
formant_shift_ratio
speaking_rate
intensity_db
spatial_azimuth
spatial_externalization
loudness_normalization_level
duration_sec
```

## 7. Part A：声音库校准与评分

目的：

```text
voice item -> subject-level perceptual ratings
```

评分字段：

| 评分 | 范围 |
| --- | --- |
| pitch | 0-100 |
| brightness | 0-100 |
| roughness | 0-100 |
| breathiness | 0-100 |
| speaker_similarity | 0-100 |
| style_strength | 0-100 |
| familiarity | 0-100 |
| confidence | 0-100 |

校准任务：

```text
播放 voice item
-> 受试者完成 2-4 个快速评分
-> 记录反应时间和评分
```

每名受试者评分 120-200 条代表性 voice items。评分结果用于构建 subject-specific voice attribute labels。

## 8. Part B：EEG 主采集

### 8.1 Block 结构

EEG 主采集包含三个核心 block：

| Block | Trials | 任务 | 主要标签 |
| --- | --- | --- | --- |
| B1 Content listening | 120-160 | 听短语音，偶发内容判断 | phoneme、word、content embedding |
| B2 Voice attribute listening | 120-160 | 同内容不同 voice，判断音调/音色/风格 | F0、formant、timbre、style |
| B3 Voice retrieval | 80-120 | 听目标 voice，随后候选匹配 | voice_id、speaker_id、Top-K labels |

总 trials 约 320-440。每个 block 之间休息 2-5 分钟。

### 8.2 Trial 设计

通用 trial：

```text
fixation: 500-800 ms jitter
audio: 0.8-3.0 s
post-audio blank: 500 ms
task response or rating: 1000-2500 ms
ITI: 800-1500 ms jitter
```

B1 内容判断：

```text
audio
-> catch question: 是否听到目标音节/词
```

B2 声音属性判断：

```text
audio A
audio B
-> higher pitch / brighter timbre / same speaker / same style forced choice
```

B3 声音检索：

```text
target audio
delay: 1000 ms
candidate audio 1..4
-> select closest voice
```

可选短 block：

```text
voice cue
-> silent voice replay 2-3 s
-> vividness / confidence rating
```

该 block 仅用于探索无外部声波条件下 token 是否接近 voice embedding，不进入主训练闭环。

## 9. 对照设计

| 对照 | 固定 | 变化 | 检验 |
| --- | --- | --- | --- |
| Same content, different speaker | content | speaker / timbre | 音色和说话人编码 |
| Same speaker, different content | speaker | content | 内容编码 |
| Same content/speaker, F0 shift | content, speaker | F0 | 音调编码 |
| Same content/speaker, formant shift | content, speaker | formant | 音色编码 |
| Same content/speaker, style shift | content, speaker | style | 情绪/风格编码 |
| Speech-shaped noise | low-level energy | speech structure absent | 低层声学控制 |
| Silence / fixation | no audio | baseline | 非语音 baseline |

## 10. 数据格式

目录结构：

```text
VoiceImageEEG/
  dataset_description.json
  participants.tsv
  participants.json
  phenotype/
    hearing_screening.tsv
    language_background.tsv
    voice_rating_summary.tsv
  stimuli/
    voice_bank/
      voice_0001.wav
      voice_0002.wav
    voice_bank_metadata.tsv
  sub-001/
    eeg/
      sub-001_task-voiceimage_eeg.vhdr
      sub-001_task-voiceimage_eeg.eeg
      sub-001_task-voiceimage_eeg.vmrk
      sub-001_task-voiceimage_events.tsv
      sub-001_task-voiceimage_channels.tsv
      sub-001_task-voiceimage_electrodes.tsv
      sub-001_task-voiceimage_coordsystem.json
    beh/
      sub-001_task-voicebankratings_beh.tsv
  derivatives/
    audio_features/
    eeg_preproc/
    eeg_tokens/
    voice_embeddings/
```

### 10.1 EEG events.tsv

核心列：

```text
onset
duration
trial_type
block_id
stim_file
content_id
transcript
speaker_id
voice_id
target_voice_id
candidate_voice_ids
correct_candidate_id
emotion_style
f0_shift_semitone
formant_shift_ratio
spatial_azimuth
spatial_externalization
response
response_time
rating_pitch
rating_brightness
rating_roughness
rating_breathiness
rating_style_strength
rating_familiarity
rating_confidence
audio_loopback_delay_ms
trigger_id
```

### 10.2 Audio derivatives

每条 wav 提取：

```text
f0_mean
f0_median
f0_std
f0_contour
voiced_ratio
intensity_rms
speaking_rate
mfcc_1..mfcc_20
mel_embedding
spectral_centroid
spectral_bandwidth
spectral_flatness
formant_f1_f2_f3
speaker_embedding
style_embedding
hubert_or_wav2vec_content_embedding
```

## 11. 预处理

EEG：

```text
raw EEG
-> trigger/audio-loopback delay correction
-> bad channel inspection
-> 0.1-40 Hz bandpass
-> 50/60 Hz notch
-> re-reference: average 或 mastoids
-> ICA / SSP 去眼动和心电
-> resample 到 250 Hz
-> epoch by audio onset
-> artifact rejection
-> save continuous cleaned + epochs
```

音频：

```text
resample 到 16 kHz 或 24 kHz
trim leading/trailing silence
loudness normalization
extract F0 / mel / MFCC / formant / speaker embedding / content embedding
```

## 12. 模型训练任务

### 12.1 Stage 1: EEG tokenizer

```text
input: EEG windows
target: normalized EEG reconstruction
loss: time-domain + frequency-domain + PCC + RVQ commitment
output: discrete EEG tokens
```

公开数据可进入预训练：

| 数据 | 用途 |
| --- | --- |
| ds006104 | phoneme、CV/VC、happy/angry、F0/timbre probe |
| ds005345 | single male/female/mix speaker stream retrieval |

### 12.2 Stage 2: 内容与属性监督

```text
EEG token -> phoneme / word / content embedding
EEG token -> F0 / intensity / timbre / style labels
```

监督信号：

| 目标 | 标签 |
| --- | --- |
| content | phoneme、word、HuBERT/wav2vec content embedding |
| pitch | F0 mean、F0 contour、high/low bin |
| timbre | MFCC、formant、centroid、brightness |
| speaker | speaker id、speaker embedding |
| style | emotion/style class、style embedding |

### 12.3 Stage 3: 声音形象检索

```text
EEG token embedding <-> voice item embedding
loss: InfoNCE / triplet / candidate CE
metric: Top-1, Top-5, MRR
```

候选集设置：

| 候选集 | 用途 |
| --- | --- |
| same content, different speaker | 说话人/音色检索 |
| same speaker, different content | 内容检索 |
| same content/speaker, different F0 | 音调检索 |
| same content/speaker, different formant | 音色检索 |
| mixed candidates | 综合声音形象检索 |

### 12.4 Stage 4: 重构接口

v0 不直接训练 waveform generator。重构以检索和属性向量形式输出：

```text
EEG token
-> Top-K voice items
-> predicted content embedding
-> predicted pitch/timbre/style vector
-> downstream voice conversion or conditional vocoder
```

## 13. 评估指标

| 任务 | 指标 |
| --- | --- |
| EEG reconstruction | L1、PCC、frequency amplitude error |
| content decoding | accuracy、macro F1、retrieval accuracy |
| pitch/timbre regression | Pearson、Spearman、MAE |
| speaker/style classification | balanced accuracy、macro F1 |
| voice retrieval | Top-1、Top-5、MRR |
| cross-subject generalization | leave-one-subject-out performance |
| disentanglement | content-only、speaker-only、F0-only、timbre-only ablations |

Baseline：

| Baseline | 说明 |
| --- | --- |
| random candidate | chance retrieval |
| acoustic envelope only | 低层声学 tracking |
| text/content only | 只用文本内容 |
| speaker only | 只用说话人 ID |
| no-RVQ continuous latent | 检验 tokenization 增益 |
| no-sensor embedding | 检验 montage-aware 设计 |

## 14. Pilot 最小闭环

| 模块 | 数量 |
| --- | --- |
| 受试者 | 12-20 |
| voice-bank items | 180-240 |
| EEG trials | 320-440 |
| 单 trial 音频 | 0.8-3.0 s |
| 有效 EEG 任务时间 | 55-70 min |

最小闭环：

```text
voice bank wav + metadata
-> audio feature extraction
-> EEG listening task
-> EEG tokenization
-> content / pitch / timbre / speaker / style alignment
-> Top-K voice retrieval
```

## 15. 与当前公开数据的关系

| 数据集 | 已有能力 | 在本实验中的位置 |
| --- | --- | --- |
| ds006104 | controlled phoneme、CV/VC、happy/angry、刺激 wav | 语音单位、F0、timbre、style probe |
| ds005345 | single male、single female、mixed speech、preprocessed EEG、acoustic CSV | speaker stream contrastive retrieval |
| Voice Image EEG Dataset | 同内容多说话人、多 F0/formant/style、完整 voice bank | 目标训练和评估数据 |

`ds006104` 和 `ds005345` 用于模型预训练、probe 和工程验证。自采数据提供同内容多说话人、多音调、多音色、多风格的可控监督。

## 16. 数据安全

- 原始声音文件与身份信息分离保存。
- 真实说话人声音使用独立 consent 和 speaker_id。
- 公开共享版本包含匿名 metadata、派生音频特征、去身份化合成替代刺激。
- 模型输出用于研究分析，不用于个体身份识别、诊断或决策。

## 17. 参考规范

- EEG-BIDS 规范说明 `events.tsv`、`channels.tsv`、`electrodes.tsv` 和 `coordsystem.json` 对 EEG 数据复现的重要性。https://www.nature.com/articles/s41597-019-0104-8
- BIDS EEG specification 1.11.1。https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html
- `ds006104` 提供 controlled phoneme、CV/VC、emotion-style speech stimuli 和 EEG 事件标签，用于声音内容、音调和音色 probe。
- `ds005345` 提供 single male、single female、mixed speech 条件，用于说话人 stream retrieval 和多说话人对齐。
