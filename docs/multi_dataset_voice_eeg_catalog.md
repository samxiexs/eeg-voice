# Multi-Dataset EEG Voice Catalog

## 1. 研究目标与筛选标准

当前项目的数据目标是把多个公开数据集组合成一个可训练的 EEG-Voice token corpus。主链路固定为：

```text
EEG -> discrete token -> content / pitch / timbre / speaker / style alignment -> voice image retrieval / reconstruction interface
```

公开数据集不再以“同一句话由大量说话人重复朗读”为硬门槛。现实筛选标准改为：

```text
只要数据集中存在大量不同语音材料 / 不同说话人 / 不同声音条件，
并且 EEG 与听到或产生的语音有时间对应关系，
即可进入候选池。
```

进入候选池的数据集至少需要满足其中一项：

- 听觉语音 EEG：被试听真实或合成语音，EEG 与音频 onset / segment / trial 对齐。
- 多说话人或竞争语音：存在不同 speaker stream、attention target 或 competing talker。
- 发声或想象语音：被试产生、轻声发音、默读、想象音素/词/短句，可作为无外部声波的代理。
- 受控声音属性：存在 phoneme、CV/VC、F0、formant、emotion/style、空间化、噪声或听觉注意操控。
- 弱代理声音数据：不是 speech 主集，但可提供 pitch、timbre、style、affect 或 attention 预训练信号。

## 2. 数据集优先级定义

| 优先级 | 定义 | 进入训练的位置 |
| --- | --- | --- |
| P0 主训练集 | 听觉语音 EEG，存在音频刺激，适合 token-to-audio / token-to-speaker retrieval | 第一批下载和建模 |
| P1 辅助预训练集 | 自然语音、AAD、speech envelope tracking，适合训练通用 speech EEG tokenizer | tokenizer pretraining |
| P2 代理/控制集 | 发声、想象语音、情绪声音、受控音素、合成声音 | phoneme、pitch、timbre、style probe |
| P3 弱相关数据 | 音乐、情绪视频、非语音听觉任务 | 只做辅助，不进入主 speech 结论 |

字段解释：

| 字段 | 含义 |
| --- | --- |
| 数据集名 | 公开名称或常用简称 |
| 来源链接 | OpenNeuro、Zenodo、Dryad、OSF、GitHub、Data portal 或论文页面 |
| 语言/声音类型 | 英文、中文、粤语、合成声音、受控声音、代理声音 |
| modality | EEG、MEG、iEEG、EEG+sEMG、EEG+audio 等 |
| 说话人多样性 | 单 narrator、多 narrator、多被试自发声、合成男女声、竞争语音等 |
| 语音音频 | 是否可定位 wav/audio/stimulus |
| 时间对齐 | 是否有 TextGrid、events、trial onset、word timing、audio onset 或 run mapping |
| 当前用途 | 对 EEG voice token 模型的具体作用 |
| 风险 | 使用前需要控制的问题 |

## 3. 英文数据集

| 数据集名 | 来源链接 | 语言/声音类型 | modality | 说话人多样性 | 语音音频 | 时间对齐 | 当前用途 | 优先级 | 风险 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ds004408` continuous naturalistic speech | [OpenNeuro ds004408](https://openneuro.org/datasets/ds004408) | 英语自然有声书 | EEG | 单 audiobook narrator，20 段连续语音 | 有 wav | TextGrid word/phoneme timing，EEG run 与音频对齐 | 英语自然语音 tokenizer 预训练；phoneme/onset branch；segment retrieval | P0 | 说话人单一；高层语义标签需要额外 NLP 构造 |
| Weissbart natural speech EEG | [Zenodo 7086168](https://zenodo.org/records/7086168) | 英语连续语音 | EEG | audiobook / spoken narrative | 有刺激材料 | continuous speech timing / surprisal predictor | acoustic tracking、surprisal/control predictor | P1 | 与 voice image 关系偏内容/预测加工，不是 speaker bank |
| `ds006434` ABR to natural speech and selective attention | [OpenNeuro ds006434](https://openneuro.org/datasets/ds006434) | 英语双 narrator audiobook | EEG | female narrator 与 male narrator；diotic/dichotic attention | 有 wav | 64 s epoch、attention trigger、cortical/subcortical task timing | 高精度 timing；attended male/female stream；speech tracking stress test | P0/P1 | 主要两个 narrator；任务目标偏 ABR/attention |
| Etard continuous speech EEG | [Zenodo 7086209](https://zenodo.org/records/7086209) | 英语连续语音与 competing speakers | EEG | audiobook / competing-speaker conditions | 有 audio | continuous speech alignment 与 competing-speaker metadata | 英语自然语音 + competing-speaker 扩展；公开可拉 metadata | P1 | 具体文件组织需要下载后统一 |
| `ds007591` speech decoding | [OpenNeuro ds007591](https://openneuro.org/datasets/ds007591) | minimally overt speech production | EEG | 被试产生少量 color words | 可能有 production/audio metadata，需核对 | events.tsv / task timing | production sanity check；EEG token 是否含 speech production content | P2 | 不适合作为听觉语音主训练 |
| SparrKULee / EEGDash speech corpus | [EEGDash NM000238](https://eegdash.org/api/dataset/eegdash.dataset.NM000238.html) | 大规模 speech EEG corpus | EEG | 85 participants，长时 speech exposure | 需按数据页核对 | EEGDash metadata / task timing | 大规模 speech EEG tokenizer pretraining 候选 | P1 | 下载和具体刺激访问方式需再核对 |

英文数据集的定位：

```text
英文数据集负责提供自然语音 EEG、cocktail party/competing-speaker attention 和 production sanity check。
它们训练通用 speech EEG tokenizer，但单个数据集通常不能提供丰富 speaker identity。
```

## 4. 中文数据集

| 数据集名 | 来源链接 | 语言/声音类型 | modality | 说话人多样性 | 语音音频 | 时间对齐 | 当前用途 | 优先级 | 风险 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ds005345` LPP Multi-talker | [OpenNeuro ds005345](https://openneuro.org/datasets/ds005345) | 普通话 computer-synthesized single female / single male / mixed speech | EEG + fMRI | synthetic male/female 两条 stream；single 与 mix 条件 | 有 `single_female.wav`、`single_male.wav`、`mix.wav` | run mapping、word information、acoustic CSV、preprocessed FIF | 当前中文主数据集；speaker stream retrieval；single-to-mix transfer | P0 | 只有一个男声和一个女声；speaker identity 空间很小 |
| ESAA | [Zenodo 7078451](https://zenodo.org/records/7078451) | 普通话 speech auditory attention | EEG + audio | female/male Mandarin storytellers；目标 stream 随 trial 变化 | 有语音材料 | trial onset / AAD labels | Mandarin AAD；target speaker retrieval；tonal language generalization | P0 | 需要下载后核对 speaker、trial、audio 文件结构 |
| NJU AAD | [Zenodo 7253438](https://zenodo.org/records/7253438) | 普通话竞争语音注意解码 | EEG + audio | competing Mandarin speakers | 有语音材料 | trial/attention target timing | 中文多说话人 AAD 扩展；speaker-stream contrastive learning | P0/P1 | 需要核对任务条件和音频许可 |
| AASD | [Zenodo 17413336](https://zenodo.org/records/17413336) | 普通话 spontaneous auditory attention switch decoding | EEG + audio | 多说话人、多目标流切换 | 有语音材料 | switch timing / trial labels | 注意切换版 Mandarin AAD；更贴近动态 target stream | P0/P1 | EEG/audio 包较大，需后续统一格式 |
| MS-AASD | [Zenodo 17149387](https://zenodo.org/records/17149387) | 普通话 mixed-speech attention switch decoding | EEG + audio | mixed speech + self-initiated attention switch | 有语音材料 | switch metadata / trial labels | Mandarin 多说话人注意切换扩展 | P0/P1 | 需要下载后统一事件定义 |
| `ds006465` / 3M-CPSEED | [OpenNeuro ds006465](https://openneuro.org/datasets/ds006465) | Mandarin pinyin overt / mouthed / imagined speech | EEG | 20 subjects 自己产生 pinyin | 发声条件可能有 audio 或 production metadata，需核对 | prompt/event timing | 中文 imagined/overt speech proxy；拼音、声母、韵母、声调 probe | P2 | 不是听觉语音；运动和想象成分强 |

中文数据集的定位：

```text
中文部分以 ds005345、ESAA、NJU AAD、AASD、MS-AASD 为听觉语音主线。
3M-CPSEED 作为普通话拼音、声母、韵母、声调的 production/imagined proxy。
```

## 5. 粤语数据集

| 数据集名 | 来源链接 | 语言/声音类型 | modality | 说话人多样性 | 语音音频 | 时间对齐 | 当前用途 | 优先级 | 风险 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ds004718` LPPHK | [OpenNeuro ds004718](https://openneuro.org/datasets/ds004718) | 粤语《小王子》自然语音 | EEG + fMRI | 单 Cantonese narrator；52 older Cantonese speakers as participants | 有 section/sentence wav | word timing、sentence trigger、f0/intensity、POS、word frequency | 粤语主数据集；word/prosody/F0/intensity alignment；跨语言 tokenizer | P0 | 被试为老年人；说话人单一；重建连续时间轴需要 timing 文件 |
| Cantonese tone/syllable production ERP dataset | [Zenodo 7750292](https://zenodo.org/records/7750292) | 粤语声调/音节 production 或 ERP | EEG/ERP | 粤语音节/声调材料，speaker 多样性需核对 | 需核对是否含 wav | event/trial timing | 粤语 tone/pitch probe；声调与 F0 表征评估 | P2 | 不是自然 voice image 主数据；音频和 speaker 信息需核对 |

粤语部分的判断：

```text
粤语公开 EEG-speech 数据明显少于英文和普通话。
ds004718 是当前最关键粤语数据集；
粤语 tone / syllable ERP 数据可用于 pitch/tone probe，
但不能替代自然 voice image 数据。
```

## 6. 合成声音 / 受控声音 / 代理声音

| 数据集名 | 来源链接 | 语言/声音类型 | modality | 说话人多样性 | 语音音频 | 时间对齐 | 当前用途 | 优先级 | 风险 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ds006104` speech decoding | [OpenNeuro ds006104](https://openneuro.org/datasets/ds006104) | 受控 phoneme、CV/VC、word、happy/angry speech | EEG + TMS | 多短语音刺激；speaker identity 需从 stimuli 核对 | 本地已定位大量 wav | events.tsv stimulus onset、phoneme/manner/place/voicing labels | controlled token probe；phoneme、F0、timbre、style | P0/P2 | TMS confound 强；不是自然连续语音 |
| `ds005345` LPP Multi-talker | [OpenNeuro ds005345](https://openneuro.org/datasets/ds005345) | computer-synthesized Mandarin male/female speech | EEG + fMRI | 合成男声、合成女声、mixed speech | 有 wav | run mapping + word/acoustic CSV | 合成 male/female voice retrieval；中文主线 | P0 | speaker 数量少 |
| `ds006434` ABR attention | [OpenNeuro ds006434](https://openneuro.org/datasets/ds006434) | male/female narrator attention | EEG | 两个 audiobook narrators | 有 wav | 高精度 event/trial timing | controlled attention；attended stream timing | P1 | 不是 rich speaker bank |
| KUL AAD | [Zenodo 4004271](https://zenodo.org/records/4004271) | competing speech AAD | EEG + audio | 多 Dutch stories / competing talker | 有 audio | trial/attention labels | speaker stream tracking；AAD baseline | P1 | 语言非中英粤；必须做 leave-trial/story split |
| DTU AAD | [Zenodo 1199011](https://zenodo.org/records/1199011) | reverberant competing speech | EEG + audio | competing talkers / room conditions | 有 audio | trial/attention labels | room robustness；speaker-stream retrieval | P1 | 房间/混响是 confound，同时也是泛化测试 |
| 255ch EEG-AAD | [Zenodo 4518754](https://zenodo.org/records/4518754) | high-density competing speech | 255-channel EEG + audio | competing speakers | 有 audio | trial/attention labels | 高密度空间 tokenizer；sensor ablation | P1 | 数据较大；通道体系与 64/128ch 不同 |
| OpenMIIR | [OpenMIIR GitHub](https://github.com/sstober/openmiir) | music perception/imagination | EEG + music | 非 speech | 有 music stimuli | beat/downbeat/timing labels | pitch/beat/tempo token probe | P3 | 不进入 speech 主结论 |
| MUSIN-G `ds003774` | [OpenNeuro ds003774](https://openneuro.org/datasets/ds003774) | natural music listening | EEG + music | 非 speech | 有 music wav | trial/event timing | timbre/pitch self-supervised token pretraining | P3 | 不是 voice |
| MAD-EEG | [Zenodo 4537751](https://zenodo.org/records/4537751) | target instrument attention | EEG + polyphonic music | 非 speech | 有 music stems/stimuli | attention/trial timing | target-source attention proxy | P3 | 不是 speech speaker stream |

这一类数据集的定位：

```text
合成声音、受控声音和代理声音不能单独证明 voice image reconstruction。
它们的价值是让 tokenizer 学会 phoneme、pitch、timbre、style、attention 和 source selection 的可读出结构。
```

## 7. 推荐组合训练路线

```text
Stage 1: 通用 speech EEG tokenizer
  English natural speech + AAD + Mandarin/Cantonese natural speech

Stage 2: voice attribute probe
  ds006104 + Cantonese tone + 3M-CPSEED

Stage 3: speaker stream retrieval
  ds005345 + ESAA + NJU AAD + KUL/DTU/255ch AAD

Stage 4: target voice image dataset
  self-collected multi-speaker / multi-style / multi-F0 voice bank
```

### Stage 1: 通用 speech EEG tokenizer

目标是训练稳定 EEG discrete token，而不是先做 waveform generation。

推荐数据：

- `ds004408`
- Weissbart natural speech EEG
- Etard continuous speech EEG
- SparrKULee / EEGDash speech corpus
- `ds004718`
- `ds005345`
- `ds006434`

训练目标：

```text
EEG reconstruction
+ masked EEG modeling
+ speech envelope / mel / phoneme-onset alignment
+ segment-level audio retrieval
+ token usage / perplexity / dead-code metrics
```

### Stage 2: voice attribute probe

目标是确认 token 是否保留声音内容、音调、音色、发音结构和风格。

推荐数据：

- `ds006104`
- Cantonese tone/syllable ERP
- `ds006465` / 3M-CPSEED

训练目标：

```text
token -> phoneme / CV / VC / word category
token -> manner / place / voicing
token -> F0 high-low / pitch contour
token -> spectral centroid / brightness / MFCC statistics
token -> happy-angry / affective style proxy
```

### Stage 3: speaker stream retrieval

目标是让 EEG token 在自然或竞争语音中对齐目标说话人 stream。

推荐数据：

- `ds005345`
- ESAA
- NJU AAD
- AASD
- MS-AASD
- `ds006434`
- KUL AAD
- DTU AAD
- 255ch EEG-AAD

训练目标：

```text
InfoNCE(token embedding, attended stream embedding)
+ target vs masker speaker retrieval
+ single-to-mix transfer
+ room / spatial / density generalization
```

### Stage 4: target voice image dataset

公开数据仍缺少最终目标所需的系统化 voice bank。自采数据负责补足：

- 多说话人。
- 多风格。
- 多 F0。
- 多 formant / timbre。
- 同一 subject 的 voice perception rating。
- EEG 与每条 voice item 的精确 trigger / audio-loopback 对齐。

## 8. 当前最小可行数据组合

第一批 Core：

```text
ds006104
ds005345
ds004408
ds004718
ds006434
ESAA
KUL AAD
DTU AAD
255ch EEG-AAD
```

第一批 Core 的分工：

| 数据集 | 分工 |
| --- | --- |
| `ds006104` | controlled phoneme / pitch / timbre / style probe |
| `ds005345` | Mandarin synthetic male/female/mix speaker stream retrieval |
| `ds004408` | English natural speech phoneme/onset pretraining |
| `ds004718` | Cantonese word/prosody/F0/intensity alignment |
| `ds006434` | attention + high-precision speech timing |
| ESAA | Mandarin AAD speaker-stream retrieval |
| KUL AAD | classic AAD baseline |
| DTU AAD | reverberant competing speech robustness |
| 255ch EEG-AAD | high-density spatial encoding and sensor ablation |

Expansion：

```text
3M-CPSEED
Weissbart natural speech EEG
Etard continuous speech EEG
SparrKULee
NJU AAD
AASD
MS-AASD
```

弱代理，不进入主 speech 结论：

```text
OpenMIIR
MUSIN-G
MAD-EEG
```

## 9. 明确不够的部分

公开数据可以训练和验证：

- EEG token 的稳定性。
- speech envelope / onset / phoneme / word alignment。
- pitch、voicing、F0、intensity、timbre proxy。
- single speaker vs mixed speaker retrieval。
- attended target stream decoding。
- imagined / overt speech proxy。

公开数据不能完整解决：

- 大规模 speaker identity manifold。
- 同一 subject 对同一 voice bank 的主观相似度评分。
- 多 F0 / 多 formant / 多 style 的系统化操控。
- 个体化 voice image retrieval。
- waveform-level voice reconstruction 的最终 ground truth。

因此当前数据路线的结论应写成：

```text
公开多数据集组合用于训练 EEG speech tokenizer 和 voice-representation alignment；
最终 voice image reconstruction 需要自采 multi-speaker / multi-style / multi-F0 voice bank 数据。
```

这份目录的使用方式：

```text
先用 Core 训练 tokenizer 和 retrieval/probe baseline；
再用 Expansion 扩大语言、说话人、任务和 modality 覆盖；
弱代理数据只用于 pitch/timbre/style 的辅助预训练或 sanity check。
```
