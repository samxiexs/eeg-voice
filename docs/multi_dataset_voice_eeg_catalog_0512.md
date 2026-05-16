# Multi-Dataset EEG Voice Catalog（0512）

## 研究主链路

```
EEG -> discrete token -> content / pitch / timbre / speaker / style alignment
                                        |
                        voice image retrieval / reconstruction
```

---

## 1. 筛选标准、优先级与可用性口径

### 入池条件（满足其一即可）

| 条件 | 说明 |
| --- | --- |
| 听觉语音 EEG | 被试听真实或合成语音，EEG 与音频 onset / segment / trial 对齐 |
| 多说话人或竞争语音 | 存在不同 speaker stream、attention target 或 competing talker |
| 发声或想象语音 | 被试产生、轻声发音、默读或想象音素/词/句，作为无外部声波的代理 |
| 受控声音属性 | 存在 phoneme、CV/VC、F0、formant、emotion/style、空间化或听觉注意操控 |
| 弱代理声音数据 | 可提供 pitch、timbre、style、affect 或 attention 预训练信号 |

### 优先级定义

| 优先级 | 定义 | 进入训练的位置 |
| --- | --- | --- |
| **P0** 主训练集 | 听觉语音 EEG，存在音频刺激，适合 token-to-audio / token-to-speaker retrieval | 第一批下载和建模 |
| **P1** 辅助预训练集 | 自然语音、AAD、speech envelope tracking，训练通用 speech EEG tokenizer | tokenizer pretraining |
| **P2** 代理/控制集 | 发声、想象语音、情绪声音、受控音素、合成声音 | phoneme / pitch / timbre / style probe |
| **P3** 弱相关数据 | 音乐、情绪视频、非语音听觉任务 | 辅助预训练，不进入主 speech 结论 |

### 可用性口径

这里的“真正可用”按 **MD 已选入池、公开可获取或可申请** 判断，不按本地是否已下载判断。本地下载和转换状态只用于执行进度。

| 口径 | 含义 |
| --- | --- |
| `selected_public` | MD 已选，公开可取或公开可申请 |
| `selected_large` | 公开可取，但体量很大，需要分批下载或只先取 metadata / 单被试 |
| `selected_contact` | 需联系作者或额外申请原始语音、刺激材料、权限或完整说明 |
| `local_ready` | 本地已下载或已转格式；这是执行状态，不影响是否入池 |

---

## 2. 数据集全览

### 2.1 英文 / speech-decoding 扩展数据集

| 数据集 | 链接 | 语言 | Modality | 说话人 | 音频 | 时间对齐 | 用途 | 优先级 | 可用性口径 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ds004408` naturalistic speech | [OpenNeuro](https://openneuro.org/datasets/ds004408) | 英语有声书 | EEG | 单 narrator，20 段 | wav | TextGrid word/phoneme | 英语 tokenizer 预训练；phoneme/onset | P0 | `selected_public`; `local_ready` |
| `ds006434` ABR + attention | [OpenNeuro](https://openneuro.org/datasets/ds006434) | 英语双 narrator | EEG | female + male narrator | wav | 64s epoch，attention trigger | 高精度 timing；attended stream | P0/P1 | `selected_public` |
| `ds007630` EEG-Speech Brain Decoding | [EEGDash](https://eegdash.org/api/dataset/eegdash.dataset.DS007630.html) / [OpenNeuro](https://openneuro.org/datasets/ds007630) | speechopen/listening，文本语言需 probe | EEG + vocal/audio | 3 subjects；1974 recordings | beh wav + EEG EDF | events.tsv + BIDS-like sessions | 大体量 speech EEG tokenizer；listening / production 双任务 | P0/P2 | `selected_large` |
| Weissbart natural speech | [Zenodo 7086168](https://zenodo.org/records/7086168) | 英语连续语音 | EEG | audiobook narrative | 有刺激材料 | continuous speech timing | acoustic tracking；surprisal predictor | P1 | `selected_public` |
| Etard competing speech | [Zenodo 7086209](https://zenodo.org/records/7086209) | 英语 + competing | EEG | audiobook + competing | 有 audio | continuous alignment | 英语 competing-speaker 扩展 | P1 | `selected_public` |
| SparrKULee / EEGDash | [EEGDash NM000238](https://eegdash.org/api/dataset/eegdash.dataset.NM000238.html) | 大规模 speech | EEG | 85 participants | 需核对 | EEGDash metadata | 大规模 tokenizer pretraining | P1 | `selected_contact` |
| Fuglsang 2020 | [Zenodo 3618205](https://zenodo.org/record/3618205) | 丹麦语 AAD | EEG + audio | 2 说话人；44 被试含听障 | 有 audio | trial/attention labels | 大样本 AAD；听障泛化 | P1 | `selected_public` |
| Rotaru 2024 | [Zenodo 11058711](https://zenodo.org/records/11058711) | 荷兰语 AAD | EEG + audio | 2 说话人；每被试 80 min | 有 audio | trial/attention labels | 长时录音；长序列稳定性 | P1 | `selected_public` |
| Geirnaert 2025 | [Zenodo 16536441](https://zenodo.org/records/16536441) | 丹麦语 AAD | scalp+around-ear+in-ear | 2 说话人；15 被试 | 有 audio | 设备同步 metadata | 多设备 sensor ablation | P1 | `selected_public` |
| `ds007591` speech decoding | [OpenNeuro](https://openneuro.org/datasets/ds007591) | overt speech production | EEG | 被试产生 color words | 需核对 | events.tsv | production sanity check | P2 | `selected_public` |
| `ds007602` EEG-Speech Brain Decoding | [EEGDash](https://eegdash.org/api/dataset/eegdash.dataset.DS007602.html) / [OpenNeuro](https://openneuro.org/datasets/ds007602) | overt speech production | EEG + vocal audio（文件路径需 probe） | 3 subjects；113 recordings | README/EEGDash 描述 vocal audio，S3 路径需确认 | events.tsv + BIDS-like sessions | overt speech production probe；EEG-audio generation sanity check | P2 | `selected_large` |
| Kara One | [Toronto](https://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html) | 英语 phoneme / word prompts | EEG + audio + face tracking | 14 participants | vocalized audio | trial states + epoch indices | imagined + vocalized phonological category probe | P2 | `selected_public` |

### 2.2 中文普通话数据集

| 数据集 | 链接 | 语言 | Modality | 说话人 | 音频 | 时间对齐 | 用途 | 优先级 | 可用性口径 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ds005345` LPP Multi-talker | [OpenNeuro](https://openneuro.org/datasets/ds005345) | 普通话合成语音 | EEG + fMRI | 合成男声 + 女声 + mix | female/male/mix wav | run mapping + acoustic CSV | 中文主数据集；speaker stream retrieval | P0 | `selected_public`; `local_ready` |
| ESAA | [Zenodo 7078451](https://zenodo.org/records/7078451) | 普通话 AAD | EEG + audio | female/male storytellers | 有语音材料 | trial onset / AAD labels | Mandarin AAD；target speaker retrieval | P0 | `selected_public` |
| NJU AAD | [Zenodo 7253438](https://zenodo.org/records/7253438) | 普通话竞争语音 | EEG + audio | competing Mandarin speakers | 有语音材料 | trial/attention timing | 中文多说话人 AAD；contrastive learning | P0/P1 | `selected_public` |
| AASD | [Zenodo 17413336](https://zenodo.org/records/17413336) | 普通话注意切换 | EEG + audio | 多说话人，多目标流 | 有语音材料 | switch timing / trial labels | 动态 target stream；注意切换 | P0/P1 | `selected_large` |
| MS-AASD | [Zenodo 17149387](https://zenodo.org/records/17149387) | 普通话 mixed-speech | EEG + audio | mixed + self-initiated switch | 有语音材料 | switch metadata | 多说话人注意切换扩展 | P0/P1 | `selected_large` |
| **Four-Talker AAD** (Yan 2024) | [Zenodo 10803261](https://zenodo.org/records/10803261) | 普通话，4 说话人空间化 | EEG 64ch + cEEGrid | **2 男 2 女真实说话人**；+/-90/+/-30 deg | 有语音材料 | trial/attention + 空间角度 | **4-speaker identity 扩充**；空间化 stream retrieval | P0 | `selected_large` |
| **Four-Direction AAD** (Yan 2024) | [Zenodo 10803229](https://zenodo.org/records/10803229) | 普通话，4 方向空间化 | EEG 64ch | **4 说话人**；消声室 | 有语音材料 + 代码 | trial/attention + 方向 | 4-speaker 消声室基准；与 Four-Talker 合并 | P0 | `selected_large` |
| **Non-block AAD** (Yan 2025) | [Zenodo 14887886](https://zenodo.org/records/14887886) | 普通话，非 block 切换 | EEG 64ch + cEEGrid | **4 说话人**；自由切换 | 有语音材料 + 代码 | switch timing | 4-speaker 注意切换；贴近自然聆听 | P0/P1 | `selected_large` |
| ASA (Lin 2024) | [Zenodo 11541114](https://zenodo.org/records/11541114) | 普通话，多空间角度 | EEG 64ch + audio | 2 说话人；+/-5 deg 到 +/-90 deg | 有语音材料 | trial/attention + 空间角度 | 空间泛化；+/-5 deg 近距离难度最高 | P1 | `selected_large` |
| `ds006465` / 3M-CPSEED | [OpenNeuro](https://openneuro.org/datasets/ds006465) | 普通话拼音 production | EEG | 20 subjects 自产 pinyin | 需核对 | prompt/event timing | 拼音/声母/韵母/声调 probe | P2 | `selected_public` |
| `ds005170` Chisco | [EEGDash](https://eegdash.org/api/dataset/eegdash.dataset.DS005170.html) / [OpenNeuro](https://openneuro.org/datasets/ds005170) | 中文 imagined speech | EEG | 5 participants；225 recordings | 无外部 speech ground-truth；有文本刺激 | sessions/runs + raw/preprocessed FIF/PKL | 中文 imagined sentence/semantic probe；P2 核心补强 | P2 | `selected_large` |
| CIRE | [Scientific Data](https://www.nature.com/articles/s41597-025-05957-y) | 普通话 prosodic emotion / intention | 128ch EEG + audio features | 2 professional actors；38 listeners | 原始音频 + Wav2Vec2 features | BIDS-like EDF/events.tsv | prosody / intention / emotion-style probe | P2 | `selected_public` |

### 2.3 粤语数据集

| 数据集 | 链接 | 语言 | Modality | 说话人 | 音频 | 时间对齐 | 用途 | 优先级 | 可用性口径 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ds004718` LPPHK | [OpenNeuro](https://openneuro.org/datasets/ds004718) | 粤语《小王子》 | EEG + fMRI | 单 Cantonese narrator；52 被试 | section/sentence wav | word timing + F0/intensity + POS | 粤语主数据集；prosody/F0/intensity alignment | P0 | `selected_public`; `local_ready` |
| Cantonese tone/syllable ERP | [Zenodo 7750292](https://zenodo.org/records/7750292) | 粤语声调/音节 | EEG/ERP | 需核对 | 需核对 | event/trial timing | 粤语 tone/pitch probe | P2 | `selected_public` |

### 2.4 受控声音 / 代理声音 / 音乐

| 数据集 | 链接 | 语言/类型 | Modality | 说话人 | 音频 | 时间对齐 | 用途 | 优先级 | 可用性口径 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ds006104` speech decoding | [OpenNeuro](https://openneuro.org/datasets/ds006104) | 受控 phoneme/CV/VC/style | EEG + TMS | 多短刺激 | 本地已有大量 wav | events.tsv + phoneme/manner/place labels | controlled token probe；phoneme/F0/timbre/style | P0/P2 | `selected_public`; `local_ready` |
| KUL AAD | [Zenodo 4004271](https://zenodo.org/records/4004271) | 荷兰语竞争语音 | EEG + audio | competing Dutch stories | 有 audio | trial/attention labels | AAD baseline；speaker stream tracking | P1 | `selected_public` |
| DTU AAD | [Zenodo 1199011](https://zenodo.org/records/1199011) | 混响竞争语音 | EEG + audio | competing talkers + room | 有 audio | trial/attention labels | room robustness；空间泛化 | P1 | `selected_public` |
| 255ch EEG-AAD | [Zenodo 4518754](https://zenodo.org/records/4518754) | 高密度竞争语音 | 255ch EEG + audio | competing speakers | 有 audio | trial/attention labels | 高密度空间 tokenizer；sensor ablation | P1 | `selected_large` |
| `ds003626` Inner Speech | [EEGDash](https://eegdash.org/api/dataset/eegdash.dataset.DS003626.html) / [OpenNeuro](https://openneuro.org/datasets/ds003626) | Spanish inner/pronounced/visualized commands | EEG | 10 subjects；5640 trials | pronounced speech condition，无连续语音重建音频 | session/event labels + derivatives | inner vs pronounced vs visualized speech probe | P2 | `selected_public` |
| FEIS | [Zenodo 3554128](https://zenodo.org/records/3554128) | English phonemes + Chinese syllables | 14ch EEG + audio | 21 English + 2 Chinese participants | recorded audio + flashcards | phase CSVs per epoch | low-density heard/imagined/spoken phoneme probe | P2 | `selected_public` |
| UGR-MINDVOICE | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0885230826000276) / [OSF](https://osf.io/6sh5d) | Iberian Spanish overt/covert speech | 64ch EEG + synchronized audio | 15 native speakers | overt audio；covert same stimuli | LSL synchronized EEG/audio/events | phoneme/word/pseudoword overt-covert transfer probe | P2 | `selected_public` |
| `ds004306` semantic imagination/perception | [EEGDash](https://eegdash.org/api/dataset/eegdash.dataset.DS004306.html) / [OpenNeuro](https://openneuro.org/datasets/ds004306) | auditory/visual/orthographic semantic categories | 128ch EEG + audio/visual stimuli | 12 subjects；15 recordings | short auditory stimuli | event/task structure + BIDS | semantic imagination/perception proxy；P2/P3 bridge | P2/P3 | `selected_public` |
| OpenMIIR | [GitHub](https://github.com/sstober/openmiir) | 音乐感知/想象 | EEG + music | 非 speech | music stimuli | beat/downbeat/timing | pitch/beat/tempo token probe | P3 | `selected_public` |
| MUSIN-G `ds003774` | [OpenNeuro](https://openneuro.org/datasets/ds003774) | 自然音乐聆听 | EEG + music | 非 speech | music wav | trial/event timing | timbre/pitch 辅助预训练 | P3 | `selected_public` |
| MAD-EEG | [Zenodo 4537751](https://zenodo.org/records/4537751) | 目标乐器注意 | EEG + polyphonic music | 非 speech | music stems | attention/trial timing | target-source attention proxy | P3 | `selected_public` |

---

## 3. 当前数据情况（按 MD selected 口径）

### 3.1 结论

- **P0/P1 已经足够支撑第一版 tokenizer + AAD retrieval。** 中文 AAD、英文/粤语自然语音、KUL/DTU/255ch AAD 组合后，足够训练稳定的 speech EEG token 和 attended-stream retrieval。
- **真正短板是 P2。** 原 catalog 里 P2 主要靠 `ds006104`、`ds006465`、`ds007591` 和粤语 tone ERP，覆盖太窄；新增后 P2 覆盖 imagined speech、overt speech、inner speech、phoneme/syllable、prosody/emotion/intention、semantic imagination。
- **公开数据仍不能替代最终 target voice image 自采。** 新增 P2 可以训练 probe 和 overt-to-covert transfer，但缺少同一被试对系统 voice bank 的主观相似度评分和可控 F0/formant/style 全因子设计。

### 3.2 数量统计（去重数据集行，按主用途归类）

复合优先级如 `P0/P2` 仍保留在表格中；下面统计按主用途去重计数，避免把一个数据集重复算成多个“可用数据集”。

| 语言/类型 | P0 | P1 | P2 | P3 | MD selected 数据集数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 英文 / speech-decoding 扩展 | 3 | 6 | 3 | 0 | 12 |
| 中文（普通话） | 8 | 1 | 3 | 0 | 12 |
| 粤语 | 1 | 0 | 1 | 0 | 2 |
| 受控/代理/音乐 | 1 | 3 | 4 | 3 | 11 |
| **合计** | **13** | **10** | **11** | **3** | **37** |

### 3.3 本地训练就绪数量

本地状态不参与“真正可用”的判断。当前明确 `local_ready` 的样例仍主要是：

```
ds004408 · ds005345 · ds004718 · ds006104
```

其余 selected 数据集先按 metadata probe、单被试/单 run、再分批下载的节奏推进。

---

## 4. 四阶段训练路线

```
Stage 1  ->  通用 speech EEG tokenizer
Stage 2  ->  voice attribute / phoneme / prosody probe
Stage 3  ->  speaker stream retrieval
Stage 4  ->  target voice image（需自采）
```

### Stage 1：通用 speech EEG tokenizer

目标：训练稳定 EEG discrete token，不先做 waveform generation。

**推荐数据**：`ds004408` · `ds007630` · Weissbart · Etard · SparrKULee · `ds004718` · `ds005345` · `ds006434`

**训练目标**：

```
EEG reconstruction
+ masked EEG modeling
+ speech envelope / mel / phoneme-onset alignment
+ segment-level audio retrieval
+ token usage / perplexity / dead-code metrics
```

### Stage 2：voice attribute probe

目标：确认 token 是否保留声音内容、音调、音色、发音结构、想象/发声状态和风格。

**推荐数据**：`ds006104` · `ds006465` / 3M-CPSEED · `ds005170` Chisco · `ds003626` Inner Speech · Kara One · FEIS · UGR-MINDVOICE · CIRE · Cantonese tone/syllable ERP · `ds004306`

**训练目标**：

```
token -> phoneme / CV / VC / word category
token -> manner / place / voicing
token -> F0 high-low / pitch contour
token -> spectral centroid / brightness / MFCC statistics
token -> imagined vs overt vs inner / visualized condition
token -> happy-angry / prosody / intention / affective style proxy
```

### Stage 3：speaker stream retrieval

目标：EEG token 在自然或竞争语音中对齐目标说话人 stream。

**推荐数据**：`ds005345` · ESAA · NJU AAD · AASD · MS-AASD · Yan 系列 · ASA · `ds006434` · KUL AAD · DTU AAD · 255ch EEG-AAD

**训练目标**：

```
InfoNCE(token embedding, attended stream embedding)
+ target vs masker speaker retrieval
+ single-to-mix transfer
+ room / spatial / density generalization
```

### Stage 4：target voice image（自采）

公开数据缺少系统化 voice bank，自采数据需覆盖：

| 维度 | 要求 |
| --- | --- |
| 说话人数量 | 50-100 人 |
| 风格多样性 | 多风格、多情绪 |
| 音调覆盖 | 多 F0 范围 |
| 音色覆盖 | 多 formant / timbre |
| 主观评分 | 同一 subject 对同一 voice bank 的相似度评分 |
| 对齐精度 | EEG 与每条 voice item 的精确 trigger / audio-loopback 对齐 |

---

## 5. 最小可行数据组合（Core + Expansion + P2 Probe）

### Core（第一批主训练）

| 数据集 | 分工 |
| --- | --- |
| `ds006104` | controlled phoneme / pitch / timbre / style probe |
| `ds005345` | Mandarin synthetic male/female/mix speaker stream retrieval |
| `ds004408` | English natural speech phoneme/onset pretraining |
| `ds004718` | Cantonese word/prosody/F0/intensity alignment |
| `ds006434` | attention + high-precision speech timing |
| ESAA | Mandarin AAD speaker-stream retrieval |
| Four-Talker AAD (Yan 2024) | 4-speaker Mandarin identity 扩充；空间化 stream retrieval |
| Four-Direction AAD (Yan 2024) | 4-speaker 消声室基准；与 Four-Talker 合并使用 |
| Non-block AAD (Yan 2025) | 4-speaker 注意切换；自然聆听场景 |
| KUL AAD | classic AAD baseline |
| DTU AAD | reverberant competing speech robustness |
| 255ch EEG-AAD | high-density spatial encoding and sensor ablation |

### P2 Probe（新增后的第一批补强）

| 数据集 | 分工 |
| --- | --- |
| `ds005170` Chisco | 中文 imagined speech；大词表/句子级 semantic-imagined probe |
| `ds003626` Inner Speech | inner / pronounced / visualized Spanish commands；状态分类 probe |
| Kara One | imagined + vocalized phoneme/word；EEG-audio-face 多模态 probe |
| FEIS | heard / imagined / spoken English phonemes + Chinese syllables；低密度快速 sanity check |
| `ds007602` | overt speech production + vocal audio；EEG-to-audio production sanity check |
| UGR-MINDVOICE | overt/covert Spanish phoneme/word/pseudoword；overt-to-covert transfer |
| CIRE | 中文 prosody emotion / intention；style and intention probe |
| `ds004306` | auditory perception + semantic imagination；semantic proxy |
| `ds006465` / 3M-CPSEED | 普通话 pinyin production；声母/韵母/声调 probe |
| Cantonese tone/syllable ERP | 粤语 tone/pitch probe |

### Expansion（第二批）

```
ds007630 · 3M-CPSEED · Weissbart · Etard · SparrKULee
NJU AAD · AASD · MS-AASD · ASA (Lin 2024)
Fuglsang 2020 · Rotaru 2024 · Geirnaert 2025
```

### 弱代理（不进入主 speech 结论）

```
OpenMIIR · MUSIN-G · MAD-EEG
```

---

## 6. 多数据集组合策略

单个公开数据集的 speaker identity 空间通常很小，AAD 数据尤其常见 2-4 个说话人。跨数据集组合是在不自采前提下最大化 speaker identity、phoneme、prosody 和 speaking-mode 覆盖的唯一路径。

### 组合原则

| 原则 | 说明 |
| --- | --- |
| 说话人不重叠 | 不同数据集的说话人是不同真实人，组合后 speaker identity 空间是各数据集的并集 |
| EEG 不需同一被试 | tokenizer 训练不要求同一被试听过所有说话人，只要 EEG-audio 对齐即可 |
| 语言可以混合 | 跨语言组合扩大 phoneme/prosody 覆盖，speaker identity 标签按语言分组管理 |
| 任务条件需归一化 | trial 长度、采样率、通道数不同，需统一预处理后再合并 |
| P2 不等同于 retrieval | imagined/overt/inner 数据主要用于 probe 与 transfer，不直接等价于多说话人 attended retrieval |

### 当前可组合的 speaker / producer 来源

| 数据集组 | 真实说话人或 producer 上限 | 语言 | 备注 |
| --- | ---: | --- | --- |
| Yan 系列（3 个数据集） | 4 | 普通话 | 同一研究组，说话人可能重叠，需核对 |
| ESAA / NJU / AASD / MS-AASD / ASA | 8-15+ | 普通话 | AAD retrieval 主力，需核对说话人重叠 |
| `ds006434` | 2 | 英语 | female/male audiobook narrators |
| KUL / DTU / Fuglsang / Rotaru / Geirnaert | 4-8+ | 荷兰语/丹麦语 | 跨语言 speaker identity |
| `ds004718` | 1 | 粤语 | 单 narrator，粤语声调提供 F0 多样性 |
| `ds006104` | 多短刺激 | 受控 | happy/angry 多条短语音，style 多样性 |
| Kara One | 14 | 英语 | imagined + vocalized participant speech |
| FEIS | 23 | 英语/中文 | 低密度 heard/imagined/spoken phoneme/syllable |
| UGR-MINDVOICE | 15 | 西班牙语 | overt audio 已匿名化，适合 phoneme/word/pseudoword transfer |
| CIRE | 2 actors + 38 listeners | 普通话 | prosodic emotion/intention listening，不是 voice-bank retrieval |
| `ds007602` / `ds007630` | 3 subjects | speech production/listening | 大体量，但需先做 metadata 和单 run probe |

**估计**：

- retrieval-ready 自然/竞争语音 speaker identity：约 20-30 人，足以训练 stream retrieval 基础模型。
- P2 controlled / imagined / overt producer 覆盖：新增后约 50+ 人次，可明显补强 phoneme、speaking-mode、prosody/style probe。
- 最终 target voice image 仍需自采，因为公开数据没有统一 voice bank、同一被试主观相似度评分和系统化 F0/formant/style 操控。

### 技术要点

**跨数据集 speaker identity 对齐：**

```
- 说话人 ID：dataset_id + speaker_id（如 esaa_spk01）
- 不同数据集说话人不共享 embedding，各自独立初始化
- 训练目标：InfoNCE(EEG token, attended speaker embedding)
- 负样本来自同一 batch 内所有数据集的所有说话人
```

**跨数据集 EEG 归一化：**

```
- 统一重采样到 128 Hz 或 256 Hz
- 统一通道子集（64ch 公共子集，或 interpolation 对齐标准 montage）
- 每个 run 独立 z-score，消除数据集间幅度差异
- trial 长度统一截取（1s / 2s / 4s window），不足 pad，过长 stride 切分
```

**跨语言 phoneme/prosody 对齐：**

```
- 英语：CMU Pronouncing Dictionary / Montreal Forced Aligner
- 普通话：MFA Mandarin model 或 pypinyin + forced alignment
- 粤语：ds004718 自带 TextGrid
- 西班牙语：espeak-ng / MFA Spanish model，统一到 IPA 或 articulatory feature
- 统一映射到 IPA 或 articulatory feature（manner/place/voicing）
```

---

## 7. 公开数据的能力边界

### 公开数据可以训练和验证

- EEG token 的稳定性
- speech envelope / onset / phoneme / word alignment
- pitch、voicing、F0、intensity、timbre proxy
- single speaker vs mixed speaker retrieval
- attended target stream decoding
- imagined / overt / inner / pronounced speech proxy
- prosody emotion、intention、semantic imagination probe
- overt-to-covert / heard-to-imagined transfer 的初步可行性

### 公开数据不能完整解决

| 缺口 | 原因 |
| --- | --- |
| 大规模可控 speaker identity manifold | AAD 说话人少；production 数据多是被试自产，和 listening retrieval 不是同一任务 |
| 主观相似度评分 | 无同一 subject 对同一 voice bank 的评分数据 |
| 系统化声音属性操控 | 无多 F0 / 多 formant / 多 style 的全因子设计 |
| 个体化 voice image retrieval | 缺乏个体化 voice bank |
| waveform-level ground truth | imagined/covert speech 没有真实声波；UGR overt audio 还做了匿名化 |

> **结论**：公开多数据集组合已经足够训练 EEG speech tokenizer、AAD speaker-stream retrieval、P2 phoneme/prosody/imagined-overt probes；最终 voice image reconstruction 仍需要自采 multi-speaker / multi-style / multi-F0 voice bank 数据。

---

## 8. Metadata Probe 与下载执行计划

### 新增数据集的优先 probe

| 数据集 | 先验 probe |
| --- | --- |
| `ds007630` | EEGDash dataset page 已保存；OpenNeuro S3 object GET 对 BIDS/EDF/WAV 返回 403，后续需 EEGDash/OpenNeuro client 拉取 |
| `ds007602` | `dataset_description.json`、`participants.tsv`、single run `events.tsv/channels.tsv/eeg.json`、EDF byte range；未在 probed `beh/` prefix 暴露 vocal wav |
| `ds005170` | `dataset_description.json`、`README`、sub-01 raw EDF byte range、preprocessed FIF byte range |
| `ds003626` | `dataset_description.json`、`README`、derivative events.dat、epoched FIF byte range |
| Kara One | dataset webpage、participant archive list、helper scripts |
| FEIS | Zenodo record metadata、file list、archive checksum/size |
| UGR-MINDVOICE | OSF folder listing、GitHub `Readme.md`、config / scripts |
| `ds004306` | `dataset_description.json`、`participants.tsv`、README、short auditory stimulus、preprocessed FIF byte range |
| CIRE | Scientific Data page、ScienceDB repository metadata、README / participants / sentences / events when downloaded |

### 样例根目录

```
data/voice_eeg_dataset_samples/   （已加入 .gitignore）
├── manifest.json
├── README.md
├── _unified_index/
│   ├── manifest_compact.json
│   ├── sample_files.tsv
│   └── sample_status.md
├── _unified_samples/ # 指向各数据集样例文件的统一 symlink 目录
└── <category>/<dataset_slug>/
    ├── local/           # 本地完整样例
    ├── remote/          # 公开 metadata / 小文件
    ├── probe_artifacts/ # EEG 预览图 + 探测文件
    ├── status.json
    └── README.md
```

### 自动样例下载结果（2026-05-13）

本轮按 MD 已选 37 个数据集逐个生成样例目录。结果：

- `data/voice_eeg_dataset_samples/manifest.json`: 37 / 37 datasets present, 37 / 37 `ready_or_partial_sample`
- `data/voice_eeg_dataset_samples/_unified_index/sample_files.tsv`: 253 个样例文件记录
- `data/voice_eeg_dataset_samples/_unified_samples/`: 253 个统一 symlink
- 23 个数据集已拿到 EEG 小样例或 EEG 文件头；9 个数据集已拿到 audio 小样例或 audio/archive 文件头；24 个数据集至少有 audio 或 EEG
- 13 个数据集当前仅 metadata 或大 archive header，尚未展开到单 trial audio/eeg
- 真正自动失败项：`ds007630` 的 OpenNeuro S3 object GET 返回 403；已保存 EEGDash metadata，EEG/audio 需用 EEGDash/OpenNeuro client 或 DataLad/OpenNeuro CLI 继续拉

| 数据集 | Audio | EEG | Metadata/other | 自动样例状态 |
| --- | ---: | ---: | ---: | --- |
| `ds004408` | 6 | 8 | 9 | 已拿到 audio/eeg 小样例或文件头 |
| `weissbart_natural_speech` | 0 | 0 | 2 | metadata 或大 archive header；未展开到单 trial audio/eeg |
| `ds006434` | 2 | 7 | 6 | 已拿到 audio/eeg 小样例或文件头 |
| `ds007630_eeg_speech_brain_decoding` | 0 | 0 | 1 | 仅 EEGDash metadata；OpenNeuro S3 object GET 403，EEG/audio 需 EEGDash/OpenNeuro client |
| `ds007602_eeg_speech_overt` | 0 | 4 | 4 | 已拿到 audio/eeg 小样例或文件头 |
| `etard_continuous_speech_7086209` | 0 | 0 | 2 | metadata 或大 archive header；未展开到单 trial audio/eeg |
| `ds007591` | 0 | 4 | 8 | 已拿到 audio/eeg 小样例或文件头 |
| `kara_one` | 0 | 0 | 1 | 仅网页 metadata；participant archive 需人工选包 |
| `sparrkulee_eegdash` | 0 | 0 | 1 | metadata 或大 archive header；未展开到单 trial audio/eeg |
| `ds005345` | 6 | 8 | 16 | 已拿到 audio/eeg 小样例或文件头 |
| `esaa_7078451` | 0 | 0 | 7 | metadata 或大 archive header；未展开到单 trial audio/eeg |
| `nju_aad_7253438` | 0 | 0 | 3 | metadata 或大 archive header；未展开到单 trial audio/eeg |
| `ds006465_3m_cpseed` | 0 | 6 | 2 | 已拿到 audio/eeg 小样例或文件头 |
| `ds005170_chisco` | 0 | 2 | 3 | 已拿到 audio/eeg 小样例或文件头 |
| `cire_2025` | 0 | 0 | 1 | 仅论文页 metadata；ScienceDB 数据需人工进入仓库 |
| `aasd_17413336` | 1 | 0 | 2 | 已拿到 audio/eeg 小样例或文件头 |
| `ms_aasd_17149387` | 0 | 0 | 5 | metadata 或大 archive header；未展开到单 trial audio/eeg |
| `ds004718` | 3 | 3 | 9 | 已拿到 audio/eeg 小样例或文件头 |
| `cantonese_tone_syllable_7750292` | 0 | 1 | 1 | 已拿到 audio/eeg 小样例或文件头 |
| `ds006104` | 8 | 5 | 19 | 已拿到 audio/eeg 小样例或文件头 |
| `ds003626_inner_speech` | 0 | 3 | 2 | 已拿到 audio/eeg 小样例或文件头 |
| `feis_3554128` | 0 | 0 | 2 | metadata 或大 archive header；未展开到单 trial audio/eeg |
| `ugr_mindvoice` | 0 | 0 | 3 | OSF/GitHub metadata；subject EEG/audio 需按 OSF 文件树继续选包 |
| `ds004306_semantic_imagination` | 1 | 1 | 3 | 已拿到 audio/eeg 小样例或文件头 |
| `kul_aad_4004271` | 0 | 1 | 5 | 已拿到 audio/eeg 小样例或文件头 |
| `dtu_aad_1199011` | 1 | 1 | 3 | 已拿到 audio/eeg 小样例或文件头 |
| `eeg_aad_255ch_4518754` | 0 | 1 | 5 | 已拿到 audio/eeg 小样例或文件头 |
| `openmiir` | 0 | 0 | 6 | metadata 或大 archive header；未展开到单 trial audio/eeg |
| `musin_g_ds003774` | 2 | 4 | 7 | 已拿到 audio/eeg 小样例或文件头 |
| `mad_eeg_4537751` | 0 | 1 | 7 | 已拿到 audio/eeg 小样例或文件头 |
| `four_talker_aad_10803261` | 0 | 1 | 1 | 已拿到 audio/eeg 小样例或文件头 |
| `four_direction_aad_10803229` | 0 | 1 | 1 | 已拿到 audio/eeg 小样例或文件头 |
| `non_block_aad_14887886` | 0 | 2 | 1 | 已拿到 audio/eeg 小样例或文件头 |
| `asa_lin2024_11541114` | 0 | 1 | 2 | 已拿到 audio/eeg 小样例或文件头 |
| `fuglsang2020_3618205` | 0 | 0 | 1 | metadata 或大 archive header；未展开到单 trial audio/eeg |
| `rotaru2024_11058711` | 0 | 1 | 2 | 已拿到 audio/eeg 小样例或文件头 |
| `geirnaert2025_16536441` | 0 | 2 | 2 | 已拿到 audio/eeg 小样例或文件头 |

### 本地完整样例（可直接跑代码）

| 数据集 | 样例内容 | EEG 预览图 |
| --- | --- | --- |
| `ds004408` | BrainVision `.eeg/.vhdr/.vmrk` + `audio01.wav` + TextGrid | `probe_artifacts/eeg_preview.vhdr.png` |
| `ds005345` | run-1~4 EEG `.npz` + female/male/mix wav + acoustic CSV | `probe_artifacts/eeg_preview.npz.png` |
| `ds004718` | 粤语句子 wav + `.set` + timing/acoustic probe files | `probe_artifacts/eeg_preview.png` |
| `ds006104` | sub-P01/S01 EEG `.npz` + events/channels + 8 条 wav | `probe_artifacts/eeg_preview.npz.png` |

### 样例管理命令

```bash
# 重新生成所有样例
python3 scripts/download_voice_eeg_dataset_samples.py

# 只更新某个数据集
python3 scripts/download_voice_eeg_dataset_samples.py --dataset ds005170_chisco

# 允许下载较大远程文件
python3 scripts/download_voice_eeg_dataset_samples.py --allow-large

# 生成所有本地 EEG 文件的预览图
python3 scripts/visualize_eeg_samples.py

# 轻量 metadata probe
python3 scripts/probe_eeg_audio_datasets.py --only ds007602 ds005170 ds003626
```
