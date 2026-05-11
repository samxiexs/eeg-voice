# Multi-Dataset EEG Voice Catalog（0512）

## 研究主链路

```
EEG → discrete token → content / pitch / timbre / speaker / style alignment
                                        ↓
                        voice image retrieval / reconstruction
```

---

## 1. 筛选标准与优先级

### 入池条件（满足其一即可）

| 条件 | 说明 |
| --- | --- |
| 听觉语音 EEG | 被试听真实或合成语音，EEG 与音频 onset/segment/trial 对齐 |
| 多说话人或竞争语音 | 存在不同 speaker stream、attention target 或 competing talker |
| 发声或想象语音 | 被试产生/轻声发音/默读/想象音素词句，作为无外部声波的代理 |
| 受控声音属性 | 存在 phoneme、CV/VC、F0、formant、emotion/style、空间化或听觉注意操控 |
| 弱代理声音数据 | 可提供 pitch、timbre、style、affect 或 attention 预训练信号 |

### 优先级定义

| 优先级 | 定义 | 进入训练的位置 |
| --- | --- | --- |
| **P0** 主训练集 | 听觉语音 EEG，存在音频刺激，适合 token-to-audio / token-to-speaker retrieval | 第一批下载和建模 |
| **P1** 辅助预训练集 | 自然语音、AAD、speech envelope tracking，训练通用 speech EEG tokenizer | tokenizer pretraining |
| **P2** 代理/控制集 | 发声、想象语音、情绪声音、受控音素、合成声音 | phoneme / pitch / timbre / style probe |
| **P3** 弱相关数据 | 音乐、情绪视频、非语音听觉任务 | 辅助预训练，不进入主 speech 结论 |

---

## 2. 数据集全览

### 2.1 英文数据集

| 数据集 | 链接 | 语言 | Modality | 说话人 | 音频 | 时间对齐 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ds004408` naturalistic speech | [OpenNeuro](https://openneuro.org/datasets/ds004408) | 英语有声书 | EEG | 单 narrator，20 段 | wav | TextGrid word/phoneme | 英语 tokenizer 预训练；phoneme/onset | P0 |
| `ds006434` ABR + attention | [OpenNeuro](https://openneuro.org/datasets/ds006434) | 英语双 narrator | EEG | female + male narrator | wav | 64s epoch，attention trigger | 高精度 timing；attended stream | P0/P1 |
| Weissbart natural speech | [Zenodo 7086168](https://zenodo.org/records/7086168) | 英语连续语音 | EEG | audiobook narrative | 有刺激材料 | continuous speech timing | acoustic tracking；surprisal predictor | P1 |
| Etard competing speech | [Zenodo 7086209](https://zenodo.org/records/7086209) | 英语 + competing | EEG | audiobook + competing | 有 audio | continuous alignment | 英语 competing-speaker 扩展 | P1 |
| SparrKULee / EEGDash | [EEGDash NM000238](https://eegdash.org/api/dataset/eegdash.dataset.NM000238.html) | 大规模 speech | EEG | 85 participants | 需核对 | EEGDash metadata | 大规模 tokenizer pretraining | P1 |
| Fuglsang 2020 | [Zenodo 3618205](https://zenodo.org/record/3618205) | 丹麦语 AAD | EEG + audio | 2 说话人；44 被试含听障 | 有 audio | trial/attention labels | 大样本 AAD；听障泛化 | P1 |
| Rotaru 2024 | [Zenodo 11058711](https://zenodo.org/records/11058711) | 荷兰语 AAD | EEG + audio | 2 说话人；每被试 80 min | 有 audio | trial/attention labels | 长时录音；长序列稳定性 | P1 |
| Geirnaert 2025 | [Zenodo 16536441](https://zenodo.org/records/16536441) | 丹麦语 AAD | scalp+around-ear+in-ear | 2 说话人；15 被试 | 有 audio | 设备同步 metadata | 多设备 sensor ablation | P1 |
| `ds007591` speech decoding | [OpenNeuro](https://openneuro.org/datasets/ds007591) | overt speech production | EEG | 被试产生 color words | 需核对 | events.tsv | production sanity check | P2 |

### 2.2 中文数据集

| 数据集 | 链接 | 语言 | Modality | 说话人 | 音频 | 时间对齐 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ds005345` LPP Multi-talker | [OpenNeuro](https://openneuro.org/datasets/ds005345) | 普通话合成语音 | EEG + fMRI | 合成男声 + 女声 + mix | female/male/mix wav | run mapping + acoustic CSV | 中文主数据集；speaker stream retrieval | P0 |
| ESAA | [Zenodo 7078451](https://zenodo.org/records/7078451) | 普通话 AAD | EEG + audio | female/male storytellers | 有语音材料 | trial onset / AAD labels | Mandarin AAD；target speaker retrieval | P0 |
| NJU AAD | [Zenodo 7253438](https://zenodo.org/records/7253438) | 普通话竞争语音 | EEG + audio | competing Mandarin speakers | 有语音材料 | trial/attention timing | 中文多说话人 AAD；contrastive learning | P0/P1 |
| AASD | [Zenodo 17413336](https://zenodo.org/records/17413336) | 普通话注意切换 | EEG + audio | 多说话人，多目标流 | 有语音材料 | switch timing / trial labels | 动态 target stream；注意切换 | P0/P1 |
| MS-AASD | [Zenodo 17149387](https://zenodo.org/records/17149387) | 普通话 mixed-speech | EEG + audio | mixed + self-initiated switch | 有语音材料 | switch metadata | 多说话人注意切换扩展 | P0/P1 |
| **Four-Talker AAD** (Yan 2024) | [Zenodo 10803261](https://zenodo.org/records/10803261) | 普通话，4 说话人空间化 | EEG 64ch + cEEGrid | **2 男 2 女真实说话人**；±90°/±30° | 有语音材料 | trial/attention + 空间角度 | **4-speaker identity 扩充**；空间化 stream retrieval | P0 |
| **Four-Direction AAD** (Yan 2024) | [Zenodo 10803229](https://zenodo.org/records/10803229) | 普通话，4 方向空间化 | EEG 64ch | **4 说话人**；消声室 | 有语音材料 + 代码 | trial/attention + 方向 | 4-speaker 消声室基准；与 Four-Talker 合并 | P0 |
| **Non-block AAD** (Yan 2025) | [Zenodo 14887886](https://zenodo.org/records/14887886) | 普通话，非 block 切换 | EEG 64ch + cEEGrid | **4 说话人**；自由切换 | 有语音材料 + 代码 | switch timing | 4-speaker 注意切换；贴近自然聆听 | P0/P1 |
| ASA (Lin 2024) | [Zenodo 11541114](https://zenodo.org/records/11541114) | 普通话，多空间角度 | EEG 64ch + audio | 2 说话人；±5°–±90° | 有语音材料 | trial/attention + 空间角度 | 空间泛化；±5° 近距离难度最高 | P1 |
| `ds006465` / 3M-CPSEED | [OpenNeuro](https://openneuro.org/datasets/ds006465) | 普通话拼音 production | EEG | 20 subjects 自产 pinyin | 需核对 | prompt/event timing | 拼音/声母/韵母/声调 probe | P2 |

### 2.3 粤语数据集

| 数据集 | 链接 | 语言 | Modality | 说话人 | 音频 | 时间对齐 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ds004718` LPPHK | [OpenNeuro](https://openneuro.org/datasets/ds004718) | 粤语《小王子》 | EEG + fMRI | 单 Cantonese narrator；52 被试 | section/sentence wav | word timing + F0/intensity + POS | 粤语主数据集；prosody/F0/intensity alignment | P0 |
| Cantonese tone/syllable ERP | [Zenodo 7750292](https://zenodo.org/records/7750292) | 粤语声调/音节 | EEG/ERP | 需核对 | 需核对 | event/trial timing | 粤语 tone/pitch probe | P2 |

### 2.4 受控声音 / 代理声音

| 数据集 | 链接 | 语言/类型 | Modality | 说话人 | 音频 | 时间对齐 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ds006104` speech decoding | [OpenNeuro](https://openneuro.org/datasets/ds006104) | 受控 phoneme/CV/VC/style | EEG + TMS | 多短刺激 | 本地已有大量 wav | events.tsv + phoneme/manner/place labels | controlled token probe；phoneme/F0/timbre/style | P0/P2 |
| KUL AAD | [Zenodo 4004271](https://zenodo.org/records/4004271) | 荷兰语竞争语音 | EEG + audio | competing Dutch stories | 有 audio | trial/attention labels | AAD baseline；speaker stream tracking | P1 |
| DTU AAD | [Zenodo 1199011](https://zenodo.org/records/1199011) | 混响竞争语音 | EEG + audio | competing talkers + room | 有 audio | trial/attention labels | room robustness；空间泛化 | P1 |
| 255ch EEG-AAD | [Zenodo 4518754](https://zenodo.org/records/4518754) | 高密度竞争语音 | 255ch EEG + audio | competing speakers | 有 audio | trial/attention labels | 高密度空间 tokenizer；sensor ablation | P1 |
| OpenMIIR | [GitHub](https://github.com/sstober/openmiir) | 音乐感知/想象 | EEG + music | 非 speech | music stimuli | beat/downbeat/timing | pitch/beat/tempo token probe | P3 |
| MUSIN-G `ds003774` | [OpenNeuro](https://openneuro.org/datasets/ds003774) | 自然音乐聆听 | EEG + music | 非 speech | music wav | trial/event timing | timbre/pitch 辅助预训练 | P3 |
| MAD-EEG | [Zenodo 4537751](https://zenodo.org/records/4537751) | 目标乐器注意 | EEG + polyphonic music | 非 speech | music stems | attention/trial timing | target-source attention proxy | P3 |

---

## 3. 数据集数量统计

| 语言/类型 | P0 | P1 | P2 | P3 | 合计 |
| --- | --- | --- | --- | --- | --- |
| 英文 | 2 | 7 | 1 | — | 10 |
| 中文（普通话） | 8 | 1 | 1 | — | 10 |
| 粤语 | 1 | — | 1 | — | 2 |
| 受控/代理/音乐 | 1 | 3 | — | 3 | 7 |
| **合计** | **12** | **11** | **3** | **3** | **29** |

---

## 4. 四阶段训练路线

```
Stage 1  ──►  通用 speech EEG tokenizer
Stage 2  ──►  voice attribute probe
Stage 3  ──►  speaker stream retrieval
Stage 4  ──►  target voice image（需自采）
```

### Stage 1：通用 speech EEG tokenizer

目标：训练稳定 EEG discrete token，不先做 waveform generation。

**推荐数据**：`ds004408` · Weissbart · Etard · SparrKULee · `ds004718` · `ds005345` · `ds006434`

**训练目标**：
```
EEG reconstruction
+ masked EEG modeling
+ speech envelope / mel / phoneme-onset alignment
+ segment-level audio retrieval
+ token usage / perplexity / dead-code metrics
```

### Stage 2：voice attribute probe

目标：确认 token 是否保留声音内容、音调、音色、发音结构和风格。

**推荐数据**：`ds006104` · Cantonese tone/syllable ERP · `ds006465` / 3M-CPSEED

**训练目标**：
```
token → phoneme / CV / VC / word category
token → manner / place / voicing
token → F0 high-low / pitch contour
token → spectral centroid / brightness / MFCC statistics
token → happy-angry / affective style proxy
```

### Stage 3：speaker stream retrieval

目标：EEG token 在自然或竞争语音中对齐目标说话人 stream。

**推荐数据**：`ds005345` · ESAA · NJU AAD · AASD · MS-AASD · Yan 系列 · `ds006434` · KUL AAD · DTU AAD · 255ch EEG-AAD

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
| 说话人数量 | 50–100 人 |
| 风格多样性 | 多风格、多情绪 |
| 音调覆盖 | 多 F0 范围 |
| 音色覆盖 | 多 formant / timbre |
| 主观评分 | 同一 subject 对同一 voice bank 的相似度评分 |
| 对齐精度 | EEG 与每条 voice item 的精确 trigger / audio-loopback 对齐 |

---

## 5. 最小可行数据组合（Core + Expansion）

### Core（第一批）

| 数据集 | 分工 |
| --- | --- |
| `ds006104` | controlled phoneme / pitch / timbre / style probe |
| `ds005345` | Mandarin synthetic male/female/mix speaker stream retrieval |
| `ds004408` | English natural speech phoneme/onset pretraining |
| `ds004718` | Cantonese word/prosody/F0/intensity alignment |
| `ds006434` | attention + high-precision speech timing |
| ESAA | Mandarin AAD speaker-stream retrieval |
| Four-Talker AAD (Yan 2024) | **4-speaker** Mandarin identity 扩充；空间化 stream retrieval |
| Four-Direction AAD (Yan 2024) | **4-speaker** 消声室基准；与 Four-Talker 合并使用 |
| Non-block AAD (Yan 2025) | **4-speaker** 注意切换；自然聆听场景 |
| KUL AAD | classic AAD baseline |
| DTU AAD | reverberant competing speech robustness |
| 255ch EEG-AAD | high-density spatial encoding and sensor ablation |

### Expansion（第二批）

```
3M-CPSEED · Weissbart · Etard · SparrKULee
NJU AAD · AASD · MS-AASD
ASA (Lin 2024) · Fuglsang 2020 · Rotaru 2024 · Geirnaert 2025
```

### 弱代理（不进入主 speech 结论）

```
OpenMIIR · MUSIN-G · MAD-EEG
```

---

## 6. 多数据集组合策略

单个公开数据集的 speaker identity 空间都很小（通常 2–4 个说话人）。跨数据集组合是在不自采前提下最大化 speaker identity 覆盖的唯一路径。

### 组合原则

| 原则 | 说明 |
| --- | --- |
| 说话人不重叠 | 不同数据集的说话人是不同真实人，组合后 speaker identity 空间是各数据集的并集 |
| EEG 不需同一被试 | tokenizer 训练不要求同一被试听过所有说话人，只要 EEG-audio 对齐即可 |
| 语言可以混合 | 跨语言组合扩大 phoneme/prosody 覆盖，speaker identity 标签按语言分组管理 |
| 任务条件需归一化 | trial 长度、采样率、通道数不同，需统一预处理后再合并 |

### 当前可组合的 speaker identity 来源

| 数据集组 | 真实说话人数 | 语言 | 备注 |
| --- | --- | --- | --- |
| Yan 系列（3 个数据集） | **4** | 普通话 | 同一研究组，说话人可能重叠，需核对 |
| ESAA | 2+ | 普通话 | female/male storytellers |
| NJU AAD | 2+ | 普通话 | 需核对是否与 ESAA 重叠 |
| AASD / MS-AASD | 多（需核对） | 普通话 | 多目标流切换 |
| `ds006434` | 2 | 英语 | female/male audiobook narrators |
| KUL AAD | 2+ | 荷兰语 | Dutch stories |
| DTU AAD / Fuglsang 2020 | 2 | 丹麦语 | 跨语言 speaker identity |
| `ds004718` | 1 | 粤语 | 单 narrator，粤语声调提供 F0 多样性 |
| `ds006104` | 多短刺激 | 受控 | happy/angry 多条短语音，style 多样性 |

**组合后估计**：普通话 8–15 人，英语 2–4 人，荷兰/丹麦语 4–6 人。

### 技术要点

**跨数据集 speaker identity 对齐：**
```
- 说话人 ID：dataset_id + speaker_id（如 esaa_spk01）
- 不同数据集说话人不共享 embedding，各自独立初始化
- 训练目标：InfoNCE(EEG token, attended speaker embedding)
  负样本来自同一 batch 内所有数据集的所有说话人
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
- 统一映射到 IPA 或 articulatory feature（manner/place/voicing）
```

### Speaker identity 空间上限估计

| 数据来源 | 说话人数上限 | 备注 |
| --- | --- | --- |
| 公开数据组合 | ~20–30 人 | 普通话为主，足以训练 stream retrieval 基础模型 |
| 自采目标 | 50–100 人 | 覆盖多 F0 / 多 formant / 多 style，支撑 voice image retrieval |

---

## 7. 公开数据的能力边界

### 公开数据可以训练和验证

- EEG token 的稳定性
- speech envelope / onset / phoneme / word alignment
- pitch、voicing、F0、intensity、timbre proxy
- single speaker vs mixed speaker retrieval
- attended target stream decoding
- imagined / overt speech proxy

### 公开数据不能完整解决

| 缺口 | 原因 |
| --- | --- |
| 大规模 speaker identity manifold | 即使加入 Yan 系列，最多 4 个真实说话人 |
| 主观相似度评分 | 无同一 subject 对同一 voice bank 的评分数据 |
| 系统化声音属性操控 | 无多 F0 / 多 formant / 多 style 的受控设计 |
| 个体化 voice image retrieval | 缺乏个体化 voice bank |
| waveform-level ground truth | 无最终重建的 ground truth |

> **结论**：公开多数据集组合用于训练 EEG speech tokenizer 和 voice-representation alignment；最终 voice image reconstruction 需要自采 multi-speaker / multi-style / multi-F0 voice bank 数据。

---

## 8. 本地样例状态

### 样例根目录

```
data/voice_eeg_dataset_samples/   （已加入 .gitignore）
├── manifest.json
├── README.md
└── <category>/<dataset_slug>/
    ├── local/          # 本地完整样例
    ├── remote/         # 公开 metadata / 小文件
    ├── probe_artifacts/ # EEG 预览图 + 探测文件
    ├── status.json
    └── README.md
```

### 本地完整样例（可直接跑代码）

| 数据集 | 样例内容 | EEG 预览图 |
| --- | --- | --- |
| `ds004408` | BrainVision `.eeg/.vhdr/.vmrk` + `audio01.wav` + TextGrid | `probe_artifacts/eeg_preview.vhdr.png` |
| `ds005345` | run-1~4 EEG `.npz` + female/male/mix wav + acoustic CSV | `probe_artifacts/eeg_preview.npz.png` |
| `ds004718` | 粤语句子 wav + `.set` + timing/acoustic probe files | `probe_artifacts/eeg_preview.png` |
| `ds006104` | sub-P01/S01 EEG `.npz` + events/channels + 8 条 wav | `probe_artifacts/eeg_preview.npz.png` |

### Zenodo/OpenNeuro metadata 已落盘（28 个数据集）

Yan 系列（Four-Talker / Four-Direction / Non-block）、ESAA、NJU AAD、AASD、MS-AASD、KUL AAD、DTU AAD、255ch EEG-AAD、ASA、Fuglsang 2020、Rotaru 2024、Geirnaert 2025 等均已下载 `zenodo_record.json`。

Yan 系列实际 EEG 文件为 2–9 GB 大 zip，需手动下载后放入 `local/` 目录，再运行：

```bash
python3 scripts/visualize_eeg_samples.py
```

### 样例管理命令

```bash
# 重新生成所有样例
python3 scripts/download_voice_eeg_dataset_samples.py

# 只更新某个数据集
python3 scripts/download_voice_eeg_dataset_samples.py --dataset ds005345

# 允许下载较大远程文件
python3 scripts/download_voice_eeg_dataset_samples.py --allow-large

# 生成所有本地 EEG 文件的预览图
python3 scripts/visualize_eeg_samples.py
```
