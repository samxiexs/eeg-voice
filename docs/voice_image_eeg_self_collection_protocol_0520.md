# Voice-Image EEG 自采实验需求协议（0520）

## 0. 文档定位

本文档用于把当前 `EEG -> grouped discrete token -> speech / voice alignment -> voice-image reconstruction` 的研究目标转化为可执行的自采数据协议。它面向三类用途：

1. 论文方法部分和预注册方案。
2. IRB / 伦理申请与临床合作沟通。
3. 后续数据采集、BIDS 整理、模型训练和统计分析的共同接口。

本协议不是普通 `EEG -> text` 数据集方案。核心目标是采集足以训练和检验以下链路的数据：

```text
EEG
-> grouped discrete EEG tokens
-> content / pitch / prosody / timbre / speaker / style alignment
-> subject-specific voice-image retrieval / reconstruction
-> optional downstream speech codec or vocoder rendering
```

研究主张必须保持清晰边界：健康受试者数据用于证明外部声音和想象声音的 EEG token alignment；幻听患者数据用于检验 auditory verbal hallucination (AVH) episode 是否能在个体化 voice manifold 中检索患者主观确认的声音形象。若临床样本不足，只能主张 healthy voice-image token alignment，不能声称已经还原幻听。

---

## 1. 论文主张与科学问题

### 1.1 中心主张

本研究应写成：

```text
Controlled self-collected EEG-voice data can support a discrete EEG token interface
that aligns not only with speech content but also with pitch, prosody, timbre,
speaker identity, style, and subject-specific voice-image perception.
```

中文表述：

```text
用自采可控 EEG-voice 数据建立 EEG discrete token，并证明这些 token 不只编码 speech content，
还可对齐和检索 pitch、prosody、timbre、speaker identity、style，以及幻听患者主观听到的 voice image。
```

### 1.2 分层结论

| 层级 | 样本 | 主要结论 | 不可过度声称 |
| --- | --- | --- | --- |
| Healthy listening | 健康成人听外部 voice bank | EEG token 能对齐 content、prosody、timbre、speaker、style，并检索对应 voice item | 不能声称恢复主观幻听 |
| Healthy imagery | 健康成人想象/静默 replay voice cue | 想象声音 token 与听觉 token 在 subject-level voice manifold 中部分接近 | 不能替代 AVH 患者证据 |
| Clinical AVH | 当前有 AVH 的患者 | AVH episode EEG token 可检索患者事后确认的 voice prototype 或 morph | 不能声称恢复客观存在的真实声波 |

### 1.3 具体科学问题

**Q1. EEG discrete token 是否稳定？**

连续 EEG 是否可以被压缩成 codebook usage 合理、跨 session 稳定、跨 subject 可迁移的 grouped discrete token。这里关注的是 token 质量，不是 waveform quality。

**Q2. EEG token 是否只编码内容，还是也编码声音形象？**

在固定 content 的条件下，token 是否仍然能区分 speaker、formant、timbre、style、F0 和 spatial/externalization。若只能解码 phoneme 或 word，则不足以支撑 voice-image claim。

**Q3. Voice/timbre/prosody 能否与 content 解耦？**

通过 same-content different-speaker、same-speaker different-content、F0-only、formant-only、style-only hard negatives，检验 token 的 attribute-specific routing 是否成立。

**Q4. AVH episode 是否能映射到患者主观声音空间？**

患者自然发生的 AVH episode 是否能在其个体化 voice bank 中检索到事后选择的 Top-K closest voice prototypes，并且显著优于 shuffled-time、content-only、speaker-only 和 random baselines。

---

## 2. 文献支撑与实验逻辑

### 2.1 顶刊证据链

| 设计决定 | 支撑文献 | 对本实验的约束 |
| --- | --- | --- |
| Voice/timbre 不是文本内容的附属变量 | Belin et al., *Nature*, 2000, voice-selective areas in human auditory cortex. https://www.nature.com/articles/35002078 | voice bank 必须操控 speaker、timbre、formant、style，不能只采 transcript |
| 非侵入式 EEG/MEG 可做 speech perception segment retrieval | Défossez et al., *Nature Machine Intelligence*, 2023. https://www.nature.com/articles/s42256-023-00714-5 | EEG-audio 对齐可采用约 3 s window 和 retrieval 评价，但要报告 EEG 的性能上限 |
| 神经语音合成需要中间语音表示 | Chen et al., *Nature Machine Intelligence*, 2024. https://www.nature.com/articles/s42256-024-00824-8 | EEG 不直接预测 full waveform，先预测 interpretable speech / voice parameters |
| 侵入式 speech decoding 已可个体化合成声音 | Anumanchipalli et al., *Nature*, 2019. https://www.nature.com/articles/s41586-019-1119-1；Metzger et al., *Nature*, 2023. https://www.nature.com/articles/s41586-023-06443-4 | 非侵入式 EEG 不能照搬 ECoG 结论，应降低为 token alignment 和 retrieval |
| Speech rhythm 2-8 Hz 和 envelope 对 intelligibility 关键 | Poeppel & Assaneo, *Nature Reviews Neuroscience*, 2020. https://www.nature.com/articles/s41583-020-0304-4 | q0-q1 base auditory 和 q4 prosody 必须保留 envelope、rhythm、F0/energy |
| Connected speech 有 word / phrase / sentence 层级 cortical tracking | Ding et al., *Nature Neuroscience*, 2016. https://www.nature.com/articles/nn.4186 | content token 不能只使用 acoustic envelope baseline，需包含 phoneme/word/unit targets |
| 幻听与 perceptual priors overweighting 有关 | Powers et al., *Science*, 2017. https://doi.org/10.1126/science.aan3458 | AVH block 不能当作普通外部听觉刺激，需要记录 confidence、prior-like voice matching 和临床量表 |
| AVH 与 inner speech misattribution、insula/STG/auditory cortex 网络有关 | Barber et al., *Translational Psychiatry*, 2021. https://www.nature.com/articles/s41398-021-01670-7；Magioncalda et al., *Nature Mental Health*, 2025. https://www.nature.com/articles/s44220-025-00493-5 | 临床分析必须区分 resting AVH、imagery、external listening，避免 reverse inference |
| EEG 数据应使用 BIDS/EEG-BIDS | Pernet et al., *Scientific Data*, 2019. https://www.nature.com/articles/s41597-019-0104-8；BIDS EEG specification. https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html | `events.tsv`、`channels.tsv`、`electrodes.tsv`、`coordsystem.json`、sidecar JSON 必须完整 |
| Audio token 应分层处理 semantic/prosody/voice/codec | HuBERT, wav2vec 2.0, WavLM, NaturalSpeech 3/FACodec, X-Codec, AudioLM, VoiceCraft；仓库综述见 `docs/audio_decoder_ccf_a_papers_0520.md` | EEG 主监督对齐 semantic/prosody/voice token；codec residual 只作为 decoder 后端 |

### 2.2 研究空缺

公开数据池已经可以支持 tokenizer pretraining、speech / voice attribute probe 和 general speaker retrieval，但无法直接支撑 personalized subjective voice-image reconstruction。主要缺口如下：

| 缺口 | 公开数据问题 | 自采数据补救 |
| --- | --- | --- |
| 统一 voice bank | 不同数据集说话人、语言、设备和任务不一致 | 所有受试者听同一批可控 voice items |
| Subject-level perceptual voice labels | 公开数据通常没有同一受试者对 voice timbre 的主观评分 | Part A 采 subject-specific ratings |
| 同 content 多 speaker / timbre | AAD 数据通常是自然故事，不是系统操控 | same-content different-speaker/formant/style hard negatives |
| 幻听主观声音形象 | 公开 EEG 数据没有 AVH episode + voice matching | Clinical AVH block 记录 episode-level voice prototype |
| q7 nuisance 检验 | 跨数据集难以分离 device、dataset、subject shortcut | 单一协议内采集，并显式报告 q7 predictability |

---

## 3. 样本与分组

### 3.1 总体样本设计

| 阶段 | 目标样本 | 有效样本目标 | 目的 |
| --- | ---: | ---: | --- |
| Pilot healthy | 16 | >=12 | 验证任务、同步、评分稳定性、artifact rate、初步 retrieval |
| Healthy main | 80 | >=60 | 主模型训练、cross-subject token、attribute alignment、voice retrieval |
| Healthy retest | main 中 30 人 | >=25 | 7-14 天 test-retest，检验 voice manifold 和 token 稳定性 |
| Clinical AVH | 30 | >=20 有可用 AVH episode | 临床 voice-image retrieval 验证 |
| Clinical/control comparison | 30 | >=24 | 匹配对照，控制 medication/general psychosis/resting state |

### 3.2 Healthy inclusion criteria

| 项目 | 标准 |
| --- | --- |
| 年龄 | 18-55 岁 |
| 听力 | 纯音或简化听阈筛查正常；左右耳差异记录 |
| 语言 | 普通话母语或高熟练度；英文能力记录但不作为主分析前提 |
| 利手 | 右利手优先；左利手可纳入但作为协变量 |
| 任务能力 | 能完成 2 小时内听觉任务、按键、评分 |
| 同意 | 完成书面知情同意，允许匿名健康数据开放 |

### 3.3 Healthy exclusion criteria

| 项目 | 排除标准 |
| --- | --- |
| 神经系统疾病 | 癫痫、脑外伤、脑卒中、严重偏头痛等 |
| 精神病史 | psychosis、bipolar mania、当前重度抑郁或严重焦虑发作 |
| 听力异常 | 未校正听力损失，耳鸣严重影响任务 |
| 药物 | 当前使用明显影响 EEG/arousal 的药物，或记录后由 PI 决定剔除 |
| 状态 | 实验当天严重睡眠不足、醉酒、急性疾病 |
| 数据质量 | 有效 epochs <70%、catch accuracy <70%、过多肌电或运动 artifact |

### 3.4 Clinical AVH inclusion criteria

| 项目 | 标准 |
| --- | --- |
| 诊断范围 | schizophrenia-spectrum 或其他经临床确认的 psychosis spectrum，当前有 auditory verbal hallucination |
| 年龄 | 18-55 岁，和 healthy group 匹配 |
| 稳定性 | 当前状态允许 60-90 分钟低负荷 EEG；无急性危险 |
| AVH 频率 | 近期有足够频繁 AVH，使 20-30 分钟 resting/low-load block 有合理 episode 捕获概率 |
| 临床记录 | PANSS、PSYRATS-AH、AVHRS-Q、药物、病程、住院/门诊状态、睡眠、咖啡因、尼古丁 |
| 安全 | 精神科医生或训练过的临床人员在场；明确停机标准 |

### 3.5 Clinical AVH exclusion criteria

| 项目 | 排除标准 |
| --- | --- |
| 急性风险 | 明确自伤/他伤风险、严重激越、无法理解任务 |
| 诱发风险 | 近期症状极不稳定，任何听觉任务可能明显恶化症状 |
| 听觉条件 | 严重听力障碍，无法进行 voice bank calibration |
| 药物/物质 | 急性中毒、戒断、镇静过深 |
| EEG 禁忌 | 皮肤问题、无法佩戴 EEG cap、强烈不适 |

### 3.6 Clinical safety rules

1. AVH block 不诱发、不强化、不训练幻听。
2. 只采自然发生 AVH 或低负荷 resting/imagery 状态下自发 episode。
3. 患者可随时暂停或结束。
4. 若 distress rating 达预设阈值，例如 80/100 或临床人员判断风险升高，立即停止。
5. 所有自由文本、临床量表、voice matching 结果和患者音色描述放入 controlled access。

---

## 4. Voice Bank 与刺激设计

### 4.1 Voice bank 总体要求

| 项目 | 规格 |
| --- | --- |
| 语言 | 普通话为主；保留 IPA / pinyin / English gloss |
| 说话人 | 12 名真实说话人，平衡性别、年龄印象、音域 |
| 内容 | 80 个短词/短句，覆盖 phoneme、pinyin、syllable、word、short phrase |
| 时长 | 0.8-2.5 s；超过 3 s 的 item 不进入主 EEG trials |
| 音质 | 安静录音室或等价条件，统一麦克风链路 |
| 响度 | loudness normalization；保留 raw wav 和 normalized wav |
| 格式 | WAV，48 kHz/24 bit 原始保存；分析可派生 16 kHz 或 24 kHz |
| 同步 | EEG 呈现时必须记录 audio loopback |

### 4.2 内容层级

| 层级 | 数量建议 | 目的 |
| --- | ---: | --- |
| Phoneme / CV / VC | 20-30 | q2-q3 phoneme / articulatory feature probe |
| Pinyin / syllable | 20-30 | Mandarin tone / syllable probe |
| Short word | 20-30 | word-level content retrieval |
| Short phrase | 20-30 | natural prosody、style、rhythm |

内容选择原则：

1. 避免强情绪或临床敏感词。
2. 避免过长、罕见、多义或强语义冲突短句。
3. 每个 content 需要能由至少 6 个说话人录制。
4. 至少 40 个 content 进入 same-content multi-speaker 设计。

### 4.3 参数操控

采用 fractional factorial，而不是全组合。全组合为 `12 speakers x 80 contents x 3 F0 x 3 formant x 6 style x 3 spatial`，不可执行，也会让 trial 数爆炸。主设计只保留单维操控和少量交互操控。

| 维度 | 水平 | 主用途 |
| --- | --- | --- |
| Speaker | 12 real speakers | speaker identity / timbre retrieval |
| F0 | -4, 0, +4 semitones | pitch/prosody code |
| Formant | 0.9, 1.0, 1.1 ratio | timbre / vocal tract proxy |
| Style | neutral, happy, angry, whisper, command, soft | style and emotional vocal quality |
| Spatial | center, left 60 deg, right 60 deg | externalization / spatial control |
| Noise/control | speech-shaped noise, scrambled voice, non-vocal sound, silence | acoustic shortcut control |

### 4.4 Core item sets

| Set | 组成 | 数量目标 | 用途 |
| --- | --- | ---: | --- |
| Core natural | neutral speech, 12 speakers, selected contents | 300-500 | Part A ratings 和主 EEG |
| Same-content speaker set | 同 content 多 speaker | 120-180 | speaker/timbre hard negatives |
| Same-speaker content set | 同 speaker 多 content | 120-180 | content hard negatives |
| F0-only set | 同 speaker/content/style，只变 F0 | 90-120 | pitch alignment |
| Formant-only set | 同 speaker/content/style，只变 formant | 90-120 | timbre/formant alignment |
| Style-only set | 同 speaker/content，只变 style | 120-160 | style alignment |
| Control set | noise/scrambled/non-vocal/silence | 80-120 | low-level and non-voice baseline |

### 4.5 Stimulus metadata

每条 voice item 必须有如下字段：

```text
stim_file
raw_file
normalized_file
content_id
transcript
pinyin
ipa
english_gloss
language
speaker_id
speaker_gender_recorded
speaker_age_bin_recorded
speaker_region
speaker_type
emotion_style
f0_shift_semitone
formant_shift_ratio
speaking_rate
intensity_db
loudness_lufs
spatial_azimuth
spatial_externalization
duration_sec
syllable_count
phoneme_sequence
word_count
recording_session_id
microphone_chain
quality_flag
```

### 4.6 Audio derivatives

每条音频生成 `AudioTokenBundle`：

```text
AudioTokenBundle
  waveform_id
  stim_file
  sample_rate
  frame_times
  content_units
  content_unit_model
  phoneme_labels
  pinyin_labels
  prosody_tokens
  f0_contour
  f0_mean
  f0_std
  energy_contour
  rhythm_features
  voiced_mask
  voice_tokens
  speaker_embedding
  timbre_embedding
  style_embedding
  mfcc_1..mfcc_20
  formant_f1_f2_f3
  spectral_centroid
  spectral_bandwidth
  spectral_flatness
  roughness
  breathiness
  codec_codes
  codec_group_names
  valid_mask
```

Audio token 分层原则：

| Token 类别 | 目标模型/特征 | EEG 对齐角色 |
| --- | --- | --- |
| Semantic/content | HuBERT, WavLM, Whisper, phoneme, pinyin | q2-q3 content |
| Prosody | F0, energy, rhythm, duration, voicing | q4 prosody |
| Voice/timbre | speaker embedding, formant, MFCC, WavLM speaker, FACodec timbre | q5-q6 voice |
| Codec | EnCodec, FACodec, X-Codec, SpeechTokenizer | V3 decoder 后端，不作为 V1 主监督 |

---

## 5. 实验流程

### 5.1 总体流程

| 部分 | 时长 | 内容 |
| --- | ---: | --- |
| Consent and screening | 10-15 min | 知情同意、听阈、语言背景、状态问卷 |
| Part A voice calibration | 35-45 min | subject-specific voice ratings |
| EEG setup | 25-45 min | cap、impedance、EOG/ECG/EMG、audio loopback |
| Part B controlled listening | 90 min 有效任务 | B1-B5 |
| Clinical AVH block | 20-30 min | 仅临床组，低负荷 resting/AVH episode marking |
| Debrief | 5-10 min | 不适检查、临床安全确认、数据共享确认 |

Healthy main 总时长预计 2.5-3.5 小时。Clinical AVH session 可以拆成两次，避免疲劳。

### 5.2 Part A: Subject-specific voice calibration

目的：

```text
voice item -> subject-level perceptual ratings -> subject-specific voice manifold
```

每名受试者评分 200 条 voice items。Pilot 可降低到 120-160 条，但 main 不应低于 200 条。

评分维度：

| 字段 | 范围 | 含义 |
| --- | --- | --- |
| rating_pitch | 0-100 | 主观音高 |
| rating_brightness | 0-100 | 明亮度 |
| rating_roughness | 0-100 | 粗糙/沙哑感 |
| rating_breathiness | 0-100 | 气声感 |
| rating_gender_impression | 0-100 | 主观性别印象，非真实性别标签 |
| rating_age_impression | 0-100 | 主观年龄印象 |
| rating_speaker_similarity | 0-100 | 与参照 voice prototype 的相似度 |
| rating_style_strength | 0-100 | 风格强度 |
| rating_familiarity | 0-100 | 熟悉感 |
| rating_externalization | 0-100 | 声音像来自外部空间的程度 |
| rating_confidence | 0-100 | 评分信心 |

Trial timing：

```text
fixation: 400-700 ms jitter
audio: 0.8-2.5 s
rating screen: max 4000 ms
ITI: 500-900 ms jitter
```

质量要求：

| 指标 | 门槛 |
| --- | --- |
| 重复 item rating ICC | >=0.60，低于门槛则 subject-level manifold 标低可信 |
| catch voice identity accuracy | >=70% |
| 平均 response missing rate | <=15% |
| 单维 rating 全为同一值 | 标记 low-engagement |

### 5.3 Part B: EEG controlled listening

总目标：同一协议内采集 content、prosody、timbre、speaker、style 和 retrieval 所需 EEG。

| Block | Trials | 主要操控 | 目标 token group |
| --- | ---: | --- | --- |
| B1 Content Listening | 160 | 同 speaker 变 content | q2-q3 content |
| B2 Timbre/Speaker Listening | 180 | 同 content 变 speaker/formant | q5-q6 voice |
| B3 Prosody/Style Listening | 180 | 同 content/speaker 变 F0/rhythm/style | q4 prosody, q5-q6 style |
| B4 Voice Retrieval | 120 | target voice + 4 candidates | q5-q6 retrieval |
| B5 Imagined Voice | 120 | cue 后静默 replay / voice imagery | transfer to AVH-like internal voice |

总 EEG trials 为 760。若疲劳风险过高，可拆成两次 session：

| Session | Blocks |
| --- | --- |
| Session 1 | B1, B2, B4 |
| Session 2 | B3, B5, retest subset |

#### 5.3.1 通用 EEG trial timing

```text
baseline fixation: 700-1100 ms jitter
audio or cue: 0.8-2.5 s
post-audio blank: 500 ms
response or rating: 1000-3000 ms
ITI: 900-1500 ms jitter
```

EEG epoch 初步定义：

```text
listening epoch: -0.5 to +3.5 s relative to audio onset
imagery epoch: -0.5 to +4.0 s relative to imagery cue onset
retrieval target epoch: -0.5 to +3.5 s
candidate epoch: per candidate onset, -0.2 to +1.5 s
```

#### 5.3.2 B1 Content Listening

设计：

```text
same speaker / neutral style / controlled loudness
content changes across phoneme, pinyin, word, phrase
```

任务：

1. 被试听 voice item。
2. 20%-25% trials 出现 catch question，例如是否听到目标音节/词。
3. 非 catch trials 只要求注视 fixation，减少运动。

主要标签：

```text
phoneme_sequence
pinyin
syllable_id
word_id
content_units
transcript
```

主要分析：

| 分析 | 指标 |
| --- | --- |
| content classification | accuracy, macro F1 |
| content unit prediction | CE/CTC, top-k unit accuracy |
| content retrieval | Recall@K, MRR |
| acoustic control | content model vs envelope-only |

#### 5.3.3 B2 Timbre/Speaker Listening

设计：

```text
same content
different speaker / formant / timbre
style fixed or counterbalanced
```

任务：

1. 单 item listening：偶发 same/different speaker judgment。
2. Pairwise trials：A/B 两个 voice item，判断是否同说话人或哪一个更明亮。

Hard negatives：

| Negative 类型 | 固定 | 变化 |
| --- | --- | --- |
| same-content different-speaker | content, style, F0 | speaker / timbre |
| same-speaker formant-only | speaker, content, F0, style | formant |
| matched-acoustic speaker | duration, intensity, content | speaker embedding |

主要分析：

| 分析 | 指标 |
| --- | --- |
| speaker retrieval | Top-1, Top-5, MRR |
| timbre regression | Pearson, Spearman, MAE |
| formant decoding | MAE, bin accuracy |
| subject-level voice manifold retrieval | distance to rated closest voice |

#### 5.3.4 B3 Prosody/Style Listening

设计：

```text
same content / same speaker
F0, rhythm, intensity, style manipulated independently
```

任务：

1. A/B forced choice：higher pitch、stronger emotion、faster rhythm。
2. 部分 trials 做 0-100 style strength rating。

主要分析：

| 分析 | 指标 |
| --- | --- |
| F0 contour prediction | temporal correlation, MAE |
| F0 bin classification | balanced accuracy |
| energy/rhythm tracking | Pearson, spectral coherence |
| style classification | macro F1, balanced accuracy |

#### 5.3.5 B4 Voice Retrieval

Trial format：

```text
target audio
delay: 800-1200 ms
candidate audio 1
candidate audio 2
candidate audio 3
candidate audio 4
response: choose closest voice / same speaker / same style
```

Candidate construction：

| Candidate | 设计 |
| --- | --- |
| correct | same voice item or nearest same voice prototype |
| hard negative 1 | same content, different speaker |
| hard negative 2 | same speaker, different content |
| hard negative 3 | same speaker/content, F0-only or formant-only |
| hard negative 4 | style-only or mixed foil |

模型训练时不只使用行为选择，也使用完整 candidate set 作为 retrieval targets。

指标：

```text
Top-1
Top-5
MRR
subject-level bootstrap CI
candidate-level mixed-effects model
```

#### 5.3.6 B5 Imagined Voice

目的：建立外部听觉和内部 voice imagery 的桥接，但不替代 AVH。

Trial format：

```text
voice cue: 0.8-2.5 s
blank delay: 500 ms
silent replay / voice imagery: 2.0-3.0 s
rating: vividness, confidence, closest voice
ITI: 1000-1800 ms
```

Control conditions：

| 条件 | 用途 |
| --- | --- |
| auditory cue + imagery | 目标条件 |
| text cue + imagery | 减少外部 acoustic carry-over |
| visualized text | control for semantic imagery |
| silence/fixation | baseline |
| motor suppression check | 结合 jaw/neck EMG 排除 subvocal artifact |

主要报告：

1. imagery token 是否接近同一 subject 听觉 voice token。
2. imagery token 是否优于 text-only 和 silence baseline。
3. imagery 仅作为 AVH-like internal voice 的工程桥接。

### 5.4 Clinical AVH block

Clinical block 不做诱发。目标是捕获自然 AVH episode，并把 episode-level EEG 与患者事后主观 voice prototype 绑定。

流程：

```text
resting / low-load fixation: 20-30 min
patient button press: AVH onset
patient button release or second press: AVH offset
post-episode rating screen
voice bank Top-K matching
pitch/formant/style sliders
distress and confidence rating
```

Post-episode 字段：

| 字段 | 范围 | 含义 |
| --- | --- | --- |
| avh_vividness | 0-100 | 幻听清晰度 |
| avh_externality | 0-100 | 外部来源感 |
| avh_distress | 0-100 | 痛苦程度 |
| avh_controllability | 0-100 | 可控性 |
| avh_semantic_clarity | 0-100 | 内容清晰度 |
| avh_voice_confidence | 0-100 | 对声音匹配的信心 |
| avh_top1_voice_id | categorical | 最接近 voice |
| avh_top5_voice_ids | list | Top-5 voice prototypes |
| avh_pitch_adjust | numeric | 患者调节的 F0 offset |
| avh_formant_adjust | numeric | 患者调节的 formant ratio |
| avh_style_adjust | categorical/numeric | style 或 style strength |

AVH episode inclusion for primary analysis：

| 指标 | 门槛 |
| --- | --- |
| episode duration | >=1.5 s and <=20 s |
| confidence | >=50/100 for primary, all episodes retained for secondary |
| EEG artifact | episode window usable after artifact rejection |
| EMG contamination | jaw/neck EMG not dominating episode window |
| distress | if high distress, stop session and mark safety termination |

Clinical negative windows：

1. Pre-AVH baseline: episode 前 10-20 s 内无 AVH 标记片段。
2. Post-AVH baseline: episode 后恢复期且无 AVH。
3. Matched resting windows: 同 session 同长度随机窗口。
4. Imagery windows: 患者主动想象 voice cue，和自发 AVH 分开建模。

---

## 6. EEG 采集规格

### 6.1 硬件

| 模块 | 最低要求 | 推荐 |
| --- | --- | --- |
| EEG channels | 64 | 128 |
| Sampling rate | 500 Hz | 1000 Hz |
| Reference | Cz or mastoid online | record exact reference; offline average + mastoid |
| Ground | 按设备规范 | 记录在 sidecar JSON |
| EOG | vertical + horizontal | required |
| ECG | 1 channel | required |
| EMG | jaw/neck 2-4 channels | required for imagery/AVH |
| Trigger | TTL | required |
| Audio loopback | stereo or mono line-in | required |
| Audio output | insert or closed-back headphones | calibrated SPL |

### 6.2 同步要求

| 指标 | 门槛 |
| --- | --- |
| TTL trigger present | 100% trials |
| audio loopback present | 100% auditory trials |
| corrected onset error | target <5 ms |
| low-quality onset error | >10 ms run flagged |
| missing trigger | trial excluded unless recoverable from loopback |

### 6.3 声压与听阈

1. 每名受试者做简化纯音听阈筛查或等价听力 screening。
2. 主刺激呈现为个体舒适音量，目标 60-70 dB SPL。
3. 左右耳差异记录入 `phenotype/hearing_screening.tsv`。
4. 对 clinical AVH，不因声音任务诱发 distress；若 auditory stimulation 增加症状，立即降低负荷或停止。

### 6.4 Impedance and quality

| 指标 | 目标 | 最高容忍 |
| --- | --- | --- |
| EEG impedance | <10 kOhm | <20 kOhm |
| EOG/ECG/EMG impedance | <20 kOhm | device-dependent |
| bad channels | <10% | 15% |
| line noise | notch 后无强残留 | run flagged |
| movement artifact | block-level inspection | bad block excluded |

---

## 7. BIDS 数据结构

### 7.1 目录结构

```text
VoiceImageEEG/
  dataset_description.json
  README.md
  participants.tsv
  participants.json
  phenotype/
    hearing_screening.tsv
    language_background.tsv
    state_questionnaire.tsv
    clinical_scales.tsv
    voice_rating_summary.tsv
  stimuli/
    voice_bank/
      raw/
      normalized/
      control/
    voice_bank_metadata.tsv
    voice_bank_features.tsv
  sub-001/
    ses-01/
      eeg/
        sub-001_ses-01_task-voiceimage_eeg.vhdr
        sub-001_ses-01_task-voiceimage_eeg.eeg
        sub-001_ses-01_task-voiceimage_eeg.vmrk
        sub-001_ses-01_task-voiceimage_events.tsv
        sub-001_ses-01_task-voiceimage_channels.tsv
        sub-001_ses-01_task-voiceimage_electrodes.tsv
        sub-001_ses-01_task-voiceimage_coordsystem.json
      beh/
        sub-001_ses-01_task-voicebankratings_beh.tsv
        sub-001_ses-01_task-voiceimage_beh.tsv
  derivatives/
    audio_features/
    audio_tokens/
    eeg_preproc/
    eeg_epochs/
    eeg_tokens/
    voice_embeddings/
    clinical_ratings/
    model_inputs/
```

### 7.2 participants.tsv

```text
participant_id
group
age
sex
handedness
native_language
mandarin_proficiency
english_proficiency
music_training_years
speech_training_years
hearing_left_threshold_db
hearing_right_threshold_db
clinical_diagnosis
current_avh
medication_antipsychotic
medication_benzodiazepine
illness_duration_years
panss_positive
panss_negative
panss_general
psyrats_ah_total
avhrs_q_total
sleep_hours
caffeine_today
nicotine_today
session_notes
```

### 7.3 events.tsv required fields

```text
onset
duration
trial_type
block_id
trial_index
stim_file
content_id
transcript
pinyin
ipa
speaker_id
voice_id
target_voice_id
candidate_voice_ids
correct_candidate_id
f0_shift_semitone
formant_shift_ratio
emotion_style
spatial_azimuth
spatial_externalization
response
response_time
catch_trial
catch_correct
rating_pitch
rating_brightness
rating_roughness
rating_breathiness
rating_gender_impression
rating_age_impression
rating_speaker_similarity
rating_style_strength
rating_familiarity
rating_externalization
rating_confidence
imagery_vividness
imagery_confidence
avh_onset
avh_offset
avh_duration
avh_vividness
avh_externality
avh_distress
avh_controllability
avh_semantic_clarity
avh_voice_confidence
avh_top1_voice_id
avh_top5_voice_ids
avh_pitch_adjust
avh_formant_adjust
avh_style_adjust
audio_loopback_delay_ms
trigger_id
quality_flag
```

### 7.4 channels.tsv additions

```text
name
type
units
low_cutoff
high_cutoff
reference
sampling_frequency
status
status_description
sensor_type
is_auxiliary
```

Auxiliary channels should use explicit types such as `EOG`, `ECG`, `EMG`, `TRIG`, `AUDIO`.

### 7.5 Controlled access split

| 数据 | 访问 |
| --- | --- |
| Healthy anonymous EEG/audio/ratings | 可公开或申请公开 |
| Clinical EEG with de-identified events | controlled access |
| Clinical free text | controlled access only |
| Patient voice matching and symptom ratings | controlled access only |
| Any potentially identifying audio description | controlled access only |

---

## 8. 预处理与质量控制

### 8.1 EEG preprocessing

推荐 pipeline：

```text
raw EEG
-> verify TTL and audio loopback
-> correct audio onset using loopback
-> mark bad channels
-> notch 50 Hz and harmonics if needed
-> bandpass 0.1-40 Hz for ERP / decoding baseline
-> optional high-gamma not used for scalp primary claim
-> re-reference average and mastoid variants
-> ICA / SSP for EOG and ECG
-> EMG-informed artifact annotation
-> resample to 250 Hz for model training
-> epoch by audio onset / imagery cue / AVH episode
-> save continuous cleaned and epoched derivatives
```

保留原始 500/1000 Hz 文件，不得只保存降采样结果。

### 8.2 Artifact policy

| Artifact | 处理 |
| --- | --- |
| Eye blink | ICA/SSP component removal, report components |
| Saccade | reject or regress depending on severity |
| Jaw/neck EMG | mark and report; imagery/AVH primary analysis excludes high EMG windows |
| ECG | ICA/SSP or regression |
| Bad channel | interpolate after marking |
| Bad block | exclude if sustained movement/noise |

### 8.3 Quality metrics

```text
valid_epoch_ratio
bad_channel_ratio
mean_impedance
audio_loopback_delay_mean_ms
audio_loopback_delay_sd_ms
catch_accuracy
rating_missing_rate
voice_rating_icc
emg_contamination_score
q7_subject_predictability
q7_device_predictability
q7_clinical_predictability
```

最低进入 primary analysis：

| 指标 | 门槛 |
| --- | --- |
| valid_epoch_ratio | >=70% |
| catch_accuracy | >=70% |
| voice_rating_icc | >=0.60 for subject-level manifold primary |
| bad_channel_ratio | <=15% |
| audio_loopback_delay corrected | mean error <5 ms target; >10 ms run flagged |
| AVH confidence | >=50/100 for clinical primary |

---

## 9. 模型接口

### 9.1 EEG token grouping

与现有 `EEGVoiceTokenV1` 保持一致：

| Quantizer | Group | 对齐目标 | 禁止用途 |
| --- | --- | --- | --- |
| q0-q1 | base | onset, envelope, shared auditory response | 不单独声称 content 或 voice |
| q2-q3 | content | phoneme, pinyin, syllable, word, HuBERT/Whisper units | 不读 speaker/style labels |
| q4 | prosody | F0, energy, rhythm, intonation, voicing | 不承担 speaker identity |
| q5-q6 | voice | timbre, speaker, style, voice identity, subject voice manifold | 不直接吸收 content shortcut |
| q7 | residual | weak EEG reconstruction residual | 不进入 alignment, retrieval, clinical heads |

### 9.2 Training targets

```text
EEG token -> AudioTokenBundle
```

Losses：

| Loss | 输入 groups | target | 用途 |
| --- | --- | --- | --- |
| reconstruction aligned | q0-q6 | EEG | token sanity |
| reconstruction full | q0-q7 | EEG | weak residual only |
| content CE/CTC | q0-q3 | phoneme/pinyin/content units | content alignment |
| prosody regression | q0-q1 + q4 | F0/energy/rhythm | prosody alignment |
| timbre regression | q0-q1 + q5-q6 | MFCC/formant/timbre embedding | timbre alignment |
| speaker/style CE | q0-q1 + q5-q6 | speaker/style labels | voice identity |
| InfoNCE retrieval | q0-q1 + q5-q6 | voice embeddings/candidates | voice-image retrieval |
| AVH prototype retrieval | q0-q1 + q5-q6 | patient-selected voice prototype | clinical endpoint |

### 9.3 Training phases

| Phase | 数据 | 目的 |
| --- | --- | --- |
| P0 public pretraining | selected public datasets | auditory/speech tokenizer initialization |
| P1 healthy listening | self-collected B1-B4 | main token alignment and voice retrieval |
| P2 healthy imagery | B5 | heard-to-imagined transfer |
| P3 subject adaptation | subject calibration + selected trials | subject-specific voice manifold |
| P4 clinical validation | AVH episodes | held-out AVH voice-image retrieval |

### 9.4 Split policy

必须同时报告：

| Split | 目的 |
| --- | --- |
| held-out trial | sanity check only |
| held-out content | 检验不是记住句子 |
| held-out speaker | 检验 speaker/timbre generalization |
| held-out style | 检验 style transfer |
| held-out session | test-retest generalization |
| leave-one-subject-out | cross-subject generalization |
| clinical held-out episode | AVH validation |

Subject ID、session ID、clinical label 不得作为 retrieval target 或 shortcut。若 q7 或任何 token group 可强预测 subject/device/clinical label，必须报告为 nuisance leakage。

---

## 10. 主要结果指标

### 10.1 Primary endpoints

| Endpoint | 数据 | 指标 | 成功标准 |
| --- | --- | --- | --- |
| Voice retrieval | Healthy B4 | Top-1, Top-5, MRR | subject-level 显著高于 chance |
| Held-out speaker retrieval | Healthy B2/B4 | Top-K, MRR | held-out speaker 下仍高于 baseline |
| Prosody prediction | Healthy B3 | Pearson, Spearman, MAE | 高于 envelope-only 和 shuffled baseline |
| Timbre/formant alignment | Healthy B2 | MAE, embedding retrieval | same-content hard negatives 下成立 |
| Style classification | Healthy B3 | balanced accuracy, macro F1 | 高于 chance 和 content-only |
| AVH voice prototype retrieval | Clinical AVH | Top-K, rank distance, manifold distance | AVH window 优于 matched resting windows |

### 10.2 Secondary endpoints

| Endpoint | 指标 |
| --- | --- |
| EEG reconstruction | L1, PCC, frequency amplitude error |
| Codebook usage | perplexity, usage entropy, dead code ratio |
| Imagery transfer | heard-imagery embedding similarity |
| Subject manifold stability | rating ICC, embedding Procrustes similarity |
| Device robustness | seen-device vs held-out-device, if applicable |
| q7 nuisance | q7 predictability for subject/device/session/clinical label |

### 10.3 Baselines

| Baseline | 目的 |
| --- | --- |
| random candidate | chance floor |
| acoustic envelope only | 低层声学 tracking |
| content-only HuBERT/Whisper | 证明不是只解码内容 |
| speaker-only target | 证明不是只用 speaker ID |
| no-RVQ continuous latent | 检验 discrete token 增益 |
| shuffled EEG/audio alignment | 排除时间错配 shortcut |
| no-device-context | 检验 acquisition context |
| q7-in-head ablation | 证明 residual shortcut 污染结论 |
| EMG-only decoder | 排除 subvocal muscle artifact, especially imagery/AVH |

---

## 11. 统计方案

### 11.1 Preregistration

预注册必须写明：

1. Primary endpoints。
2. Exclusion criteria。
3. Split policy。
4. AVH episode inclusion rule。
5. Clinical safety stop rule。
6. Baselines and ablations。
7. Multiple-comparison correction。
8. 何种情况下降级论文主张。

### 11.2 EEG inference

| 分析 | 统计 |
| --- | --- |
| ERP / time-domain | cluster-based permutation correction |
| time-frequency | permutation cluster correction across time-frequency-electrode |
| channel/source exploratory maps | FDR or FWE; clearly label exploratory |
| decoding metrics | subject-level bootstrap confidence intervals |
| repeated measures | mixed-effects model |

### 11.3 Retrieval inference

推荐 mixed-effects formulation：

```text
retrieval_success ~ condition + token_group + hard_negative_type + (1|subject) + (1|speaker) + (1|content)
```

Clinical AVH：

```text
prototype_rank_or_distance ~ window_type + confidence + psyrats_ah_total + medication_covariates
                            + (1|subject) + (1|voice_prototype)
```

其中 `window_type` 包括 AVH episode、pre-AVH baseline、post-AVH baseline、matched resting、imagery。

### 11.4 Multiple comparisons

| 场景 | 方法 |
| --- | --- |
| EEG sensor/time/frequency maps | cluster-level permutation/FWE |
| 多个 voice attributes | Benjamini-Hochberg FDR |
| 多个 retrieval splits | Holm or FDR |
| Exploratory clinical correlations | FDR and clearly marked exploratory |

### 11.5 Effect sizes

必须报告：

```text
Cohen's d or paired effect size
odds ratio for binary success
rank-biserial or Cliff's delta for rank metrics
Pearson/Spearman r with CI
bootstrap CI for Top-K/MRR
Bayesian posterior interval if Bayesian model used
```

### 11.6 Reverse inference control

允许表述：

```text
The model retrieves the patient-confirmed nearest voice prototype in an individualized voice manifold.
```

禁止表述：

```text
The model objectively reconstructs the actual hallucinated waveform.
```

原因：AVH 没有外部声波 ground truth，患者事后 voice matching 是主观但可量化的 target。

---

## 12. 临床与伦理要求

### 12.1 Mental privacy

非侵入式 neural decoding 研究必须明确 mental privacy 边界：

1. 模型训练需要被试合作和个体校准。
2. 不声明可在未合作情况下读取任意想法。
3. 临床 AVH 只解码患者主动标记并事后确认的 episode。
4. 自由文本和临床评分受 controlled access 保护。

### 12.2 AVH risk management

| 风险 | 控制 |
| --- | --- |
| 症状加重 | 低负荷任务，不诱发；实时 distress monitoring |
| 隐私泄露 | controlled access, de-identification |
| 误解为读心 | consent 中明确模型限制 |
| 声音材料触发不适 | 避免威胁性/侮辱性内容；患者可跳过 |
| 疲劳 | clinical session 可拆分 |

### 12.3 Data sharing

| 数据类型 | 建议 |
| --- | --- |
| healthy EEG and stimulus metadata | OpenNeuro/BIDS after de-identification |
| healthy voice ratings | de-identified public |
| clinical EEG | controlled access |
| clinical ratings | controlled access |
| raw patient reports/free text | controlled access or not released |
| code and derived non-identifying schemas | public |

---

## 13. 论文结构建议

### 13.1 第一篇主论文

Title:

```text
Discrete EEG Voice Tokens Align with Speech Content, Prosody and Timbre
for Subject-Specific Voice-Image Reconstruction
```

核心图：

1. Study design and voice bank factorization。
2. EEGVoiceToken grouped RVQ architecture。
3. Content/prosody/timbre/speaker/style decoding。
4. Voice retrieval with hard negatives。
5. Imagery transfer and subject-specific voice manifold。
6. Ablations: content-only, envelope-only, no-RVQ, q7-in-head。

主结论：

```text
Healthy controlled listening and imagery data support an EEG token interface
that aligns with both linguistic content and non-linguistic voice-image attributes.
```

### 13.2 第二篇临床扩展

Title:

```text
Subject-Specific EEG Voice Tokens Retrieve the Perceptual Timbre
of Auditory Verbal Hallucinations
```

核心图：

1. Clinical AVH capture protocol。
2. Patient-specific voice manifold and AVH matching。
3. AVH episode vs resting baseline retrieval。
4. Relation to PSYRATS-AH / AVHRS-Q / confidence。
5. Ethical and mental privacy boundary。

主结论：

```text
Patient-marked AVH episodes can be mapped to patient-confirmed voice prototypes
within an individualized voice manifold, without claiming objective waveform recovery.
```

### 13.3 降级策略

| 情况 | 论文主张 |
| --- | --- |
| 无 clinical recruitment | healthy voice-image token alignment |
| clinical episodes too few | case-series / feasibility only |
| timbre retrieval 不稳定 | content/prosody/speaker retrieval, timbre exploratory |
| q7 leakage 强 | 报告 nuisance absorption，不能声称 clean disentanglement |
| imagery EMG contamination | imagery 降级为 exploratory |

---

## 14. 最小可执行版本

若资源有限，最小版本不应删掉 voice-image 核心：

| 模块 | 最小值 |
| --- | --- |
| healthy pilot | 16 人 |
| voice bank | 12 speakers, 40 contents, 240-360 items |
| ratings | 每人 160-200 items |
| EEG blocks | B1, B2, B3, B4 必须保留；B5 可缩短 |
| trials | 每人 >=500 usable auditory trials |
| controls | speech-shaped noise, scrambled voice, silence |
| model | q0-q7 grouped RVQ + content/prosody/voice heads |
| metrics | Top-K/MRR, F0 correlation, timbre retrieval, q7 nuisance |

不可删除：

1. Same-content different-speaker hard negatives。
2. Same-speaker different-content hard negatives。
3. Subject-level voice ratings。
4. Audio loopback。
5. q7 不进入 alignment/retrieval heads。

---

## 15. 采集前检查清单

### 15.1 Stimulus checklist

- [ ] 12 speakers 完成录音。
- [ ] 80 contents 完成 transcript / pinyin / IPA / metadata。
- [ ] 所有 raw wav 和 normalized wav 保存。
- [ ] F0/formant/style/spatial/control stimuli 生成并质检。
- [ ] AudioTokenBundle extraction pipeline 跑通。
- [ ] Voice bank metadata 无缺失主键。

### 15.2 EEG checklist

- [ ] EEG cap、EOG、ECG、jaw/neck EMG 可同步采集。
- [ ] TTL trigger 测试通过。
- [ ] Audio loopback 测试通过。
- [ ] SPL 校准流程固定。
- [ ] Pilot 中 corrected onset error <5 ms。
- [ ] BIDS writer 输出通过 validator。

### 15.3 Clinical checklist

- [ ] IRB approved clinical protocol。
- [ ] Psychiatrist or trained clinical staff available。
- [ ] Stop rule written in consent and task script。
- [ ] PANSS、PSYRATS-AH、AVHRS-Q 表格准备。
- [ ] Controlled access data policy written。
- [ ] Post-episode distress handling protocol ready。

### 15.4 Analysis checklist

- [ ] Preprocessing scripts versioned。
- [ ] Primary endpoints preregistered。
- [ ] Split policy frozen before model tuning。
- [ ] Baselines implemented。
- [ ] q7 nuisance metrics implemented。
- [ ] Clinical AVH primary and secondary analyses separated。

---

## 16. References

- Anumanchipalli, G. K., Chartier, J., & Chang, E. F. (2019). Speech synthesis from neural decoding of spoken sentences. *Nature*, 568, 493-498. https://www.nature.com/articles/s41586-019-1119-1
- Barber, L., Reniers, R., & Upthegrove, R. (2021). A review of functional and structural neuroimaging studies to investigate the inner speech model of auditory verbal hallucinations in schizophrenia. *Translational Psychiatry*, 11, 582. https://www.nature.com/articles/s41398-021-01670-7
- Belin, P., Zatorre, R. J., Lafaille, P., Ahad, P., & Pike, B. (2000). Voice-selective areas in human auditory cortex. *Nature*, 403, 309-312. https://www.nature.com/articles/35002078
- Chen, X., Wang, R., Khalilian-Gourtani, A., et al. (2024). A neural speech decoding framework leveraging deep learning and speech synthesis. *Nature Machine Intelligence*, 6, 467-480. https://www.nature.com/articles/s42256-024-00824-8
- Défossez, A., Caucheteux, C., Rapin, J., et al. (2023). Decoding speech perception from non-invasive brain recordings. *Nature Machine Intelligence*, 5, 1097-1107. https://www.nature.com/articles/s42256-023-00714-5
- Ding, N., Melloni, L., Zhang, H., Tian, X., & Poeppel, D. (2016). Cortical tracking of hierarchical linguistic structures in connected speech. *Nature Neuroscience*, 19, 158-164. https://www.nature.com/articles/nn.4186
- Magioncalda, P., Yadav, A., & Martino, M. (2025). An umbrella review of neuroimaging studies and conceptual framework linking pathophysiology and psychopathology in schizophrenia. *Nature Mental Health*, 3, 1241-1255. https://www.nature.com/articles/s44220-025-00493-5
- Metzger, S. L., Littlejohn, K. T., Silva, A. B., et al. (2023). A high-performance neuroprosthesis for speech decoding and avatar control. *Nature*, 620, 1037-1046. https://www.nature.com/articles/s41586-023-06443-4
- Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., et al. (2019). EEG-BIDS, an extension to the brain imaging data structure for electroencephalography. *Scientific Data*, 6, 103. https://www.nature.com/articles/s41597-019-0104-8
- Poeppel, D., & Assaneo, M. F. (2020). Speech rhythms and their neural foundations. *Nature Reviews Neuroscience*, 21, 322-334. https://www.nature.com/articles/s41583-020-0304-4
- Powers, A. R., Mathys, C., & Corlett, P. R. (2017). Pavlovian conditioning-induced hallucinations result from overweighting of perceptual priors. *Science*, 357, 596-600. https://doi.org/10.1126/science.aan3458

