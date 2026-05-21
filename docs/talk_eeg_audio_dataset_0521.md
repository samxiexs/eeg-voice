# EEG-Audio Dataset 汇报讲稿（0521）

来源：

- Notion: https://www.notion.so/36625594c17b80ffaf86eb9f19333d3e
- 本地补充：`docs/selected_voice_eeg_datasets_detailed_0519.md`
- 本地补充：`docs/multi_dataset_voice_eeg_catalog_0518.md`

建议时长：12-15 分钟

---

## 开场

大家好，我今天汇报的是 EEG-Audio Dataset 这部分工作。这个汇报的核心不是简单列数据集，也不是说“我们找到了很多 EEG 数据”。我真正想回答的问题是：现有公开 EEG-audio / EEG-speech 数据，能不能共同支撑一个稳定的研究链路：

```text
EEG -> discrete token
    -> content / pitch / timbre / speaker / style alignment
    -> voice / speaker retrieval / voice-image foundation
```

也就是说，我们要判断这些数据是否足够支持 EEG voice token foundation，而不是直接支持最终的个体化声音形象重构。这个边界非常重要，因为公开数据可以支撑 tokenizer、attribute alignment 和 retrieval，但还不能直接支撑每个受试者主观听到的 voice image reconstruction。

---

## 1. 研究口径：为什么不是单个大数据集问题

首先我想强调，这个数据池不能用“哪个数据集最大”来判断。它本质上是一个多范式拼图。

自然语音数据主要提供连续听觉追踪，告诉我们 EEG 如何跟随 speech envelope、word timing 和 phoneme timing。AAD，也就是 auditory attention decoding 数据，主要提供目标说话人或目标声源的选择信息。受控 phoneme、pinyin 和 tone 数据提供更干净的语音属性标签。imagined、inner 和 overt speech 数据则把研究从外部听觉推进到发声准备和内隐言语。

所以我的判断标准是：这些范式加在一起，能不能覆盖 EEG voice token 所需要的几个轴：content、pitch/prosody、timbre/speaker、style、mode，以及 retrieval。

这里有一个数据清洗原则：设备参数只采用 BIDS sidecar、README、channels.tsv、数据页和原论文中能够互相印证的信息。没有被 metadata 或论文支持的采集细节，一律标成待确认，避免把推断写成事实。

---

## 2. 数据池总览：37 个 selected datasets

目前我们把 37 个 selected EEG-voice / EEG-audio / speech-proxy 数据集放入研究池。这里的 selected 不是说本地都完整下载了，而是说它们已经满足公开可获取、可申请或存在可追踪访问路径。

这 37 个数据集可以压缩成四个层次。

第一类是 English natural / decoding，包括 `ds004408`、Weissbart、`ds006434`、`ds007630`、`ds007602`、Etard、`ds007591`、Kara One 和 SparrKULee。这一类支撑英文自然语音、speech perception、production 和 imagined/overt speech。

第二类是 Mandarin / Cantonese speech，包括 `ds005345`、ESAA、NJU AAD、AASD、MS-AASD、Yan 系列、ASA、`ds004718` 和 tone/syllable ERP。这一类支撑中文多说话人、声调、F0/intensity、spatial stream 和 speaker-stream retrieval。

第三类是 controlled speech / inner speech，包括 `ds006104`、`ds006465`、Chisco、Inner Speech、FEIS、UGR-MINDVOICE、CIRE 和 `ds004306`。这类数据提供 phoneme class、pinyin unit、mode label、emotion/intention 和 semantic category。

第四类是 AAD expansion / music proxy，包括 KUL、DTU、255ch AAD、Fuglsang、Rotaru、Geirnaert、OpenMIIR、MUSIN-G 和 MAD-EEG。这类数据用于 retrieval 泛化、设备泛化、音乐 pitch/timbre proxy 和非语音 auditory boundary。

几个关键数字是：

- selected 数据集总数是 37 个。
- 优先级分布是 P0 13 个、P1 10 个、P2 11 个、P3 3 个。
- 本地已有完整可播放音频样例的是 22 / 37 个。
- 设备跨度很大：通道数从 14 到 255，采样率从 128 Hz 到 8192 Hz，还有 scalp、ear、in-ear 等不同传感器形态。

这直接推导出模型设计要求：不能假设固定 montage，必须显式处理 channel mask、sensor embedding、sampling rate、device context 和 montage normalization。

---

## 3. 英文自然语音与 speech decoding 数据

英文数据是第一版 tokenizer 和 content alignment 的核心入口。

`ds004408` 是最干净的英文自然语音 EEG 数据之一。被试听英文有声书，每段音频有 `.wav` 和 TextGrid，TextGrid 提供 word 和 phoneme timing。设备是 BioSemi ActiveTwo，128 通道，512 Hz。它适合做 speech envelope tracking、phoneme/word onset alignment、match-mismatch retrieval 和 tokenizer 预训练。

`ds006434` 是 natural speech selective attention / ABR 数据。它的价值在于把早期 auditory response 和 attention-modulated cortical response 区分开。对我们的模型来说，这提示 q0-q1 的 base auditory code 不应该和 q5-q6 的 attention/voice retrieval code 混在一起。

`ds007630` 和 `ds007602` 共同提供 perception / production 方向的英文扩展。`ds007630` 体量很大，包含 listening 和 speechopen；`ds007602` 更偏 overt speech production，并且有 MIC-channel 可用于 production probe。但它们也有访问和音频质量限制，所以不能作为第一版唯一核心。

Kara One 则是 imagined/overt phonological probe。它有 imagined speech 和 vocalized prompt，并同步 Kinect audio/face。它不能证明 EEG 可直接生成 speech waveform，但能证明 EEG 中存在可解码的 phonological category。

这一组英文数据的作用可以概括为：建立 base auditory code 和 content code，并给 speaking mode 提供英文侧桥接。

---

## 4. 普通话与粤语数据：speaker、tone 和 prosody

中文数据主要补足英文数据缺少的几个维度：多说话人、声调、中文 pinyin/syllable 和 Mandarin AAD。

`ds005345` 是 Le Petit Prince multi-talker 数据，包含 single male、single female 和 mixed speech。它同时有 EEG、annotation 和多个 wav 样例，因此非常适合 speaker-stream retrieval 和 timbre/content disentanglement。

ESAA、NJU AAD、AASD、MS-AASD 和 Yan 系列提供普通话 competing speech / attention decoding。这里最重要的是，它们不是只给我们“听到语音”的 EEG，而是给了 attended stream、speaker stream、switch label 和 spatial stream。对于 voice / speaker retrieval，这些数据天然适合做 contrastive learning。

`ds004718` 是粤语自然语音数据，包含 F0、intensity、word timing 等 annotation。它对 q4 prosody code 非常关键，因为粤语 tone 和自然语音 prosody 能提供英文数据不具备的 pitch/tone 监督。

Cantonese tone/syllable ERP 则是更受控的 pitch/tone probe。虽然它不是完整语音重构数据，但可以帮助我们验证 token 是否保留 tone-level neural response。

所以中文和粤语数据的核心贡献是：把模型从 English content decoding 推向 speaker-stream retrieval、tone/prosody alignment 和跨语言 robustness。

---

## 5. Controlled speech、inner speech 和 imagined speech

第三类数据提供的是属性解释和 speaking mode 的关键证据。

`ds006104` 是受控 speech decoding 数据，包含 phoneme、CV/VC、real words 和 pseudowords，并且有高采样率 EEG 和音频刺激。它是 content、phoneme、coarticulation 和 style/emotion prompt 的核心 probe。这里要注意，细粒度 phoneme/word decoding 本身很难，因此我们不能用单一高准确率分类作为成功标准，而要看粗粒度 articulatory feature、content embedding 和 token alignment。

`ds006465` 提供普通话 pinyin 的 speak、mouthed 和 imagined 条件。Chisco 提供中文 sentence-level imagined speech。Inner Speech 提供西语 inner、pronounced 和 visualized command 条件。UGR-MINDVOICE 则同时有 overt 和 covert speech production，并有 audio。

这组数据的作用不是训练最终 audio decoder，而是让模型学习 heard、imagined、inner、overt、mouthed 这些 speaking mode 的边界。它们支撑 mode head，也帮助我们理解 EEG token 是否只对外部声波有效，还是能在内隐言语或想象语音中保持部分结构。

CIRE 也很重要，因为它提供普通话 prosodic emotion 和 speech intention，直接服务 q4 prosody code 与 style/intention head。

---

## 6. AAD(Auditory Attention Decoding) expansion 与 music proxy

第四类数据是泛化和边界条件。

KUL 和 DTU 是经典 AAD benchmark。它们提供荷兰语/丹麦语 competing speech 和 room/reverberation conditions。255ch AAD 提供高密度 EEG，可用于 sensor-density ablation。

Fuglsang 提供更大样本和听障群体，Rotaru 明确指出 gaze/spatial shortcut 的风险，Geirnaert 同步记录 scalp、around-ear 和 in-ear EEG。这些数据告诉我们，retrieval evaluation 最大的风险不是模型不够强，而是模型可能利用 trial identity、spatial cue、eye gaze 或 device shortcut。

OpenMIIR、MUSIN-G 和 MAD-EEG 属于音乐和非语音 auditory proxy。它们不支撑 speech 主结论，但能测试 tokenizer 是否只学到 speech-specific pattern，还是能捕获更一般的 auditory dynamics，比如 pitch、beat、timbre 和 target-source attention。

这一类数据的结论是：我们必须把 retrieval evaluation 设计得非常严格，包括 leave-subject-out、leave-speaker/story-out、leave-dataset-out、gaze/spatial negative control 和 cross-device evaluation。

---

## 7. 对 V1 模型设计的直接含义

从数据池出发，模型设计自然落到 grouped EEG token。

q0-q1 应该承担 base auditory response，比如 onset、envelope 和共享听觉动态。q2-q3 对应 content，比如 phoneme、syllable、word 和 HuBERT-like content unit。q4 对应 prosody，包括 F0、energy、rhythm、tone 和 intonation。q5-q6 对应 voice，包括 speaker、timbre、style 和 attended stream identity。q7 只应该作为 residual nuisance，用于弱重构，不能进入 alignment 和 retrieval head。

这个分组不是拍脑袋来的，而是由数据结构决定的。自然语音和 ABR 数据支撑 base auditory；phoneme/pinyin 数据支撑 content；粤语 tone、CIRE 和音乐 proxy 支撑 prosody；AAD 和 multi-talker 数据支撑 speaker/voice retrieval；inner/imagined/overt 数据支撑 mode head。

同时，跨设备差异非常大，因此 acquisition device context 不是可选项。模型要知道采集设备、montage、reference、sampling rate 和 channel count，否则跨数据集训练很容易把设备差异误当成声音差异。

---

## 8. 当前边界：公开数据能做什么，不能做什么

这部分是汇报里最重要的边界。

公开数据已经足够支撑 foundation + retrieval。也就是说，它可以帮助我们训练 EEG tokenizer，做 content / pitch / timbre / speaker / style alignment，并做 voice / speaker retrieval。

但公开数据还不够支撑 personalized subjective voice image reconstruction。原因是它缺少统一 voice bank、同一受试者对同一批声音的主观相似度评分，以及系统化 F0/formant/style 操控。

所以第一版主张应该是：

```text
在跨数据集 EEG-audio 研究池上学习可复用的 speech/voice EEG discrete token，
并检验这些 token 是否携带 content、pitch、timbre、speaker 和 style 信息。
```

个体化主观 voice image reconstruction 应该放到自采实验中完成，而不是强行用公开数据声称已经完成。

---

## 结尾

总结一下，这 37 个数据集的价值不在于每一个都完美，而在于它们共同覆盖了 EEG voice token 所需的监督轴。

英文自然语音提供 base auditory 和 content，中文 AAD 提供 speaker-stream retrieval，粤语和 CIRE 提供 tone/prosody，受控 phoneme 和 imagined/inner/overt speech 提供 attribute probe 和 mode transfer，AAD expansion 和 music proxy 提供泛化和边界验证。

因此，这个数据池足以支持 V1 的模型目标：EEG discrete token、attribute alignment 和 voice/speaker retrieval。它不直接支持最终个体化幻听或主观声音形象重构，这部分需要后续自采 voice bank 数据来补齐。

---

## Q&A 预设回答

**Q: 为什么不直接训练 EEG-to-waveform？**

A: 因为当前公开数据没有统一 voice bank，也没有主观相似度标签。直接训练 waveform 会把 decoder 能力和 EEG token 能力混在一起。第一阶段更稳的是 token alignment 和 retrieval。

**Q: 37 个数据集任务不一致，为什么还能放一起？**

A: 训练 tokenizer 和 retrieval 不要求所有数据集完全同构。关键是每个数据集内部有可靠的 EEG-audio、event、speaker、attention 或 condition label。跨数据集差异反而可以作为泛化测试。

**Q: 最大风险是什么？**

A: retrieval shortcut。模型可能利用 subject、trial、device、spatial cue 或 gaze，而不是学到 EEG-voice 对齐。因此 split policy 和 negative control 比单纯提高 accuracy 更重要。

**Q: 下一步是什么？**

A: 工程上是 DatasetRegistry、real-data collator、AudioTokenBundle extraction 和 evaluation scripts；实验上是自采统一 voice bank，补齐 personalized voice-image reconstruction 所需的主观评分和系统化声学操控。

---

1. **缺少统一 voice bank**
   公开数据里的声音材料通常来自不同数据集、不同说话人、不同录音条件、不同语言、不同任务。
   所以模型即使学到 EEG 和声音的对应关系，也很难判断它是不是学到了稳定的“音色/声像空间”。

2. **缺少同一受试者对同一批声音的主观评分**
   你的目标不是只问“这个声音是谁说的”或“内容是什么”，而是问：

> 这个受试者主观觉得这个声音像不像他听到/想象/幻听中的声音？

公开数据通常没有让每个受试者对同一批声音做 pitch、brightness、roughness、breathiness、speaker similarity、externalization 等评分。
没有这些评分，就无法建立 **subject-specific voice manifold**，也就是每个人自己的“声音形象空间”。

3. **缺少系统化 F0/formant/style 操控**
   如果一个声音和另一个声音不同，差异可能来自很多因素：内容、说话人、音高、共振峰、情绪、语速、响度、录音设备。
   如果没有系统操控，比如：

- 同一句话，不同说话人；
- 同一个说话人，同一句话，只改变 F0；
- 同一个说话人，同一句话，只改变 formant；
- 同一句话，只改变 happy/angry/whisper/style；

那么模型检索成功时，你无法证明它到底学到的是 **内容**，还是 **音高**，还是 **音色**，还是某种数据集 shortcut。

所以这句话的核心含义是：

**公开数据够做 EEG-speech alignment 或 speaker/content retrieval 的预训练和基线验证，但不够支撑“个体化主观声像重构”这个更高级的论文主张。要做这个主张，你需要自采一个受控 voice bank，并让每个受试者对这批声音做主观声像评分。**
