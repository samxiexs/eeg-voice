# Audio Decoder Methods/Paper 汇报讲稿（0521）

来源：

- Notion: https://www.notion.so/35925594c17b80a28233e920cb4033ab
- 本地补充：`docs/audio_decoder_ccf_a_papers_0520.md`
- 本地补充：`docs/audio_decoder_reading_guide_0520.md`

建议时长：12-15 分钟

---

## 开场

大家好，我今天汇报的是 Audio Decoder Methods/Paper 这部分。这个汇报要解决的不是“我们最后选哪个 TTS decoder”，而是一个更前置的问题：如果 EEG 侧只能稳定恢复一部分声音信息，那么 audio 侧应该被拆成什么 token，才能和 EEG token 对齐？

我的核心结论是：

```text
V2 不应直接做 waveform generation，
也不应让 EEG 直接预测 full residual codec token。

更合理的路线是：
audio waveform -> semantic / prosody / voice / codec tokens
EEG token 先对齐 semantic、prosody、voice
V3 再由 audio token LM 或 decoder foundation model 补全 codec detail
最后由 waveform renderer 生成声音。
```

这条路线把 EEG 研究问题和 audio generation 工程问题拆开，避免一开始就把所有声学细节压给 EEG。

---

## 1. 总体链路：从 waveform 到 EEG alignment target

我们先看总链路：

```text
audio waveform
-> audio tokenizer / codec encoder
-> semantic content / prosody / voice / codec token
-> EEG-token alignment target
-> audio token LM or voice decoder foundation model
-> waveform renderer
```

这里最关键的思想是：audio waveform 不是一个整体目标，而是可以拆成四类信息。

第一类是 semantic/content token，回答“说了什么”。对应 phoneme、syllable、word-like unit 和 speech content。

第二类是 prosody token，回答“怎么说”。对应 F0、energy、rhythm、duration、intonation 和 stress。

第三类是 voice token / timbre，回答“是谁、什么音色、什么风格”。对应 speaker identity、timbre、style 和 vocal quality。

第四类是 codec token，回答“如何还原波形”。它包含相位、细粒度谱纹理、录音条件和 residual detail。

EEG 更可能稳定捕获前三类，也就是 semantic、prosody 和 voice。完整 codec residual 对高保真声音重建很重要，但不应该作为 EEG 主监督目标。

---

## 2. 为什么 full codec residual 不适合作为 EEG 主目标

普通 neural codec 的目标是压缩并重建 waveform。它会保留大量细节，比如相位、局部频谱纹理、录音环境和 decoder residual。这些信息对 MOS 或 waveform fidelity 有帮助，但不一定是 EEG 能稳定恢复的神经表征。

如果我们让 EEG 直接预测 full codec token，风险是任务会变成噪声拟合：模型在追逐 EEG 本来就很难支持的声学细节，而不是学习可解释的 speech/voice representation。

所以 V2 的目标应该是：

```text
EEG token q2-q3 -> semantic / content token
EEG token q4    -> F0 / energy / rhythm / prosody token
EEG token q5-q6 -> speaker / timbre / style / voice token
EEG token q7    -> no audio alignment; residual nuisance only
```

codec token 应该留给 V3 decoder，用来从高层条件补齐可听声音。

---

## 3. 第一层：Semantic Representation / Audio Tokenizer

第一层文献回答的是：audio 先被表示成什么内容单位？

HuBERT 是最直接的 content token 起点。它通过 masked prediction 学习 speech hidden units，不依赖人工 phoneme 标注，而是先用 clustering 形成 pseudo-label，再预测被遮挡的 speech unit。对我们来说，HuBERT 的价值不是 ASR 分数本身，而是它能提供相对干净的 discrete content unit，用作 EEG q2-q3 的监督目标。

wav2vec 2.0 是前置背景，它提供 speech SSL representation 和 contrastive learning 思路。WavLM 更偏 full-stack speech representation，同时覆盖 speaker、emotion、denoising 等任务，所以它既能做 content feature，也能做 voice/speaker feature。

Whisper 的价值在 robustness。跨数据集音频质量和语种不一致时，Whisper 可以作为 transcript sanity check 和 language-aware content reference。

BEATs 要单独保留，因为我们的数据池里还有 OpenMIIR、MUSIN-G、MAD-EEG 这些非语音 auditory proxy。BEATs 可以为非语音声音提供可比的 audio token 参照。

SpeechT5 则提供 speech/text shared space 和 cross-modal vector quantization，适合作为 speech-text bridge，但不替代 neural codec。

这一层的结论是：V2 的 content target 首选 HuBERT / wav2vec / Whisper / SpeechTokenizer semantic units，而不是 waveform。

---

## 4. 第二层：Factorized Codec / Acoustic Token

第二层文献回答的是：waveform 如何变成可重建 token？哪些 token 可以给 EEG 对齐，哪些只是 decoder residual？

NaturalSpeech 3 / FACodec 是这里最关键的一篇。它把 speech 拆成 content、prosody、timbre 和 acoustic detail。这个拆分几乎直接对应我们的 EEG grouped token：content 对 q2-q3，prosody 对 q4，timbre/speaker 对 q5-q6，acoustic detail 更接近 q7 或 decoder-only residual。

这篇论文的意义不是让我们复现 NaturalSpeech 3，而是告诉我们 audio codec 可以被设计成可解释的 factorized space。这样 EEG 对齐就不需要面对一个混在一起的 codec codebook，而是可以分别对齐内容、韵律和音色。

X-Codec / Codec Does Matter 则进一步说明普通 codec 有 semantic shortcoming。它在 quantization 前融合 HuBERT/WavLM semantic feature，并引入 semantic reconstruction loss，目标是让 codec code 不只高保真，还保留语义一致性。

这对 EEG 非常关键。如果 codec token 自己都不稳定表达内容，EEG 去预测它就会被错误目标牵引。因此，V2 应该优先使用 semantic-enhanced 或 factorized codec 的 coarse subset。

SoundStream、EnCodec、RVQGAN 和 ESC 更适合作为 V3 waveform backend。它们说明如何用 neural codec 从离散 code 恢复高质量声音，但 full residual codebook 不进入 V2 主监督。

---

## 5. 第三层：Audio Token LM / Voice Decoder Foundation

第三层文献回答的是：如果 EEG 已经预测出高层 token，后面如何补全成可听语音？

AudioLM 是这条路线的骨架。它把 audio generation 写成 semantic token 和 acoustic token 的语言建模。semantic token 保持长程内容结构，acoustic token 补充局部声学细节。对 EEG-Voice 而言，这说明 decoder 可以分工：EEG 提供高层条件，audio LM 负责补全时间结构和声学 token。

VoiceCraft 更接近我们的实际场景。它把 speech editing 和 TTS 做成 neural codec token infilling。EEG token 往往是不完整、带噪、低带宽的，所以更像“给 decoder 一组不完整条件”，而不是给出完整 codec sequence。VoiceCraft 这类 infilling 模型可以作为 EEG-to-audio completion 的重要参考。

VALL-E 和 VALL-E 2 展示了 content token + acoustic prompt 到 codec token 的路线。Voicebox 更像 universal speech generation foundation，覆盖 infilling、editing 和 multilingual transfer。MaskGCT 提供 semantic-to-acoustic 两阶段生成结构。Moshi 和 StreamSpeech 更偏实时和 streaming decoder。

UniAudio 说明 audio token LM 可以扩展到 speech、sound、music 和 singing，这对我们后续 voice-image foundation 有意义，因为最终可能不只处理标准语音，还要处理更广义的声音形象。

这一层的结论是：V3 decoder 不必从零开始训练 waveform generator，而应接入 audio token LM / voice decoder foundation，把 EEG 预测出的 semantic、prosody、voice 条件转成完整 codec token。

---

## 6. 第四层：Voice Control / Acoustic Realization

第四层文献回答的是：speaker、timbre、style 和 prosody 如何进入最终声音？

StyleTTS 2 以 style latent variable 和 diffusion-based style modeling 建模声音风格。它对应的是 voice/timbre/style/prosody realization，而不是 content token 主线。对我们来说，它可以帮助定义 EEG 对齐出的 style 或 voice image 如何影响 decoder。

P-Flow、CoMoSpeech、E2 TTS、F5-TTS 和 Mega-TTS 2 都属于 voice control 或 fast acoustic realization 的候选家族。它们主要用于后期选择 renderer 或对照 decoder。

DASpeech 的结构尤其值得注意，因为它把 linguistic decoder 和 acoustic decoder 分离。这个分离和 EEG-Voice 的自然分工一致：EEG 侧先恢复 content-like、voice-like 和 prosody-like token，声学 decoder 侧再完成 acoustic realization。

所以第四层不是定义 audio token 的起点，而是在 token 已经存在之后，决定声音如何被渲染出来。

---

## 7. AudioTokenBundle：把文献变成工程接口

为了让不同 audio model 的输出能服务 EEG alignment，我们需要一个统一对象：`AudioTokenBundle`。

它不是某个单一模型的输出，而是一个文件级接口：

```text
AudioTokenBundle
  waveform_id
  sample_rate
  frame_times
  content_units        # HuBERT / wav2vec / Whisper / SpeechTokenizer
  prosody_tokens       # F0, energy, rhythm, envelope, voicing
  voice_tokens         # speaker, timbre, style, prompt-level voice identity
  codec_codes          # SoundStream / EnCodec / RVQGAN / FACodec / X-Codec
  codec_group_names
  valid_mask
  voiced_mask
```

这样组织之后，EEG alignment 可以先关注前面三类：content、prosody、voice。codec code 只是后续 decoder completion 的材料。

这也让数据集接入更清楚。每段音频先离线转成 AudioTokenBundle，再和 EEG window 对齐。模型不用直接面对 wav，也不用在训练时实时跑各种 audio foundation model。

---

## 8. V2 / V3 路线

根据上面的文献结构，我们可以把路线分成 V2 和 V3。

V2 的中心是 audio token alignment：

```text
EEG token q0-q1  <-> audio envelope / onset / broad auditory response
EEG token q2-q3  <-> HuBERT / wav2vec / Whisper / SpeechTokenizer semantic units
EEG token q4     <-> F0 / energy / rhythm / prosody token
EEG token q5-q6  <-> speaker / timbre / style / voice embedding
EEG token q7     <-> no audio alignment; residual nuisance only
```

这个阶段的评估应该落在 content accuracy、prosody correlation、speaker retrieval、voice embedding retrieval 和 token-to-token alignment 上。waveform MOS、speaker similarity MOS 和 naturalness 不作为 V2 主指标。

V3 才进入 audio token completion：

```text
EEG tokens
-> predicted semantic / prosody / voice tokens
-> audio token LM or voice decoder foundation model
-> codec / acoustic tokens
-> waveform renderer
```

也就是说，V2 证明 EEG token 有声音相关信息；V3 才证明这些信息可以被 decoder 渲染成可听声音。

---

## 9. 最优先精读的五篇论文

如果只精读五篇，我建议顺序是：

第一，HuBERT。它定义 content unit 和时间对齐方式，是 EEG q2-q3 content target 的起点。

第二，NaturalSpeech 3。它定义 factorized token space，帮助我们确认 content、prosody、timbre 和 acoustic detail 的分工。

第三，X-Codec / Codec Does Matter。它解释普通 codec 的 semantic failure，避免我们把 residual codec token 误当成 EEG 主目标。

第四，AudioLM。它建立 semantic-to-acoustic token LM 的 V3 decoder 骨架。

第五，VoiceCraft。它建立 incomplete/noisy upstream token 到 decoder completion 的实现参照。

这五篇正好覆盖最小闭环：

```text
HuBERT 定义内容单位
NaturalSpeech 3 定义因子化语音 token
X-Codec 说明 codec 需要语义约束
AudioLM 定义 token LM 生成范式
VoiceCraft 定义不完整 token 条件下的补全方式
```

---

## 10. 和 EEG-Voice 项目的直接关系

回到 EEG-Voice，我们不应该把 audio decoder 当成一个孤立模块。它决定的是 EEG token 的监督目标。

如果 audio side 是 entangled full codec，那么 EEG 任务会变得不可解释。相反，如果 audio side 被拆成 semantic、prosody、voice 和 codec detail，那么 EEG token 分组就有了明确目标：

- q2-q3 对齐 semantic/content。
- q4 对齐 F0、energy、rhythm。
- q5-q6 对齐 speaker、timbre、style。
- q7 保持 residual nuisance，不进入 audio alignment。

这让 EEG 侧的评价也更清晰。我们可以报告 semantic unit prediction、F0 correlation、speaker retrieval、voice embedding retrieval，而不是一上来报告 waveform naturalness。

后续如果 waveform 质量不好，也可以判断问题来自 EEG token 还是 decoder backend，而不会把两个问题混在一起。

---

## 结尾

总结一下，audio decoder 路线的核心不是直接生成 waveform，而是先建立可解释的 audio token target。

V2 做的是：

```text
audio 先被拆成 semantic / prosody / voice，
EEG token 对齐这些稳定目标。
```

V3 做的是：

```text
audio foundation decoder 补全 codec detail，
waveform renderer 生成最终声音。
```

这个分层路线对本项目很重要，因为非侵入式 EEG 的信息带宽有限。我们应该先证明 EEG token 中有可读出的内容、韵律、音色和说话人信息，再讨论如何把这些信息渲染成声音。

---

## Q&A 预设回答

**Q: 为什么不直接用 EnCodec token 当 EEG target？**

A: EnCodec 这类 codec 更偏 waveform reconstruction，包含大量 residual detail。EEG 很难稳定恢复这些细节。V2 更适合对齐 semantic/prosody/voice，codec detail 留给 V3 decoder 补全。

**Q: HuBERT 已经是内容 token，为什么还需要 voice token？**

A: HuBERT 主要保留“说了什么”，但我们的目标还包括“谁在说、什么音色、什么风格”。这些信息需要 WavLM speaker feature、FACodec timbre code、speaker embedding 或 style latent 作为额外目标。

**Q: NaturalSpeech 3 对我们最大的启发是什么？**

A: 不是复现它的 TTS，而是借它的 factorization：content、prosody、timbre、acoustic detail。这个分解和 EEG grouped RVQ 的设计高度一致。

**Q: V3 最可能用哪类 decoder？**

A: 首选能接受不完整、高层、带噪条件的 token completion 或 infilling 模型，比如 VoiceCraft、AudioLM/MaskGCT 这一类。传统 full codec LM 可以作为 backend，但不应该决定 V2 的主监督。

**Q: Waveform quality 什么时候报告？**

A: V3 报告。V2 先报告 content accuracy、prosody correlation、speaker retrieval、voice embedding retrieval 和 token alignment。否则 waveform 分数会混合 EEG 能力和 decoder 能力。
