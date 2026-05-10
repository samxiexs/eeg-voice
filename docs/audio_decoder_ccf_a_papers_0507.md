# 近五年 CCF-A Audio Decoder 文献分类清单（0507）

## 口径

- 时间范围：2021-2025
- 会议范围：按 `ccf_2022.json` 的 A 类会议口径筛选
- 主题范围：聚焦 speech/audio generation、neural codec decoder、acoustic decoder、speech-to-speech decoder、voice conversion decoder
- 使用目标：服务于 `EEG -> token -> voice representation -> audio decoder`

## 写法说明

这份清单不再按“论文 1、论文 2”平铺，而是按**方法类型**分类，写法参考 `speech_quantization.pdf` 的组织方式：先给出方法类型，再放代表论文，再总结这一类方法解决什么问题、对当前 EEG 项目有什么帮助。

## 方法类型总览

| 方法类型 | 代表论文 | 解决的核心问题 | 对 EEG 项目的直接意义 |
| --- | --- | --- | --- |
| 统一式 encoder-decoder | SpeechT5 | speech/text 共用一个离散或共享中间空间 | 适合作为 `EEG token -> 语音解码` 的总框架 |
| neural codec / token decoder | RVQGAN, VoiceCraft, UniAudio | 把离散 audio token 还原成高保真语音 | 适合作为 `EEG token -> audio token -> waveform` 主路径 |
| style / prosody / speaker decoder | StyleTTS 2, P-Flow, CoMoSpeech | 建模音色、音调、speaker、韵律 | 适合作为 voice image 分支 |
| content / acoustic 解耦 decoder | DASpeech, SpeechT5 | 把“说什么”和“怎么说”拆开建模 | 适合把 EEG 内容信息和声音形象信息分开建模 |
| 通用多任务 audio decoder | UniAudio | speech / sound / singing 统一生成 | 适合后期扩展到更广的声音表征 |

## I. 统一式 Encoder-Decoder

这一类方法把 speech/text 或多个模态放进统一 backbone，重点不是单一 vocoder，而是建立一个稳定的中间接口。

### SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing

- 会议：ACL 2022
- 链接：[ACL Anthology](https://aclanthology.org/2022.acl-long.393/)
- 引用：OpenAlex 约 123（2026-05-07 查询）
- 方法类型：统一式 encoder-decoder
- 核心结构：
  - shared encoder-decoder backbone
  - speech / text 共用预训练空间
  - cross-modal vector quantization 作为跨模态接口
- 方法价值：
  - 它给出的是“统一中间空间 + 下游 decoder”的总范式
  - 不只是做 TTS，而是把 ASR、TTS、speech translation、voice conversion 放进同一体系
- 对当前项目的帮助：
  - 最适合借来定义 `EEG token` 的角色：不是直接输出波形，而是先进入一个能被语音端 decoder 消化的共享离散接口

## II. Neural Codec / Token Decoder

这一类方法的共同点是：**先把音频压成离散 token，再由 decoder 从 token 还原波形**。如果后面做 `EEG token -> audio token -> waveform`，这一类最关键。

### High-Fidelity Audio Compression with Improved RVQGAN

- 会议：NeurIPS 2023
- 链接：[NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/58d0e78cf042af5876e12661087bea12-Abstract.html)
- 引用：OpenAlex 约 59（2026-05-07 查询）
- 方法类型：neural codec decoder / RVQ token decoder
- 核心结构：
  - residual vector quantization
  - GAN-based decoder
  - 面向高保真重建的 codec 训练
- 方法价值：
  - 代表了“离散 token 先行，decoder 专注重建质量”的标准路线
  - token fidelity、bitrate、重建质量之间的平衡很清楚
- 对当前项目的帮助：
  - 如果 EEG 侧先学出离散 token，这篇最适合作为音频端 codec decoder 的参考原型

### VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild

- 会议：ACL 2024
- 链接：[ACL Anthology](https://aclanthology.org/2024.acl-long.673/)
- 引用：OpenAlex 约 35（2026-05-07 查询）
- 方法类型：codec token language model / token infilling decoder
- 核心结构：
  - neural codec token 序列建模
  - Transformer decoder 在 codec token 空间中做 infilling 和生成
  - 支持 zero-shot speech editing 和 TTS
- 方法价值：
  - 不是传统 mel-to-waveform，而是直接在 token 空间做生成和补全
  - 更接近后续 `EEG token -> 局部声音片段恢复` 的需求
- 对当前项目的帮助：
  - 很适合后续做 `EEG token -> voice token completion`
  - 对“补出说话人的声音形象”比纯文本 TTS 更贴近

### UniAudio: Towards Universal Audio Generation with Large Language Models

- 会议：ICML 2024
- 链接：[PMLR](https://proceedings.mlr.press/v235/yang24x.html)
- 方法类型：通用 audio token decoder
- 核心结构：
  - 离散 tokenization
  - next-token prediction
  - 统一 speech / sound / music / singing generation
- 方法价值：
  - 把 speech 从单独任务提升为统一 audio generation 框架的一部分
  - 说明 token decoder 不一定只服务语音，也可以服务更广的声音空间
- 对当前项目的帮助：
  - 如果后续要把 EEG token 对齐到更广义的 voice image，而不是狭义文本内容，这篇非常关键

## III. Style / Prosody / Speaker Decoder

这一类方法不把目标限制在“说对内容”，而是明确建模**音色、音调、韵律、speaker 风格**。这和当前项目的 voice image 目标直接相关。

### StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models

- 会议：NeurIPS 2023
- 链接：[NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3eaad2a0b62b5ed7a2e66c2188bb1449-Abstract-Conference.html)
- 引用：OpenAlex 约 23（2026-05-07 查询）
- 方法类型：style diffusion decoder
- 核心结构：
  - style latent variable
  - diffusion-based style modeling
  - 结合 speech LM feature 的 adversarial 训练
- 方法价值：
  - 核心贡献不是“读出字”，而是把 style 建成一个可控变量
  - 对 prosody、speaker likeness、naturalness 的控制强
- 对当前项目的帮助：
  - 这类方法最适合承接 `EEG token -> style / timbre / pitch branch`
  - 对“重构声音形象”比普通内容解码更关键

### P-Flow: A Fast and Data-Efficient Zero-Shot TTS through Speech Prompting

- 会议：NeurIPS 2023
- 链接：[NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Abstract-Conference.html)
- 引用：OpenAlex 约 2（2026-05-07 查询）
- 方法类型：flow-based prompt-conditioned decoder
- 核心结构：
  - speech prompt 做 speaker adaptation
  - flow matching generative decoder
  - fast zero-shot TTS
- 方法价值：
  - 不依赖超重的 neural codec LM，走的是 prompt-conditioning + fast decoder 路线
  - 更轻，更适合数据量受限场景
- 对当前项目的帮助：
  - 如果 EEG 只输出紧凑 token 或 voice profile，这篇提供了轻量 decoder 路线

### CoMoSpeech: One-Step Speech and Singing Voice Synthesis via Consistency Model

- 会议：ACM MM 2023
- 链接：[项目页](https://comospeech.github.io/)
- 辅助来源：[HKUST publication page](https://researchportal.hkust.edu.hk/en/publications/comospeech-one-step-speech-and-singing-voice-synthesis-via-consis/)
- 引用：OpenAlex 约 27；HKUST 页面显示 Scopus 26（2026-05-07 查询）
- 方法类型：consistency-based fast decoder
- 核心结构：
  - consistency model
  - one-step speech / singing synthesis
  - 兼顾生成速度和质量
- 方法价值：
  - 把传统多步生成压缩成单步或极少步
  - 很适合需要快速试错和小样本迭代的场景
- 对当前项目的帮助：
  - 如果后期真要从 EEG token 走到声音重建，fast decoder 会比慢 diffusion 更实用

## IV. Content / Acoustic 解耦 Decoder

这一类方法的核心不是“更强生成器”，而是把**内容信息**和**声学实现**拆成两个层次。对 EEG 尤其重要，因为 EEG 里内容相关信息和声音形象相关信息大概率不是同一层表示。

### DASpeech: Directed Acyclic Transformer for Fast and High-quality Speech-to-Speech Translation

- 会议：NeurIPS 2023
- 链接：[NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html)
- 引用：OpenAlex 约 4（2026-05-07 查询）
- 方法类型：two-stage content/acoustic decoder
- 核心结构：
  - 先 linguistic decoder
  - 再 acoustic decoder
  - directed acyclic transformer 提升速度
- 方法价值：
  - 明确把“内容 token”和“声学 realization”分开处理
  - 很适合需要同时控制内容和声音形象的任务
- 对当前项目的帮助：
  - 这篇几乎可以直接映射到你的建模口径：
    `EEG -> content-like token / voice-like token -> acoustic decoder`

## V. 方法演进方向

参考 `speech_quantization.pdf` 的写法，如果只看 decoder 侧，这批论文的演进可以概括为四条线：

### 1. 单一文本条件 -> 离散 token 条件

- 代表：`SpeechT5`、`RVQGAN`、`VoiceCraft`、`UniAudio`
- 含义：
  - 过去常见的是 text -> mel -> vocoder
  - 现在越来越多方法直接在离散 audio token 上建模
- 对 EEG 的意义：
  - EEG 更适合先对齐 token，再交给 decoder，而不是直接回归波形

### 2. 只解内容 -> 同时解内容与声音形象

- 代表：`StyleTTS 2`、`P-Flow`、`DASpeech`
- 含义：
  - decoder 不再只负责“句子说对”
  - 还要负责 speaker、style、prosody、timbre、pitch
- 对 EEG 的意义：
  - 这正对应你现在最关心的 voice image 问题

### 3. 慢生成 -> 快生成

- 代表：`CoMoSpeech`、`P-Flow`、`DASpeech`
- 含义：
  - 从重型自回归或多步扩散，转向 one-step、flow、并行解码
- 对 EEG 的意义：
  - EEG 侧数据量和监督强度都有限，decoder 过重通常不利于早期实验

### 4. 单任务语音 -> 通用声音生成

- 代表：`UniAudio`
- 含义：
  - speech 不再被单独看待，而是统一到 audio foundation model
- 对 EEG 的意义：
  - 后面如果把 voice image 扩展为更广义的 auditory image，这条路最自然

## 当前最值得优先看的 5 篇

如果只按“对 `EEG -> token -> voice reconstruction interface` 最有帮助”排序：

1. SpeechT5
2. High-Fidelity Audio Compression with Improved RVQGAN
3. VoiceCraft
4. StyleTTS 2
5. DASpeech

这 5 篇刚好覆盖：

- 统一中间接口
- 离散 token 重建
- token-space 生成/补全
- style / timbre / prosody 建模
- content / acoustic 解耦

## 没放进主清单但仍然值得看

### SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models

- 很相关
- 但严格按 `ccf_2022.json` 口径，`ICLR` 不在这次主筛选名单里，所以没有放进主清单

### VALL-E / NaturalSpeech 2 / Voicebox

- 都很相关
- 但不满足这次“近五年 + CCF-A 会议”的严格口径，主清单里排除

### Textually Pretrained Speech Language Models (TWIST)

- NeurIPS 2023
- 更偏 speech LM 预训练与 token 建模
- 对 decoder 设计有帮助，但不是这份清单里的主类型

## 最后一句话

从当前项目角度看，audio decoder 文献不是一条线，而是四类模块的组合：

```text
统一接口 + codec token decoder + style/prosody decoder + content/acoustic 解耦
```

因此，EEG 侧先做 tokenization，音频侧再从这四类方法中拼出对应解码器，路线最清楚。
