# CCF-A Audio Decoder 文献清单（0512）

## 研究口径

- **主题范围**：speech/audio generation、neural codec decoder、acoustic decoder、speech-to-speech decoder、voice conversion decoder
- **使用目标**：服务于 `EEG → token → voice representation → audio decoder` 主链路
- **筛选标准**：近五年 CCF-A 会议（ACL、EMNLP、NeurIPS、ICML、AAAI、ACM MM）

---

## 方法类型总览

```
EEG token
    │
    ├─► 统一接口（SpeechT5 范式）
    │       └─► 共享离散空间 → 下游 decoder
    │
    ├─► Neural Codec Decoder（RVQGAN / VoiceCraft / X-Codec）
    │       └─► audio token → 高保真波形
    │
    ├─► Style / Prosody / Speaker Decoder（StyleTTS 2 / NaturalSpeech 3）
    │       └─► voice image 分支：音色 / 音调 / 韵律
    │
    └─► Content / Acoustic 解耦（DASpeech / UniAudio 1.5）
            └─► 内容 token ≠ 声音形象 token，分开建模
```

| 方法类型 | 代表论文 | 解决的核心问题 | 对 EEG 项目的直接意义 |
| --- | --- | --- | --- |
| 统一式 encoder-decoder | SpeechT5, StreamSpeech, VoiceCraft-X | speech/text 共用离散中间空间 | 定义 EEG token 角色：先进入共享接口，再交给 decoder |
| Neural codec / token decoder | RVQGAN, VoiceCraft, UniAudio, X-Codec, ESC | 离散 audio token → 高保真语音 | `EEG token → audio token → waveform` 主路径 |
| Style / prosody / speaker decoder | StyleTTS 2, P-Flow, CoMoSpeech, NaturalSpeech 3 | 建模音色、音调、speaker、韵律 | voice image 分支，重构声音形象 |
| Content / acoustic 解耦 decoder | DASpeech, SpeechT5, NaturalSpeech 3 | 内容信息与声学实现分层建模 | EEG 内容信息与声音形象信息分开建模 |
| 通用多任务 audio decoder | UniAudio, UniAudio 1.5 | speech / sound / singing 统一生成 | 后期扩展到更广义的声音表征 |

---

## I. 统一式 Encoder-Decoder

> 核心思路：建立稳定的跨模态中间接口，而非单一 vocoder。

### SpeechT5 — ACL 2022

- **链接**：[ACL Anthology](https://aclanthology.org/2022.acl-long.393/) | [PDF](https://aclanthology.org/2022.acl-long.393.pdf)
- **引用**：~123（OpenAlex 2026-05-07）
- **核心结构**：shared encoder-decoder backbone；speech/text 共用预训练空间；cross-modal VQ 跨模态接口
- **对项目的意义**：定义 EEG token 的角色——不直接输出波形，而是进入能被语音端 decoder 消化的共享离散接口

### StreamSpeech — ACL 2024

- **链接**：[ACL Anthology](https://aclanthology.org/2024.acl-long.485/) | [PDF](https://aclanthology.org/2024.acl-long.485.pdf)
- **引用**：~18（OpenAlex 2026-05-11）
- **核心结构**：两阶段架构——自回归 speech-to-text decoder + 非自回归 text-to-unit decoder；unit vocoder 合成最终语音
- **对项目的意义**：两阶段结构直接对应 `EEG → content token → unit decoder → waveform`；非自回归 unit decoder 适合实时场景

### VoiceCraft-X — EMNLP 2025

- **链接**：[ACL Anthology](https://aclanthology.org/2025.emnlp-main.137/) | [PDF](https://aclanthology.org/2025.emnlp-main.137.pdf)
- **核心结构**：VoiceCraft 扩展至 11 种语言；自回归 neural codec LM；voice cloning prompt 控制 speaker identity
- **对项目的意义**：voice cloning prompt 机制可类比为 `EEG voice profile → decoder conditioning`；多语言泛化说明 token 空间跨域对齐可行

---

## II. Neural Codec / Token Decoder

> 核心思路：先把音频压成离散 token，再由 decoder 从 token 还原波形。`EEG token → audio token → waveform` 主路径的核心参考。

### RVQGAN — NeurIPS 2023

- **链接**：[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/58d0e78cf042af5876e12661087bea12-Abstract.html) | [PDF](https://arxiv.org/pdf/2306.06546)
- **引用**：~59（OpenAlex 2026-05-07）
- **核心结构**：Residual Vector Quantization；GAN-based decoder；高保真重建
- **对项目的意义**：EEG 侧先学出离散 token 后，最适合作为音频端 codec decoder 的参考原型

### VoiceCraft — ACL 2024

- **链接**：[ACL Anthology](https://aclanthology.org/2024.acl-long.673/) | [PDF](https://aclanthology.org/2024.acl-long.673.pdf)
- **引用**：~35（OpenAlex 2026-05-07）
- **核心结构**：neural codec token 序列建模；Transformer decoder 在 token 空间做 infilling 和生成；支持 zero-shot speech editing
- **对项目的意义**：适合 `EEG token → voice token completion`；对"补出说话人声音形象"比纯文本 TTS 更贴近

### UniAudio — ICML 2024

- **链接**：[PMLR](https://proceedings.mlr.press/v235/yang24x.html) | [PDF](https://arxiv.org/pdf/2310.00704)
- **核心结构**：离散 tokenization；next-token prediction；统一 speech / sound / music / singing 生成
- **对项目的意义**：如果 EEG token 要对齐更广义的 voice image 而非狭义文本内容，这篇非常关键

### X-Codec — AAAI 2025

- **链接**：[arXiv](https://arxiv.org/abs/2408.17175) | [GitHub](https://github.com/zhenye234/xcodec) | [PDF](https://arxiv.org/pdf/2408.17175)
- **核心结构**：RVQ 量化前融合 HuBERT/WavLM 语义特征；联合量化声学与语义；语义重建损失
- **对项目的意义**：EEG 解码出的 token 本质是语义 token，X-Codec 提供"语义 token → 高质量语音"的直接桥梁；语义重建损失设计可借鉴到 EEG token 对齐训练

### ESC — EMNLP 2024

- **链接**：[ACL Anthology](https://aclanthology.org/2024.emnlp-main.562/) | [PDF](https://aclanthology.org/2024.emnlp-main.562.pdf)
- **核心结构**：跨尺度 RVQ Transformer；decoder 侧跨尺度注意力融合；低比特率高保真重建
- **对项目的意义**：EEG 侧 token 数量受限，"少量 token → 高质量重建"的 decoder 参考；跨尺度结构可类比 EEG 不同频段特征融合

---

## III. Style / Prosody / Speaker Decoder

> 核心思路：不把目标限制在"说对内容"，明确建模音色、音调、韵律、speaker 风格——与 voice image 目标直接相关。

### StyleTTS 2 — NeurIPS 2023

- **链接**：[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3eaad2a0b62b5ed7a2e66c2188bb1449-Abstract-Conference.html) | [PDF](https://arxiv.org/pdf/2306.07691)
- **引用**：~23（OpenAlex 2026-05-07）
- **核心结构**：style latent variable；diffusion-based style modeling；结合 speech LM feature 的 adversarial 训练
- **对项目的意义**：最适合承接 `EEG token → style / timbre / pitch branch`；对"重构声音形象"比普通内容解码更关键

### P-Flow — NeurIPS 2023

- **链接**：[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Abstract-Conference.html) | [PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Paper-Conference.pdf)
- **引用**：~2（OpenAlex 2026-05-07）
- **核心结构**：speech prompt 做 speaker adaptation；flow matching generative decoder；fast zero-shot TTS
- **对项目的意义**：EEG 只输出紧凑 token 或 voice profile 时，提供轻量 decoder 路线

### CoMoSpeech — ACM MM 2023

- **链接**：[项目页](https://comospeech.github.io/) | [PDF](https://arxiv.org/pdf/2305.06908)
- **引用**：~27（OpenAlex）；Scopus 26（HKUST 页面）
- **核心结构**：consistency model；one-step speech / singing synthesis
- **对项目的意义**：fast decoder 比慢 diffusion 更适合 EEG 侧数据量受限的早期实验

### NaturalSpeech 3 — ICML 2024

- **链接**：[PMLR](https://proceedings.mlr.press/v235/ju24b.html) | [PDF](https://arxiv.org/pdf/2403.03100)
- **核心结构**：FACodec（Factorized VQ codec）将语音解耦为 content / prosody / timbre / acoustic details 四个子空间；四路 factorized diffusion model 分别生成各属性
- **对项目的意义**：FACodec 四路解耦直接对应 EEG 里可能分离的内容/音色/韵律信号；`EEG → factorized token → 各子空间 decoder` 是最自然的映射路线

---

## IV. Content / Acoustic 解耦 Decoder

> 核心思路：把内容信息和声学实现拆成两个层次——EEG 里内容相关信息和声音形象相关信息大概率不是同一层表示。

### DASpeech — NeurIPS 2023

- **链接**：[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html) | [PDF](https://arxiv.org/pdf/2310.07403)
- **引用**：~4（OpenAlex 2026-05-07）
- **核心结构**：先 linguistic decoder，再 acoustic decoder；directed acyclic transformer 提升速度
- **对项目的意义**：几乎可以直接映射到 `EEG → content-like token / voice-like token → acoustic decoder`

### UniAudio 1.5 — ACM MM 2024

- **链接**：[ACM DL](https://dl.acm.org/doi/10.1145/3664647.3681078) | [PDF](https://arxiv.org/pdf/2406.10056)
- **核心结构**：LLM 驱动 audio codec decoder；few-shot in-context learning 适配多种 audio 任务
- **对项目的意义**：EEG 数据量天然受限，few-shot decoder 路线比全监督更现实；LLM 作为 decoder controller 可承接 EEG token 的语义对齐

---

## V. 方法演进四条线

```
1. 单一文本条件 ──► 离散 token 条件
   SpeechT5 → RVQGAN → VoiceCraft → UniAudio → ESC → VoiceCraft-X
   EEG 意义：先对齐 token，再交给 decoder，而非直接回归波形

2. 只解内容 ──► 同时解内容与声音形象
   StyleTTS 2 → P-Flow → DASpeech → NaturalSpeech 3
   EEG 意义：voice image 问题的核心方向

3. 慢生成 ──► 快生成
   CoMoSpeech → P-Flow → DASpeech → StreamSpeech → UniAudio 1.5
   EEG 意义：数据量和监督强度受限，decoder 过重不利于早期实验

4. 单任务语音 ──► 通用声音生成
   UniAudio → UniAudio 1.5
   EEG 意义：voice image 扩展为更广义 auditory image 的自然路线
```

---

## VI. 优先阅读排序

按"对 `EEG → token → voice reconstruction` 最有帮助"排序：

| 优先级 | 论文 | 覆盖的能力 |
| --- | --- | --- |
| ★★★ | SpeechT5 | 统一中间接口 |
| ★★★ | RVQGAN | 离散 token 重建 |
| ★★★ | VoiceCraft | token-space 生成/补全 |
| ★★★ | StyleTTS 2 | style / timbre / prosody 建模 |
| ★★★ | DASpeech | content / acoustic 解耦 |
| ★★ | NaturalSpeech 3 | 四路 factorized 解耦 |
| ★★ | X-Codec | 语义 token → 高质量语音 |
| ★★ | UniAudio 1.5 | few-shot decoder |
| ★ | StreamSpeech | 两阶段实时解码 |
| ★ | ESC | 少量 token 高保真重建 |
| ★ | CoMoSpeech | fast one-step decoder |
| ★ | P-Flow | 轻量 prompt-conditioned decoder |
| ★ | UniAudio | 通用 audio 生成扩展 |
| ★ | VoiceCraft-X | 多语言 voice cloning |

---

## VII. 全部论文汇总

| 论文 | 会议 | 年份 | 方法类型 | 核心贡献 | PDF |
| --- | --- | --- | --- | --- | --- |
| SpeechT5 | ACL | 2022 | 统一式 encoder-decoder | speech/text 共享预训练空间，跨模态 VQ 接口 | [PDF](https://aclanthology.org/2022.acl-long.393.pdf) |
| StreamSpeech | ACL | 2024 | 统一式 encoder-decoder | 两阶段 S2ST，非自回归 unit decoder | [PDF](https://aclanthology.org/2024.acl-long.485.pdf) |
| VoiceCraft-X | EMNLP | 2025 | 统一式 encoder-decoder | 多语言 voice cloning，prompt 控制 speaker | [PDF](https://aclanthology.org/2025.emnlp-main.137.pdf) |
| RVQGAN | NeurIPS | 2023 | Neural codec decoder | RVQ + GAN decoder，高保真重建 | [PDF](https://arxiv.org/pdf/2306.06546) |
| VoiceCraft | ACL | 2024 | Neural codec decoder | Token infilling，zero-shot editing | [PDF](https://aclanthology.org/2024.acl-long.673.pdf) |
| UniAudio | ICML | 2024 | Neural codec decoder | 统一 speech/sound/music 生成 | [PDF](https://arxiv.org/pdf/2310.00704) |
| X-Codec | AAAI | 2025 | Neural codec decoder | 语义增强 codec，解决内容信息不足 | [PDF](https://arxiv.org/pdf/2408.17175) |
| ESC | EMNLP | 2024 | Neural codec decoder | 跨尺度 RVQ，低比特率高保真 | [PDF](https://aclanthology.org/2024.emnlp-main.562.pdf) |
| StyleTTS 2 | NeurIPS | 2023 | Style/prosody decoder | Style diffusion，adversarial 训练 | [PDF](https://arxiv.org/pdf/2306.07691) |
| P-Flow | NeurIPS | 2023 | Style/prosody decoder | Flow matching，fast zero-shot TTS | [PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Paper-Conference.pdf) |
| CoMoSpeech | ACM MM | 2023 | Style/prosody decoder | Consistency model，one-step 生成 | [PDF](https://arxiv.org/pdf/2305.06908) |
| NaturalSpeech 3 | ICML | 2024 | Style/prosody decoder | FACodec 四路解耦（content/prosody/timbre/acoustic） | [PDF](https://arxiv.org/pdf/2403.03100) |
| DASpeech | NeurIPS | 2023 | Content/acoustic 解耦 | 两阶段 linguistic + acoustic decoder | [PDF](https://arxiv.org/pdf/2310.07403) |
| UniAudio 1.5 | ACM MM | 2024 | Content/acoustic 解耦 | LLM-driven codec，few-shot 学习 | [PDF](https://arxiv.org/pdf/2406.10056) |

---

## VIII. 主链路映射总结

```
EEG token
    │
    ▼
[统一接口]  SpeechT5 范式 — 共享离散空间
    │
    ├──[内容分支]──► DASpeech / StreamSpeech
    │                   linguistic token → acoustic decoder
    │
    ├──[声音形象分支]──► StyleTTS 2 / NaturalSpeech 3 / P-Flow
    │                       timbre / prosody / speaker style
    │
    └──[波形重建]──► RVQGAN / VoiceCraft / X-Codec / ESC
                        audio token → high-fidelity waveform
```

> 结论：audio decoder 不是一条线，而是四类模块的组合。EEG 侧先做 tokenization，音频侧从这四类方法中拼出对应解码器，路线最清楚。
