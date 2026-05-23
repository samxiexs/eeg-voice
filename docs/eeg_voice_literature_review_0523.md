# EEG-Voice Literature Review：从 EEG Token 到 Voice/Audio Token Alignment

## 摘要

EEG-Voice 当前应被定位为一个 **token-level alignment** 问题，而不是端到端 EEG-to-waveform generation 问题。非侵入式 EEG 的优势是时间分辨率高、采集风险低、可扩展到更多被试；主要限制是信噪比低、跨被试差异大、设备和 montage 不统一。因此，第一阶段更合理的问题不是“能否直接从 EEG 生成高保真声音”，而是“EEG 是否能形成稳定的离散 token，这些 token 是否携带 auditory base、content、prosody、voice identity 等可解释信息，并且能否与 audio/voice token 建立可检验的对齐关系”[1][2][3][4]。

这一路线可以由三组文献支撑。第一组是非侵入式 speech decoding 与 EEG foundation model：Défossez 团队用 EEG/MEG brain segment 与 speech representation 做 retrieval-style decoding，Duan 团队和 Jiang 团队分别从 EEG-to-text 与 EEG foundation model 角度说明离散 neural token 可以作为跨任务接口，Xiao 团队进一步把 sensor-aware brain tokenizer 用于统一 EEG/MEG 建模[1][2][3][4]。第二组是 audio/speech tokenization：wav2vec 2.0、BEATs、Whisper、HuBERT 和 NaturalSpeech 3 说明 audio waveform 可以被拆成 semantic/content、prosody、voice/timbre、codec/detail 等层次，而不是作为单一黑盒 target[9][10][11][12][17]。第三组是 speech generation 与 neural codec：SoundStream、RVQGAN、AudioLM、Voicebox、StyleTTS 2、UniAudio 和 SEAMLESS 说明高保真声音渲染需要 codec token、style prompt、speaker condition 和 large-scale audio generation backend，但这些组件应放在后续 decoder/backend 阶段，而不是直接作为非侵入式 EEG 的第一主监督目标[13][14][15][16][18][19][21]。

本报告采用“高等级为主”的引用策略：核心论据优先使用 Nature / Nature 子刊、NeurIPS、ICML、ICLR 等来源；HuBERT、SoundStream、AudioLM 等必要系统作为工程背景使用，但不把它们单独作为核心科学 claim 的唯一证据[17][18][19]。当前仓库状态也需要明确区分：`EEGVoiceTokenV1`、grouped RVQ、alignment heads、speaking-mode adapter、retrieval queue 和 synthetic tests 已经落地；真实 selected-dataset registry、real-data collator、AudioTokenBundle extraction、target extraction、training loop 和真实评估脚本尚未完成。

## Evidence-Claim Map

| Source ID | Source                      | Usable fact                                                                                      | Supported claim                                                                                              | Citation slot                | Risk                                |
| --------- | --------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | ---------------------------- | ----------------------------------- |
| [1]       | Défossez 团队              | 非侵入式 EEG/MEG 可以用 3 s brain segment 与 speech representation 做 retrieval-style decoding。 | EEG-Voice 第一阶段更适合做 representation retrieval 和 token alignment，而不是直接 waveform reconstruction。 | EEG 侧、Alignment 侧         | Nature 子刊，强证据                 |
| [2]       | Duan 团队                   | DeWave 用离散 EEG encoding 连接 EEG 与文本生成，并减少对 word-level marker 的依赖。              | 离散 neural token 可以作为 EEG 与语言/语音表征之间的中间接口。                                               | EEG tokenization             | NeurIPS，强证据                     |
| [3]       | Jiang 团队                  | NeuroLM 将 EEG 视作可 token 化、可与 LLM 交互的信号，并通过 neural tokenizer 形成离散 token。    | EEG token 不只服务分类，也可作为跨任务 foundation interface。                                                | EEG foundation               | ICLR，强证据                        |
| [4]       | Xiao 团队                   | BrainOmni 提出 BrainTokenizer 和 sensor-aware 机制以统一异构 EEG/MEG。                           | 跨设备、跨 sensor 配置的 EEG 建模需要 sensor/device-aware 表征。                                             | EEG preprocessing / backbone | NeurIPS，强证据                     |
| [5]       | Tang 团队                   | 非侵入式 fMRI 可重建连续语言语义，但依赖个体训练和受试者配合。                                   | EEG-Voice 的 content claim 应定位为 semantic/content alignment，不应过度声称逐字读心。                       | EEG-text / semantic boundary | Nature Neuroscience，强证据但非 EEG |
| [6]       | Ding 团队                   | 皮层活动可追踪 connected speech 中的层级语言结构。                                               | content token 不能只对齐 acoustic envelope，还需要 phoneme / syllable / word-like target。                   | Content alignment            | Nature Neuroscience，强证据         |
| [7]       | Belin 团队                  | 人类听觉皮层存在 voice-selective areas。                                                         | voice identity、timbre、speaker/style 应作为独立对齐目标，而不是 content 的附属标签。                        | Voice identity alignment     | Nature，强证据                      |
| [8]       | Anumanchipalli 团队         | 侵入式皮层记录可以通过 articulatory representation 合成 speech。                                 | 神经语音合成是可行方向，但该证据不能直接外推为非侵入式 EEG-to-waveform 已成熟。                              | Boundary / future decoder    | Nature，强证据但侵入式              |
| [9]       | Baevski 团队                | wav2vec 2.0 从未标注 speech 中学习 self-supervised speech representation。                       | audio semantic target 可以来自 SSL speech representation，而不是 waveform 本身。                             | Audio semantic token         | NeurIPS，强证据                     |
| [10]      | Chen 团队                   | BEATs 通过 acoustic tokenizer 产生语义丰富的离散 audio label。                                   | 非语音 auditory proxy 也可进入 audio token 视角，支持 auditory base pretraining。                            | Audio tokenizer              | ICML，强证据                        |
| [11]      | Radford 团队                | Whisper 通过大规模弱监督形成鲁棒 speech recognition / transcription model。                      | transcript 或 ASR output 可作为 content target 的 sanity check。                                             | Audio preprocessing          | ICML，强证据                        |
| [12]      | Ju 团队                     | NaturalSpeech 3 / FACodec 将 speech 拆成 content、prosody、timbre 和 acoustic detail。           | AudioTokenBundle 应采用 factorized token，而不是单一 entangled embedding。                                   | Audio tokenization           | ICML，强证据                        |
| [13]      | Kumar 团队                  | RVQGAN 将高维 audio 压缩为离散 token 以服务高保真 audio modeling。                               | codec token 适合作为 decoder backend，不适合作为 EEG 主监督。                                                | Codec boundary               | NeurIPS，强证据                     |
| [14]      | Le 团队                     | Voicebox 使用 text-guided speech infilling 和 audio context 做大规模 speech generation。         | 后续 decoder 可以把 EEG 预测出的高层条件转成声音，但这属于后阶段。                                           | Future decoder               | NeurIPS，强证据                     |
| [15]      | Li 团队                     | StyleTTS 2 将 style diffusion 和 large speech language model discriminator 用于 TTS。            | voice style 应作为 audio side 的独立因素，而不是混入 content。                                               | Voice/style token            | NeurIPS，强证据                     |
| [16]      | SEAMLESS Communication Team | SEAMLESSM4T 系列统一 speech/text translation 多任务系统。                                        | speech/text/audio 可以在统一系统中组织，但 EEG-Voice 需要先建立更小的 token interface。                      | System boundary              | Nature，强证据                      |
| [17]      | Hsu 团队                    | HuBERT 通过 hidden-unit prediction 学习 speech units。                                           | HuBERT-like units 是 content token extraction 的工程起点。                                                   | Engineering background       | IEEE/TASLP，基础系统                |
| [18]      | Zeghidour 团队              | SoundStream 是 neural audio codec 代表工作。                                                     | codec encoder/decoder 是未来 waveform backend 的工程基础。                                                   | Engineering background       | IEEE/TASLP，基础系统                |
| [19]      | Borsos 团队                 | AudioLM 将 audio generation 写成 semantic token 与 acoustic token 的语言建模。                   | EEG 预测高层 token 后，可由 audio token LM 补全 acoustic/detail token。                                      | Engineering background       | IEEE/TASLP，基础系统                |
| [20]      | Fang 团队                   | DASpeech 将 speech-to-speech translation 分成 linguistic decoder 与 acoustic decoder。           | speech-to-speech 系统本身也支持“内容层先行、声学层后渲染”的分层路线。                                      | Future decoder               | NeurIPS，强证据                     |
| [21]      | Yang 团队                   | UniAudio 用 LLM-style audio generation 支持 speech、sound、music、singing voice 等多任务。       | audio generation backend 可以统一多种 audio token，但 EEG-Voice 应先建立可解释条件接口。                     | Future decoder               | ICML，强证据                        |

## 1. EEG 侧：从连续脑电到可解释 Neural Token

### 1.1 任务定位：先做 representation retrieval，而不是 waveform regression

EEG 侧首先要处理的是信号边界。非侵入式 EEG 能捕获快速神经动态，但它从头皮电位间接观察皮层活动，容易受到肌电、眼电、设备噪声、参考方式、通道布局和个体差异影响。Défossez 团队没有把 EEG/MEG 直接回归到 waveform，而是把 3 s brain segment 与 speech representation 对齐，并用 contrastive retrieval 判断模型能否在大量候选 speech segment 中找回对应片段[1]。这个 framing 对 EEG-Voice 很关键：它把问题从“直接生成声音”改写为“脑信号表征是否和 speech/audio 表征共享可检索结构”[1]。

沿着这个方向，EEG-Voice 的第一阶段应采用 segment-level 或 token-level alignment。auditory base token 可以先承接 onset、envelope、broad auditory response；content token 再承接 phoneme、syllable、word-like 或 semantic unit；prosody token 承接 F0、energy、rhythm 和 intonation；voice identity token 承接 speaker、timbre、style 和 stream identity。这样的分层比直接预测 waveform 更符合非侵入式 EEG 的可观测能力，也更容易通过 Recall@K、phoneme accuracy、pitch correlation、speaker retrieval 和 held-out-device split 评估。

### 1.2 离散化：EEG token 是跨任务接口，不只是压缩码

Duan 团队的 DeWave 说明，EEG-to-text 不一定依赖 word-level marker 或 eye fixation segmentation；它可以通过离散 EEG encoding 把连续 EEG 映射到更接近语言模型输入的中间表示[2]。Jiang 团队的 NeuroLM 进一步把 EEG 视为可被 neural tokenizer 编码的“类语言”信号，并通过离散 token 与 LLM-style 多任务学习连接[3]。这两条线索共同支持一个更通用的判断：EEG token 不应只被当成 reconstruction code，而应被当成跨任务接口。它既要保留足够的时序和空间信息，又要支持 content probe、prosody probe、voice retrieval、mode classification 和 cross-dataset generalization[2][3]。

当前仓库的实现方向与这一判断一致。`EEGVoiceTokenizerV1` 已经实现 normalization / windowing、sensor-aware encoder、device context embedding、latent token former 和 grouped residual vector quantization。模型把功能 token 组织为 auditory base、content、prosody、voice identity、residual/noise 五类；其中 residual/noise 只服务弱重构和 nuisance diagnostics，不进入主要 alignment 或 retrieval 目标。这一点很重要：如果 residual/noise 也被用于 voice retrieval，模型可能学习到设备、数据集或噪声 shortcut，而不是稳定的 voice-relevant neural representation。

### 1.3 Sensor-aware 与 device-aware 是 EEG token 的必要前提

EEG 数据不是统一相机图像或统一采样文本。不同数据集的通道数、坐标系统、参考电极、采样率、设备类型和实验范式都可能不同。Xiao 团队的 BrainOmni 将 EEG/MEG 的 sensor layout、sensor type 和 recording device 纳入 BrainTokenizer 设计，说明跨设备 brain foundation model 不能忽略传感器结构[4]。这为 EEG-Voice 的 sensor-aware encoder 和 device context embedding 提供了直接依据：设备信息应作为 acquisition covariate 用来校正观测差异，而不应作为 content、prosody 或 speaker label 的捷径[4]。

从数据处理角度看，EEG 侧至少需要四层 pipeline。第一层是基础清理，包括重采样、滤波、bad channel 处理、artifact control 和统一时间轴；第二层是 epoching/windowing，把连续 EEG 切成与 audio frame 或 speech segment 可对齐的窗口；第三层是 sensor/device metadata 注册，记录 channel position、sensor type、montage、reference、sampling rate 和 device；第四层才是 tokenization，把 windowed EEG 送入 sensor-aware encoder 和 grouped quantizer。当前仓库已经具备模型侧接口，但真实 selected-dataset registry、real-data collator 和统一 target extraction 尚未完成。

### 1.4 内容、韵律、声音身份需要分开建模

Ding 团队显示，皮层活动可以追踪 connected speech 中的层级语言结构；这意味着 content token 不能只对齐 acoustic envelope，也需要 phoneme、syllable、word-like 或 semantic unit 作为监督和 probe[6]。Tang 团队在 fMRI 上展示了连续语言语义重建，但同时强调非侵入式语言解码依赖个体训练和受试者配合，因此 EEG-Voice 的 content 轴应表述为 semantic/content alignment，而不是逐字读心或无条件文本恢复[5]。这两个工作共同限定了 content token 的合理边界：它可以对齐语义和语言层级，但必须通过严格 split 和行为控制验证[5][6]。

Belin 团队显示人类听觉皮层存在 voice-selective areas，因此 speaker、timbre、style 应作为独立 voice identity 目标，而不是被合并进 content target[7]。这对 EEG-Voice 尤其重要，因为 voice image 并不只包含“说了什么”。同一句话可以由不同 speaker、timbre、emotion、speaking style 和 rhythm 表达；如果模型只对齐 text/content，就无法解释被试想象或感知到的“声音形象”。因此，EEG 侧 token 的目标不是单一语义向量，而是多轴信息结构：auditory base 负责听觉事件，content 负责语言内容，prosody 负责时间和音高动态，voice identity 负责说话人和音色风格，residual/noise 负责诊断不可解释残差。

## 2. Audio 侧：从 Waveform 到 AudioTokenBundle

### 2.1 Audio waveform 应拆成多层 token target

Audio 侧的核心判断是：waveform 不是适合作为 EEG 主监督的单一 target。Baevski 团队的 wav2vec 2.0 说明 speech representation 可以通过 self-supervised learning 从未标注语音中学习，并在低标注条件下服务 speech recognition[9]。Chen 团队的 BEATs 进一步把一般 audio 预训练改写为 acoustic tokenizer 产生离散 label 后的 masked prediction，说明 audio token 不只适用于 speech，也适用于更广义 auditory event[10]。Radford 团队的 Whisper 通过大规模弱监督形成鲁棒 transcription 能力，可以作为 content target 的 sanity check 或 transcript reference[11]。

这些工作共同说明，AudioTokenBundle 的 content 部分可以有多种来源。最直接的是 phoneme、syllable 或 transcript-derived label；更鲁棒的是 HuBERT-like hidden unit、wav2vec-like SSL embedding 或 BEATs-like acoustic token；更工程化的是 Whisper transcript 或 ASR confidence 作为检查信号[9][10][11][17]。对于 EEG-Voice，content target 不应绑定单一 tokenizer，而应保留多源字段：离散 semantic unit 用于 token-level classification，continuous SSL embedding 用于 contrastive alignment，transcript/phoneme 用于可解释评估。

### 2.2 Prosody 与 voice identity 不能被 content 吞掉

Ju 团队的 NaturalSpeech 3 / FACodec 明确把 speech 拆成 content、prosody、timbre 和 acoustic detail，这几乎正好对应 EEG-Voice 需要的 audio side 分层[12]。content 回答“说了什么”，prosody 回答“怎么说”，voice identity 回答“谁在说/是什么音色/是什么风格”，codec/detail 回答“怎样渲染成高保真 waveform”[12]。如果把这些信息压成一个 entangled embedding，EEG 侧即使对齐成功，也很难判断成功来自 content、speaker shortcut、dataset bias 还是声学残差。

Li 团队的 StyleTTS 2 和 Le 团队的 Voicebox 进一步说明，style、speaker prompt、audio context 和 infilling 对自然语音生成非常重要[14][15]。不过这些系统主要解决的是 decoder 或 renderer 问题，不等价于 EEG 主监督目标。换句话说，AudioTokenBundle 可以包含 style embedding、speaker embedding、timbre vector 和 codec code，但 EEG 第一阶段应优先学习可解释的高层条件：content、prosody 和 voice identity。codec/detail 可以作为 optional backend target 或未来 decoder input，不应主导当前 EEG token learning。

### 2.3 Codec token 是未来 backend，不是当前 EEG 主目标

Zeghidour 团队的 SoundStream 和 Kumar 团队的 RVQGAN 说明 neural codec 可以把高维 audio 压缩为离散 token，并服务高保真 audio modeling[13][18]。Borsos 团队的 AudioLM 说明 audio generation 可以被写成 semantic token 与 acoustic token 的语言建模问题，即先生成或条件化高层 token，再补全 acoustic/detail token[19]。这为 EEG-Voice 提供了清晰未来路线：EEG 侧先预测高层 AudioTokenBundle 条件，后续再由 audio token LM 或 codec decoder 完成 waveform rendering[13][18][19]。

但 full codec residual 不应成为 EEG 侧第一主监督。codec residual 包含相位、微小频谱纹理、录音条件、麦克风响应、房间效应、压缩细节和不可控噪声。非侵入式 EEG 很难稳定携带这些信息。若强行把 full codec residual 作为主目标，模型可能更容易学习 dataset/device shortcut，或者把重构 loss 降低为平均化声学纹理，而不是得到可解释 voice representation。因此，AudioTokenBundle 中的 codec codes 应标注为 optional path：未来用于 decoder backend、diagnostic reconstruction 或 ablation，而不是当前核心 claim。

### 2.4 Audio preprocessing 需要服务 token alignment

Audio side 的预处理也应围绕 alignment 设计，而不是只保存 waveform。一个合理的 AudioTokenBundle 至少应包含：semantic/content units、phoneme/transcript reference、F0、energy、rhythm、voiced mask、speaker/timbre/style embedding、optional codec codes、frame time、valid mask 和 source metadata。content units 可以来自 HuBERT 或 wav2vec-like models；prosody 可以来自 F0 tracker、energy contour 和 duration/rhythm features；speaker/timbre/style 可以来自 speaker embedding、style encoder 或 voice bank metadata；codec codes 可以来自 SoundStream/RVQGAN 类 tokenizer[9][12][13][17][18]。

SEAMLESS Communication Team 展示了 speech/text translation 多任务系统可以在统一框架中组织 ASR、speech-to-text、text-to-speech 和 speech-to-speech translation[16]。Yang 团队的 UniAudio 也展示了 large language model 风格的 universal audio generation 可以覆盖 speech、sound、music 和 singing voice 多任务[21]。Fang 团队的 DASpeech 把 speech-to-speech translation 拆成 linguistic decoder 和 acoustic decoder，也支持“先语言/内容，后声学渲染”的分层思想[20]。这些系统提示 EEG-Voice 可以预留 future decoder 接口，但当前更需要把 audio side 的 high-level target 抽出来，形成稳定、可解释、可评估的 AudioTokenBundle[16][20][21]。

## 3. EEG-Audio Alignment：从时间同步到分层检索

### 3.1 Alignment 的第一层是时间，而不是模型

EEG-audio alignment 的第一层是 time alignment。trigger、audio loopback onset、audio frame time、EEG sampling clock、neural lag allowance 和 epoch window 如果没有对齐，后面的 token target 会系统性错位。Défossez 团队的 3 s segment retrieval 能成立，是因为 brain segment 与 speech segment 有明确配对关系[1]。对 EEG-Voice 来说，最小可用样本不应只是 `eeg` 和 `audio_embedding`，还应包括 onset、duration、valid mask、speaker metadata、task condition、language、device 和 montage。没有这些元数据，模型即使训练成功，也难以解释成功来自真实 neural alignment 还是数据泄漏。

时间同步之后才是 representation alignment。auditory base 可以对齐 envelope、onset、spectral flux 或 low-level acoustic event；content 可以对齐 phoneme、syllable、HuBERT-like unit、wav2vec-like semantic embedding 或 transcript reference；prosody 可以对齐 F0、energy、rhythm 和 duration；voice identity 可以对齐 speaker embedding、timbre/style embedding 或 voice-bank candidate；residual/noise 只用于 reconstruction sanity check、device leakage diagnostics 和 ablation，不参与主要 voice alignment。

| EEG token group | Audio/voice target                                        | Alignment form                                          | Evaluation                                         |
| --------------- | --------------------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------- |
| auditory base   | envelope, onset, broad auditory event                     | regression / contrastive matching                       | onset accuracy, envelope correlation               |
| content         | phoneme, syllable, HuBERT-like unit, transcript reference | classification / sequence probe / contrastive alignment | phoneme accuracy, unit accuracy, content retrieval |
| prosody         | F0, energy, rhythm, duration                              | regression / binned classification                      | pitch correlation, rhythm accuracy                 |
| voice identity  | speaker, timbre, style, stream identity                   | contrastive retrieval / embedding regression            | Recall@K, MRR, held-out speaker retrieval          |
| residual/noise  | reconstruction residual, nuisance signal                  | diagnostic only                                         | token usage, leakage, device predictability        |

### 3.2 EEG-text 是 content 轴，不是 voice 轴

Duan 团队和 Jiang 团队说明 EEG token 可以连接文本或语言模型任务，但 EEG-text 只覆盖 EEG-Voice 的一个轴，即 content/semantic alignment[2][3]。对于 voice image 或 voice perception，“说了什么”并不等于“听到了谁的声音”。Belin 团队关于 voice-selective areas 的结果提示，speaker、timbre、style 有独立神经基础，不能被 content target 替代[7]。因此 EEG-Voice 应把 EEG-text 结果作为 content 证据，而不是把文本生成作为全部目标。

Tang 团队的语义重建工作也提示需要谨慎表述。非侵入式语言解码可以恢复语义层面的信息，但该工作基于 fMRI、个体训练和受试者合作，不应被简单改写成 EEG 可以无条件逐字恢复语言[5]。对 EEG-Voice 来说，更稳妥的 claim 是：content token 可以尝试对齐语义和语言单位；prosody token 可以尝试对齐音高、能量和节奏；voice identity token 可以尝试对齐 speaker/timbre/style embedding；最终通过检索、probe 和 split evaluation 判断每个轴是否真的存在可泛化信息[5][7]。

### 3.3 当前仓库已实现的是模型骨架，不是真实数据闭环

当前仓库已经实现 `EEGVoiceTokenizerV1`、grouped residual vector quantization、content/phoneme heads、pitch/prosody heads、timbre/style heads、speaking-mode adapter、voice retrieval head、memory queue hard negatives、reconstruction losses 和 synthetic tests。这说明模型接口已经能够表达 token-level alignment 的方向：输入端接收 EEG、sensor metadata、dataset/language/domain metadata 和 device context；target 端支持 content labels、phoneme labels、pitch/prosody targets、timbre/style targets、mode labels 和 audio embedding；输出端可以返回 grouped tokens、attribute heads、retrieval logits 和 residual diagnostics。

但这仍然不是完整 EEG-to-voice 系统。真实 selected-dataset registry 尚未建立，real-data collator 尚未把本地 EEG/audio 样例转为统一 `EEGVoiceBatch`，AudioTokenBundle extraction 尚未实现，phoneme、F0、prosody、style、speaker/audio embedding 等 target 还没有统一提取管线，training loop 和真实评估脚本也尚未完成。因此，报告中应把“已实现模型 skeleton”与“文献支持的未来路线”分开：前者是工程接口，后者是研究设计，二者都不能被写成“已经从 EEG 生成声音”。

### 3.4 Candidate voice retrieval 是当前最合适的检验目标

在当前阶段，candidate voice retrieval 比 waveform generation 更适合作为主实验。给定一个 EEG segment，模型可以在候选 audio/voice embeddings 中检索匹配项；评价指标可以使用 Recall@1/5/10、MRR、within-dataset retrieval、cross-dataset retrieval、held-out speaker retrieval 和 held-out device retrieval。Défossez 团队的 segment retrieval framing 已经证明这种评估方式适合非侵入式 brain-to-speech representation 对齐[1]。相比 waveform MOS 或听感评分，retrieval 更容易做 hard negative、更容易控制 split，也更容易判断 voice identity token 是否真的携带 speaker/timbre/style 信息。

未来如果接入 AudioLM、Voicebox、UniAudio 或其他 audio foundation decoder，较合理的路线是：EEG token 先预测 content/prosody/voice identity 条件，audio token LM 再补全 codec/detail token，最后由 neural codec decoder 渲染 waveform[14][19][21]。Anumanchipalli 团队的侵入式语音合成工作能证明 neural speech synthesis 是重要方向，但其信号来源和任务设置不同，不能直接作为非侵入式 EEG 高保真生成的证据[8]。因此，当前 EEG-Voice 的科学表述应是“可解释 EEG-to-voice token interface”，不是“已完成 EEG-to-waveform generator”。

## 4. 研究定位与下一步缺口

综合上述文献，EEG-Voice 的合理路线是三步。第一步，建立 EEG discrete token foundation：把连续 EEG 通过 sensor-aware encoder 和 grouped quantization 转为可解释 token，并验证 auditory base、content、prosody、voice identity、residual/noise 是否具有不同功能[2][3][4]。第二步，建立 AudioTokenBundle alignment：从 audio side 提取 semantic/content units、prosody features、speaker/timbre/style embedding 和 optional codec codes，并让 EEG token 与这些目标分层对齐[9][10][12][17]。第三步，在前两步可靠之后，才考虑 audio token completion 和 waveform rendering，让 decoder backend 负责高保真细节[13][14][18][19][21]。

当前仓库最重要的工程缺口不是再增加复杂 decoder，而是补齐真实数据链路。下一阶段应优先建立 selected-dataset registry，记录每个数据集的语言、任务、speaker、device、montage、sampling rate、channel layout、trigger/loopback 方式和可用 target。随后实现 real-data collator，把 EEG window 与 audio item、onset、duration、speaker metadata 和 task condition 对齐。第三步实现 AudioTokenBundle extraction，至少先跑通 content units、F0/energy、basic timbre/speaker embedding 和 frame-level masks。最后补真实评估脚本，包括 content accuracy、prosody correlation、speaker/voice retrieval Recall@K、hard-negative retrieval、token usage、residual leakage diagnostics 和 held-out-device split。

最终，EEG-Voice 的贡献应表述为：构建一个可解释的 EEG-to-voice token interface。EEG token 是否保留 auditory base、content、prosody 和 voice identity 信息；audio token 是否提供稳定、可分解的监督目标；二者是否能在 controlled split、hard negatives 和 held-out evaluation 中稳定对齐。这一定位既符合非侵入式 EEG 的信号边界，也为未来接入 AudioLM、Voicebox、UniAudio 或其他 audio foundation decoder 留出清晰接口[1][13][14][19][21]。

## References

[1] Défossez, A., Caucheteux, C., Rapin, J., Kabeli, O., & King, J.-R. (2023). Decoding speech perception from non-invasive brain recordings. *Nature Machine Intelligence, 5*, 1097-1107. https://doi.org/10.1038/s42256-023-00714-5

[2] Duan, Y., Zhou, J., Wang, Z., Wang, Y.-K., & Lin, C.-T. (2023). DeWave: Discrete encoding of EEG waves for EEG to text translation. In *Advances in Neural Information Processing Systems, 36*. https://papers.nips.cc/paper_files/paper/2023/hash/1f2fd23309a5b2d2537d063b29ec1b52-Abstract-Conference.html

[3] Jiang, W.-B., Wang, Y., Lu, B.-L., & Li, D. (2025). NeuroLM: A universal multi-task foundation model for bridging the gap between language and EEG signals. In *International Conference on Learning Representations*. https://proceedings.iclr.cc/paper_files/paper/2025/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html

[4] Xiao, Q., Cui, Z., Zhang, C., Chen, S., Wu, W., Thwaites, A., Woolgar, A., Zhou, B., & Zhang, C. (2025). BrainOmni: A brain foundation model for unified EEG and MEG signals. In *Advances in Neural Information Processing Systems, 38*. https://nips.cc/virtual/2025/poster/117066

[5] Tang, J., LeBel, A., Jain, S., & Huth, A. G. (2023). Semantic reconstruction of continuous language from non-invasive brain recordings. *Nature Neuroscience, 26*, 858-866. https://doi.org/10.1038/s41593-023-01304-9

[6] Ding, N., Melloni, L., Zhang, H., Tian, X., & Poeppel, D. (2016). Cortical tracking of hierarchical linguistic structures in connected speech. *Nature Neuroscience, 19*, 158-164. https://doi.org/10.1038/nn.4186

[7] Belin, P., Zatorre, R. J., Lafaille, P., Ahad, P., & Pike, B. (2000). Voice-selective areas in human auditory cortex. *Nature, 403*, 309-312. https://doi.org/10.1038/35002078

[8] Anumanchipalli, G. K., Chartier, J., & Chang, E. F. (2019). Speech synthesis from neural decoding of spoken sentences. *Nature, 568*, 493-498. https://doi.org/10.1038/s41586-019-1119-1

[9] Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. In *Advances in Neural Information Processing Systems, 33* (pp. 12449-12460). https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html

[10] Chen, S., Wu, Y., Wang, C., Liu, S., Tompkins, D., Chen, Z., Che, W., Yu, X., & Wei, F. (2023). BEATs: Audio pre-training with acoustic tokenizers. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR, Vol. 202, pp. 5178-5193). https://proceedings.mlr.press/v202/chen23ag.html

[11] Radford, A., Kim, J. W., Xu, T., Brockman, G., Mcleavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR, Vol. 202, pp. 28492-28518). https://proceedings.mlr.press/v202/radford23a.html

[12] Ju, Z., Wang, Y., Shen, K., Tan, X., Xin, D., Yang, D., Liu, E., Leng, Y., Song, K., Tang, S., Wu, Z., Qin, T., Li, X., Ye, W., Zhang, S., Bian, J., He, L., Li, J., & Zhao, S. (2024). NaturalSpeech 3: Zero-shot speech synthesis with factorized codec and diffusion models. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR, Vol. 235, pp. 22605-22623). https://proceedings.mlr.press/v235/ju24b.html

[13] Kumar, R., Seetharaman, P., Luebs, A., Kumar, I., & Kumar, K. (2023). High-fidelity audio compression with improved RVQGAN. In *Advances in Neural Information Processing Systems, 36*. https://papers.nips.cc/paper_files/paper/2023/hash/58d0e78cf042af5876e12661087bea12-Abstract-Conference.html

[14] Le, M., Vyas, A., Shi, B., Karrer, B., Sari, L., Moritz, R., Williamson, M., Manohar, V., Adi, Y., Mahadeokar, J., & Hsu, W.-N. (2023). Voicebox: Text-guided multilingual universal speech generation at scale. In *Advances in Neural Information Processing Systems, 36*. https://papers.nips.cc/paper_files/paper/2023/hash/2d8911db9ecedf866015091b28946e15-Abstract-Conference.html

[15] Li, Y. A., Han, C., Raghavan, V. S., Mischler, G., & Mesgarani, N. (2023). StyleTTS 2: Towards human-level text-to-speech through style diffusion and adversarial training with large speech language models. In *Advances in Neural Information Processing Systems, 36*. https://proceedings.neurips.cc/paper_files/paper/2023/hash/3eaad2a0b62b5ed7a2e66c2188bb1449-Abstract-Conference.html

[16] SEAMLESS Communication Team. (2025). Joint speech and text machine translation for up to 100 languages. *Nature, 637*, 587-593. https://doi.org/10.1038/s41586-024-08359-z

[17] Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A. (2021). HuBERT: Self-supervised speech representation learning by masked prediction of hidden units. *IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29*, 3451-3460. https://doi.org/10.1109/TASLP.2021.3122291

[18] Zeghidour, N., Luebs, A., Omran, A., Skoglund, J., & Tagliasacchi, M. (2022). SoundStream: An end-to-end neural audio codec. *IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30*, 495-507. https://doi.org/10.1109/TASLP.2021.3129994

[19] Borsos, Z., Marinier, R., Vincent, D., Kharitonov, E., Pietquin, O., Sharifi, M., Roblek, D., Teboul, O., Grangier, D., Tagliasacchi, M., & Zeghidour, N. (2023). AudioLM: A language modeling approach to audio generation. *IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31*, 2523-2533. https://doi.org/10.1109/TASLP.2023.3288409

[20] Fang, Q., Zhou, Y., & Feng, Y. (2023). DASpeech: Directed acyclic transformer for fast and high-quality speech-to-speech translation. In *Advances in Neural Information Processing Systems, 36*. https://proceedings.neurips.cc/paper_files/paper/2023/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html

[21] Yang, D., Tian, J., Tan, X., Huang, R., Liu, S., Guo, H., Chang, X., Shi, J., Zhao, S., Bian, J., Zhao, Z., Wu, X., & Meng, H. M. (2024). UniAudio: Towards universal audio generation with large language models. In *Proceedings of the 41st International Conference on Machine Learning* (PMLR, Vol. 235, pp. 56422-56447). https://proceedings.mlr.press/v235/yang24x.html
