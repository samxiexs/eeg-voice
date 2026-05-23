# EEG 与 Voice Token 建模关系说明图

这是一张概念设计图，不是当前实验结果图。它只想表达一个核心思路：左边从 EEG 提取神经表征，右边从 audio 提取声音表征，中间做 alignment，最后得到一组可解释的 voice-related tokens。

说明图已经单独放在 PowerPoint 文件中：

- [`eeg_voice_token_connection_overview_0523.pptx`](eeg_voice_token_connection_overview_0523.pptx)

PPT 中的图采用三列结构：

- 左边：EEG 如何经过 preprocessing、epoching、sensor-aware encoder，形成 EEG functional tokens。
- 右边：audio waveform / voice bank 如何经过 frozen audio tokenizers，形成 AudioTokenBundle 和 audio / voice tokens。
- 中间：说明 EEG token 与 audio token 的对齐关系，包括 content、prosody、voice identity 和 candidate retrieval。

## Token 对应关系

| 功能 token | EEG 侧含义 | Voice/audio 侧对应目标 | 主要用途 |
| --- | --- | --- | --- |
| auditory base | 声音出现、包络、早期听觉反应 | envelope、onset | 时间校正和基础听觉对齐 |
| content | 语音内容相关神经表征 | phoneme、syllable、HuBERT-like unit | 判断 EEG 是否携带“说了什么” |
| prosody | 音高、能量、节奏相关表征 | F0、energy、rhythm | 判断 EEG 是否携带“怎么说” |
| voice identity | 说话人、音色和风格相关表征 | speaker、timbre、style embedding | 做 voice alignment 和候选声音检索 |
| residual / noise | 难以解释的剩余变化 | 不作为主要 audio 对齐目标 | 只做重构诊断和噪声检查 |

## 设计边界

当前设计重点是建立 EEG 与 audio 之间的 token-level connection：content 对齐语音内容，prosody 对齐音高和节奏，voice identity 对齐说话人、音色和风格。

这张图不表示已经能从 EEG 直接生成自然、高保真的 waveform。高保真声音生成属于后续 decoder 阶段；当前图只说明两边 token 如何联系起来。
