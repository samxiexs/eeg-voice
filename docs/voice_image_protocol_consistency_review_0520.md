# Voice-Image EEG Protocol 一致性与可执行性核查（0520）

## 核查对象

源文档：`docs/voice_image_eeg_self_collection_protocol_0520.md`

核查口径：

- P1 = 会导致数据不可用或 pilot 难以启动的严重问题。
- P2 = 会增加分析复杂度、降低效力或影响投稿说服力的中等问题。
- P3 = 措辞模糊、实现细节待确认或文档可读性问题。

本核查优先检查数字一致性、术语一致性、任务时序、缺失步骤、统计可执行性和 BIDS 合规性。

---

## P1 严重问题

| 优先级 | 位置 | 问题描述 | 建议修复 |
| --- | --- | --- | --- |
| P1 | 4.4, 5.2, 14 | Voice bank 规模没有明确区分“唯一录音 item”“派生 manipulation item”“每名受试者实际评分/听到的 item”。4.4 写 Core natural 300-500，另列多个 90-180 的 derived sets；14 写最小 240-360 items；5.2 写每人评分 200 items。若这些 set 被理解为互不重叠，总刺激规模会超过 pilot 能承受的范围。 | 增加一个 `Stimulus accounting` 小节，明确定义 `source_recording_id`、`voice_item_id`、`derived_item_id`、`analysis_set_id`。规定 pilot master bank = 240-360 derived items，main master bank = 600-900 derived items；每名受试者只抽样评分 200 items。 |
| P1 | 5.1, 5.3 | Healthy main 总时长写 2.5-3.5 小时，但流程加总为 consent 10-15 + Part A 35-45 + setup 25-45 + Part B 90 + debrief 5-10 = 165-205 分钟，尚未计入任务说明、cap 调整、休息、失败 trial 和 B4 多候选呈现。实际很容易超过 3.5 小时并造成疲劳 artifact。 | 把 main 默认改为两 session：Session A = screening + Part A + EEG setup + B1/B2；Session B = B3/B4/B5 或 B4/B5 + retest subset。若坚持单日，pilot 必须压缩到 B1/B2/B4 + shortened B3，总有效 EEG <=75 分钟。 |
| P1 | 4.6, 15.1 | AudioTokenBundle 目前只列字段，没有可执行的 stimulus generation pipeline。F0/formant/style/spatial/control 如何生成、质检和版本化没有步骤说明，会导致不同批次刺激不可复现。 | 增加 `Stimulus generation pipeline`：recording -> trim -> loudness normalization -> F0/formant/style/spatial transforms -> objective QC -> human listening QC -> feature/token extraction -> frozen manifest。明确工具优先级，例如 Praat/Parselmouth 或 WORLD 做 F0/formant，pyloudnorm/EBU R128 做响度。 |
| P1 | 6.2, 15.2 | Audio loopback 阈值定为 corrected onset error <5 ms，但没有写 pilot 前的硬件校准测试。消费级音频链路即使用 TTL，也可能存在 buffer jitter；如果只记录 TTL 而 loopback 不稳定，EEG-audio 对齐会失效。 | 增加 `Synchronization validation`：采集前用 100-200 个短 click/pulse 事件估计 TTL-loopback delay、jitter、dropout；pilot 允许 `<10 ms` 作为启动阈值，main 冻结为 `<5 ms` 目标。 |
| P1 | 7.3 | `candidate_voice_ids` 和 `avh_top5_voice_ids` 是列表型字段，但 TSV 单元格只能是字符串。若直接写 Python list，会破坏解析和 BIDS validator 兼容性。 | 规定列表字段使用 `|` 分隔字符串，例如 `v001|v014|v032|v077`，并在 `events.json` 中写明 `Delimiter: "|"`, `Levels` 或 `Description`。复杂对象移入 `beh/*.jsonl` 或 derivatives。 |

---

## P2 中等问题

| 优先级 | 位置 | 问题描述 | 建议修复 |
| --- | --- | --- | --- |
| P2 | 3.1 | Healthy main n=80/effective n>=60、Clinical AVH n=30/effective >=20 没有独立 power analysis 或 precision justification。顶刊审稿会要求说明这是 formal power、prior-study convention 还是 feasibility-driven sample size。 | 新增 `Power and precision rationale` 章节。若没有 formal power，明确写：no formal power calculation was conducted；健康组样本量基于 Défossez 等非侵入式 speech decoding、大样本 EEG speech datasets 和预期 dropout；clinical AVH 为 feasibility + hierarchical model precision。 |
| P2 | 3.2, 5.1 | Healthy inclusion 写“能完成 2 小时内听觉任务”，但协议总流程可能 3 小时以上。 | 改为“能完成单次不超过 2 小时的任务段；完整实验可拆分为 2 次 session”。 |
| P2 | 4.1, 4.5, 15.1 | 12 名真实说话人的招募、录音 consent、声音版权、重录标准、口音/音域质控没有写。 | 增加 `Speaker recruitment and recording QC`：说话人知情同意、可公开/受限授权、噪声底、峰值、LUFS、F0 range、发音错误重录规则。 |
| P2 | 4.3, 4.4 | F0-only、formant-only、style-only set 的 counterbalancing 没有写。若 F0 操控总是来自固定 speaker/content，会形成 stimulus shortcut。 | 增加 Latin-square 或 balanced incomplete block 规则：每个 content/speaker 在 F0/formant/style 条件中出现次数近似平衡，trial order subject-wise randomized。 |
| P2 | 4.5, 5.2, 9.2 | `voice item`、`voice_id`、`voice prototype`、`voice profile`、`voice manifold` 未统一定义。模型和行为任务里这些词承担不同层级，后续写代码会混淆。 | 增加术语表：`voice_item_id` = 一条具体刺激；`voice_id` = item 或 derived item 主键；`voice_prototype_id` = subject/clinical matching 的代表点；`voice_profile` = 属性向量；`voice_manifold` = subject rating + audio embedding 的低维空间。 |
| P2 | 5.3 | B4 Voice Retrieval 的 120 trials 实际包含 target + 4 candidates，音频呈现次数是 600 段。它对总时长、疲劳和 auditory adaptation 的影响没有计入。 | 把 B4 单独估时：每 trial 约 10-14 s，120 trials 约 20-30 分钟。pilot 可降到 72-96 trials，保留 hard-negative 平衡。 |
| P2 | 5.3.6, 6.1, 8.2 | B5 的 motor suppression check 依赖 jaw/neck EMG，6.1 写 required，但没有说明 EMG placement、阈值或 exclusion rule。 | 写明 jaw/neck EMG 电极位置、采样方式、EMG bandpower 阈值、B5/AVH primary analysis 的 EMG exclusion rule。 |
| P2 | 7.3 | AVH episode 同时作为 `events.tsv` 字段 `avh_onset/avh_offset`，又可被表达为独立 event row。当前方案容易产生重复编码。 | 采用两层编码：`trial_type=avh_episode` 的独立 rows 使用 `onset/duration` 表示 episode；post-episode rating rows 用 `avh_episode_id` 关联。 |
| P2 | 9.1, 9.2 | `q0-q7 grouped RVQ` 在协议中既像既有模型事实，又像实验假设。需要明确数据采集不依赖训练出 q0-q7，模型分析阶段才使用。 | 在 9.1 开头加一句：采集协议不要求在线 tokenization；q0-q7 是离线模型分析接口，现有代码已有 skeleton，但真实数据训练仍需后续实现。 |
| P2 | 10.1, 11.3 | Primary endpoints 与统计公式不是一对一映射。例如 prosody prediction、style classification、AVH prototype retrieval 的主模型、固定效应和 correction family 未逐项绑定。 | 建立 `Endpoint-to-analysis mapping` 表：每个 endpoint 对应数据 block、metric、primary contrast、statistical model、multiple-comparison family、success criterion。 |
| P2 | 11.3 | Pilot n=12 有效样本下，`(1|speaker) + (1|content)` 的 mixed-effects 模型可能估计不稳，尤其 hard-negative 类型多时。 | Pilot 统计降级为 descriptive + subject-level bootstrap；main 才使用 full mixed-effects。pilot 可用 speaker/content fixed blocking 或 cluster bootstrap。 |
| P2 | 11.3 | Clinical AVH 每人 episode 数量不定，n=20 有 episode 不保证 hierarchical mixed-effects 有足够功效。 | Clinical primary endpoint 应先写为 feasibility/within-subject rank improvement；confirmatory model 需要最低 episode 数，例如 total usable episodes >=80 且 >=10 subjects 有 >=3 episodes。 |
| P2 | 15.2 | BIDS validator 只写在 checklist，但自定义列和 derivatives 需要 sidecar JSON schema；否则验证可通过但语义不可复用。 | 增加 `events.json`, `participants.json`, `voice_bank_metadata.json` 的字段字典要求。所有 custom columns 必须有 Description、Units、Levels 或 Delimiter。 |

---

## P3 细节问题

| 优先级 | 位置 | 问题描述 | 建议修复 |
| --- | --- | --- | --- |
| P3 | 2.1 | 文献证据链整体合理，但缺少对 EEG spatial resolution 和 non-invasive decoding 上限的专门约束。 | 加一行文献/约束：EEG 不能精确定位 STS voice areas，主要通过 sensor-level temporal decoding 和 source-localization exploratory analysis 支撑。 |
| P3 | 2.1, 9.1 | 缺少 RVQ/VQ-VAE 或 discrete neural token 在 EEG/brain foundation models 中的应用先例的明确文献位置。 | 在证据链中加入 DeWave、NeuroLM、BrainOmni/LUNA 或 VQ-VAE/RVQ 基础引用，区分“神经信号 tokenization 先例”和“audio codec tokenization 先例”。 |
| P3 | 4.2 | 四类内容层级各 20-30，总数可能为 80-120，但 4.1 固定写 80 contents。 | 改成“总计 80 contents，其中各层级约 20 个；pilot 可只保留每类 10 个”。 |
| P3 | 4.4 | Control set 写 non-vocal environmental sounds，但最小版本只列 speech-shaped noise、scrambled voice、silence。 | 最小版本补上 non-vocal sounds，或说明 non-vocal 为 main-only。 |
| P3 | 5.2 | `rating_speaker_similarity` 写“与参照 voice prototype 的相似度”，但 Part A 尚未定义 prototype 如何出现。 | 改为 pairwise similarity 或 anchor voices；若无 anchor，删除该评分或放到 AVH matching 阶段。 |
| P3 | 5.3.2 | B1 catch question 只写 20%-25%，没有规定 catch 类型和正确答案平衡。 | 写明 catch 包括 target syllable/word presence，yes/no 平衡，不能总是同一 response。 |
| P3 | 5.3.3 | `matched-acoustic speaker` 需要说明 matching 方法，否则很难生成。 | 定义 matching features：duration、LUFS、F0 mean、energy envelope correlation，并设置 tolerance。 |
| P3 | 6.4, 8.3 | Impedance 用 kOhm 纯 ASCII，但 EEG 厂商常用 kΩ；文档可保持 ASCII，但需统一。 | 保持 `kOhm` 全文一致，避免混用。 |
| P3 | 7.4 | `TRIG` 和 `AUDIO` 作为 channel type 需和具体 BIDS validator 支持的 type 列表核对。 | 若 validator 不接受，使用 `MISC` 并在 `description` 中标注 trigger/audio loopback。 |
| P3 | 8.1 | `optional high-gamma not used for scalp primary claim` 容易被误读为 scalp EEG 也要分析 high-gamma。 | 改成“no high-gamma primary analysis for scalp EEG; any high-frequency EMG-related analysis is artifact/QC only”。 |
| P3 | 10.3 | `q7-in-head ablation` 表述很强，但 q7-in-head 是人为违反设计约束的 negative control。 | 改为“q7 leakage / q7-in-head negative-control ablation”。 |
| P3 | 13.1 | 第一篇主论文 title 含 `Reconstruction`，但 primary endpoint 是 retrieval/alignment。 | 为避免审稿人认为 waveform reconstruction，title 可改为 `Voice-Image Retrieval` 或 `Voice-Image Reconstruction in a Subjective Voice Manifold`。 |

---

## BIDS 合规性说明

BIDS 1.11.1 的 `events.tsv` 允许任意数量 additional columns，但 `trial_type` 和所有 additional columns SHOULD 在 accompanying JSON sidecar 中描述。也就是说，自定义字段不需要强制加 `X_` 前缀；真正的风险是字段没有 sidecar 描述、列表值没有 delimiter 规则、复杂对象直接塞进 TSV 单元格。

参考：BIDS events specification 1.11.1: https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/events.html

---

## 总体评估

协议科学方向清晰，但 pilot 前必须先修 stimulus accounting、session 时长、audio pipeline、同步验证和 TSV 列表编码。否则数据可解释性和复用性会受影响。

