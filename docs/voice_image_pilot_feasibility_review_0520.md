# Voice-Image EEG Pilot 可执行性评估（0520）

## 评估前提

当前假设资源：

- 1 名 EEG 操作员 + 1 名实验助手。
- EEG 设备已到位：64ch，1000 Hz，支持 TTL trigger。
- Python 环境可用：MNE-Python、librosa、transformers/HuBERT。
- 无专业录音室，但有安静房间 + 一支电容麦。
- 目标：3 个月内完成 healthy pilot，n=16。

结论先行：可以启动 pilot，但不能按 full protocol 单日完整执行。必须先冻结一个 pilot-minimum design，并在第 1 周完成音频与同步技术验证。

---

## 阻塞项列表

| 问题 | 当前状态 | 解决方案建议 | 是否必须 Day 1 前解决 |
| --- | --- | --- | --- |
| 12 名说话人招募和录音授权 | 未在协议中具体化 | 先招募 6-8 名说话人做 technical pilot；正式 pilot 若要检验 speaker/timbre，至少需要 8-12 名。必须签声音使用授权，明确是否可公开。 | 是，如果 Day 1 指第一名 EEG 受试者 |
| 录音环境无专业录音室 | 安静房间 + 电容麦 | 可作为 pilot 非阻塞项，但要固定麦克风距离、采样率、增益、房间、噪声底；每条录音记录 `microphone_chain` 和 `noise_floor_db`。主实验前建议升级录音条件。 | 否，但录音 SOP 必须 Day 1 前完成 |
| F0/formant shift 工具未指定 | 未定 | 选择一条固定路线：Praat/Parselmouth 或 WORLD vocoder。librosa 可做辅助特征提取，但不建议作为高质量 formant manipulation 的唯一工具。 | 是 |
| Loudness normalization 未落地 | 只写原则 | 用 pyloudnorm / EBU R128 生成 normalized wav，保留 LUFS、peak、clip flag；写入 `voice_bank_features.tsv`。 | 是 |
| 音频人工质检标准缺失 | 未定 | 每条刺激至少由 2 人听检：发音正确、无爆音、无明显房间噪声、无剪切、manipulation 不失真。 | 是 |
| AudioTokenBundle pipeline | 字段已定义，脚本未定义 | Pilot Day 1 前不必全部跑通 codec codes，但必须跑通 manifest、F0/energy、MFCC、HuBERT content units 和 basic speaker embeddings。 | 部分是 |
| TTL + audio loopback 校准 | 未见技术验证结果 | 在正式 EEG subject 前做 hardware sync test：100-200 个 click/pulse，估计 TTL-loopback delay、jitter、dropout。 | 是 |
| SPL calibration | 只写 60-70 dB SPL | 若没有声级计/耳 coupler，至少使用固定设备音量和主观舒适度记录；但主实验/IRB 最好配声级计。 | 是，至少要有替代校准流程 |
| 刺激呈现软件 | 未指定 | 固定 PsychoPy/Psychtoolbox/Python presentation stack，要求写出 TTL marker、audio path、event log、crash recovery。 | 是 |
| BIDS writer | checklist 有要求，脚本未定 | Day 1 前至少保证 raw event log 可无损转换为 BIDS；正式 BIDS writer 可在前 2 名 subject 后补齐。 | 否，但原始日志格式必须 Day 1 前冻结 |
| jaw/neck EMG | 协议要求，但资源只说明 EEG 64ch | 确认设备是否可把 EMG 作为 aux channels。若不行，B5 Imagined Voice 只能 exploratory，不能作为主分析。 | 是，如果运行 B5 |
| 受试者排班 | 3 个月 n=16 | 需要每周 2 名完成两 session 或每周 3 名完成单 session；考虑 dropout，建议招募 20 人。 | 否，但需第 1 周排班 |

---

## EEG 同步风险评估

### 风险判断

消费级音频链路的未校正 latency 往往不稳定，TTL marker 只能标记程序发出命令或声卡 buffer 写入时间，不等于实际声波到达耳机时间。若 audio loopback 被同步采进 EEG，后处理后达到接近 5 ms 的 onset precision 是可争取的；若只依赖 TTL，<5 ms 不可信。

### 必做测试

1. 生成 100-200 个 click stimuli，随机 ISI，完整走正式 presentation stack。
2. 同时记录 TTL channel 和 audio loopback channel。
3. 计算每个 trial 的 TTL-to-loopback delay、delay SD、missing TTL、missing loopback、audio onset detection failure。
4. 分别测试短音、真实 voice item、B4 target-candidate sequence。
5. 测试长 block 下是否有 buffer drift 或 dropped audio。

### Pilot 阈值建议

| 阶段 | 阈值 |
| --- | --- |
| Technical pilot | corrected onset error <10 ms 可接受；目标 <5 ms |
| Healthy pilot | run-level mean corrected error <10 ms；delay SD 尽量 <5 ms |
| Main study | target <5 ms；>10 ms run flagged or excluded |

如果 pilot 第一周无法稳定 <10 ms，应暂停 EEG subject 采集，先修 presentation/audio chain。

---

## 受试者时间负担

### Full protocol 单日估算

| 模块 | 保守估计 |
| --- | ---: |
| Consent/screening | 15-25 min |
| Part A ratings, 200 items | 35-50 min |
| 64ch EEG setup + aux + loopback | 45-60 min |
| Instructions and practice | 10-15 min |
| B1 Content, 160 trials | 15-20 min |
| B2 Timbre/Speaker, 180 trials | 25-35 min |
| B3 Prosody/Style, 180 trials | 25-35 min |
| B4 Retrieval, 120 trials with 4 candidates | 25-35 min |
| B5 Imagery, 120 trials | 20-30 min |
| Breaks and troubleshooting | 20-30 min |
| Debrief | 5-10 min |

实际总时长约 4.1-5.0 小时。单日完整执行不适合 n=16 pilot，会显著增加疲劳和肌电 artifact。

### 推荐 pilot session 拆分

#### 方案 A：两次到场，质量优先

| Session | 内容 | 预计时长 |
| --- | --- | ---: |
| Session 0/1 | consent + hearing + Part A ratings + optional cap size check | 60-80 min |
| Session 2 | EEG setup + B1 + B2 + B4 + shortened B3 | 150-180 min |

说明：B5 不进入首轮 pilot primary endpoint；若要测试 B5，只在最后加 30-40 个 trials。

#### 方案 B：两次 EEG，到位但更费排班

| Session | 内容 | 预计时长 |
| --- | --- | ---: |
| Session 1 | consent + Part A + EEG setup + B1 + B2 | 180-210 min |
| Session 2 | EEG setup + B3 + B4 + shortened B5 | 150-180 min |

说明：科学完整性更好，但 3 个月内 n=16 需要更强排班能力。

### Pilot 推荐

在当前 1 名 EEG 操作员 + 1 名助手条件下，采用方案 A。第一轮 pilot 成功标准应聚焦 B1/B2/B4：content、timbre/speaker、voice retrieval 和同步质量。

---

## 数据处理最小闭环

Pilot 结束后 2 周内，应只跑最小闭环，不要等待完整 foundation model 训练。

1. **Manifest and synchronization QC**  
   生成 `voice_bank_metadata.tsv`、event log、TTL-loopback delay report；剔除同步失败 runs。

2. **EEG preprocessing and epoch QC**  
   用 MNE-Python 跑 0.1-40 Hz、notch、bad channel、ICA/SSP、epoch、artifact rejection；输出 valid epoch ratio、bad channel ratio、catch accuracy。

3. **Audio features and lightweight tokens**  
   提取 F0/energy/MFCC/formants/HuBERT content units/basic speaker embedding；不等待 codec codes。

4. **Baseline alignment checks**  
   跑 envelope-only、content-only、speaker/timbre retrieval 的轻量 baseline，例如 ridge/CCA/simple contrastive embedding；先不训练完整 q0-q7 model。

5. **Pilot go/no-go metrics**  
   报告 corrected onset error、valid epoch ratio、voice rating ICC、catch accuracy、B4 behavioral accuracy、EEG-to-voice retrieval 是否高于 chance 的初步结果。

---

## Go/No-Go 标准

### Go

满足以下条件即可进入 main protocol refinement：

- 受试者完成率 >=75%。
- corrected onset error run-level <10 ms，目标接近 <5 ms。
- valid epoch ratio >=70%。
- catch accuracy >=70%。
- Part A voice rating ICC >=0.60 或可通过删除低质量 items 达到。
- B4 行为 retrieval 高于 chance，说明任务可理解。
- 至少一个 EEG baseline 在 B1/B2/B4 中显示高于 shuffled baseline 的趋势。

### No-Go / Pause

任一条件满足则暂停扩招：

- audio loopback 丢失或 jitter 无法校正。
- 任务单次时长超过 3.5 小时且 artifact 明显升高。
- voice bank 评分稳定性低，无法构建 subject-level voice manifold。
- B4 行为准确率接近 chance，说明候选设计或任务说明失败。
- EEG aux channels 无法记录 EMG，却计划把 B5/imagery 作为主结论。

---

## 总体建议

当前资源足以启动 healthy pilot，但必须先完成音频生成、同步校准和 session 缩减。建议首轮只把 B1/B2/B4 作为 primary pilot，B3 缩短，B5 仅做技术测试；clinical AVH 不进入本轮 pilot。

