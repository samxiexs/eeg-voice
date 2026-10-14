# NeuroSpeech v2.1 进展报告：数据阶段（A0）

日期：2026-10-13 · 对应方案版本 `neurospeech_plan_v2.1` · 英文版：[neurospeech_progress_2026-10-13.md](neurospeech_progress_2026-10-13.md)

## 当前状态

- **本报告范围：**[研究方案](neurospeech_research_plan_2026-10-13.zh.md)写完之后，2026-10-13 当天完成的工作：数据获取与处理、同步、标注、数据决策和 registry。
- **核验状态：**五个数据集全部处理完成并通过 checksum 校验；registry 和窗口列表已生成。**没有训练或评估任何模型。**训练与评估代码目前只在合成测试数据上运行过。
- **在方案中的位置：**方案第 11 节的 A0 阶段已完成。A1–D 阶段（copy / oracle、TRF、encoder、可靠度、flow、评估）尚未开始。
- **存储：**只保留处理后的数据（27 GB）。每一批原始文件在其处理结果核验通过后即删除；累计约 184 GB 原始文件下载后被清除，Etard（约 18 GB）直接在内存中流式处理，没有落盘。

## 1. 数据集

| 数据集 | 获取人数 | 使用人数 | 配对 EEG–音频时长（使用部分） | EEG 通道数 | EEG 与音频的配对依据 | 词 / 音素 |
|---|---|---|---|---|---|---|
| ds004408 | 19 | 19 | 19.2 小时 | 128 | source-aligned（官方 README 声明） | 官方 TextGrid：11,419 个词，35,248 个音素 |
| Brennan2019 | 45 | 31 | 6.0 小时（已处理 8.7 小时） | 60 | 28 人由音频监测通道实测，3 人采用作者的事件表 | 官方词表（2,125 个词）；强制对齐得到 6,781 个音素 |
| Etard 英语部分 | 18 | 18 | clean 2.9 小时 + babble 噪声 11.3 小时 | 63 | source-aligned（作者已对齐的 HDF5） | 书籍原文转写：8,395 个词，强制对齐得到 27,677 个音素 |
| Kymata 英文部分 | 20 | 20 | 18.1 小时 | 64 | 每次重复呈现都由音频监测通道实测 | 官方标注：1,396 个词，4,837 个音素 |
| KaraOne（后续阶段） | 14 | 14 | 1,913 个 trial | 62 | `epoch_inds.mat` 中的 trial 状态区间 | 每个 trial 的提示标签 |

处理后的 EEG 以 200 Hz（0.5–40 Hz）保存，另存一份 15 Hz 低通、64 Hz 的副本，单位 µV；每段刺激保留一份 24 kHz 的 canonical 音频。音频监测、EOG / ECG、触发和 MEG 通道都不进入任何记录。

## 2. 各数据集的发现与修复

### 2.1 Brennan2019

- **14 名被试的片段标签错误或缺失。**在这个重发布版本中，6 名被试没有 `events.tsv`，另有 8 名被试的标签整体晚了一个故事片段。对整段记录的音频监测通道（AUD）做全局搜索，再用与其他记录相同的三点核验，找回了这些片段。可用配对的被试从 26 人增加到 40 人。
- **5 名被试的音频监测通道无信号**（sub-005、sub-009、sub-022、sub-023、sub-038），见数据决策 3。
- **官方词表中有 4 个词的撇号字符损坏**（现已修复为 "what's"、"she'd"、"who'd"、"it's"）。
- **发布包自带原作者的质量控制结果。**`participants.tsv` 标明 31 名被试进入了 Brennan & Hale (2019) 的分析，14 名因理解成绩低和/或噪声大被排除，见数据决策 2。

### 2.2 Kymata 英文部分

- **原先没有 reader**，已新写：每次只下载一个 EEG/MEG 混合 FIF，与已审计的 SHA256 比对，提取 EEG 后删除。
- **文件中实际是 64 个 EEG 通道**，不是方案和论文所写的 70 个。电极位置使用 FIF 中保存的个体数字化坐标。
- **音频使用的是另一个时钟。**音频在 trigger 3 之后约 19 ms 才开始，并且比采集时钟慢约 0.054%，在 400 秒刺激上累计约 0.2 秒。每次重复呈现都在音频监测通道上测量起点和时钟比例，并把 EEG 重采样到音频时钟上，残余误差约 1 ms。
- **有一次重复呈现出现 14 ms 的播放跳变**（sub-E12，run 1，第 2 次重复）。该次改为按内容块逐块对齐，只扣除包含跳变的那一个 60 秒内容块。
- **有一个记录没有重复呈现的触发**（sub-E6，run 2）。起点由音频监测通道搜索得到，两次重复均已恢复。
- **结果：**81 个记录全部完成，162 次重复呈现全部对齐，没有被隔离的记录。`sub-E3b`（第二次 session）归入 `sub-E3`。400 秒刺激被切成 7 个固定的 60 秒内容块。

### 2.3 Etard

- 从 4 人续传到 18 人；文件通过 HTTP Range 直接读入内存，不保存压缩包。连接中途停滞时现在会自动重试，不再导致整个任务结束。
- **16 段英语故事全部是不同的语音内容**（4 段 clean，12 段带噪声）。方案中的待核实问题已有答案：clean 与噪声条件之间没有共享内容，因此这个数据集不存在“训练见过内容的噪声迁移”。

### 2.4 KaraOne

- 方案把 KaraOne 安排在后续的 Imagined 阶段。由于此前已处理了一半，这次直接补完。
- reader 在 MM15 上崩溃。部分 session 的 `speaking_inds` 条目顺序错乱、重复，或重复了其他区间。现在改为按位置归属：rest 与 imagined 区间在所有 session 中都是一致的，提示与发声区间按其落在两者之间的位置确定。没有区间的位置直接留空，不做猜测：3,826 个提示 / 发声区间中有 123 个被留空（MM15 42 个，MM16 1 个，MM20 80 个）。所有 rest 与 imagined 区间均已发布。

### 2.5 ds004408

- 此前已完成。唯一的变化是数据决策 1 中的坏导规则。

## 3. 数据源没有提供的标注

| 数据集 | 缺少的内容 | 生成方式 | 自动核验 |
|---|---|---|---|
| Brennan2019 | 音素 | 用 Montreal Forced Aligner 3.4.3（`english_us_arpa`）对官方词表做强制对齐 | 对齐得到的词起点与官方词时间相比：中位差 13–20 ms，95 分位 45–61 ms |
| Etard | 转写文本、词时间、音素 | 这些故事是《The Children of Odin》的若干章节（Project Gutenberg 电子书 24737）；原始 Zenodo 压缩包中的刺激文件名为 `odin_<章>_<段>`。参考文本取自书籍原文。Whisper 草稿只用于定位每段在书中的位置及其停顿。词和音素时间由对齐工具给出 | 每段中 95–99.6% 的书籍词被草稿确认；对齐工具与草稿的词起点中位差为 40–50 ms |
| Kymata 英文部分 | 说话人重叠标注（三名说话人；数据源未提供） | 官方词标注的停顿区间内达到语音强度的声音；共 15 个区间，合计 11.4 秒（刺激总长 401 秒；约 7 秒声音，两侧各加 0.1 秒），写入 `configs/neurospeech_v2/content_masks.json` | 无可用的核验手段 |

- 方案（第 6.1 节）要求每个来源人工抽查 200 个边界。目前用上表的自动核验代替；**没有做人工检查。**
- Kymata mask 找到的是词与词之间未被转写的声音（附和声、笑声、插话）。与已转写的词同时发生的重叠语音无法用这种方法检出。文件标记为 `human_review: false`；与 mask 区间重叠的窗口被丢弃（约占 Kymata 窗口的 10%）。

## 4. 数据决策

1. **已标注的坏导不再导致窗口被拒绝。**ds004408 提供了逐 run 的坏导标注（128 个通道中 0–34 个）。原始研究保留了全部 19 名被试，并对噪声通道做插值。这里的做法是把已标注坏导从模型输入中 mask 掉。10% 的阈值现在只统计未被标注为坏导的通道中平坦或超幅的通道，并要求至少一半的电极仍然有效。修改之前 sub-002 的全部窗口、sub-013 的大部分窗口都被拒绝；ds004408 的窗口拒绝率从 10.2% 降到 0.2%。
2. **Brennan2019 采用原作者的分析样本。**使用发布包中标记为 `analyzed` 的 31 名被试；被原作者排除的 14 人已处理但不使用。这由 `configs/neurospeech_v2/plan.yaml` 中的一行（`subject_policy`）控制。
3. **3 名进入原作者分析但没有音频监测信号的 Brennan 被试标为 source-aligned。**sub-005、sub-022 和 sub-038 使用原作者的事件表：12 个片段按正确顺序首尾相接；起点按其他被试实测得到的偏移量修正（305 个片段的中位数为 −5.0 ms，范围 −13 到 +27 ms）。这些记录带有 `status: source-aligned`。sub-009 和 sub-023 存在同样的问题，且被原作者因理解成绩低而排除，因此不使用。

## 5. Registry 与数据划分

`python scripts/neurospeech.py prepare` 从处理后的数据生成以下内容。所有数字都只统计 registry 中的被试。

| 数据集 | 人数 | 内容组（train / validation / test） | 窗口数（train / validation / test） |
|---|---|---|---|
| ds004408 | 19 | 14 / 3 / 3 段刺激 | 20,534 / 2,298 / 2,240 |
| Brennan2019 | 31 | 8 / 2 / 2 个片段 | 3,809 / 487 / 500 |
| Etard clean | 18 | 60 秒内容块 | 1,422 / 360 / 324 |
| Etard 噪声条件（只用于评估） | 18 | 60 秒内容块 | 共 9,648 |
| Kymata 英文部分 | 20 | 5 / 1 / 1 个 60 秒内容块 | 11,297 / 1,132 / 1,296 |

- **合计：**55,347 个 4 秒窗口（训练 stride 2 秒，评估 stride 4 秒）。
- **P1 协议**（四个来源）：共 88 人；13 名开发被试（ds004408 的 sub-001–003，Brennan 5 人，Etard 2 人，Kymata 3 人）；五个测试 fold 分别为 18 / 15 / 14 / 14 / 14 人。
- **P2 协议**（Brennan 整体留出）：源侧 57 人，其中 8 人用于开发；31 名 Brennan 被试作为外部测试集。

## 6. 新增或修改的代码

| 文件 | 修改 |
|---|---|
| `eegspeech/neurospeech/sources/kymata.py` | 新 reader：白名单下载、基于监测通道的时钟测量、逐块回退、无触发时的起点搜索 |
| `eegspeech/neurospeech/sources/brennan.py` | 整段 AUD 搜索、重试选项、source-aligned 回退、撇号修复、重跑时保留后加的音素标注 |
| `eegspeech/neurospeech/sources/karaone.py` | 按位置确定 trial 状态、重试选项 |
| `eegspeech/neurospeech/sources/etard.py` | Range 下载停滞时重试 |
| `eegspeech/neurospeech/annotation.py` | 新增：书籍原文转写、强制对齐、Kymata mask |
| `eegspeech/neurospeech/data.py` | 内容 mask、坏导规则、被试策略、按方案固定 ds004408 开发被试、缓存键包含音素标注 |
| `eegspeech/neurospeech/evaluation.py`、`scripts/neurospeech.py` | `annotate` 命令；`evaluate --mode zero-shot / few-shot / noise` |
| `configs/neurospeech_v2/plan.yaml`、`content_masks.json` | `min_valid_fraction`、`subject_policy`、Kymata mask |
| `tests/test_neurospeech_sources.py` | reader 与标注的新测试；60 个单元测试全部通过 |
| `README.md`、`report/` | 英文文档；中文原文保留为 `*.zh.md` |

强制对齐工具安装在项目内的 conda 环境 `.venv-mfa`（1.3 GB），其模型位于 `artifacts/neurospeech_v2/mfa`。

## 7. 与方案不一致的地方

- KaraOne 现在就已处理完，而不是等到 Perceived 验证之后。
- Kymata 文件中是 64 个 EEG 通道，不是 70 个。
- Etard 的 clean 与噪声条件没有共享内容，因此方案第 3.3 节中“训练见过内容”的噪声对比无法进行。
- 人工抽查 200 个音素边界（第 6.1 节）由自动比较代替。
- Kymata 重叠 mask（第 2.1、3.1 节）为自动生成，未经人工检查。
- Brennan2019 只使用原作者的分析样本（31 人而不是 45 人），其中 3 人为 source-aligned。
- 窗口 QC 规则把已标注坏导视为 mask，而不是拒绝窗口的依据。

## 8. 尚未完成

1. **全部训练和评估**（方案 A1–D 阶段）。按要求没有运行任何训练。这部分代码只在合成数据上跑过，第一次真实运行时预计需要调试。
2. **Etard 的 babble 噪声音轨。**发布包为带噪声的故事提供了 `part_k_distractor.wav`，目前没有处理，因此无法对实际呈现的带噪混合音做 ASR（第 3.3 节）。
3. **消融变体 A1–A6** 的命令行选项，以及 72 次训练矩阵的运行器。
4. **REVE 预训练语料重叠审计**（第 3.2 节）。
5. 如有需要，对音素边界和 Kymata mask 做**人工检查**。

## 9. 如何核对

```bash
python scripts/neurospeech.py status
```

```bash
python scripts/neurospeech.py verify-data
```

```bash
python scripts/neurospeech.py prepare
```

```bash
python -m pytest tests -q
```

各数据集的证据（来源回执、同步测量、标注核验、清理日志）与数据放在一起：`data/processed/neurospeech/{dataset}/manifest.json`、`metadata/` 和 `cleanup.jsonl`。当前的命令和数据决策也写在[项目 README](../README.md) 中。

## 10. 补充：把 Imagined 语音做成生成任务

在数据阶段之后补写，起因是要求把 Imagined 语音当作生成而不是分类来做。**目前只有代码，没有训练任何模型。**

- `eegspeech/neurospeech/imagined.py` 把听觉语音模型迁移到 KaraOne：Imagined EEG → 冻结的听觉语音 encoder → 一个小的重定时模块（唯一被训练的部分，初始时为恒等映射）→ 同样的连续条件 → 冻结的 flow 生成器 → 2 秒波形。整条路径中没有分类器。
- 回归目标是同一 trial 的发声录音，因为想象阶段本身没有声音。EEG 窗口（4.5 秒）完全位于 5 秒的 imagined 状态之内。
- 两种测试：未见过的被试（四个 fold，每个留出三人），以及未见过的提示（11 个提示中有一个完全不参与迁移训练）。对照条件：同一被试另一个提示的 imagined EEG、同一 trial 的 rest EEG、空条件，以及不带重定时模块的 zero-shot 模型。
- 所有指标都在生成的音频上计算：与该被试自己发声录音的相似度、Mel 误差、真实提示在 11 个提示中的排名。提示身份只出现在指标和数据划分中。
- 已用未训练的权重在真实的 KaraOne trial 上核对：张量形状、梯度路径（只有重定时模块得到梯度）以及各对照条件音频的写出。这一设计取代了方案第 14 节中以分类指标为先的验证。
- 局限：11 个很短的提示无法证明自由形式的想象语音；发声录音只是替代目标，其时间关系未知。
