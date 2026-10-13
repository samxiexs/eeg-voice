# EEG-Voice 阶段报告（2026-10-09）

本文件夹整理了截至 2026-10-07 最新一轮 true reconstruction 的方法、结果和模型 framework。对应的代码版本是 commit `255d361f`，结果文件位于 `../outputs/`（总览 `../outputs/index.html`）。本报告只读取已经保存的结果，没有重新运行任何训练或评估。

我们要做的是 within-person 的 imagined-speech EEG → speech reconstruction：用 TTS 合成语音构建一个 CLIP 式的 speech embedding space，为每个人训练一个把 EEG 映射到这个空间的 encoder，再用一个只在语音上训练的 mel diffusion decoder，把 EEG 得到的连续 embedding 加上这个人的 voice traits 生成语音。生成路径中不做任何 classification 选择，候选只用于评估。实验覆盖 Thinking Out Loud、BCI2020（小词表 items）和 Chisco（约 6,500 个中文句子）。

结果可以概括为：decoder/renderer 一侧基本到位（items oracle 95–100%，声音性别约 95%、pitch error 1–2 semitones；句子 oracle rank 0.71、duration r 0.94，但对未见过的句子尚不可懂），而 EEG 中可读出的内容信息非常少。Items 的 EEG 重建只比 wrong-trial control 高 1–2 个百分点（TOL 7/10 人 p = 0.027，BCI2020 10/15 人 p = 0.048），绝对水平接近 chance；Chisco 的 encoder 排序信号约为 0.51（chance 0.5），在生成语音中已经测不出来。主要瓶颈是 EEG 信息量。

| 文件 | 内容 |
|---|---|
| [01_methods.md](01_methods.md) | 做了什么：数据、三部分架构、训练细节、评估协议，以及改变结论的数据审计（BCI2020 泄漏、Chisco 标签与 topic block） |
| [02_results.md](02_results.md) | 效果如何：items 与 sentences 的完整数字、与 decision rendering 的对比、两个瓶颈的分析和下一步建议 |
| [03_framework.md](03_framework.md) | 模型 framework 图（Figure 1 整体结构，Figure 2 Chisco deep encoder）及说明 |
| [04_foundation_models.md](04_foundation_models.md) | EEG foundation model（REVE、CBraMod、LaBraM）能否带来更好的结果：文献、方案、frozen probe 与 fine-tuning 的结果（2026-10-11）：都没有超过现有的 log-power encoder |
| `figures/` | `framework.png`、`deep_encoder.png` 以及生成它们的 `make_figures.py` |
