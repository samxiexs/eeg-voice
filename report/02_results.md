# 2. 效果如何：最新一轮 true reconstruction 的结果（2026-10-07）

一句话概括：**speech side 已经接近 ceiling，而 EEG 中可读出的内容信息非常少**。Items 上，从 EEG 生成的语音与 wrong-trial control 相比有统计显著但只有 1–2 个百分点的优势，绝对水平仍在 chance 附近；句子上，从 EEG 生成的语音在 run 内候选中的排名与 chance 无法区分。所有数字都是 within-person、5 folds、held-out block 的结果（`outputs/reconstruct/summary.json`、`outputs/sentences/summary.json`、`outputs/sentences_deep/summary.json`，总览在 `outputs/index.html`，可以在 `outputs/reconstruct/index.html` 和 `outputs/sentences/index.html` 里直接试听）。

## 2.1 Items：TOL 与 BCI2020

下表是生成语音被三个 measure 识别为真实 item 的比例（Whisper / MCD / HuBERT，取每人均值再平均）。"Encoder" 一列是 encoder 自身 top-1 的准确率，只作为 diagnostic，不参与生成。

| Dataset (chance) | Encoder top-1 | **From EEG** | Wrong trial | Prior | Oracle (ceiling) | EEG > wrong（人数, Wilcoxon p） |
|---|---|---|---|---|---|---|
| TOL (0.25), 10 人, 2,236 trials | 0.292（9/10 > chance, p = 0.003） | **0.248 / 0.246 / 0.248** | 0.234 / 0.235 / 0.242 | 0.247 / 0.255 / 0.258 | 0.975 / 1.000 / 0.945 | 7/10, p = 0.027（MCD p = 0.024） |
| BCI2020 (0.20), 15 人, 6,000 trials | 0.222（11/15 > chance） | **0.206 / 0.204 / 0.205** | 0.202 / 0.201 / 0.200 | 0.192 / 0.192 / 0.193 | 0.992 / 0.993 / 0.992 | 10/15, p = 0.048 |

Oracle 一列说明 decoder 本身没有问题：只要给出正确 item 的 anchor，生成语音有 95–100% 被识别为该 item，MCD 到真实语音约 1.8–2.1 dB。声音部分同样达标：EEG 重建语音的性别正确率 TOL 95.8%、BCI2020 94.7%，pitch error 分别为 0.8 和 1.7 semitones，speaker identification 0.63 和 0.56（8 选 1，chance 0.125）。因此 content 和 voice 两个部分在 decoder 侧都是可控的。

问题在 EEG 侧。Encoder 本身只比 chance 高 2–4 个百分点（TOL 0.292 vs 0.25，BCI2020 0.222 vs 0.20），这在 BCI2020 已经是去除 cue-episode 泄漏后的诚实数字。换成 posterior-mean conditioning 后，生成语音的识别率几乎回到 chance（TOL 0.248，BCI2020 0.206），只是仍然稳定地高于 wrong-trial control（TOL 7/10 人，p = 0.027；BCI2020 10/15 人，p = 0.048）。作为对比，2026-10-06 的 decision rendering（先 argmax 再让 decoder 说出该 item，`outputs/reconstruct_v3_decision`）得到 TOL 0.290、BCI2020 0.229，几乎等于 encoder 的 top-1，因为它本质上就是 classification 加 rendering。我们的解释是：encoder 的 calibrated posterior 非常平（4 选 1 只有 29% 的准确率），所以 $E[a\mid\text{EEG}]$ 基本落在所有 anchor 的中心附近，decoder 收到的是一个"哪个都不太像"的条件，结果接近 prior control；argmax 则会把很小的 margin 放大成一个完整的 item。也就是说，在 true reconstruction 的约束下，现有的弱信号大部分被稀释掉了，剩下的只够在 paired test 中显著。

## 2.2 Sentences：Chisco

Chisco 的评估把生成语音的 HuBERT embedding 投影回 speech space，看真句子在同一 run 约 130 个候选（训练中都没见过）中的 rank percentile（1 = 最好，0.5 = chance）。每人每折渲染 60 个 trial，共 1,500 个；encoder diagnostic 用全部 29,336 个 held-out trial。

| | Encoder diagnostic（29,336 trials） | **Generated speech from EEG** | Wrong trial | Prior | Oracle（renderer ceiling） | EEG > wrong（人数, p） |
|---|---|---|---|---|---|---|
| Linear encoder | 0.510（4/5 人 > 0.5, p = 0.063） | **0.500** | 0.495 | 0.493 | 0.710 | 2/5, p = 0.41 |
| Deep encoder | 0.505（5/5 人 > 0.5, p = 0.031） | **0.496** | 0.488 | 0.493 | 0.710 | 4/5, p = 0.063 |

其他 measure 给出相同的结论：从 EEG 生成的语音 top-1 0.73%（chance 0.78%），top-10 8.9% / 7.9%（chance 7.8%），与真句子 duration 的相关 r = 0.06 / 0.03，Whisper 转写听出真句子的比例 0.4% / 0.6%；生成语音的性别正确率 99%。Encoder 自身在 run 内的排序有一个很小但可重复的信号：linear encoder 0.510，circular-shift 检验中 sub-01 在 5 折中全部 p ≤ 0.016，sub-02、sub-03 在部分折显著，sub-04、sub-05 基本不显著；deep encoder 0.505，5 个人都在 0.5 以上，但比 linear 弱。这与前一天（2026-10-06）的分析一致：Chisco 的信息在 band power 的 time course 里（time-resolved 0.512 vs whole-trial 0.503），而且很粗——topic（6.8% vs chance 6.8%）和精确句子（top-1）都恢复不出来。0.51 的排序信号经过 renderer 之后就消失了。

Sentences 这边还有第二个瓶颈：**renderer 对没听过的句子不可懂**。Oracle（给真句子的 anchor）生成的语音 rank percentile 0.710、top-10 25.4%、top-1 5.0%，duration 相关 r = 0.94，说明 renderer 确实能把 speech space 里的点还原成时长和粗略声学特征正确的语音；但 Whisper 只听出 1.0% 的真句子，也就是说生成的是"像中文、长度对、但词听不清"的语音。原因在于 16 维 + duration 的 speech space 只保留了句子的粗略信息，而 renderer 只用每折训练集中的句子（held-out run 里出现过的句子全部剔除）× 6 个 TTS voice 从头训练。所以即使 EEG 完全正确，目前这条句子 pipeline 也只能得到粗略重建，而不是可懂的句子。

## 2.3 总结与判断

把两条 pipeline 放在一起看，结论比较清楚。（1）**Speech side 不是瓶颈（items）**：oracle 95–100%，声音的性别、pitch 和 timbre 都可控；句子的 renderer 能还原时长和粗略声学，但还不够可懂。（2）**EEG 是主要瓶颈**：within-person 的 imagined speech 信号真实存在但很小（TOL +4 pt、BCI2020 +2 pt、Chisco rank 0.51），跨人则完全没有。（3）**从 classification 改为 true reconstruction 会损失大部分信号**：items 从 decision rendering 的 0.290 / 0.229 降到 0.248 / 0.206，只剩下相对 wrong trial 的 paired 优势；Chisco 的 0.51 排序信号在生成语音中不再可见。（4）**之前看起来更好的结果主要来自数据泄漏**：BCI2020 的 0.36 是 cue-repetition 泄漏造成的，修正后是 0.23。

基于这些结果，下一步最值得做的是能直接增加 EEG 信息量的事情，而不是继续优化 decoder。具体包括：在 Chisco 的 reading epoch 上跑 positive control（`bash scripts/run_plan.sh control`，已写好但未运行），用来确认 pipeline 在内容肯定存在于 EEG 时能达到什么水平；每人更多的 imagery 数据或更强的 within-person 先验；对于句子，再加一个更强的 speech space 和 renderer（例如更高维、包含音素信息的空间，或使用预训练的 TTS-style decoder），让 oracle 先变得可懂。只要 encoder 的信号仍然停留在 0.51 / +2–4 pt 的水平，任何 renderer 都无法让 EEG 重建出可懂的语音。
