# 4. EEG foundation model 能否带来更好的结果？（2026-10-11）

## 4.1 文献怎么说

过去两年出现了一批在大规模无标注 EEG 上做 self-supervised pretraining 的 foundation model。LaBraM（Jiang et al., ICLR 2024）在约 2,500 小时、20 个数据集上用 masked neural-token prediction 预训练，5.8M 参数，每个 10-05 电极名有一个可学习的 spatial embedding；CBraMod（Wang et al., ICLR 2025）在 TUEG 上做 patch-based masked reconstruction，用 criss-cross（空间/时间分开的）attention 和卷积式 positional encoding，4.9M 参数；REVE（El Ouahidi et al., NeurIPS 2025）是目前规模最大的公开模型，在 92 个数据集、25,000 名被试、6 万小时 EEG 上做 masked autoencoding，69M 参数，用 (x, y, z, t) 的 4D Fourier positional encoding 适配任意 montage，并且 pretraining 频带是 0.5–99.5 Hz（LaBraM 和 CBraMod 只到 75 Hz）。这三个模型的权重都公开，braindecode 1.8 提供了统一实现。此外还有 EEGPT、CSBrain、CodeBrain、BrainOmni、LUNA 等。

这些论文都在 BCI Competition 2020 Track 3（就是我们的 BCI2020，5 个 imagined 短语）上报告了结果：LaBraM 0.506、CBraMod 0.537、REVE 0.564、CSBrain 0.600（balanced accuracy，full fine-tuning）；REVE 的 frozen linear probe 是 0.390。但这些数字都用的是官方 train/validation split，而我们在 2026-10-05 的审计中发现这个 split 有 cue-repetition 泄漏：每个 cue 被连续想 4 次，4 次重复几乎是 near-duplicate，被分散到 train、validation 和 test 文件里，一个测试 trial 在训练集里的最近邻有 67% 是同一个 item（chance 20%）。我们自己在官方 split 上只用 > 55 Hz 的 log-power 就能得到 46%，按 cue episode 划分 fold 后降到 23%。因此这些 0.5–0.6 的数字主要反映的是重复样本的匹配能力，不能和我们去泄漏后的数字直接比。

专门针对语音的评测给出了更冷静的图景。2026 年 7 月的 benchmark（arXiv 2607.27268）在同一套预处理和 fine-tuning 流程下比较了 LaBraM、EEGMamba 和 EEGNet、ShallowFBCSPNet、EEGConformer：BCI2020 within-subject 上 LaBraM 27.7%、EEGConformer 30.2%、EEGNet 26.7%（LaBraM vs EEGNet +1.0 pt，p = 0.41），leave-one-subject-out 上所有模型都在 chance；作者的结论是大规模 EEG pretraining 对语音解码没有带来优势。更一般的 benchmark（AdaBrain-Bench arXiv 2507.09882、EEG-FM-Bench arXiv 2508.17742、OmniEEG-Bench arXiv 2606.00815、NeuroAtlas arXiv 2605.14698）也发现：frozen linear probe 通常明显弱于 full fine-tuning；模型排名随协议变化很大；foundation model 经常打不过简单 baseline 或 random-init 的同一架构；还有工作（arXiv 2607.24519）指出这些模型的表示里编码了很多 dataset identity，需要 negative control。Chisco 上目前唯一的 foundation-model 工作是 NeuroSpeak（ICLR 2026），它在 subject-agnostic 设置下用 masked modeling + VQ 表示对齐语言 embedding，再交给预训练语言模型生成文本，报告 47.7% 的 semantic similarity；这个指标和我们的 rank percentile 不同，而且有语言模型先验，不能直接比较。

由此可以得到三个预期。第一，foundation model 不会凭空增加 EEG 里的信息量，它能帮上忙的方式只有两种：把已有的微弱信号变得更容易被线性读出，或者压制与内容无关的变异（噪声、session drift、个体差异）。第二，LaBraM 和 CBraMod 的 pretraining 频带只到 75 Hz，而我们的 items 信号有一部分在 > 55 Hz（很可能是 covert articulation 的 EMG），REVE 保留到 99.5 Hz，在这一点上更合适。第三，任何提升都必须在我们去泄漏的 block-wise 协议下、和同一批 held-out trial 上的 log-power encoder 成对比较才算数。

## 4.2 方案

测试只评估 encoder，因为重建语音是把 encoder 给出的 CLIP 空间里的点渲染出来，它能携带的内容不可能多于 encoder 读到的内容；encoder 没有提升就不值得重跑整个重建。除 EEG 表示外的一切都保持不变：fold（TOL 按录制顺序的连续 block，BCI2020 按完整 cue episode，Chisco 按完整 run，held-out run 里出现过的句子不进训练）、held-out trial、inner cross-validation、anchor（合成语音的 HuBERT）、top-3 ensemble 都和 `reconstruct.py` / `sentences.py` 完全一致，所以结果可以和 `outputs/reconstruct`、`outputs/sentences` 里 log-power encoder 的结果按 trial 一一配对。代码在 `eegspeech/foundation.py` 和 `scripts/foundation.py`，运行入口是 `bash scripts/run_plan.sh foundation`。

输入按每个模型自己的 pretraining 约定准备：重采样到 200 Hz，按模型频带 band-pass（REVE 0.5–99.5 Hz，CBraMod/LaBraM 0.3–75 Hz），去掉当地工频（TOL、Chisco 50 Hz，BCI2020 60 Hz）；REVE 对每个 trial 的每个通道做 z-score 并 clip 到 ±15 SD，CBraMod/LaBraM 用 100 µV 为单位。这些处理只用 trial 自身，不依赖 fold，所以 held-out trial 不会进入任何统计量；fold 专用的 Euclidean alignment 不用于这些模型，因为它们是在未对齐的 EEG 上预训练的。通道方面，REVE 直接用 store 里的电极坐标（与它的 position bank 同一坐标系），CBraMod 用全部有效通道，LaBraM 只认 10-05 电极名，所以没有标准名的电极（TOL 的 Biosemi A1–D32、Chisco 的部分半位置）映射到 2 cm 以内最近的空闲标准电极，映射不上的就丢弃（TOL 保留 63/128，Chisco 80/115，BCI2020 64/64）。

**Stage 1（frozen probe）**：每个 trial 过一遍冻结的模型，取中间层和最后一层（REVE 第 11、22 层，CBraMod/LaBraM 第 6、12 层）的 token embedding（channel × 1 s patch × dim），pool 成三种向量：所有 token 的均值（mean）、每个 patch 对通道取均值（time，即时间进程）、每个通道对 patch 取均值后投影到该人训练 trial 的前 16 个主方向（chan，保留空间信息）。这些特征替换（`fm`）或拼接到（`fm+lp`）原来的 aligned log-power 特征上，再进入原来的 personal linear CLIP encoder（items：InfoNCE + L-BFGS；sentences：标准化 + PCA 512 + ridge 回归到 anchor）。层、pooling 方式、drift removal、weight decay 由同样的 coordinate search 在 inner CV 上选出，最好的 3 组做 ensemble。

**Stage 2（fine-tuning）**：把 backbone 和每个人自己的 head（每个 token 做 LayerNorm、flatten、linear 映射到 CLIP 空间）一起训练，loss 是 s·cos(z, a) 对 anchor 的 cross-entropy（items 对 K 个 item；sentences 对所有训练句子，相当于冻结 speech tower 的 CLIP，和现有的 deep encoder 一样）。CBraMod 和 LaBraM 全部参数参与 fine-tuning；REVE 太大，只训练最上面 4 层（下面 18 层的输出预先缓存）。backbone 学习率 1e-4（文献指出 1e-3 会破坏预训练权重），head 1e-3，AdamW，最多 20 个 epoch，用每个人训练数据的最后一个 inner block 做 early stopping 和 temperature calibration。Items 上三个模型都做；Chisco 上只做 CBraMod（REVE 在 Chisco 上 fine-tune 一个 epoch 约需 1 小时，在这台 16 GB 的 Mac 上不现实）。

**Stage 3**：只有当某个 foundation-model encoder 相对 log-power encoder 有稳定的提升时，才用它重跑完整的语音重建。

## 4.3 结果

所有数字都是 5 folds、每人先在 held-out trial 上取平均、再在人之间取平均，和同一批 trial 上的 log-power encoder（`outputs/reconstruct`、`outputs/sentences` 里的 encoder diagnostic）成对比较，p 是人之间的 Wilcoxon 检验。Items 报告 top-1 accuracy（TOL chance 0.25，BCI2020 0.20），Chisco 报告真句子在 run 内约 130 个候选中的 rank percentile（chance 0.5）。完整表格在 `outputs/foundation/summary.md`，逐 trial 结果在 `outputs/foundation/<run>/<dataset>_f<k>.csv`。

**Stage 1：frozen probe。** 三个模型、两种用法（只用 foundation-model 特征 `fm`，或与 log-power 拼接 `fm+lp`）、三个数据集，一共 18 组，没有一组优于现有的 log-power encoder。

| 模型 | 用法 | TOL（log-power 0.292） | BCI2020（log-power 0.222） | Chisco（log-power 0.5104） |
|---|---|---|---|---|
| REVE | fm | 0.263（1/10 人更好, p = 0.03） | 0.207（6/15, p = 0.13） | 0.5067（2/5, p = 0.63） |
| CBraMod | fm | 0.251（1/10, p = 0.01） | 0.220（8/15, p = 0.93） | 0.5040（1/5, p = 0.31） |
| LaBraM | fm | 0.246（2/10, p = 0.02） | 0.213（5/15, p = 0.40） | 0.5073（1/5, p = 0.44） |
| REVE | fm+lp | 0.270（4/10, p = 0.32） | 0.228（8/15, p = 0.53） | 0.5106（1/5, p = 0.63） |
| CBraMod | fm+lp | 0.266（3/10, p = 0.08） | 0.226（8/15, p = 0.35） | 0.5080（2/5, p = 0.31） |
| LaBraM | fm+lp | 0.268（2/10, p = 0.11） | 0.223（8/15, p = 0.73） | 0.5111（2/5, p = 0.81） |

只用 foundation-model 特征时，TOL 上三个模型都显著变差（掉 3–5 个百分点），BCI2020 和 Chisco 上低于或等于 log-power（Chisco 上和现有 deep encoder 的 0.505 相当）。把它们加到 log-power 上，BCI2020 和 Chisco 的变化在 ±0.6 个百分点以内（Chisco top-10 最多从 8.0% 到 8.3%），都不显著；TOL 上反而降低 2–3 个百分点，说明额外的几千维特征只是稀释了 log-power 里已有的弱信号。inner CV 选出的层和 pooling 方式在 fold 之间不稳定（mean+time、chan 都有，中间层和最后一层都有），inner log-loss 始终贴着 uniform（BCI2020 1.604–1.608 vs log 5 = 1.609），这本身就说明 frozen 表示里与内容相关的成分非常少。

**Stage 2：fine-tuning（items）。** fine-tuning 后的结果比 frozen probe 还差，全部回到 chance：

| 模型 | TOL（log-power 0.292） | BCI2020（log-power 0.222） |
|---|---|---|
| CBraMod（全部参数） | 0.242（1/10, p = 0.008） | 0.203（6/15, p = 0.04） |
| LaBraM（全部参数） | 0.244（2/10, p = 0.010） | 0.198（5/15, p = 0.04） |
| REVE（最上面 4 层） | 0.247（1/10, p = 0.006） | 0.208（5/15, p = 0.14） |

**Stage 2：fine-tuning（Chisco，CBraMod）。** 测试 rank percentile 0.5006，同一批 trial 上 log-power encoder 是 0.5104，现有 deep encoder 是 0.5047；5 个人里只有 sub-05 高于 log-power（0.507 vs 0.494，而 log-power 在这个人身上本来就低于 chance），sub-01（log-power 最强的人，0.526）降到 0.497；top-10 是 7.9%（log-power 8.0%，chance 7.6%），p = 0.44。每个 fold 训练 8–10 个 epoch（每个 epoch 约 6 分钟），validation rank percentile 最高只到 0.503–0.508，而且是在训练 loss 还远高于下限时就达到峰值、之后随着训练 loss 下降而回落；用这个峰值选出的模型在 held-out run 上回到 0.50。

训练曲线的形状在所有模型和 fold 上都一样：训练 loss 稳定下降（例如 CBraMod BCI2020 fold 4 从 1.72 降到 0.50），validation log-loss 却一直贴在 uniform（1.6094 / 1.3863），最好的 epoch 也只低 0.001–0.004。也就是说，模型在记忆每个人的几百个训练 trial，而这些记忆没有迁移到新的 cue episode 或新的录制 block 上。这和 arXiv 2607.27268 的结论一致（在泄漏的官方 split 上 LaBraM 也只是和 EEGNet 持平），而在我们去泄漏的协议下连这一点优势都不存在。


**补充检验：小 head 的 fine-tuning。** 上面 fine-tuning 用的 head 把所有 token 展平（TOL 每人 128 × 2 × 200 = 51k 个输入，只有几百个训练 trial），一个可能的解释是失败来自 head 过拟合。为此把 head 换成只读取平均 token 和每个 patch 的通道平均（每人约 3.6k 个参数，原来约 154k），在 items 上对 CBraMod 和 REVE 重跑（输出在 `outputs/foundation/finetune_*_meanhead`）。训练 loss 这次几乎不降（BCI2020 从 1.73 到 1.63），说明不再记忆训练 trial，但 validation 同样停在 uniform，测试结果没有变化：CBraMod TOL 0.251（1/10, p = 0.006）、BCI2020 0.198（4/15, p = 0.03）；REVE TOL 0.240（0/10, p = 0.002）、BCI2020 0.195（4/15, p = 0.01）。所以 fine-tuning 无效不是 head 设计的问题。

## 4.4 结论

在我们去泄漏的 within-person 协议下，EEG foundation model 没有带来更好的结果。三个公开的代表性模型（REVE、CBraMod、LaBraM），无论冻结使用、与 log-power 拼接，还是 fine-tuning（全部参数或顶层、大 head 或小 head），在 TOL、BCI2020 和 Chisco 上都没有超过现有的 filter-bank log-power encoder；冻结特征单独使用和所有 fine-tuning 版本在 TOL 上都显著更差，fine-tuning 的结果全部回到 chance。只有"冻结特征 + log-power"在 BCI2020 和 Chisco 上与 log-power 持平（差异 ±0.6 个百分点以内，都不显著）。因此没有进入 Stage 3：没有任何一个 foundation-model encoder 值得拿来重跑语音重建。

这和 4.1 的预期一致。这些模型从临床、睡眠、运动想象等数据里学到的是通用的频谱和空间结构，而我们的 log-power 特征已经直接、低方差地描述了这些结构；imagined speech 的内容信息（如果存在）很弱，并且可能部分来自 > 55 Hz 的肌电，LaBraM 和 CBraMod 的 pretraining 频带根本不覆盖这一段，REVE 虽然覆盖，但它对每个 trial 做的 z-score 去掉了绝对功率水平，而 TOL 的信号恰恰在功率水平上（这也解释了为什么冻结特征在 TOL 上掉得最多）。在每人只有几百个（items）到几千个（Chisco）标注 trial 的情况下，fine-tuning 只是让大模型记住了训练 trial。文献里 foundation model 在 BCI2020 上的 0.51–0.60 依赖官方 split 里的重复样本，在 cue-episode 划分下这种优势不存在。

需要说明的局限：只测了三个模型（没有 EEGPT、CSBrain 等）；REVE 在 Chisco 上没有 fine-tune（在这台 16 GB 的 Mac 上每个 epoch 约 1 小时），只做了 frozen probe；fine-tuning 的超参数（学习率、epoch、head）只试了文献推荐的一组和一个小 head 变体，没有大规模搜索；跨人的 multi-subject 预训练再 within-person 微调这种用法也没有测试（之前的跨人实验在所有模型上都是 chance）。

这次测试对下一步的含义是：瓶颈仍然是 EEG 中可读出的内容信息，而不是表示学习的能力。更有希望的方向不是换一个更大的预训练模型，而是增加每个人的有效信息量：例如 Chisco 的 reading-epoch positive control（`bash scripts/run_plan.sh control`，已写好未运行），确认整条 pipeline 在内容肯定存在时能达到什么水平；或者使用每人数据更多、并且同时记录了语音或肌电的数据集。

## 4.5 复现

```bash
bash scripts/run_plan.sh foundation                       # embed → frozen probe (fm, fm+lp) → fine-tune → report
python scripts/foundation.py finetune --head mean --models cbramod reve --datasets bci2020 thinking_out_loud --fold 0 1 2 3 4
python scripts/foundation.py report                       # outputs/foundation/summary.md, summary.json
```

运行时间（16 GB Apple Silicon，MPS）：特征提取约 30 分钟，frozen probe 约 3 小时（90 个 fold），items fine-tuning 约 2.5 小时，Chisco CBraMod fine-tuning 约 4.5 小时，小 head 补充检验约 1.8 小时。需要的额外依赖是 braindecode 1.8.1（已装入 `.venv-aligned-local`，pandas 保持 2.2.x），权重在 `models/reve_base`、`models/reve_positions`、`models/labram_pretrained`、`models/cbramod_pretrained`。

**参考文献**：LaBraM (Jiang et al., ICLR 2024)；CBraMod (Wang et al., ICLR 2025, arXiv 2412.07236)；REVE (El Ouahidi et al., NeurIPS 2025, arXiv 2510.21585)；CSBrain (arXiv 2506.23075)；Does EEG Foundation Models Transfer to Speech? (arXiv 2607.27268)；AdaBrain-Bench (arXiv 2507.09882)；EEG-FM-Bench (arXiv 2508.17742)；OmniEEG-Bench (arXiv 2606.00815)；NeuroAtlas (arXiv 2605.14698)；EEG foundation models 的 negative-control 研究 (arXiv 2607.24519)；NeuroSpeak (ICLR 2026)；NeuroTTT (arXiv 2509.26301)。
