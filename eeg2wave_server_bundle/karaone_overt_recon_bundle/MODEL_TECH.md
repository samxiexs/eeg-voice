# KaraOne EEG→语音 模型技术文档（detailed）

> 本文档说明 `karaone_overt_recon_bundle` 当前模型**是什么、怎么做的、每一步细节**。
> 代码事实均来自 `app/src/` 与 `app/configs/karaone.yaml`，可逐行对照。
> 配套文档：[METHOD.md](METHOD.md)（方法与论文对照）、[DIFFUSION_PLAN.md](DIFFUSION_PLAN.md)（重做方案）。

---

## 0. 一句话定位

从 **KaraOne overt（发声）EEG** 重建语音。流水线**全可切换**，默认：
**`EEG → 编码器(可选通道MoE) → 解码头(回归) → mel 目标 → Griffin-Lim 声码器 → 波形`**，
对齐论文（NeuroTalk/Park/FESDE）的"mel + 对齐 + 声码器"配方。**模型完全不使用被试 ID**
（`forward(eeg, subject_idx, stage_idx)` 接收 `subject_idx` 仅为接口兼容，内部 `del subject_idx`；
已验证同段 EEG 不同 ID 输出逐位相同）。

### 开关总览（`configs/karaone.yaml`）
| 开关 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `target.kind` | mel / encodec_latent | **mel** | 声学目标表示 |
| `vocoder.kind` | griffinlim / encodec | **griffinlim** | mel→wav / latent→wav |
| `model.decoder` | regression / diffusion | **regression** | 解码头（选 train_recon vs train_diffusion） |
| `--model` | baseline / moe | — | 编码器通道 MoE 开关 |
| `train.lambda_dtw` | 0 / >0 | **1.0** | DTW 对齐重建（治跨 trial 错位） |
| `train.lambda_gan` | 0 / >0 | **0** | 对抗损失（治均值塌缩/糊；设 0.1 开） |

---

## 1. 数据（KaraOne）

- **EEG**：62 通道 @ 256Hz，带通 1–40Hz + 60Hz 陷波，已 z-score；每 trial 截 `eeg_len=1280`（≈5s，发声段 ~718 点，其余补零）。
- **阶段**：`clearing/stimulus_like/thinking/overt_like`；默认用 `overt_like`（发声中 EEG）。
- **标签**：11 个 prompt（7 音素 `/iy/ /uw/ /m/ /n/ /diy/ /piy/ /tiy/` + 4 词 `gnaw knew pat pot`）。
- **音频**：同 trial 的 overt 录音，定长 **2.0s @ 16kHz**，RMS 归一化。**实测很轻、~12% 有声、同音素跨 trial 不对齐**——这是本任务最大的结构性难点（见 §7）。
- **目标表示**（[targets.py]）：
  - `mel`：`karaone_trial_mel.npz`，`[N, 122, 80]` log-mel，按维 z-score。
  - `encodec_latent`：`karaone_trial_encodec_latents.npz`，`[N, 150, 128]` EnCodec 连续潜在。
- **划分**（[data.py]）：
  - `trial`（同被试留出 trial）：按 `(subject,label,stage)` 分组，组内留最后 1 trial 作 test、倒 2 作 val、其余 train；
    **已排除 heldout 被试**（P02/MM21）——防止它们既进 trial-train 又进 subject_test 造成泄漏。
  - `subject_holdout`：heldout 被试（P02/MM21）→ 真·跨被试测试。

---

## 2. 编码器（EEG → 特征序列）

### 2.1 通道 MoE 前端 `ChannelMoEFrontend`（[encoder.py]，`--model moe` 开）
把"62 通道不是每个都有用"做成显式机制：
1. **每通道时序描述子**：depthwise conv → 取每通道 `(mean, std)` 统计 `[B, 62, 2]`。
2. **门控（筛通道）**：`gate = sigmoid(MLP(stats) + 学习偏置) ∈[0,1]`——本 trial 哪些通道有用。
3. **聚类（相似通道归一簇）**：通道 embedding 与 `E` 个专家原型做相似度 → softmax 软分配 `[B,62,E]`。
4. **路由专家**：每个专家只混合 `gate·assign` 加权后的通道 → `Conv1d(62→d_model/E)`，拼成 `d_model`。
5. 产出 `channel_balance` 负载均衡正则（防簇塌缩）。
`baseline` 模式则退化为普通 `Conv1d(62→d_model,k=1)` 空间混合。

### 2.2 时空主干 `SpatialTemporalEEGEncoder`
- 前端输出 → FiLM(stage 条件) → `num_blocks=6` 个 `ResidualTemporalBlock`（Conv-GN-GELU-Conv-FiLM 残差，
  stride `[2,2,2,2,1,1]` 下采样、dilation `[1,1,2,4,8,8]` 扩感受野）→ `adaptive_avg_pool1d` 到 `target_steps`
  （mel=122 / encodec=150）。
- 注：`stage_embedding(stage_idx)` 是任务条件（非身份）；`subject_condition`/`speaker_embedding` 已删除。

---

## 3. 回归解码头 `KaraOneEEG2Codec`（[model.py]，默认）

1. 编码序列 `seq [B,T,d_model]`；**有效长度掩码池化** `_masked_time_mean`：用数据集已算的 `eeg_valid_len`
   只对**非补零帧**求均值 → 句向量 `pooled`（避免 44% 补零稀释表示）。
2. 多个头（`content_dim` 绑定到目标维 D，使 proto 余弦维度一致）：
   - `content_seq_head`：逐帧内容特征 `[B,T,D]`。
   - `content_embed_head` / `content_classifier`：句级内容向量 + 11 类分类。
   - `global_head`：`pooled → speaker_dim` 的 **EEG 推断全局音色/语境向量**（取代被删的 speaker 查表）。
   - `clip_head`：`pooled → D`，用于跨模态 InfoNCE 对齐（见 §5）。
   - `log_rms_head`：响度。
3. 输出头（`num_experts=1` 时是普通 MLP）：`concat(content_seq, global)` → `pred_latent/pred_mel [B,T,D]`。

---

## 4. 扩散解码头 `EEGLatentDiffusion`（[diffusion.py]，`model.decoder=diffusion`）

真扩散，作用于所选目标（mel=80 或 encodec=128 自动适配）：
- **cosine 噪声调度**，`timesteps=1000`；**ε 预测**训练（`loss = MSE(ε̂, ε)`）。
- 去噪器：复用编码器出逐帧 EEG 条件，1D 卷积块用 **timestep 的 FiLM** 调制。
- **DDIM 采样**（默认 50 步），x0 带 `x0_clip` 钳制（防 ᾱ→0 时发散）。
- 为什么用它：回归是"求条件均值→灰糊"；扩散从噪声采样→**保方差、不塌缩**（但弱信号下会"多样但不准"）。

---

## 5. 损失（[losses.py `compute_losses`]，权重见 yaml）

- `recon_cos`(1−cos) + `recon_mse`：逐帧重建（朴素，对错位敏感）。
- **`dtw_recon_loss`（核心修复）**：带状(Sakoe-Chiba)**硬 DTW** 在 detach 的代价上求对齐路径，再按路径算 L1；
  梯度经对齐后回传。→ 对**跨 trial 起始/语速错位**不敏感（NeuroTalk 式）。
- **`clip_alignment`（跨模态对齐，Défossez 2022）**：EEG 句向量(`clip_head`) ↔ 音频目标摘要做**对称 InfoNCE**
  （同 trial 正、batch 内其它负，音频侧冻结）→ 治逐帧回归的均值化。
- `content_ce`/`supervised_contrastive`/`proto_cos`：基于**音素标签**的内容监督（标签是要解码的内容，非身份）。
- `log_rms_loss`、`std_match`、`channel_balance`（编码器 MoE）。
- 可选 **GAN**（[discriminator.py `AcousticDiscriminator`]，LSGAN + 特征匹配）在训练循环里交替 G/D，治糊。

---

## 6. 声码器 & 评估

### 6.1 声码器（[audio_features.py]）
- **mel → Griffin-Lim**：纯 scipy/numpy（无 torchaudio/HiFi-GAN，离线）。用**带动量的 Fast Griffin-Lim**
  （Perraudin，迭代 100，momentum 0.99）大幅减少"气泡/水声"伪迹（实测频谱收敛 0.254→0.119）。
- `encodec_latent → EnCodec 解码器`（音质最好）。

### 6.2 评估（[eval.py]，**诚实指标是重点**）
KaraOne 可"靠记类均值作弊"，所以原始 cosine 会误导。eval 同时报：
- `zeroeeg`（EEG 置零的预测）、`mean_latent`（全局均值）两个基线；
- **`pred_over_mean_cos_gain`**（相对均值的纯增益，**模型选择就用它**）、`pred_over_zero_cos_gain`；
- `pred_pcc`（mel 皮尔逊，对齐论文）、`pred_std_ratio_median`（→1 不塌缩）、
  `pred_pairwise_corr_median`（↓ 更多样）、within-subject retrieval top-1。

---

## 7. 当前结果与诚实局限（最干净一次：mel+MoE+DTW+GAN，20 epoch）

| | test（同被试，n=132） | subject_test（跨被试 P02/MM21，n=297） |
|---|---|---|
| `pred_over_mean_gain` | **+0.073** | **−0.003（≈0）** |
| `pred_pcc` | 0.679 | 0.695 |
| `std_ratio` | 0.537 | 0.481 |
| `content_acc` | 0.129（随机 0.091） | 0.108（≈随机） |

- **同被试**：有"微弱但真实"的正信号（包络层面 + 一丝音素，content_acc 比随机高约 40% 相对）。
- **跨被试**：≈0（不能泛化到没见过的人）——诚实且预期内。
- **结构性天花板**：KaraOne 音频又轻又短、**同音素跨 trial 不对齐**、为分类而非重建设计；逐词可懂度难起来。
  真正的杠杆在**数据**（为重建设计、说满整段、有对齐音频），不在调模型。
- 历史教训：最初"EnCodec 连续 latent + cosine 回归"会**塌缩**（std 0.15、两两相关 0.94、gain +0.02）；
  换 mel + DTW(+GAN) 后不塌缩（std 0.54–0.66、gain +0.07~0.09）。扩散头不塌缩但弱信号下"多样不准"。

---

## 8. 怎么跑

```bash
# 根目录一键（带时间戳输出，不覆盖）：训练→重建
bash run_mel.sh 20                       # mel+MoE+回归+DTW，20 epoch（治过拟合）
bash run_mel.sh 20 moe gan               # 开对抗
bash run_mel.sh 20 moe nogan diffusion   # 扩散解码头
bash run_mel.sh 20 moe gan regression -1 myexp   # 第5参=重建条数(-1全部)，第6参=运行名

# 单独重建全部测试样本（用训练好的 last.pt）
cd app && python scripts/synthesize_karaone.py \
  --checkpoint ../artifacts/outputs_karaone/<run>/checkpoints/last.pt \
  --split test --limit -1 --device cpu
```
产物：`artifacts/outputs_karaone/<run>/`：`metrics/training_curves.png`、`metrics/test_metrics.json`、`wav_*/`。
