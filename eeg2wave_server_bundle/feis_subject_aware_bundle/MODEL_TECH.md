# FEIS EEG→语音 模型技术文档（detailed）

> 本文档说明 `feis_subject_aware_bundle` 当前模型**是什么、怎么做的、每一步细节**。
> 代码事实均来自 `app/src/` 与 `app/configs/`，可逐行对照。

---

## 0. 一句话定位

从 **FEIS 想象/感知语音 EEG** 直接生成语音声学表示，**完全不使用任何身份信息**
（无 subject/speaker/id 输入；代码用 `assert_*_identity_free_keys` 在 batch/字段层面强制拦截）。
当前有**两条并行的可运行路线**：

| 路线 | 包 | 目标表示 | 声码器 | 入口 |
|---|---|---|---|---|
| **A. direct（主线）** | `src/direct_eeg2speech` | EnCodec 连续潜在 [75×128] | 冻结 EnCodec 解码器 | `scripts/direct_train.py` / `run_direct_eeg_only.sh` |
| **B. mel（新增）** | `src/feis_mel` | log-mel [64×80] | Griffin-Lim(默认)/EnCodec | `scripts/feis_mel_train.py` |

> `configs/config.yaml`（含 VQ/STFT 字段）是**遗留配置**，与当前损失代码不符；当前实际用
> `configs/direct_eeg2speech.yaml`（A 路）与 `configs/feis_mel_align.yaml`（B 路）。

---

## 1. 数据（FEIS）

- **EEG**：14 通道（Emotiv EPOC），每个 trial 截成定长 `eeg_len=1280`（不足补零）。
- **阶段 stage**：`stimuli / thinking / speaking / resting` 等；每个 stage 是同一 trial 的不同时间窗。
- **标签 label**：FEIS 的词/音素提示（`num_labels`，约 16 类）。
- **音频**：FEIS 的 imagined/thinking **没有逐 trial 对齐音频**。两条路线用不同方式解决：
  - A 路：用 `feis_templates_encodec_latents.npz`——每条音频模板的 EnCodec 潜在（`audio_path` 为键）。
  - B 路：用 `feis_label_mel_banks.npz`——**按 label 聚合的 mel 参考库**（见 §3.1）。
- **清洗**：`is_clean_subject` 字段过滤异常被试（默认排除 `05`，可 `include_anomalous` 打开）。
- **划分**（[data.py `_assign_splits`]）：按 `(音频键/源, stage/label)` 分组，组内按 trial 顺序留出
  最后 1 个做 `test`、倒数第 2 个做 `val`、其余 `train`（`train/val_seen/test_seen/test_holdout`）。

---

## 2. 路线 A：`direct_eeg2speech`（EEG→EnCodec 潜在）

输入只有 **EEG + stage_idx**。整体：`EEG → 编码器 → Transformer → 粗潜在头 + 潜在扩散 → EnCodec 解码`。

### 2.1 编码器 `SpatialTemporalEEGEncoder`（[encoder.py]）
分两段，stage 通过 **FiLM**（`γ·h+β`，γ/β 由 cond 线性映射，零初始化）注入每一层：

1. **空间适配器**（二选一，开关 `use_channel_moe`）：
   - `SpatialAdapter`（默认关）：随机通道 dropout → `Conv1d(14→d_model, k=1)` → GroupNorm → GELU → FiLM。
   - **`ChannelClusterMoEAdapter`（开，老师要的"通道 MoE"）**：
     a. 每通道算 4 维统计量 `[mean, std, abs_mean, diff_std]`（时间维上）。
     b. `router`（统计量+cond → `num_experts` logits）→ **top-k softmax** 得到每通道的专家路由权重；
        `channel_gate`（同输入 → sigmoid）得到每通道**保留门**（"这通道有没有用"）。
     c. `num_experts` 个 `Conv1d(14→d_model,k=1)` 专家；每个专家吃 `x·(route_k·gate)`，求和混合 → FiLM。
     d. 产出 4 个 **MoE 正则量**：`load_balance`（专家使用均衡）、`channel_sparsity`（门稀疏）、
        `route_entropy`（路由置信）、`cluster_cohesion`（相似时序通道应同簇）。
2. **时间主干 `TemporalTrunk`**：`num_blocks` 个 `ResidualTemporalBlock`（Conv→GN→GELU→Conv→FiLM 残差），
   stride `[2,2,2,2,1…]` 逐步下采样、dilation `[1,1,2,4,8…]` 扩感受野，最后 `adaptive_avg_pool1d` 到
   `target_steps`（=75，对齐 EnCodec 帧数）。

### 2.2 序列模型 + 头（[model.py `DirectEEG2Speech`]）
- 编码器输出 `[B, d_model, 75]` → 转置 `[B,75,d_model]` → **3 层 TransformerEncoder**（pre-norm，GELU）。
- 三个头：
  - `latent_head`：每帧 → `target_dim=128` 的 **EnCodec 粗潜在** `pred_latent [B,75,128]`。
  - `content_classifier`：池化向量 → `num_labels`（辅助内容 CE）。
  - `log_rms_head`：池化向量 → 标量（预测响度 log-RMS）。

### 2.3 潜在扩散 `LatentDiffusion`（[diffusion.py]，开关 `use_latent_diffusion`）
**这是真扩散**（DDPM 训练 + DDIM 采样），在归一化 EnCodec 潜在空间：
- 调度：线性 β（1e-4→2e-2），`num_steps=200`，预计算 `ᾱ_t`。
- 去噪器 `LatentDenoiser`：Transformer，输入 = `noisy_latent` 线性 + **EEG 条件序列**线性 +
  **coarse_latent（粗潜在）**线性 + **timestep 正弦嵌入**；预测噪声 ε。
- 训练损失：`q_sample` 加噪 → 预测 ε 的 MSE（主）+ 由 ε 反推 x0 的 SmoothL1（辅）。
- 采样：`sample_ddim`（24 步，确定性），从 N(0,I) 反推 x0，条件用 EEG 序列 + 粗潜在。
- **角色**：`latent_head` 的粗潜在 = 既是非扩散消融基线，又作为扩散的条件/起点；扩散负责"精修+保多样"。

### 2.4 损失（[losses.py `compute_direct_losses`]，权重见 `direct_eeg2speech.yaml`）
针对**音频时序特性**设计，不是图像式逐像素：
- `recon_cos`(1−cos) + `recon_smoothl1`：逐帧潜在重建。
- `delta`/`delta2`：一阶/二阶**时间动态**匹配（治时间结构）。
- `temporal_envelope`：每帧能量包络匹配。
- `content_ce`：标签分类（弱内容监督）。
- `log_rms`：响度回归。
- `std_match` + `diversity`(批内两两相关) + `mean_margin`(**把预测推离全局均值**)：**抗塌缩**三件套。
- MoE 4 项正则 + `diffusion_loss`（ε-MSE）。

### 2.5 生成 & 四阶段运行
- 生成：`generate_full` → （粗潜在→）DDIM 采样潜在 → 反归一化 → **冻结 EnCodec 解码器** → 波形 → 按 log-RMS 调音量。
- `run_full_four_stage_moe_diffusion.sh`：对 **4 个 FEIS 阶段**（stimuli/thinking/speaking/resting）**各独立训练一个模型** +
  画图 + 合成（`test_holdout`，DDIM 24 步）。"四阶段"= 四个语音阶段，不是四个模型变体。

---

## 3. 路线 B：`feis_mel`（EEG→mel，标签库 + DTW 对齐）

输入**只有 EEG**（连 stage 都不用：`null_condition` 是常量；`stage_idx` 被显式列入禁止字段）。

### 3.1 目标：标签 mel 参考库（解决"想象语音无对齐 GT"）
[targets.py `MelLabelTargets`] 加载 `feis_label_mel_banks.npz`：
`banks [num_labels, K≤20, T=64, D=80]` = **每个 label 的 K 条真实朗读音频的 log-mel**（按维度 z-score）。
因为同一个词的不同朗读在时间上不对齐，所以不给单一目标，而给一**库**参考，训练时对齐到最接近的几条。

### 3.2 模型 `FEISEEGToMel`（[model.py]）
- 编码器：同 A 路的 `SpatialTemporalEEGEncoder`（带通道 MoE），输出 EEG memory token `[B,T_enc,d_model]`。
- **交叉注意力解码**：`target_steps=64` 个**可学习 query** → `TransformerDecoder`（2 层）对 EEG memory 做
  cross-attention → `mel_head` → `pred_mel [B,64,80]`。（这点和 KaraOne 的卷积头不同，用 query+cross-attn 显式生成 64 帧。）
- 另有 `content_classifier`、`contrast_head`（标签对比）、`log_rms_head`。

### 3.3 损失（[losses.py `compute_feis_mel_losses`]）
- **softmin-DTW mel 损失（核心）**：对 label 库里**朴素 L1 最近的 top-k(3) 条**参考，各做**带状 DTW** 对齐后算 L1，
  再按 DTW 代价 **softmin 加权**求和。→ 既解决跨朗读的**时间错位**，又自动挑最像的参考（不强迫匹配某一条）。
- `content_ce`（标签分类）+ `label_contrastive`（句向量 vs 标签原型的 InfoNCE 检索）+ `log_rms` + MoE 正则。
- 可选 **GAN**（[gan.py `AcousticPatchDiscriminator`]，BCE 对抗 + 特征匹配，治糊）；
  可选 **扩散**（[diffusion.py] 复用 A 路 `LatentDiffusion`，作用于 mel）。开关在 `feis_mel_align.yaml`。

### 3.4 声码器（[audio.py]）
- `wav_to_logmel` / `logmel_to_wav`：纯 scipy STFT + mel 滤波器组；mel→linear 用伪逆，再 **Griffin-Lim**（默认 48 迭代）。
- 也可切 EnCodec（`vocoder.kind`）。

---

## 4. 关键设计点（两条路线共有）

1. **彻底身份无关**：模型/批/字段层面拦截任何 subject/speaker/id；A 路允许 stage，B 路连 stage 都不用。
2. **通道 MoE**：把"14 个 EEG 通道不是每个都有用"做成**显式路由+门控+聚类正则**，在进入时间编码前筛选/聚合通道。
3. **抗塌缩**：A 路用 std/diversity/mean-margin 三正则；B 路靠 DTW 对齐 + 可选 GAN；两路都可选真扩散（采样保方差）。
4. **真扩散**：`LatentDiffusion` = DDPM(ε 预测) + DDIM 采样，A 路在 EnCodec 潜在、B 路在 mel；粗预测作条件。

---

## 5. 诚实局限

- FEIS imagined/thinking **无逐 trial 对齐音频**：A 路用音频模板潜在、B 路用 label 库——目标本质是"该类的参考发音"，
  不是"这一 trial 真实说了什么"。所以可学的主要是**类别级声学包络**，不是个体逐 trial 内容。
- 14 通道、低 SNR、短窗（1s）→ 跨被试/逐 trial 可懂度的天花板很低；指标应看**类别检索/包络相似**而非逐词还原。
- Griffin-Lim 是无神经声码器，音质偏机械（mel 级指标 PCC 不受影响）。

---

## 6. 怎么跑

```bash
cd app
# A 路（主线，EnCodec 潜在 + 扩散）
bash run_direct_eeg_only.sh                  # 或 run_full_four_stage_moe_diffusion.sh 跑四个阶段
# B 路（mel + DTW，按 feis_mel_align.yaml）
python scripts/build_feis_mel_targets.py     # 先建 label mel 库
python scripts/feis_mel_train.py --config configs/feis_mel_align.yaml
python scripts/feis_mel_synthesize.py --checkpoint <run>/checkpoints/best.pt
```
