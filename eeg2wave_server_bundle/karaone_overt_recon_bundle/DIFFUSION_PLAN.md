# 生成式重建方案：用条件潜在扩散摆脱"均值塌缩"

> 状态：规划 → 实现。本文件是实现依据；代码按此落地。

## 1. 背景与目标

当前回归模型（`KaraOneEEG2Codec`）逐帧回归 EnCodec 连续潜在，用 MSE+cos，
**结果塌缩到条件均值**（实测：`pred_std_ratio ≈ 0.15`、不同 trial 预测两两相关
`0.94`、`pred − mean` 增益仅 `0.02`）。也就是模型对每个输入吐出几乎相同的"平均声"。

**根因**：回归确定目标 = 求 `E[latent | EEG]`。当 EEG 信息弱时，条件均值≈全局均值 → 灰糊。

**目标**：换成**生成式**，从分布 `p(latent | EEG)` 里**采样**而不是求均值，
这样输出天然保方差、不塌缩。

## 2. 方案选择

| | A. 连续潜在条件扩散（**推荐**） | B. 离散 token 自回归 |
|---|---|---|
| 目标表示 | 复用现有连续 EnCodec 潜在缓存（150×128，已 z-score） | 需重抽 EnCodec 离散码 [8,150] |
| 解码 | 复用现成 EnCodec decoder（直接喂 latent） | EnCodec quantizer+decoder |
| 目标函数 | ε 预测（扩散）→ 采样保方差 | 逐 token 交叉熵 → 分类天然不塌缩 |
| 改动量 | **小**（加 1 个模块 + 1 个训练脚本 + 1 个采样脚本） | 大（改目标缓存 + AR 解码 + RVQ 展开） |
| 代表工作 | ATM/扩散先验、low-density EEG diffusion | DeWave、AVDE、AudioLM/VALL-E |

**选 A**：复用现有连续 latent + EnCodec decoder，改动最小；采样即不塌缩；
也正好把之前那个"假 diffusion"的 `refiner.py` 升级成**真**扩散。B 列为后续选项。

## 3. 架构（简单但是"真"扩散）

```text
                 ┌─────────── EEG 条件器（复用 SpatialTemporalEEGEncoder, 可带通道MoE）
EEG [62×1280] ──▶│  → cond_seq [B,150,Ccond]   （逐帧条件，和 latent 的 150 帧对齐）
                 └───────────────────────────────────────────────┐
                                                                  ▼
x_t [B,150,128]（加噪潜在）  +  t（扩散步）  ──▶  1D 条件去噪器  ──▶  ε̂ [B,150,128]
   去噪器：concat(x_t, cond_seq) → Conv1d 残差块 ×N，每块用 timestep 嵌入做 FiLM 调制
```

- **EEG 条件器**：复用 `SpatialTemporalEEGEncoder`，输出 `[B,d_model,150]` 转成逐帧
  条件 `cond_seq`；与扩散去噪器**联合训练**（不冻结，保证 EEG 真正参与）。
- **去噪器**（1D，无需上/下采样，序列短）：
  - 输入：`concat(x_t[128], cond_proj(cond_seq)[Cc])` → `Conv1d(in,H,1)`。
  - timestep：`sinusoidal(t) → MLP → t_emb`，在每个残差块做 **FiLM**（scale/shift）。
  - N 个残差块：`GroupNorm→GELU→Conv(k=5)→FiLM(t_emb)→GELU→Conv` + 残差。
  - 输出：`Conv1d(H,128,1)` = ε̂。
- 这本质上是把现有的 `ResidualDenoisingRefiner` 升级为：真 timestep + EEG 逐帧条件
  + 全噪声调度训练 + 迭代采样。

## 4. 扩散过程

- 目标 `x_0` = 数据集里**已 z-score 归一化**的 latent（`target_seq`，近似单位方差，适合扩散）。
- 前向：`x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε`，`ε ~ N(0,I)`。
- 噪声调度：**cosine**（Nichol & Dhariwal），`T=1000`。
- 目标函数：**ε 预测**，`loss = MSE(ε̂, ε)`（可选小权重 x0/latent 辅助，先不加）。
- 采样：**DDIM**（确定性，`η=0`，约 50 步），从 `N(0,I)` 反推到 `x_0`，
  再反归一化 + EnCodec decode → 波形。

## 5. 为什么这能解决塌缩（以及诚实的局限）

- **解决塌缩**：推理是从噪声采样并去噪，输出是分布的**样本**，方差天然保留
  （目标 `std_ratio → ≈1`，不再是 0.15），不同输入的样本**不再 0.94 高相关**。
- **诚实局限**：扩散**不会凭空造出 EEG 里没有的信息**。EEG 信噪比低时，样本会是
  "像语音但未必是对的词"——把"灰糊均值"换成了"多样但保真有限的语音"。
  这是已知权衡（视觉综述："高 CLIP ≠ 像原图"）。所以评估要同时看**保真**和**多样性**。
- 后续可加 **classifier-free guidance**（条件 dropout + 引导采样）提升 EEG 利用率；v1 先不加，避免过复杂。

## 6. 文件与接口

- 新增 `app/src/karaone_recon/diffusion.py`
  - `DiffusionConfig`（latent_dim, hidden, num_blocks, cond_ch, timesteps, ddim_steps, schedule…）
  - `make_cosine_schedule(T)`；`EEGLatentDiffusion(nn.Module)`：
    - `encode_cond(eeg, eeg_valid_len) -> cond_seq`
    - `denoise(x_t, t, cond_seq) -> eps_hat`
    - `loss(x0, eeg, eeg_valid_len) -> scalar`
    - `@torch.no_grad() sample(eeg, eeg_valid_len, steps) -> x0_hat`（DDIM）
  - 与现有 `KaraOneEEG2Codec` **互不影响**（新增路径，不改回归模型）。
- 新增 `app/scripts/train_karaone_diffusion.py`：训练循环 + 逐 epoch 画曲线（复用
  `plotting.py`）+ **采样式评估**（见 §7）。
- 新增 `app/scripts/synthesize_karaone_diffusion.py`：载入扩散 ckpt → 采样 latent →
  反归一化 → EnCodec decode → 写 wav（含 `original/oracle_codec/mean_latent/sample` 对照）。
- `app/configs/karaone.yaml` 增加 `diffusion:` 配置段。
- 文档：`METHOD.md`/`README.md` 增加扩散路径说明，并标注 `refiner.py` 被它取代。

## 7. 评估指标（重点：证明塌缩消失）

采样 val 集潜在后计算：
- `pred_std_ratio_median`：应 **≈1.0**（回归是 0.15）→ 塌缩消失的硬证据。
- `pred_pairwise_corr_median`：应从 **0.94 明显下降** → 样本有多样性。
- `pred_over_mean_cos_gain` / `pred_over_zero`：保真增益（大概率仍小，**诚实报告**）。
- within-subject retrieval top-1：采样 latent 是否比随机更能匹配到正确 trial。
- 多次采样的 trial 内方差（可选）：体现"生成多样性"。

## 8. 验证计划

1. `py_compile` 全部新文件。
2. 单元冒烟（随机张量，CPU）：`loss` 可反向；`sample` 形状正确；调度系数单调。
3. **抗塌缩对照**：随机小数据上训练几十步后采样，断言 `std_ratio` 明显 > 回归基线、
   `pairwise_corr` 明显 < 0.9。
4. 端到端：真数据上 `--max-steps` 跑几步训练 + 采样 + EnCodec decode 出 wav，确认链路通；删冒烟产物。

## 9. 不做什么（避免过度复杂）
- v1 不做 classifier-free guidance、不做离散 token AR、不做多尺度 U-Net（序列短，无需）。
- 不改动现有回归模型/脚本（扩散是并行新增路径）。
- 这些都在文档里列为"后续可选"。
