# FEIS Factored v2 方案（falsification-first，科学与工程分家）

> 针对 `factored_visual_report_20260611_0102` 的诊断结论制定。
> 核心判断：当前有**两个互相独立的失败**，必须分开处理——
> - **P1 内容（科学问题）**：EEG→content 在随机线，zero-EEG 持平甚至反超。**任何 loss 重配都修不了**，这是信号/SNR 上限。
> - **P2 重建塌缩（工程问题）**：pred latent 趋均值 → wav 偏小声（仅 ref 的 17%）、互相关高、谱平。codec 本身健康（oracle 解码 RMS 0.077、有结构），病在 EEG→latent 生成器。**可修，但修好主要恢复"嗓音"那免费的一半，不是内容。**
>
> 战略：**先用最便宜的实验把 P1 钉死（go/no-go），再决定要不要投入 P2 的机器**。
> 不堆 5 个新损失；不加 phoneme 头（FEIS label 本身就是音素，等于现有内容分类器）；
> 反塌缩验收只认一个不可被自身方法刷掉的指标：`content_top1 − zeroeeg`。

## 成功 / 失败判据（全局）

| 维度 | 通过线 | 不可造假性 |
|---|---|---|
| **内容增益(核心)** | val `content_top1 − zeroeeg ≥ 0.03`，且置换检验 p<0.05 | 高：zero-EEG 对照 + 标签置换 |
| codec 健康 | oracle 解码 vs 原始音频：RMS 比 0.7–1.3、mel 距离明显 < mean-latent | 高 |
| 能量/音量 | pred_scaled / 原始 RMS 中位数 ∈ [0.5,1.5] | 中（用模型预测 RMS，不偷 target） |
| 反塌缩(次要) | pred/target std 比 ≥0.5、pred 互相关中位数 <0.25 | **低：会被 std-match 等方法刷，仅作诊断** |

---

## Stage 0 — Codec QC（半天，定 P2 前提）

脚本 `scripts/factored_recon_eval.py`，对每个 checkpoint/split 抽样 cell，生成五路对照并量化：

- `original_ref.wav`：原始录音文件（`audio_paths`，RMS 归一到提取时的 0.08）。
- `target_oracle.wav`：真实 target latent + 本 cell `decoder_scales` 解码 = **重建上限**。
- `mean_latent.wav`：全局 target latent 均值解码 = **塌缩下限参照**。
- `pred_unscaled.wav`：模型预测 latent 直接解码。
- `pred_scaled.wav`：预测 latent 解码后，用**模型预测的 RMS**（`pred_log_rms`）缩放，禁止偷用 target RMS。

输出：`audio_qc.json`（含 codec 健康判定）、`collapse_diagnostics.json`、`recon_pairs.csv`、`listening_manifest.csv`。

**闸门**：若 `target_oracle` 的 RMS 中位数 < 原始的 0.5，或 oracle 的 mel/STFT 距离不明显优于 mean-latent → **codec 提取/解码路径有问题，先修 extraction/scales，暂停训练**。（依现有证据预期会通过。）

---

## Stage 1 — 先把"内容可解性"（P1）钉死（优先于任何新损失）

不碰生成器。两件事：

1. **独立解码探针** `scripts/content_probe.py`：受试内 16 类，简单分类器（多项 logistic / 浅层），k 折 CV + **标签置换检验**（≥200 次）出真实 p 值与 chance 带；分 `stimuli`/`thinking`；粗类别按**正确的多数类基线**比。这一步回答"EEG 里到底有没有内容信号"，生成模型不可能超过它能拿到的信息量。
2. **eval 修干净** `src/feis_factored/eval.py`：headline 改成 `content_top1 − zeroeeg`；粗类别也加 zero-EEG 基线；holdout 留格子提供 `holdout_random` 随机排布，堵掉常数预测器 gaming（或只报 EEG−zeroeeg 差）。

**决策闸门**：若探针在 stimuli/thinking 两阶段都 ≤ chance（置换 p>0.05）→ **内容在 FEIS 上不可解，停止堆生成模型**，按路线图转听觉感知数据集 / 预训练。这是一个对照扎实、可发表、能向导师交代的干净负结果。

---

## Stage 2 — 仅当 Stage 1 证明存在可复现、超 chance 的内容信号，才修塌缩（最小化）

- **必做且唯一第一优先：能量/scale 建模**。模型加 `pred_log_rms` 头预测 decoded wav 目标能量；合成时按预测 scale 反归一化/缩放——这一条就解决"小声"（17% RMS）。
- 最多再加**一个**分布项：`latent_std` 匹配（默认关，`lambda_std=0`，需要时再开并消融）。`recon_mse` 权重从 1.0 降到 0.25，减少均值奖励。
- **不实现** InfoNCE / 一阶差分 / phoneme 头——它们要么与现有项重复，要么主要给"虚假希望"。
- 验收只认 val `content_top1 − zeroeeg ≥ 0.03`；std 比、互相关只当诊断并标注"可被方法刷"。

---

## Stage 3 — checkpoint / val 卫生（无条件做）

`scripts/factored_train.py`：
- 新增 `val_seen` 划分：seen cell 的**倒数第 2 个重复**做 validation，最后 1 个留给 `test_seen`（test 不再参与选模型）。
- `best.pt` 用 **val `content_top1 − zeroeeg`（内容增益）**选择；若最优增益 ≤0，checkpoint 标记 `no_eeg_content_gain=true`。
- 每 epoch 写 `metrics/history.jsonl` 与 `history.csv`（train loss、content acc、zero-EEG acc、recon cos、std ratio、pred RMS、adv acc）。

---

## Stage 4 — speaker 主张（出结果后再做）

`factored_synthesize.py` + 验证脚本：固定内容、两受试间 speaker 插值的听感集；
ECAPA/x-vector 相似度、同受试 vs 异受试检索。**框成"嗓音=免费的一半"，不是"从脑信号解码"**。

---

## 退出条件（提前写死，避免无限堆模型）

- Stage 0 不过：修 codec，不训模型。
- Stage 1 探针 ≤ chance：**停 FEIS-only**，转 KaraOne/预训练或更强 EEG encoder。
- Stage 1 过但 Stage 2 仍打不过 zero-EEG（增益<0.03）：同样停 FEIS-only，转更大数据 / 预训练。

## 不做的事（明确边界）

- 不重抽 FEIS EnCodec target（cache 已含 `target_rms/log_rms/decoder_scales/audio_paths`）。
- 不改 FEIS EEG 预处理。
- 不把真实 label 或 target RMS 注入推理期 decoder（只作训练监督）。
- 不回退旧的 raw-waveform 回归基线。
