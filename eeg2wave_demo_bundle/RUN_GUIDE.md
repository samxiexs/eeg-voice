# EEG2Wave Demo 运行说明

## 1. 这个 bundle 是做什么的

这是第一版最小化 demo，用来完成：

- `FEIS thinking EEG -> 离散 EEG token -> 固定长度 waveform`

它的目标不是高保真语音合成，也不是最终版 speech decoding。

它的目标是先验证一件更基础的事：

- imagined speech 的 EEG，在经过离散 token bottleneck 之后，是否还保留足够信息，去恢复正确 prompt 对应的 waveform 原型

## 2. 这版模型到底在学什么

对于 FEIS，这个 bundle 里的每条训练样本是：

- 输入：一段 `5 s` 的 `thinking` EEG
- 目标：同一个 `subject + label` 对应的 canonical wav

所以它学的不是：

- 同 trial 的自然语音复现
- 跨被试语音合成
- 完整句子的可懂语音生成

它学的是一个更窄、但更适合第一版 demo 的任务：

- 从 imagined speech EEG 中，恢复该 prompt 对应的 waveform 模板

最准确的理解方式是：

- `imagined EEG -> prompt-conditioned waveform prototype reconstruction`

## 3. 为什么这和最终语音目标只“部分对齐”

这版 demo 和最终 EEG 语音解码目标是部分对齐的。

### 已经对齐的部分

- 输入确实是 imagined-speech EEG
- 中间确实经过了离散 token bottleneck
- 输出确实是 waveform
- 如果成功，说明 EEG token 里保留了 prompt 相关的声学结构

### 还没有完全对齐的部分

- FEIS 的 target wav 是 canonical wav，不是 same-trial wav
- wav 通常很短，更像 prompt 音频，不是自然连续语音
- 当前是单被试训练，不是跨被试泛化
- 当前输出是固定长度，不是变长语音

所以这版应该理解为：

- 一个“EEG 到语音原型”的验证 demo

而不是：

- 最终版的自然语音解码系统

## 4. 文件结构

```text
eeg2wave_demo_bundle/
  RUN_GUIDE.md
  README.md
  requirements.txt
  configs/
    config.yaml
  src/
    dataset.py
    model.py
    losses.py
    utils.py
  scripts/
    train.py
    infer.py
    prepare_local_bundle.sh
  data/
    feis/
      manifest.json
      trials.csv
      segments.csv
      subjects/
      audio/
  outputs/
    checkpoints/
    recon_wavs/
    metrics/
```

## 5. 当前数据假设

这个 bundle 默认读取内部数据目录：

```text
data/feis
```

当前代码默认假设：

- 数据集：`FEIS`
- 阶段：`thinking`
- EEG shape：`[14, 1280]`
- EEG 时长：`5 s`
- 音频采样率：`16 kHz`
- 训练目标 wav 长度：`1.5 s = 24000 samples`

注意：

- FEIS 的 `5 s` 指的是 EEG 的 imagined-speech window
- 不是 `5 s` 的 wav
- FEIS 的 wav 通常只有 `1 s` 左右，只是训练时统一裁剪/补零到 `1.5 s`

## 6. 安装方式

### 第一步：创建环境

例如：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 第二步：安装 PyTorch

如果服务器有 CUDA，请先安装和 CUDA 匹配的 PyTorch 版本。

官方安装说明：

- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)

例如 CPU-only：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 第三步：安装其余依赖

在 `eeg2wave_demo_bundle/` 目录下执行：

```bash
python -m pip install -r requirements.txt
```

## 7. 第一轮训练怎么跑

建议先只跑一个 subject，确认整条链路是通的。

例如：

```bash
python scripts/train.py --subject 01
```

训练完成后会写出：

- checkpoint：`outputs/checkpoints/subject_01_best.pt`
- 训练摘要：`outputs/metrics/subject_01_train_summary.json`

然后做推理：

```bash
python scripts/infer.py --subject 01
```

推理完成后会写出：

- 重建 wav：`outputs/recon_wavs/subject_01/`
- 指标：`outputs/metrics/subject_01_metrics.json`

## 8. 批量训练怎么跑

如果单个 subject 跑通了，可以批量训练全部 FEIS subject：

```bash
for subj in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21; do
  python scripts/train.py --subject "${subj}"
done
```

## 9. 当前训练目标是什么

总 loss 形式是：

```text
L_total = L1_waveform + lambda_stft * L_stft + lambda_vq * L_vq
```

具体含义：

- `L1_waveform`
  - 约束重建波形在 sample-level 上接近目标 wav
- `multi-resolution STFT loss`
  - 约束重建结果在频谱结构上接近目标 wav
- `VQ loss`
  - 强制 encoder 必须经过离散 token bottleneck，而不是退化成纯连续 latent

## 10. 为什么第一版选这组 loss

这是当前最小但有效的组合：

- 只有 `L1` 往往会让波形发糊
- 加 `STFT` 后，频谱形状更容易对齐
- `VQ` 是必须的，因为你的研究目标明确要求 EEG 先变成 token

这版故意不加：

- GAN
- diffusion
- audio tokenizer
- speaker loss
- phoneme classifier

原因很简单：

- 第一版先确保链路最短、最容易跑通

## 11. 怎么判断训练有没有在正常工作

第一轮训练主要看四件事：

### 基础 sanity check

- total loss 能下降
- val loss 大体跟 train loss 同方向
- 输出 wav 不是全零
- 输出 wav 不是纯噪声

### 核心指标

最重要的指标是：

- `nearest_template_accuracy`

它的含义是：

1. 用 EEG 重建出 wav
2. 把这条重建 wav 和该 subject 的所有 canonical prompt wav 比较
3. 找距离最近的 template
4. 看预测 label 是否等于真实 label

对于 FEIS 的 16 个 label，随机 baseline 是：

- `1 / 16 = 6.25%`

如果这个准确率明显高于随机，就说明这个 demo 在做有意义的事。

## 12. 如何微调这版 demo

这里的“微调”分成两类：

- A. 在当前第一版框架内微调
- B. 朝最终语音目标继续推进

## A. 在当前第一版框架内微调

适用场景：

- 训练不稳定
- 重建 wav 太糊
- 最近模板准确率太低

建议按下面顺序调：

### 1. 调整 audio 长度

当前默认：

- `1.5 s`

如果词尾经常被截断：

- 改成 `2.0 s`

如果训练太难、太不稳定：

- 改成 `1.0 s`

改动位置：

- `config.yaml -> audio.duration_sec`
- `config.yaml -> audio.n_samples`

### 2. 调整 codebook size

当前默认：

- `512`

可尝试：

- `256`：如果数据太小、训练不稳
- `1024`：如果 token 容量不够

改动位置：

- `config.yaml -> model.codebook_size`

### 3. 调整 VQ 压力

当前默认：

- `vq_beta = 0.25`
- `lambda_vq = 0.1`

如果模型基本无视 token bottleneck：

- 略微提高 `lambda_vq`

如果 reconstruction 因为 VQ 约束太强而崩：

- 略微降低 `lambda_vq`

### 4. 调整 waveform 和频谱的权重平衡

当前默认：

- `lambda_stft = 0.5`

如果重建 wav 很毛躁、频谱不稳：

- 提高 `lambda_stft`

如果波形能量太弱、过于平滑：

- 降低 `lambda_stft`

## B. 如何朝最终语音目标继续推进

这部分更重要，因为它决定这版 demo 和最终研究路线怎么衔接。

推荐路线：

### Stage 1：FEIS thinking -> canonical wav

它验证的是：

- imagined EEG 经过 token 压缩后，能否恢复 prompt-level 的 waveform 原型

### Stage 2：加入更强监督

下一步最合适的是：

- 在 FEIS 中加入 `speaking` 作为辅助阶段
- 或者切到 `KaraOne`

原因：

- `KaraOne` 的 wav 更接近 same-trial overt speech
- 和真正语音目标的对齐会更强

### Stage 3：进一步逼近真正 speech decoding

如果你要更接近最终“语音复现”，后面通常需要：

- same-trial audio
- 更丰富的词汇或语音材料
- 变长 waveform 输出
- 甚至先做 spectrogram / SSL speech target，再回到 waveform

## 13. 当前训练目标和最终语音的对齐关系

### 当前已经对齐的内容

这版 loss 确实已经在训练：

- EEG 到声学目标的映射
- 离散 token 压缩
- 直接 waveform 解码
- prompt 相关的频谱结构恢复

### 当前还没完全对齐的内容

它还不是最终 speech objective，主要因为：

- FEIS 的 target wav 不是 same-trial
- 没有完整 phonetic timing 监督
- 没有显式 intelligibility 目标
- 固定长度输出简化了问题

所以这版 objective 最合适的定位是：

- 原型级 acoustic alignment objective

而不是：

- 完整自然语音 intelligibility objective

## 14. 推荐的下一步实验

建议顺序：

1. 先跑通一个 subject
2. 再跑完 21 个 FEIS subject
3. 比较不同 subject 的 nearest-template accuracy
4. 再试 `thinking` / `stimuli` / `speaking`
5. 如果 FEIS 结论稳定，再引入 KaraOne

## 15. 最短命令总结

```bash
python -m pip install -r requirements.txt
python scripts/train.py --subject 01
python scripts/infer.py --subject 01
```
