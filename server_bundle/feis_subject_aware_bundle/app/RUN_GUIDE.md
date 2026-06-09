# EEG2Wave Demo 运行说明

## 1. 这个 bundle 是做什么的

这是第一版最小化 demo，用来完成：

- `FEIS thinking EEG -> 离散 EEG token -> 固定长度 waveform`

现在这份 bundle 也支持第二条对照路径：

- `FEIS stimuli EEG -> 离散 EEG token -> 固定长度 waveform`

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
- 训练目标 wav 长度：`1.0 s = 16000 samples`

注意：

- FEIS 的 `5 s` 指的是 EEG 的 imagined-speech window
- 不是 `5 s` 的 wav
- FEIS 的 wav 通常只有 `1 s` 左右，所以当前默认直接统一到 `1.0 s`

## 6. 安装方式

这里默认你已经有可用的 Conda 环境：

- `eegvoice`

### 第一步：激活环境

```bash
conda activate eegvoice
python -m pip install --upgrade pip
```

### 第二步：确认 PyTorch 可用

如果你的 `eegvoice` 环境里已经有 PyTorch，就不用重复安装。

例如你现在环境里已经是：

```bash
python -c "import torch; print(torch.__version__)"
```

如果输出正常，例如 `2.11.0`，就说明 torch 已经可用。

### 第三步：安装其余依赖

在 `eeg2wave_demo_bundle/` 目录下执行：

```bash
python -m pip install -r requirements.txt
```

### 说明：为什么这里不再要求 torchaudio

这个 bundle 当前不依赖 `torchaudio`。

原因是：

- wav 读取使用的是 `scipy.io.wavfile`
- 重采样使用的是 `scipy.signal.resample_poly`
- 训练和推理代码中没有调用 `torchaudio`

所以如果你遇到这类报错：

```text
ERROR: Could not find a version that satisfies the requirement torchaudio<3.0,>=2.4
```

对这个 bundle 来说，正确处理方式不是继续硬装 `torchaudio`，而是直接把它从依赖中去掉。

现在这个 bundle 的 `requirements.txt` 已经按这个逻辑整理好了。

## 7. 第一轮训练怎么跑

建议先只跑一个 subject，确认整条链路是通的。

例如：

```bash
python scripts/train.py --subject 01
```

现在 `train.py` 默认是“同一 stage 下按 subject 顺序增量训练”，并且同一 `stage` 只维护一个唯一的工作 checkpoint：

- `thinking` 只有一个：`outputs/checkpoints/thinking_best.pt`
- `stimuli` 只有一个：`outputs/checkpoints/stimuli_best.pt`

- `subject 01` 默认从头开始
- `subject 02` 默认加载当前的 `thinking_best.pt` 再继续训练
- `subject 03` 默认继续加载更新后的 `thinking_best.pt`
- 依此类推

所以像下面这种写法：

```bash
for subj in 01 02 03 04; do
  python scripts/train.py --stage thinking --subject "${subj}"
done
```

含义就是：

- 先训练 `thinking_subject_01`
- 再把它的结果增量传给 `thinking_subject_02`
- 再传给 `thinking_subject_03`
- 再传给 `thinking_subject_04`

如果你想强制某个 subject 从头训练，可以加：

```bash
python scripts/train.py --stage thinking --subject 04 --scratch
```

训练完成后会写出：

- checkpoint：`outputs-thinking/checkpoints/thinking_best.pt`
- 训练摘要：`outputs-thinking/metrics/thinking_subject_01_train_summary.json`
- 逐 epoch 历史：`outputs-thinking/metrics/thinking_subject_01_history.json`
- 训练曲线图：`outputs-thinking/metrics/thinking_subject_01_training_curves.png`

然后做推理：

```bash
python scripts/infer.py --subject 01
```

推理完成后会写出：

- 重建 wav：`outputs-thinking/recon_wavs/thinking/subject_01/`
- 指标：`outputs-thinking/metrics/thinking_subject_01_metrics.json`

注意：

- `infer.py` 默认不会再按 `subject` 猜 checkpoint
- 它会直接读取当前 stage 唯一的 `best.pt`
- 也就是 `thinking -> outputs-thinking/checkpoints/thinking_best.pt`
- 或者 `stimuli -> outputs-stimuli/checkpoints/stimuli_best.pt`

## 7.1 输出目录默认分开

## 8. Phase 2: Retrieval Waveform

对齐模型训练完之后，Phase 2 的主实验入口是：

```bash
python scripts/eval_alignment_retrieval.py \
  --config configs/alignment_ssl_local.yaml \
  --protocol G \
  --split test
```

`Protocol S`：

```bash
python scripts/eval_alignment_retrieval.py \
  --config configs/alignment_ssl_local.yaml \
  --protocol S \
  --subject 01 \
  --split test
```

`Protocol U`：

```bash
python scripts/eval_alignment_retrieval.py \
  --config configs/alignment_ssl_local.yaml \
  --protocol U \
  --holdout-subject 21 \
  --split test
```

这个脚本会：

- EEG -> predicted embedding
- predicted embedding -> top-5 template retrieval
- 保存 top-1 retrieved waveform
- 输出 exact/label top-k
- 输出 waveform-space `NTA`
- `Protocol U` 下额外单列 oracle ceiling

结果默认在：

```text
../artifacts/outputs_alignment/<run_name>/retrieval/test/<policy>/
```

总汇总 JSON 在：

```text
../artifacts/outputs_alignment/<run_name>/metrics/test_retrieval_evaluation.json
```

## 9. Speech Space Analysis

```bash
python scripts/analyze_alignment_space.py --config configs/alignment_ssl_local.yaml
```

输出：

- `pca_by_subject.png`
- `pca_by_label.png`
- `space_summary.json`
- `space_summary.md`

默认目录：

```text
../artifacts/outputs_alignment/template_space/feis_subject_templates_ssl/
```

## 10. Consolidated Phase 2 Report

```bash
python scripts/report_phase2.py \
  --config configs/alignment_ssl_local.yaml \
  --protocol G \
  --split test
```

它会把：

- alignment audit
- retrieval benchmark
- speech-space summary
- Route A/B/C decoder memo
- 主方向建议

整合到一个 markdown：

```text
../artifacts/outputs_alignment/<run_name>/metrics/test_phase2_report.md
```

为了避免 `thinking` 和 `stimuli` 共用同一个 `outputs/`，当前代码默认会自动分开写：

- `thinking -> outputs-thinking/`
- `stimuli -> outputs-stimuli/`

如果你手动传了 `--output-root`，脚本就会尊重你传入的目录，不再自动加后缀。

## 8. 两条路径怎么分别跑

### 8.1 imagined 路径

也就是 `thinking EEG -> wav`：

```bash
python scripts/train.py --stage thinking --subject 01
python scripts/infer.py --stage thinking --subject 01
```

如果你顺序跑多个 subject，这条路径默认就是增量训练。

### 8.2 实际听到路径

也就是 `stimuli EEG -> wav`：

```bash
python scripts/train.py --stage stimuli --subject 01
python scripts/infer.py --stage stimuli --subject 01
```

同样地，如果你顺序跑多个 subject，这条路径也默认是增量训练。

### 8.3 批量比较两条路径

如果你想把 `thinking` 和 `stimuli` 都跑一遍，bundle 里已经放了批量脚本：

```bash
bash scripts/run_stage_compare.sh
```

这个脚本会对 `01-21` 全部 subject 依次执行：

- `thinking -> train + infer`
- `stimuli -> train + infer`
- 每个 stage 跑完后自动生成一张全量汇总图

对应输出是：

- `outputs-thinking/metrics/thinking_full_training_summary.png`
- `outputs-stimuli/metrics/stimuli_full_training_summary.png`

如果你只想单独批量训练某一条路径，也可以手动跑：

```bash
for subj in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21; do
  python scripts/train.py --stage thinking --subject "${subj}"
done
```

或者：

```bash
for subj in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21; do
  python scripts/train.py --stage stimuli --subject "${subj}"
done
```

## 9. 当前训练目标是什么

总 loss 形式是：

```text
L_total = L1_waveform + lambda_stft * L_stft + lambda_vq * L_vq + lambda_cls * L_cls
```

具体含义：

- `L1_waveform`
  - 约束重建波形在 sample-level 上接近目标 wav
- `multi-resolution STFT loss`
  - 约束重建结果在频谱结构上接近目标 wav
- `VQ loss`
  - 强制 encoder 必须经过离散 token bottleneck，而不是退化成纯连续 latent
- `prompt classification loss`
  - 用 EEG token 同时预测当前 prompt label
  - 这个分支不会直接输出 wav，但会强迫 token 保留更明确的 prompt 信息

## 10. 为什么第一版选这组 loss

这是当前最小但有效的组合：

- 只有 `L1` 往往会让波形发糊
- 加 `STFT` 后，频谱形状更容易对齐
- `VQ` 是必须的，因为你的研究目标明确要求 EEG 先变成 token
- 加 `prompt classification` 的原因，是当前 FEIS 直接做 raw waveform 很容易塌成“同一 subject 输出一个平均 wav”，所以需要一个更明确的 prompt 监督把 token 拉开

这版故意不加：

- GAN
- diffusion
- audio tokenizer
- speaker loss
- phoneme classifier

原因很简单：

- 第一版先确保链路最短、最容易跑通

## 11. 这次增强了什么

和上一版相比，当前代码做了几项专门针对“塌缩输出”的增强：

- 同时支持 `thinking` 和 `stimuli` 两条路径，方便直接比较 imagined 与 actually-heard EEG
- 加了一个轻量的 prompt 分类辅助头
- checkpoint、指标和重建结果按 `stage` 分开保存，避免混淆
- 默认把目标 wav 长度改成 `1.0 s`，更贴近 FEIS canonical wav 的真实长度
- 默认把 `codebook_size` 降到 `64`，提高 token 实际利用率，尽量缓解 `px` 过低的问题

## 12. 结果怎么看

建议你不要只看 `train loss`，而要同时看这三个：

- `classification_accuracy`
  - 看 token 是否真的带有 prompt 信息
- `nearest_template_accuracy`
  - 看重建 wav 是否更接近正确 prompt 的 canonical template
- `px`
  - 看 VQ codebook 有没有严重塌缩

一个比较理想的现象是：

- `stimuli` 路径明显优于 `thinking`
- `classification_accuracy` 明显高于随机
- `nearest_template_accuracy` 也随之提高
- `px` 不再长期贴着 `1.0`

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
conda activate eegvoice
python -m pip install -r requirements.txt
python scripts/train.py --subject 01
python scripts/infer.py --subject 01
```
