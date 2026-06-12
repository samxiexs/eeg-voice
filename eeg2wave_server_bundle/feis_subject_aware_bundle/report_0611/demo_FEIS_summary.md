# EEG-to-Speech Demo: FEIS Dataset

**2026-06-11**

---

## 1. Executive Summary

**当前思路：**

```text
EEG -> speech/content representation -> EnCodec latent -> frozen EnCodec decoder -> wav
```

**目前结果：**

- EnCodec codec path 跑得通效果也好，毕竟是很多文献了：直接用真实 target latent 解码出来的音频 与原始 wav 接近。

  真实参考 wav
  -> EnCodec encoder
  -> target latent
  -> EnCodec decoder
  -> target_oracle wav
- FEIS EEG content signal 不稳定：*真实 EEG 在内容解码上没有稳定超过 zero-EEG 的随机结果*。
- 我觉得还是需要更多数据集，正在找不同数据集来扩充

## 2. FEIS 数据集分析

**FEIS 是 14 通道低密度 imagined speech EEG 数据集。**

| 项目                       | 内容                                                                                                                                                           |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **EEG 设备**         | **Emotiv EPOC**                                                                                                                                          |
| **通道数**           | **14，事实证明还是少了**                                                                                                                                 |
| **被试**             | **21 名英文被试**                                                                                                                                        |
| **prompt (stimuli)** | **16 个音素/音节/元音词，每个被试实验期间听自己提前遵从实验录制的<br />(**p, t, k, f, s, sh, v, z, zh, m, n, ng, fleece, goose, trap, thought**)** |
| **repetition**       | **单个被试试验期间每个 prompt 约 10 次**                                                                                                                |
| **阶段**             | **`stimuli 5s`, `articulators` 1s, `thinking` 5s, `speaking` 5s, `resting 5s`**                                                                |
| **音频目标**         | **每个 `(subject, prompt)` 一条 subject-specific wav**                                                                                                 |

**FEIS 的关键限制：**

- **14 通道低密度 EEG，缺少 C3/Cz/C4 等中央运动区电极，还是粗糙了。**
- **每个 `(subject, prompt)` 只有一条音频模板，10 个 EEG trial 共享同一 wav。因此模型无法学习 trial-level pronunciation variation，只能学习 subject × prompt 的 canonical target。**

以Subject 01的第一条 stimulis 为例（从5s才是是因为前面是resting，实验人员给的csv没有记录），横坐标为时间，纵坐标是offset后的 EEG 电压幅值，offset只是为了将把不同通道错开显示


![FEIS trial 阶段时间轴示例](assets/feis_window_example.png)

![FEIS thinking 阶段 EEG 示例](assets/feis_thinking_eeg_example.png)

同一 prompt (以下取辅音 f 为例) 在不同 subject 下是不同真人录音，因此目标不是 16 个全局模板，而是 subject-specific speech target。（热图，纵轴是频率，颜色是强度，横轴是时间）

![同 prompt 不同被试的 subject-specific mel 对比](assets/feis_subject_specific_mel.png)

---

## 3. Demo 思路

### 3.1 从 waveform regression 转向 representation reconstruction

最开始，我直接尝试eeg -> waveform，就是直接全部仿照前几周讨论的论文算法和代码，走VQ-VAE 风格的 EEG-to-waveform 模型：用 1D CNN 编码 EEG，经 VQ codebook 离散化，再用反卷积 decoder 直接生成 1 秒 wav，并用 L1 + STFT/log-STFT + RMS/envelope + 分类辅助损失训练；但输出塌缩成低频平均波形，听感不可辨，出来的声音根本听不了 :(

原因可能是原始波形回归在 EEG content 弱时容易收敛到均值波形，特别是在thinking这方面。

**过去一周我尝试将路线改为：**

```text
EEG -> EnCodec latent -> frozen decoder -> wav
```

### 3.2  Content × Speaker decomposition

**FEIS 的 target 是 subject-specific 的，所以模型必须区分：**

| 因素                      | 来源                 | 处理方式               |
| ------------------------- | -------------------- | ---------------------- |
| **content**         | **EEG**        | **从 EEG 解码**  |
| **speaker / voice** | **subject id** | **作为条件输入** |

但其实肯定在 eeg 侧也有 speaker 的信息，因为是 demo 考虑得比较简单

**当前 Demo 模型结构：**

单独eeg

```text
  EEG [14, 1280]，14 个通道，1280 则是该阶段的时间采样点 256Hz * 5sec，Batch: [B, 14, 1280]
  -> Spatial-Temporal EEG Encoder 其实就是空间(通道)-时间，这里encoder直接参考了论文github，
  -> content_seq [75, 128] / content_embed [128] / content_logits [B, 16]

# subject_id [B]
#  -> speaker_embedding 查表，[num_subjects, speaker_dim], [21, 64]，查表后 [B, 64]
（不应该要)

content_seq [B, 75, 128] + speaker_embedding [B, 75, 64] (时间维度复制)，
					concat后：[B, 75, 192]
  -> generator 通过3层MLP得到 [B, 75, 192]
  -> predicted EnCodec latent [B, 75, 128]
  -> frozen decoder 格式先转为 [1, 128, 75]，由于是24KHZ，最终为[24000]
		直接用了Meta/Facebook 的 EnCodec 24 kHz decoder，事实证明他们的效果不错
  -> wav [24000]
```

![Factored content × speaker 架构](assets/arch_factored.png)

用codex生成了图，虽然看着很丑，但是直观

### 3.3 Hold-out-cell evaluation

**FEIS 可视为一个 subject × label 网格。一个 cell 表示：**

```text
某个 subject × 某个 prompt
```

Hold-out-cell 指训练时整格拿掉某些 `(subject, label)` 组合，例如保留 subject 01 的其他 label，也保留其他 subject 的 `f`，但不让模型见到 `01-f`。

该评估用于测试模型能否组合，就是为了防止模型在背模板，因为 FEIS 的每条目标音频都同时包含 **subject 信息**和  **label 内容** ，如果普通随机划分训练/测试，模型可能已经在训练中见过某个 subject 的声音、某个 label 的声音，甚至间接记住了对应组合。

```text
已知 subject voice + 已知 content -> 未见过的 subject-content combination
```

![Hold-out-cell 网格评估示意图](assets/grid_holdout.png)

---

## 4. 当前 Demo 组件

为了快速跑通，很多地方设定比较简单

| 文件                     | 功能                                                                       |
| ------------------------ | -------------------------------------------------------------------------- |
| **`targets.py`** | **加载 EnCodec target cache，构造 content/speaker prototypes**       |
| **`data.py`**    | **构造 factored FEIS dataset 与 holdout-cell splits**                |
| **`model.py`**   | **EEG encoder、content branch、speaker branch、generator、RMS head** |
| **`losses.py`**  | **content, reconstruction, speaker, adversarial, RMS losses**        |
| **`eval.py`**    | **content top1、zero-EEG、coarse metrics、collapse metrics**         |

---

## 5. 重建评估

我选择用最直观的结果来看，就是重建声音的听觉效果，以及重建声音的声图

具体而言，**每个样本生成五种 wav：**

| 类型              | 含义                                                                                   |
| ----------------- | -------------------------------------------------------------------------------------- |
| `original_ref`  | 原始参考音频                                                                           |
| `target_oracle` | 真实 target latent 经 meta decoder 解码，具体来说meta有完整一套，直接用就行            |
| `mean_latent`   | 全局平均 latent 经 decoder 解码                                                        |
| `pred_unscaled` | 模型预测 latent 直接解码                                                               |
| `pred_scaled`   | 模型预测 latent 解码后按预测 RMS 缩放（因为我之前做过的版本rec后响度太小，这为了方便听 |

**具体的意义：**

- `original_ref ≈ target_oracle`：说明 codec path 健康，直接拿声音解码复原效果可以，本身音频没问题。
- `pred_scaled` 音量恢复但谱图仍差：说明不是单纯小声问题。
- `pred` 接近 `mean_latent`：说明有平均化/塌缩风险。

---

## 6. 实验结果

~~内容识别失败~~

### 6.1 Independent content probe

**`content_probe.py` 使用线性 ridge decoder、受试内 5 折、200 次 permutation test，独立检验 EEG 是否包含 16 类 content signal。**

n = 3152：20 个 subject× 16 个 label × 10 次 repetition = 3200 trials，其中12号作者说他做到一半有事先走了，就少了48个trial，看top 模型 16 选 1 猜对的比例比chance小，目前结果就是还不如随机

| 阶段         |    n |   top1 | chance | null95 |     p | 显著 |
| ------------ | ---: | -----: | -----: | -----: | ----: | ---- |
| `stimuli`  | 3152 | 0.0457 | 0.0625 | 0.0587 | 0.920 | 否   |
| `thinking` | 3152 | 0.0466 | 0.0625 | 0.0593 | 0.896 | 否   |

**粗音系类别也低于 majority baseline：所以模型在更粗糙的分类标准依旧没学到，具体而言**

| manner          | 发音方式，比如爆破音、摩擦音、鼻音、元音 |
| --------------- | ---------------------------------------- |
| voicing         | 清浊，比如 voiceless / voiced            |
| vowel/consonant | 元音还是辅音                             |

| 阶段         |        manner |       voicing | vowel/consonant |
| ------------ | ------------: | ------------: | --------------: |
| `stimuli`  | 0.267 / 0.375 | 0.539 / 0.625 |   0.633 / 0.750 |
| `thinking` | 0.280 / 0.375 | 0.537 / 0.625 |   0.622 / 0.750 |

**结论：在当前 FEIS 特征下，`stimuli` 和 `thinking` 两阶段均未显示可泛化 content signal。**

![Content probe 结果：stimuli / thinking 均未高于随机水平](assets/content_probe_bars.png)

### 6.2 Subject positive control probe

为排除 probe 管线本身无效的问题，使用同一套 EEG 特征、同一套 cross-validation / permutation test，将预测目标从 16 类 content 改为 20 类 subject identity。

| 阶段         |    n | classes |   top1 | chance | null95 |     p | 显著 |
| ------------ | ---: | ------: | -----: | -----: | -----: | ----: | ---- |
| `stimuli`  | 3152 |      20 | 0.7846 | 0.0500 | 0.0568 | 0.005 | 是   |
| `thinking` | 3152 |      20 | 0.8115 | 0.0500 | 0.0562 | 0.005 | 是   |

结论：positive control passed。FEIS EEG 特征中确实存在可被探针稳定捕获的 subject-level signal；因此 content probe 的 null result 更可能是真负结果，而不是代码、特征或评估管线失效。

![Subject positive control：同一管线可稳定解出 subject identity](assets/probe_positive_control.png)

### 6.3 音频重建图示例

![五路 wav 对比示例：original / oracle / mean / pred_unscaled / pred_scaled](../artifacts/outputs_factored/factored_v2_visual_report_20260611_0248/images/wav_pages/recon_eval_test_holdout/page_001.png)

---

## 7. 总体结果

**demo 流程通了，但是效果不好，一是求快粗糙，二十数据集本身不足**

1. FEIS codec target 可用
   `target_oracle` 与 `original_ref` 接近，说明 EnCodec extraction/decode 没有成为主要问题。
2. 当前数据集我目前的做法，content signal 不可稳定解码
   independent probe 和 factored v2 均显示 EEG 没有稳定超过 zero-EEG。
3. subject signal 可稳定解码
   subject positive control 在 `stimuli` 和 `thinking` 上分别达到 0.7846 和 0.8115，说明同一套 EEG 特征和 probe 管线能捕获真实神经/个体差异信号；content null 不是管线 bug。
