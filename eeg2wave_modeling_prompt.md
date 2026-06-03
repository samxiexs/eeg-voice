# EEG → Token → Waveform 建模方案 Prompt（v2）

## 任务背景

你是一位专门研究 EEG-to-Speech 解码的深度学习工程师。当前目标是为一项认知神经科学研究构建第一版 demo 模型，核心任务是：

**从 imagined speech 的 EEG 信号出发，经过离散 token bottleneck，直接重建对应 prompt 的 waveform。**

这不是一个语音合成任务，而是一个 EEG-conditioned waveform prototype reconstruction 任务。

---

## 预处理说明（已完成，代码勿改）

预处理脚本为 `scripts/preprocess_thinking_waveform_pairs.py`，已产出所有阶段的 EEG 分段，**无需再跑**。

### 预处理参数（已固化）

| 参数 | 值 |
|------|----|
| EEG bandpass | 1–40 Hz |
| FEIS notch | 50 Hz |
| KaraOne notch | 60 Hz |
| 参考 | Common Average Reference |
| 输出 EEG 采样率 | **256 Hz** |
| 输出 Audio 采样率 | **16 kHz** |
| baseline | FEIS：同 trial 的 `resting`；KaraOne：同 trial 的 `clearing` |

### FEIS 导出阶段

预处理导出了全部 5 个阶段，每条 segment 均在 `segments.csv` 中有一行，`segment_stage` 列标注阶段名：

| segment_stage | 含义 | 用途建议 |
|---------------|------|----------|
| `stimuli` | 受试听到/看到 prompt | hearing / prompt perception 对照 |
| `articulators` | 1 秒 fixation point | 通常不用 |
| `thinking` | **想象说出 prompt** | **第一版主要输入** |
| `speaking` | 实际说出 prompt（有录音） | overt speech 对照 |
| `resting` | 静息基线 | 已用于 baseline 标准化 |

### KaraOne 导出阶段

| segment_stage | 含义 | 用途建议 |
|---------------|------|----------|
| `clearing` | 放空上一 trial 状态 | 已用于 baseline 标准化 |
| `stimulus_like` | 接收 prompt（~2s） | hearing-like 对照 |
| `thinking` | 想象说出 prompt（~5s，**变长**） | 第二版引入 |
| `overt_like` | 真正说出 prompt（~2.4s，**变长**） | 有 trial-synchronous wav |

---

## 数据结构说明

### 统一输出路径

```
data/processed/thinking_waveform_pairs/
├── feis/
│   ├── manifest.json
│   ├── trials.csv          ← 一行一个 trial
│   ├── segments.csv        ← 一行一个 trial × stage，筛选主入口
│   ├── subjects/
│   │   ├── 01.npz
│   │   ├── 02.npz
│   │   └── ...             ← 21 个英文被试
│   └── audio/
│       └── *.wav           ← 16 个 canonical prompt wav（共用，非 trial-sync）
└── karaone/
    ├── manifest.json
    ├── trials.csv
    ├── segments.csv
    ├── subjects/
    │   ├── MM05.npz
    │   └── ...
    └── audio/
        └── *.wav           ← trial-synchronous overt speech wav
```

### NPZ 文件结构（FEIS）

```python
data = np.load("subjects/01.npz", allow_pickle=True)

data["trial_indices"]          # [n_trials]，trial 编号
data["labels"]                 # [n_trials]，对应 prompt label
data["audio_relpaths"]         # [n_trials]，canonical wav 相对路径（同 label → 同路径）
data["channel_names"]          # [14]，通道名
data["stage_names"]            # ['stimuli', 'articulators', 'thinking', 'speaking', 'resting']

# 各阶段 EEG 数组（FEIS 全部固定长度）
data["stage__thinking"]        # [n_trials, 14, 1280]  ← 第一版主要使用
data["stage__speaking"]        # [n_trials, 14, 1280]
data["stage__stimuli"]         # [n_trials, 14, 1280]
data["stage__articulators"]    # [n_trials, 14,   ?]   ← 1秒段，较短
data["stage__resting"]         # [n_trials, 14, 1280]
```

EEG 数值说明：已做 bandpass、notch、CAR、baseline 标准化，数值为相对基线偏移量，不是原始微伏绝对值。

### NPZ 文件结构（KaraOne，供参考）

```python
data = np.load("subjects/MM05.npz", allow_pickle=True)

data["trial_indices"]
data["labels"]
data["audio_relpaths"]         # trial-synchronous overt wav
data["channel_names"]          # [62]
data["stage_names"]            # ['clearing', 'stimulus_like', 'thinking', 'overt_like']

# KaraOne 各阶段变长，需配合 valid_lengths 使用
data["stage__thinking"]               # [n_trials, 62, max_len]，行内有 padding
data["stage__thinking__valid_lengths"] # [n_trials]，实际有效样本点数
data["stage__thinking__src_ranges"]   # [n_trials, 2]，在原始 EEG 中的采样点区间
```

### segments.csv 字段（两个数据集通用）

```
subject_id, trial_index, segment_stage, label, audio_relpath, ...
```

**第一版 dataset 类筛选方式**：

```python
df = pd.read_csv("feis/segments.csv")
thinking_df = df[df["segment_stage"] == "thinking"]
```

---

## 第一版严格范围

**做**：
- FEIS `thinking` EEG → 离散 VQ token → 固定长度 waveform
- Subject-specific 训练（每个 subject 单独一个小模型）
- 输出 `.wav` 文件

**不做（第一版）**：
- 多阶段联合建模（不混 `stimuli` / `speaking`，只用 `thinking`）
- KaraOne（变长段 + 62 通道，留第二版）
- 跨被试泛化
- 变长 waveform 输出
- Audio tokenizer / codec（不做 EnCodec / SoundStream）
- GAN / diffusion decoder
- 追求语音可懂度

**预留扩展点**（代码里埋好接口，第二版容易接）：
- `segment_stage` 改成 `stimuli` 或 `speaking`，其余代码不变
- KaraOne 接入时，需用 `valid_lengths` 做 mask 或裁剪

---

## 模型架构

### 总体数据流

```
EEG Input          [B, 14, 1280]
     ↓
EEG Encoder        [B, 128, 40]     ← 1D CNN，压缩 32×
     ↓
Vector Quantizer   [B, 128, 40]     ← 单层 EMA-VQ，codebook=512
     ↓
Waveform Decoder   [B, 1, 24000]    ← ConvTranspose1d 上采样
     ↓
Waveform Output    [B, 1, 24000]    ← 16 kHz × 1.5s
```

### 模块 1：EEG Encoder

```python
# 目标：[B, 14, 1280] → [B, 128, 40]（32× 压缩）
# 每个 latent 时间步覆盖约 125ms 的 EEG

Conv1d(14,  64,  kernel=7, stride=2,  padding=3) + BN + ReLU  # → [B, 64,  640]
Conv1d(64,  128, kernel=5, stride=2,  padding=2) + BN + ReLU  # → [B, 128, 320]
Conv1d(128, 128, kernel=5, stride=4,  padding=2) + BN + ReLU  # → [B, 128,  80]
Conv1d(128, 128, kernel=5, stride=2,  padding=2) + BN + ReLU  # → [B, 128,  40]
```

不用 Transformer，原因：subject-specific 小数据（~130 训练 trial），Transformer 容易过拟合。

### 模块 2：Vector Quantizer

```python
# 标准 EMA-updated VQ（VQ-VAE 实现）
codebook_size = 512    # 若效果差可升到 1024
embedding_dim = 128    # 与 encoder 输出通道一致
beta = 0.25            # commitment loss 权重

# 前向步骤：
# 1. reshape encoder 输出为 [B*T_lat, 128]
# 2. 计算到 codebook 每个 embedding 的 L2 距离
# 3. 取最近邻 code index，产生离散 token 序列（长度 40）
# 4. straight-through gradient 回传给 encoder
# 5. EMA 更新 codebook embedding
# 输出：量化后的 latent [B, 128, 40] + vq_loss
```

commitment loss = `beta * ||sg(z_e) - z_q||^2 + ||z_e - sg(z_q)||^2`

### 模块 3：Waveform Decoder

```python
# 目标：[B, 128, 40] → [B, 1, 24000]（约 600× 上采样）
# 上采样路径：40 → 80 → 320 → 1280 → ~6400 → [interpolate] → 24000

ConvTranspose1d(128, 128, kernel=4, stride=2, padding=1)  + BN + LeakyReLU  # 40  → 80
ConvTranspose1d(128, 64,  kernel=4, stride=4, padding=0)  + BN + LeakyReLU  # 80  → 320
ConvTranspose1d(64,  32,  kernel=8, stride=4, padding=2)  + BN + LeakyReLU  # 320 → 1280
ConvTranspose1d(32,  16,  kernel=8, stride=5, padding=1)  + BN + LeakyReLU  # 1280 → ~6300
Conv1d(16, 8, kernel=5, padding=2) + LeakyReLU
# F.interpolate → 精确对齐到 24000
Conv1d(8, 1, kernel=3, padding=1) + Tanh
```

输出 raw waveform，值域 [-1, 1]，可直接用 `soundfile.write` 写出。

---

## 数据处理

### Audio 加载

```python
import librosa, numpy as np

def load_canonical_wav(relpath, audio_dir, sr=16000, duration=1.5):
    """
    relpath: audio_relpaths 字段的值，相对 audio_dir 的路径
    输出：float32 array, shape [24000]
    """
    path = Path(audio_dir) / relpath
    wav, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)   # 24000
    if len(wav) >= target_len:
        return wav[:target_len].astype(np.float32)
    return np.pad(wav, (0, target_len - len(wav))).astype(np.float32)
```

### Dataset 类接口

```python
class FEISThinkingDataset(Dataset):
    def __init__(
        self,
        subject_id: str,            # "01", "02", ...
        data_root: str,             # pointing to feis/
        stage: str = "thinking",    # 预留：也可传 "stimuli" / "speaking"
        split: str = "train",       # "train" / "val" / "test"
        train_ratio: float = 0.8,
        val_ratio:   float = 0.1,
        audio_sr:    int = 16000,
        audio_dur:   float = 1.5,
    ):
        # 1. 读 segments.csv，筛 segment_stage == stage 且 subject_id 匹配
        # 2. 读 subjects/{subject_id}.npz
        #    eeg_array = data[f"stage__{stage}"]   # [n_trials, 14, 1280]
        #    labels    = data["labels"]
        #    relpaths  = data["audio_relpaths"]
        # 3. 按 trial index（不随机打乱！）顺序划分 train/val/test
        #    train: 前 80%，val: 中 10%，test: 后 10%
        # 4. 构建 index 映射

    def __getitem__(self, idx):
        eeg = self.eeg_array[self.indices[idx]]   # [14, 1280], float32
        relpath = self.relpaths[self.indices[idx]]
        wav = load_canonical_wav(relpath, self.audio_dir)  # [24000], float32
        label = self.labels[self.indices[idx]]
        return eeg, wav, label
```

**关键约束**：
- 按 trial 顺序划分（不用 random split），防止相邻 trial 泄漏
- 同 label 所有 trial 共享同一条 canonical wav（FEIS 特性，不是 bug）
- `stage` 参数化，方便第二版切换到 `stimuli` 或 `speaking`

---

## 训练设定

### Loss

```python
def total_loss(pred_wav, target_wav, vq_loss):
    l1   = F.l1_loss(pred_wav, target_wav)
    stft = multi_resolution_stft_loss(
        pred_wav, target_wav,
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128,  256,  512],
        win_sizes=[512, 1024, 2048],
    )
    return l1 + 0.5 * stft + 0.1 * vq_loss
```

不加 GAN / speaker / phoneme / contrastive loss。

### config.yaml

```yaml
model:
  n_channels_eeg: 14
  eeg_len: 1280
  enc_channels: 128
  t_latent: 40
  codebook_size: 512
  vq_beta: 0.25

audio:
  sample_rate: 16000
  duration_sec: 1.5
  n_samples: 24000

train:
  batch_size: 32
  lr: 3.0e-4
  epochs: 200
  lambda_stft: 0.5
  lambda_vq: 0.1
  grad_clip: 1.0

data:
  stage: thinking          # 第一版固定，预留后期改 stimuli / speaking
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

### 训练流程

**Step 1：单 subject 跑通**

```bash
python train.py --subject 01 --config config.yaml
# 先确认：
#   - total_loss 前 20 epoch 内开始下降
#   - val loss 跟 train loss 方向一致（没有立即过拟合）
#   - 输出 wav 不是全零 / 全噪
```

**Step 2：批量跑所有 subject**

```bash
for subj in 01 02 03 04 05 06 07 08 09 10 11 12 \
            13 14 15 16 17 18 19 20 21; do
    python train.py --subject $subj --config config.yaml
done
# 或者用 nohup 跑后台
nohup bash run_all.sh > logs/run_all.log 2>&1 &
```

---

## 评估方案

### 核心指标：Nearest Template Accuracy

```python
def nearest_template_accuracy(model, test_loader, canonical_wavs, device):
    """
    canonical_wavs: dict[label_str -> np.array shape (24000,)]
    对每条 test trial：
        1. EEG → VQ token → recon wav
        2. 与 16 条 canonical wav 各算 multi-res STFT distance
        3. 取最近的 label，看是否等于 ground truth label
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for eeg, wav_gt, label in test_loader:
            recon, _, _ = model(eeg.to(device))
            recon_np = recon.squeeze(1).cpu().numpy()  # [B, 24000]
            for b in range(len(label)):
                dists = {
                    lbl: stft_distance(recon_np[b], cw)
                    for lbl, cw in canonical_wavs.items()
                }
                pred_label = min(dists, key=dists.get)
                if pred_label == label[b]:
                    correct += 1
                total += 1
    return correct / total

# Random baseline: 1/16 = 6.25%
```

### 全套指标

| 指标 | 类型 | 说明 |
|------|------|------|
| Waveform L1 | 定量 | 整体波形距离 |
| Multi-res STFT distance | 定量 | 频谱层面距离 |
| **Nearest Template Accuracy** | **定量（核心）** | 16-class retrieval precision，random = 6.25% |
| 能量包络 | 定性目视 | envelope 是否接近 target |
| 听感抽样 | 定性 | 每 subject 各 label 各抽 1 条 |

---

## 服务器部署包结构

```
eeg2wave_demo_bundle/
├── README.md
├── requirements.txt      ← torch, torchaudio, librosa, numpy, pandas, scipy, soundfile
├── config.yaml
├── train.py              ← 主训练入口，支持 --subject / --config
├── infer.py              ← 批量推理 + 保存重建 wav + 计算所有指标
├── dataset.py            ← FEISThinkingDataset（stage 参数化）
├── model.py              ← EEGEncoder + VQ + WaveformDecoder
├── losses.py             ← L1 + multi-STFT + VQ loss
├── utils.py              ← stft_distance / load_canonical_wav / save_wav
├── data/
│   └── feis/
│       ├── manifest.json
│       ├── trials.csv
│       ├── segments.csv
│       ├── subjects/          ← 21 个 *.npz
│       └── audio/             ← 16 个 canonical wav
└── outputs/
    ├── checkpoints/           ← subject_{id}_best.pt
    ├── recon_wavs/            ← subject_{id}/{label}_{trial_idx}.wav
    └── metrics/               ← subject_{id}_metrics.json
```

数据拷贝命令（本地执行）：

```bash
cp -r \
  /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/data/processed/thinking_waveform_pairs/feis \
  eeg2wave_demo_bundle/data/feis
```

---

## 第二版扩展路径（现在不做，埋好接口）

| 扩展方向 | 触发条件 | 核心改动 |
|----------|----------|----------|
| 加入 `stimuli` 阶段对照 | 第一版 accuracy > chance 后 | `--stage stimuli`，其余代码不变 |
| 加入 `speaking` 做 teacher | 第一版跑通后 | 双流输入，`stage__speaking` 作为辅助监督 |
| 引入 KaraOne | 第一版结论稳定后 | 需处理 `valid_lengths` mask，audio 改为 trial-sync wav |
| cross-subject 泛化 | 单 subject 效果验证后 | 换成 pooled dataset + subject embedding |

---

## 向导师汇报定位语

> 这是一个 subject-specific 的 imagined-speech waveform reconstruction demo。  
> 输入：FEIS thinking 阶段 EEG（14 通道，256 Hz，5 秒，预处理后 shape [14, 1280]）；预处理包括 1–40 Hz bandpass、50 Hz notch、CAR、resting baseline 标准化，均已完成。  
> 中间：显式经过离散 EEG token bottleneck（单层 EMA VQ，codebook size=512，latent 长度 40）。  
> 输出：1.5 秒固定长度 raw waveform（16 kHz mono）。  
> 目标：不是生成自然语音，而是验证 imagined EEG 经 token 压缩后，能否恢复出与 prompt 对应的 canonical waveform prototype。  
> 核心评估：Nearest Template Accuracy（16-class retrieval precision），random baseline = 6.25%。

---

## 关键设计决策总表

| 决策 | 选择 | 理由 |
|------|------|------|
| 数据集 | 只用 FEIS thinking | EEG 固定长度、标签清晰、audio 已对齐、约 913MB 轻便 |
| 建模粒度 | Subject-specific | FEIS target 是 canonical wav，非 trial-sync，混被试引入 voice identity 干扰 |
| Audio 长度 | 1.5s / 24000 samples@16kHz | 不截断词尾；比 5s 更容易收敛 |
| EEG 段筛选 | `segments.csv`，`segment_stage == thinking` | 预处理已导出全阶段，直接筛取，无需重跑 |
| Token 方式 | 单层 EMA-VQ，codebook=512 | 满足路线定义，复杂度最低 |
| Decoder | ConvTranspose1d 直接上采样 | 非 autoregressive，训练快 |
| 不用 audio tokenizer | 是 | 按研究定义，audio 不经过 tokenization |
| 不加 KaraOne | 是（第一版） | 变长段 + 62 通道更复杂，第二版引入 |
| Loss | L1 + multi-STFT + VQ | 最小必要组合 |
| 数据划分 | 按 trial 顺序划分 80/10/10 | 防止相邻 trial EEG 残影泄漏 |
