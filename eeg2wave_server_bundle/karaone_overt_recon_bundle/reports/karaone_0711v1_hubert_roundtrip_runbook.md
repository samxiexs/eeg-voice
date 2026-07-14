# 0711v1 adapted-HuBERT 音频 round-trip 实验运行说明

## 实验问题

本实验检验冻结的 adapted-HuBERT 连续表征是否保留足以预测原始音频 EnCodec latent 的信息。它是完全独立于 EEG 的音频表征审计：

```text
KaraOne wav
  -> 已冻结、已缓存的 adapted-HuBERT semantic_sequence [50,768]
  -> 新训练的 HubertToEncodecDecoder
  -> predicted EnCodec latent [150,128]
  -> frozen EnCodec decoder
  -> round-trip wav
```

round-trip decoder 的输入只有 `semantic_sequence`；它不接受 EEG、subject-ID、真实 label、reference wav 或真实 EnCodec latent。

## 使用的既有 0711v1 产物

```text
artifacts/karaone_0711v1/
  karaone_0711v1_overt_like_adapted_audio_targets_s11.npz
```

其中读取：

- `semantic_sequence [1913,50,768]`：冻结 adapted-HuBERT 连续表征；
- `encodec_latent [1913,150,128]`：训练目标；
- `fit_split`、subject、label、audio path：切分审计、基线和对比音频定位。

## 切分与选择规则

| 数据 | 用途 |
|---|---|
| 1,616 条 subject_train | 训练 decoder；其中按 label 分层、确定性划出 10% 作内部 early-selection validation |
| P02（165 条） | 锁定 checkpoint 后的主要验证和音频对比图 |
| MM21（132 条） | 仅在 P02 结果审阅后，显式授权的一次最终评估/导出 |

decoder checkpoint 根据 train subjects 内部 validation 的最低 raw EnCodec-latent MSE 选择；不会为选择 checkpoint 而访问 P02 或 MM21。

## 首次完整运行：训练 + P02 指标 + P02 wav/图片

从 `karaone_overt_recon_bundle/app` 运行：

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/karaone_overt_recon_bundle/app

MPLCONFIGDIR=/tmp/matplotlib-karaone-hubert-roundtrip \
python3 scripts/train_karaone_0711v1_hubert_roundtrip.py \
  --phase all \
  --config configs/karaone_0711v1.yaml \
  --stage overt_like \
  --seed 11 \
  --device mps
```

如果机器没有 MPS，可把最后一行替换为 `--device cpu`；CUDA 机器则使用 `--device cuda`。

默认训练 80 epochs。先做小规模运行检查可加 `--epochs 1 --limit 5`；它会训练 1 epoch，并只导出 5 条 P02 wav/图片，不能作为正式结果。

## 中断后继续训练

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/karaone_overt_recon_bundle/app

MPLCONFIGDIR=/tmp/matplotlib-karaone-hubert-roundtrip \
python3 scripts/train_karaone_0711v1_hubert_roundtrip.py \
  --phase train \
  --config configs/karaone_0711v1.yaml \
  --stage overt_like \
  --seed 11 \
  --device mps \
  --resume-training ../artifacts/outputs_karaone_0711v1/karaone_0711v1_overt_like_hubert_roundtrip_s11/checkpoints/last.pt
```

完成训练后，如需重新只导出 P02 图像：

```bash
python3 scripts/train_karaone_0711v1_hubert_roundtrip.py \
  --phase synthesize --split subject_val \
  --config configs/karaone_0711v1.yaml --stage overt_like --seed 11 --device mps
```

## MM21 最终测试（只在锁定 P02 结论后运行）

先写最终 latent 指标：

```bash
python3 scripts/train_karaone_0711v1_hubert_roundtrip.py \
  --phase evaluate --split subject_test --allow-final-test \
  --config configs/karaone_0711v1.yaml --stage overt_like --seed 11 --device mps
```

再导出 MM21 的 reference/oracle/round-trip wav 和 comparison PNG：

```bash
MPLCONFIGDIR=/tmp/matplotlib-karaone-hubert-roundtrip \
python3 scripts/train_karaone_0711v1_hubert_roundtrip.py \
  --phase synthesize --split subject_test --allow-final-test \
  --config configs/karaone_0711v1.yaml --stage overt_like --seed 11 --device mps
```

## 输出位置和含义

```text
artifacts/outputs_karaone_0711v1/
  karaone_0711v1_overt_like_hubert_roundtrip_s11/
    checkpoints/best.pt
    checkpoints/last.pt
    metrics/subject_val_latent_metrics.json
    metrics/subject_val_audio_metrics.json
    wavs/hubert_roundtrip_subject_val/
      reference/
      cache_latent_oracle/
      reconstructed/
      comparison/
      synthesis_manifest.json
```

每张 comparison PNG 含：reference、cache-latent oracle、HuBERT round-trip 的波形叠图，以及三张对应的 log-spectrogram。

`cache_latent_oracle` 是把缓存中的真实 EnCodec latent 直接交给 EnCodec decoder 的输出，用于量化 codec/cache 本身造成的损失。当前 0711 cache 没有逐条保存 EnCodec decoder scale，因此它是“cache-latent ceiling”，不是严格无损的 waveform oracle。

主要比较应看：

1. `roundtrip` 是否在 P02 上优于 mean latent baseline；
2. roundtrip 与 `cache_latent_oracle` 的 latent cosine/MSE 差距；
3. roundtrip 的 waveform correlation、SI-SDR、log-spectrogram MAE；
4. 固定导出的 P02 参考/重建图和音频是否显示发声时段、频谱结构和类别信息仍被保留。

