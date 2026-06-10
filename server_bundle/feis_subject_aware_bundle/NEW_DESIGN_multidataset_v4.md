# 多数据集统一训练设计（v4，承接 v3）

> 需求：把数据集做成可插拔单元，EEG 映射到**同一个语音向量空间**联合训练；
> config 里可自由选「只 FEIS」「只 KaraOne」「两个一起」；最终落点是 **waveform 重建保真度**。

## 1. 为什么 v3 天然支持这件事

v3 已经把目标对齐到**与数据集无关的语音表征空间**（EnCodec latent / 对比 embedding），
而不是某个数据集的标签头。音频侧是通用的：任何 16 kHz wav → 冻结 EnCodec → latent。
所以「换数据集」只需要换 EEG 侧的入口，语音目标空间天然共享。

## 2. 两个数据集的差异与桥接

| 维度 | FEIS | KaraOne | 桥接方式 |
|---|---|---|---|
| 通道 | 14 | 62 | **per-dataset 空间 adapter**（Conv1d C→d_model）|
| EEG 长度 | 1280 固定 | ~1272 变长 | valid_lengths mask |
| 音频 | canonical（16 条共享）| **trial-sync 真录音** | 统一成 EnCodec latent，公共帧窗 T_common + mask |
| 音频时长 | 1.0s | 1.3–1.8s | T_common=150 帧(2.0s)，按 valid 帧 mask |
| 标签 | 16 | 11（几乎不相交）| **per-dataset 分类头**（可选）；主监督走对比/latent，与标签无关 |
| 被试 | 21 | 14 | **全局 subject id**（跨数据集编号）|

## 3. 统一架构（EEG2SpeechMD）

```
            ┌── FEIS EEG [B,14,L] ──┐
            │                        ├─► input_adapter[ds]  (C→d_model, 1x1)
            └── KaraOne EEG [B,62,L]─┘            │
                                                  ▼
   cond = subject_emb(global_sid) + dataset_emb(ds_id)   ── FiLM ──►  共享 TemporalTrunk
                                                  │
                                                  ▼
                         ┌── content_head ──► EnCodec latent [B,T_common,128] ─┐
   共享                  ├── contrastive_head ──► embedding（InfoNCE 对齐）     │
                         └── class_heads[ds]（per-dataset，可选）              │
                                                                               ▼
                                              冻结 EnCodec decoder ──► waveform（保真目标）
```

要点：
- **input_adapter 按数据集分开**（解决通道数不同）；**trunk + content/contrastive head 共享**（这才是「同一向量空间」）。
- **dataset embedding** 经 FiLM 注入 trunk，让共享 trunk 知道当前域。
- **subject embedding 用全局 id**，跨数据集统一编号；预留 unknown 行。
- **batch 是 dataset-homogeneous**（一个 batch 只来自一个数据集），因为通道数不同没法 stack；联合训练时按权重在数据集之间轮流出 batch。

## 4. 单选 / 多选 / 联合 —— 由 config 一行控制

```yaml
datasets: [feis]            # 只训 FEIS
datasets: [karaone]         # 只训 KaraOne
datasets: [feis, karaone]   # 联合（共享 trunk，加权采样）
sampling_weights: [1.0, 1.0]
```

- 单数据集：退化为 v3 单 adapter，行为与 v3 一致。
- 多数据集：每个数据集各自 DataLoader，训练循环按 `sampling_weights` 加权轮流取 batch；
  每个数据集独立做验证/检索评测，分别报指标。

## 5. 最终目标 = waveform 重建保真

- 统一用冻结 EnCodec decode 合成（自然度保证）。
- **FEIS 是 canonical**：只能重建到「选对 16 条之一」的上限，没有 trial 级声学。
- **KaraOne 是 trial-sync**：有真实 per-trial 录音，**能训练并评测真正的逐 trial 波形保真**
  （STFT/Mel 距离 vs 该 trial 真实 wav）。所以若主目标是「还原得像」，
  KaraOne 的 overt_like / thinking 才是真正能优化保真的战场，FEIS 提供识别与对齐的额外监督。
- 评测分两类：识别（top-1/5，两数据集都有）、重建保真（Mel/STFT，仅 KaraOne trial-sync 有意义）。

## 6. 数据增强/迁移的收益来源

- KaraOne ~1900 thinking trial + 真实音频，给共享 trunk 大量「EEG→语音表征」监督。
- 推荐两条用法：
  1. **联合训练**（datasets:[feis,karaone]，加权采样）——共享 trunk 同时受益。
  2. **KaraOne 预训练 → FEIS 微调**（`--init-from`）——更稳、风险低。

## 7. 落地组件（已实现，见 README_v3 多数据集章节）

- `src/v3/datasets.py`：`DatasetSpec` + 注册表 + `KaraOneV3Dataset` + FEIS 统一封装 + 公共 T padding + 全局 subject id。
- `src/v3/encoder.py`：拆出 `SpatialAdapter`（per-dataset）+ 共享 `TemporalTrunk`。
- `src/v3/model.py`：`EEG2SpeechMD`（多 adapter + dataset/subject 条件 + 共享 head + per-dataset 分类头）。
- `scripts/v3_extract_karaone_targets.py`：KaraOne **trial 级** EnCodec 目标缓存。
- `scripts/v3_train_md.py`：多数据集训练（`datasets:[...]` + 加权调度 + 分数据集评测）。
- `configs/v3_multidataset.yaml`。
