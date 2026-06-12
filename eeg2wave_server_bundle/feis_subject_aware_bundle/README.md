# FEIS Subject-Aware EEG → Speech Bundle

从 imagined / overt speech 的 EEG 重建**该受试本人**的语音波形。核心路线：
**EEG → EnCodec latent → 冻结 EnCodec decoder → 自然 wav**，用对比 + latent + 分类目标
取代旧的裸波形回归（根治 mode collapse）。

> 目录重命名说明：`server_bundle/` 已改名为 `eeg2wave_server_bundle/`。
> **所有可运行代码（v3/v4 的 `.py`、`.yaml`）都用相对路径 + `BUNDLE_DIR`，不受重命名影响，无需改动。**
> 仅 `bundle_manifest.json` 的 provenance 绝对路径已同步更新（非功能性元数据）。

## 入口文档

- **运行手册（主）**：[`app/README_v3.md`](app/README_v3.md) — 环境、目标抽取、v3 单数据集 / v4 多数据集训练、评测、合成的完整指令。
- 设计：[`NEW_DESIGN_eeg2speech_v3.md`](NEW_DESIGN_eeg2speech_v3.md)（单数据集解耦方案）、[`NEW_DESIGN_multidataset_v4.md`](NEW_DESIGN_multidataset_v4.md)（FEIS+KaraOne 共享空间）。

## 代码结构

```
app/src/v3/        encoder / model(EEG2SpeechV3 + EEG2SpeechMD) / losses / data / datasets / eval / recon_eval / synth
app/scripts/       v3_train, v3_eval, v3_recon_eval, v3_synthesize           (单数据集 FEIS)
                   v3_train_md, v3_extract_karaone_targets                   (多数据集)
                   extract_audio_targets                                     (FEIS EnCodec 目标)
app/configs/       v3_encodec.yaml(单), v3_multidataset.yaml(多)
data/feis/         21 受试, 14ch, canonical 音频(每受试每 prompt 1 条, 336 模板)
data/karaone/      14 受试, 62ch, trial-synchronous 真录音(每 trial 1 条, 1913 条) ← 已迁入
models/            encodec_24khz, hubert-base-ls960
artifacts/audio_targets/   EnCodec/HuBERT 目标缓存
```

## 数据现状

| | FEIS | KaraOne |
|---|---|---|
| 受试 / 通道 | 21 / 14 | 14 / 62 |
| trials | 3312 | 1913 |
| 音频性质 | canonical（受试级，每 prompt 共享 1 条）| **trial-synchronous 真录音（每 trial 独立）** |
| 目标键 | `subject:label`（336） | `subject:trial`（1913） |
| 采样率 | EEG 256Hz / 音频 16kHz | 同 |

要点：FEIS 跨受试的 wav 不同（每人自己的录音），但同受试同 prompt 的多 trial 共享一条 → 没有 trial 级声学差异；
**只有 KaraOne 有逐 trial 真录音，是优化/评测"逐 trial 波形保真"的唯一数据。**

## 当前进展与关键诊断

- v3（FEIS）与 v4（FEIS+KaraOne 共享 trunk + per-dataset adapter）均已实现并跑通。
- speaking-stage teacher 已训练；`template_top1≈0.088`、`top5≈0.33`。
- **诊断**：该 `template_top1` 几乎等于"认对受试 + 受试内 16 选 1 随机"的理论值（0.0625 / 0.3125），
  而 subject-agnostic 的 `label_top1` 仅 ≈ chance。说明对比 embedding 主要编码 **subject identity（是谁）**，
  而非 **prompt content（说了什么）**。
- 为此 `eval.py` 增加了两个决定性指标：
  - `within_subject_prompt_top1`：只在该受试自己的 16 条里选 prompt（chance 1/16）——真正的内容解码指标。
  - `within_subject_prompt_top1_zeroeeg`：EEG 置零、只留 subject embedding 的对照；与上者之差 = EEG 的真实贡献。
- thinking-stage 训练出现过拟合（train 升 / val 平），符合 imagined EEG 信号弱的预期。

## 指标解读（务必看 within-subject，而非被 subject 抬高的 template_top1）

| 指标 | 含义 | chance |
|---|---|---|
| `within_subject_prompt_top1` | **真·内容**：受试内选对 prompt | 1/16=0.0625 |
| `within_subject_prompt_top1_zeroeeg` | 仅 subject 的基线（EEG 置零） | — |
| `template_top1 / top5` | subject+prompt（被 subject 身份抬高） | 1/336 |
| `label_top1 / top5` | prompt-only、跨受试 | 1/16 |
| `recon_eval`: `subject_specificity_gap` | 解码 wav 更像本人(>0) | — |

## 下一步建议

若 thinking 的 `within_subject_prompt_top1` 接近 chance 且与 `_zeroeeg` 差距很小，即 FEIS imagined 的内容上限——
转向：(1) 以 KaraOne overt（trial-sync、强信号）为保真主战场；(2) FEIS 侧目标降级为"受试 + 对齐"，不再堆 epoch。
KaraOne 目标抽取与多数据集训练命令见 `app/README_v3.md`。
