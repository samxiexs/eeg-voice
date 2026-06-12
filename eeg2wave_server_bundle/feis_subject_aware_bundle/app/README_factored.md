# FEIS Factored 模型（content × speaker 网格生成）

按"网格"思路正确使用 FEIS：把目标拆成**内容（从 EEG 解）× 说话人（已知 subject 取）**，
用解耦生成 + 留格子泛化，做到**超越分类、直接生成波形**，并诚实量出"内容到底解出多少"。

## 数据结构（你的四维网格）

```
内容(16 label) × 说话人(20 受试) × 阶段(stimuli听 / thinking想象) × 重复(~10)
每个 (受试,label) 格子 → 1 条目标 EnCodec latent（该受试自己的录音）
```

- **content**：从 EEG 解（难，科学问题）；监督 = 目标感知监督对比（同 label 为正）+ 16 类 CE + 向说话人无关原型对齐。
- **speaker**：从**已知 subject id** 取 embedding（不从 EEG 解，杜绝身份捷径）。
- **generator**：(content_seq, speaker) → EnCodec latent → 冻结声码器 → wav。

### 划分：hold-out-cell（核心）

用 Latin-square 整格挖掉若干 `(受试,label)`（每受试 1 格，且每受试/每 label 仍在 train 的别的格子里出现）：

| split | 内容 | 作用 |
|---|---|---|
| `train` | 见过格子的 reps[:-1] | 训练 |
| `test_seen` | 见过格子的最后 1 rep | 同格子识别（recognition）|
| **`test_holdout`** | **挖掉格子的全部 reps** | **生成没见过的 受试×内容 组合 = 超越分类的证据** |

实测规模（stimuli+thinking，clean 20 人）：train **5310** / test_seen **600** / test_holdout **394**，20 个 hold-out 格子。

## 代码结构

```
app/src/feis_factored/
  targets.py   # 从 EnCodec 缓存建 content 原型 / speaker 原型 / 粗音系类别（无需 transformers）
  data.py      # FactoredFEISDataset：网格 + hold-out-cell 划分 + 多阶段
  model.py     # FactoredEEG2Speech：EEG→content + subject→speaker → generator → latent
  losses.py    # 目标感知监督对比 + 内容 CE + 原型对齐 + 重建 cos/MSE
  eval.py      # within-subject 内容 top1 + zero-EEG 对照 + 粗类别 + 生成保真
app/scripts/
  factored_train.py       # 训练 + 自动评测 test_seen / test_holdout
  factored_synthesize.py  # 解码预测 latent → wav（需 EnCodec；test_holdout 可听"生成新组合"）
app/configs/factored.yaml
app/tests/test_factored_smoke.py
```

## 运行

```bash
cd app
export KMP_DUPLICATE_LIB_OK=TRUE          # macOS OpenMP

# 训练（听+想象联合）
python scripts/factored_train.py --config configs/factored.yaml --stages stimuli,thinking
# 只听 / 只想象
python scripts/factored_train.py --config configs/factored.yaml --stages stimuli
python scripts/factored_train.py --config configs/factored.yaml --stages thinking

# 合成（听 test_holdout 的"生成新组合"，与该格目标 ref 对照）
python scripts/factored_synthesize.py --config configs/factored.yaml \
  --checkpoint ../artifacts/outputs_factored/factored_stimuli_thinking_v1/checkpoints/best.pt \
  --split test_holdout --out-dir ../artifacts/outputs_factored/.../wavs --limit 24
```

## 看哪些数（决定性指标）

| 指标 | 含义 | chance |
|---|---|---|
| `within_subject_content_top1` | 受试内 16 选 1 解内容（真·内容）| 1/16=0.0625 |
| `within_subject_content_top1_zeroeeg` | EEG 置零、仅 stage 的基线；与上者之差 = EEG 贡献 | — |
| `content_top1_by_stage` | 分 stimuli/thinking 看（听 > 想象）| 0.0625 |
| `coarse_manner/voicing/vc_acc` | 粗音系类别（常是 FEIS 真信号）| 见类别数 |
| `recon_cos_to_cell` | 预测 latent 对该格目标的余弦（生成保真）| — |
| **test_holdout 上的同套指标** | **没见过的组合上还成立吗 = 超越分类** | 同上 |

## 诚实预期（需训练后实测）

- 合成自然 + 嗓音对：基本必达（冻结声码器 + 已知 subject）。
- 内容 16 选 1：stimuli 阶段大概率显著 > chance（感知诱发），thinking 阶段可能接近 chance；粗类别更高。
- test_holdout：能合成"新 受试×内容"组合（技术上超越分类），但**内容正确率 = 上面那行内容准确率**。
- 做不到：sub-phoneme 声学细节、连续/新词语音（目标里没有，无法监督）。

依赖：`torch`、`numpy<2`、`soundfile`、`transformers`（仅合成时）、本地 `models/encodec_24khz`。

---

## 嗓音（speaker）相关新增（2026-06）

为"学到跨被试嗓音特点"补了两块（来自语音/解耦文献）：

1. **嗓音原型显式监督**（`lambda_speaker`）：speaker embedding 通过 `speaker_to_proto` 对齐到
   **音频派生的 speaker 原型**（该受试 16 条录音去内容后的均值，`targets.speaker_prototype`）。
   → 让 embedding 真的承载"这个人的音色"，而非随意向量。（可换成 ECAPA-TDNN，见下。）
2. **内容↔说话人对抗解耦**（`lambda_adv` + 梯度反转 GRL）：一个 `subject_adversary` 试图从
   `content_embed` 预测 subject，GRL 把梯度反向 → **强制 content 丢掉 subject 身份**。
   训练日志里 `adv_subj_acc` **下降**就说明 content 越来越说话人无关（直接打"身份 confound"）。

### 嗓音插值 demo（证明嗓音被学到、且连续可生成）

```bash
python scripts/factored_interpolate.py --config configs/factored.yaml \
  --checkpoint <run>/checkpoints/best.pt --label f --subjects 01,10 --steps 5 \
  --out-dir <run>/interp
```
固定内容（来自 01 的一条 `f` EEG），把 speaker embedding 在 01↔10 之间插值 → 输出"同一个 `f`、
嗓音从 01 渐变到 10"的一串 wav。能听出嗓音平滑过渡 = 学到了跨被试嗓音流形。

## 参考与方法对照（哪些来自论文）

| 组件 | 对应文献（`paper-ref/`）|
|---|---|
| EEG→语音 codec 目标 + 冻结声码器 | EnCodec(Défossez 2022)；HiFi-GAN/SoundStream；Lee 2023《Voice Reconstruction from EEG during Imagined Speech》|
| 内容（音素）从 EEG 解 | Lee 2025《Listened Speech Decoding…Parallel Phoneme Sequence》；HuBERT/WavLM/wav2vec2(内容单元) |
| 跨被试对比 / 检索 | Défossez 2023《Decoding speech perception…》（subject layer + InfoNCE）|
| **speaker embedding（嗓音）** | **ECAPA-TDNN / x-vector**（可替换 `speaker_to_proto` 的目标）|
| **内容↔说话人对抗解耦** | 域对抗 DANN（GRL）；AutoVC / StyleTTS 风格 voice-conversion 解耦 |
| 离散化 EEG（备选） | DeWave(Duan 2024)；speech_quantization |
| EEG 基础模型前端（可选） | LaBraM(Jiang 2024)、NeuroLM(Jiang 2025)、BrainOmni、LUNA |

### 可选增强（需额外权重/依赖，已留接口）

- **ECAPA-TDNN 嗓音目标**：用 `speechbrain` 的 ECAPA 对每个受试音频抽 speaker embedding，替换
  `targets.speaker_proto` → 更强的音色刻画（当前用 EnCodec 均值近似）。
- **HuBERT 内容单元**：用 HuBERT 离散单元作**说话人无关**的内容目标，替换按 label 派生的 content 原型
  → 更纯的内容解耦（需 `transformers`）。
- **F0/CREPE 韵律**：把音高轮廓并入 speaker/prosody 条件。

