# v3 EEG → Speech 实现说明 / Runbook

实现了 `NEW_DESIGN_eeg2speech_v3.md` 的核心方案：**解耦识别与合成**，用
对比 + latent + 分类目标取代原来的裸波形回归（根治 mode collapse），并用
**冻结 EnCodec 解码器**保证输出自然。

## 代码结构

```
app/src/v3/
  encoder.py   # 空间-时间 EEG 编码器 + FiLM 被试条件化 + 通道 dropout
  model.py     # EEG2SpeechV3：content / class / contrastive 三个 head
  losses.py    # InfoNCE + latent(cosine/MSE) + 分类 CE + 可选 KD
  data.py      # V3Dataset（复用 FEISProtocolDataset，加 teacher-stage 配对）
  eval.py      # subject-specific 检索(template top-1/5) + label-only + class-head
  recon_eval.py # 重建保真：解码 wav vs 受试本人目标 wav 的 Mel/STFT + subject 专属性对照
  synth.py     # 预测 latent → 反归一化 → 冻结 EnCodec decode → wav
app/scripts/
  v3_train.py        # 训练（支持课程：--init-from / --distill-teacher）
  v3_eval.py         # 单独评测
  v3_synthesize.py   # 合成 wav（含 canonical reference 对照）
app/configs/v3_encodec.yaml
app/tests/test_v3_smoke.py
```

复用的既有组件：`FEISProtocolDataset`（协议 G/S/U + EnCodec-latent 目标缓存）、
`_EncodecLatentBackend`（冻结 EnCodec encode/decode）。

## 关键设计点（对应 design md）

- **不回归原始波形**：模型预测 EnCodec latent 序列 `[B,75,128]`（归一化空间），
  合成端用冻结 EnCodec decoder 还原 → 输出天生自然。
- **FiLM 被试条件化**：subject embedding 调制编码器 trunk（不是侧 head 拼接），
  从而可以跨被试池化（数据 ×20）。`num_subjects+1` 预留 unknown 行给未见被试。
- **反坍缩目标**：InfoNCE + latent cosine/MSE + 分类 CE，三者都惩罚"对所有输入给同一输出"。
- **跨阶段课程**：先用 speaking（信号最强）训 teacher，再 warm-start / 蒸馏到 thinking。

## 前置：准备 EnCodec 目标缓存（P0）

已存在 `../artifacts/audio_targets/feis_subject_templates_encodec_latents.npz`
（336 模板 × 75 帧 × 128 维）。若需重建：

```bash
python scripts/extract_audio_targets.py \
  --config configs/alignment_encodec_local.yaml --backend encodec_latent
```

验证合成端（应听到干净人声）：

```bash
python scripts/v3_synthesize.py --config configs/v3_encodec.yaml \
  --checkpoint <任意已训 best.pt> --out-dir ../artifacts/outputs_v3/_p0_probe --limit 8
```

## P1：speaking 上限探针（先证明 pipeline 能 work）

```bash
python scripts/v3_train.py --config configs/v3_encodec.yaml \
  --protocol G --stage speaking --run-suffix speaking_teacher
```

期望：speaking 的 retrieval top-1 显著 > chance(0.0625)。若 speaking 都解不出来，
说明是流程 bug，而不是 thinking 难。

## P2：thinking 主线（warm-start + 可选 KD）

```bash
python scripts/v3_train.py --config configs/v3_encodec.yaml \
  --protocol G --stage thinking --run-suffix thinking_main \
  --init-from ../artifacts/outputs_v3/g_speaking_speaking_teacher/checkpoints/best.pt \
  --distill-teacher ../artifacts/outputs_v3/g_speaking_speaking_teacher/checkpoints/best.pt \
  --teacher-stage speaking
```

- `--init-from`：用 speaking 模型权重热启动。
- `--distill-teacher` + `--teacher-stage speaking`：同一 trial 的 speaking EEG 喂 teacher，
  thinking EEG 喂 student，KD 对齐（latent MSE + logits KL）。
- 协议切换：`--protocol S --subject 01` 或 `--protocol U --holdout-subject 21`。

## 评测与合成

```bash
python scripts/v3_eval.py --config configs/v3_encodec.yaml \
  --checkpoint <run>/checkpoints/best.pt --protocol G --stage thinking --split test \
  --out <run>/metrics/test_eval.json

python scripts/v3_synthesize.py --config configs/v3_encodec.yaml \
  --checkpoint <run>/checkpoints/best.pt --protocol G --stage thinking \
  --out-dir <run>/recon_wavs --limit 32
```

每条 trial 产出 `*_pred.wav`（模型预测）与 `*_ref.wav`（canonical 目标经 EnCodec 解码，
即本 pipeline 的上限），便于 A/B 听感对照。

## 冒烟测试

```bash
python app/tests/test_v3_smoke.py        # 或 python -m pytest app/tests/test_v3_smoke.py -q
```

## 依赖

`torch`, `transformers`（EnCodec）, `numpy`, `soundfile`, `scipy`, `tqdm`。
EnCodec 权重在 `../models/encodec_24khz/`，`local_files_only: true` 离线可用。

## 多数据集（v4）：FEIS + KaraOne 统一向量空间

详见 `NEW_DESIGN_multidataset_v4.md`。把数据集做成可插拔单元，EEG 经
**per-dataset 输入 adapter** 进入**共享 trunk + 共享 content/contrastive head**，
落在同一个 EnCodec-latent 向量空间；config 一行控制单选/多选/联合。

新增组件：

```
src/v3/datasets.py    # DatasetSpec 注册表 + UnifiedEEGSpeechDataset + 全局 subject id
src/v3/model.py       # EEG2SpeechMD（多 adapter + dataset/subject FiLM + per-dataset 分类头）
src/v3/encoder.py     # 拆出 SpatialAdapter(per-dataset) + 共享 TemporalTrunk
scripts/v3_extract_karaone_targets.py  # KaraOne trial 级 EnCodec 目标
scripts/v3_train_md.py                 # 多数据集训练
configs/v3_multidataset.yaml
```

### 准备 KaraOne（一次性）

推荐直接从原始 KaraOne 受试者压缩包准备 v4 数据。脚本会逐个读取
`<repo>/data/KaraOne/*.tar.bz2`，只临时解压 `.cnt`、`epoch_inds.mat` 和
`kinect_data/*.wav/*.txt`，写入 bundle 需要的 `../data/karaone/`，随后删除临时解压目录。
最后会继续生成 trial 级 EnCodec latent cache：

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/server_bundle/feis_subject_aware_bundle/app
conda activate feis_ssl

bash prepare_karaone_v4.sh
```

只先处理少数受试或少量 target 做 smoke test：

```bash
KARAONE_SUBJECTS="MM05 MM08" TARGET_LIMIT=16 bash prepare_karaone_v4.sh
```

如果你已经有处理好的 KaraOne 目录，也可以手动放到 bundle，使其与 FEIS 同构：
`../data/karaone/segments.csv`、`../data/karaone/subjects/`、`../data/karaone/audio/`。
然后只抽 trial 级 EnCodec 目标：

```bash
python scripts/v3_extract_karaone_targets.py \
  --karaone-root ../data/karaone --codec-model ../models/encodec_24khz \
  --out ../artifacts/audio_targets/karaone_trial_encodec_latents.npz \
  --duration-sec 2.0 --extract-steps 150   # 2.0s @ 75Hz = 150 帧, 与 target_steps 对齐；--limit N 可先小样本试跑
```

`REGISTRY`（`src/v3/datasets.py`）里已登记 feis / karaone 的通道数、stage、目标缓存路径，
按需改路径即可。

### 单选 / 多选 / 联合训练

```bash
# 只 FEIS
python scripts/v3_train_md.py --config configs/v3_multidataset.yaml --datasets feis
# 只 KaraOne
python scripts/v3_train_md.py --config configs/v3_multidataset.yaml --datasets karaone
# 联合（共享 trunk，加权采样）
python scripts/v3_train_md.py --config configs/v3_multidataset.yaml --datasets feis,karaone
# KaraOne 预训练 → FEIS 微调
python scripts/v3_train_md.py --config configs/v3_multidataset.yaml --datasets feis \
  --init-from ../artifacts/outputs_v4/karaone_md/checkpoints/best.pt --run-suffix ft_from_karaone
```

- batch 是 dataset-homogeneous（通道数不同），联合时按 `sampling_weights` 在数据集间轮流出 batch。
- 每个数据集独立做检索评测，分别报 `[test:feis]` / `[test:karaone]`。
- 公共目标窗 `target_steps=150`（~2.0s）：FEIS 1.0s 自动 pad+mask，KaraOne 变长按 valid 帧 mask。

### 重建保真的落点

- 两数据集统一走冻结 EnCodec decode（自然度保证）。
- **KaraOne 有 trial-sync 真录音**，是唯一能优化并评测「逐 trial 波形保真」的地方
  （pred wav vs 该 trial 真 wav 的 Mel/STFT 距离）；FEIS canonical 只能到「选对 16 选 1」上限。
- 因此若主目标是「还原得像」，把 KaraOne（尤其 overt_like 强信号阶段）作为保真主战场、
  FEIS 作为识别/对齐增强，是最优组合。

## 指标解读（subject-specific，非 16 选 1）

FEIS 目标是**每个受试自己的录音**：`01:f` 与 `02:f` 是不同波形，全集 336=21×16 条
（clean 时 320=20×16）。所以评测以 subject-specific 为主：

- **`template_top1 / top5`（主指标）**：是否检索到**正确受试的正确 prompt**（subject+label 都对），
  chance = 1/320 ≈ 0.003。这才是"还原到他本人的那条"的度量。
- `label_top1 / top5`（辅助）：只看 prompt 对不对、忽略受试身份的弱视角，chance=1/16=0.0625。
- `class_head_accuracy`：分类头的 16 类准确率（诊断用）。
- **重建保真**（`v3_recon_eval.py`）：解码预测 wav 与**该受试本人目标 wav**的 Mel/STFT 距离；
  `subject_specificity_gap > 0` 表示重建更接近本人录音（对照同 prompt 的别人录音）。这是
  "还原得像不像他自己"的直接证据。
- 协议 U（holdout 受试不在 train）的 template 检索用 oracle test bank（`bank_split=test`，训练脚本已自动处理）。
- 务必三协议 + speaking/thinking 双阶段分层报告（speaking 是 pipeline 健康度探针）。
