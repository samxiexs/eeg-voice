# 服务器运行手册 — factored（训练 + 重建）

> 适用：已把 `eeg2wave_server_bundle` 整个 cp 到服务器（如 `~/speech_decoding/eeg2wave_server_bundle`）。
> 依赖与权重均随 bundle：encodec 权重（`models/encodec_24khz/model.safetensors`）、
> EEG 数据（`data/feis`）、目标缓存（`artifacts/audio_targets/feis_subject_templates_encodec_latents.npz`）都在本地，
> `local_files_only=True`，**不需要联网下载模型**。
>
> 所有命令都在 `feis_subject_aware_bundle/app` 目录下执行。

---

## 1. 建 conda 环境 + 装依赖

```bash
conda create -n eegvoice python=3.10 -y
conda activate eegvoice

cd ~/speech_decoding/eeg2wave_server_bundle/feis_subject_aware_bundle/app
pip install -r requirements.txt
```

> `requirements.txt` 默认装 CUDA 版 torch。若服务器 CUDA 版本特殊导致装错，改用官方指定版本，例如：
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`

## 2.（可选）确认 GPU + 强制离线

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # 权重已在本地，强制离线更稳
```

脚本自动 `cuda if available`，有卡即上 GPU，无需额外参数。

> 以下顺序对应 `V2_PLAN.md`：**先 Stage-1 决定性探针 → 训练 → Stage-0 codec QC → 重建/插值**。
> run 名 = `factored_stimuli_thinking_v2`（`--stages stimuli` 单训听时为 `factored_stimuli_v2`）。

## 3. Stage-1：内容可解性探针（最便宜的判决，**先跑这个**）

不依赖 codec，几分钟出结果。回答"EEG 里到底有没有内容信号"：

```bash
python scripts/content_probe.py \
    --config configs/factored.yaml \
    --stages stimuli,thinking --folds 5 --permutations 200
```

产物：`../artifacts/outputs_factored/content_probe/probe_content_*.json`。
看 `verdict`：若两阶段都 `NOT DECODABLE`（置换 p≥0.05）→ 内容不可解，按计划停 FEIS-only、转数据集/预训练，**下面训练只为拿"诚实负结果 + 听感 demo"**。

**阳性对照（强烈建议，封口实验）**：用同样的特征和探针改解 **subject 身份**，证明管线没坏：

```bash
python scripts/content_probe.py \
    --config configs/factored.yaml \
    --target subject --stages stimuli,thinking --folds 5 --permutations 200
```

预期身份准确率 ~0.8（≈16× chance，p<0.05）= `POSITIVE CONTROL PASSED`。
同一管线身份解得出、内容解不出 → 内容的负结果**无法被"解码器太弱/有 bug"反驳**。
（`--target stage` 可再做一个二分类对照。）

## 4. 训练（100 epoch，听 + 想象两阶段）

```bash
python scripts/factored_train.py \
    --config configs/factored.yaml \
    --stages stimuli,thinking \
    --epochs 100
```

产物（run 名 = `factored_stimuli_thinking_v2`）：

- checkpoint：`../artifacts/outputs_factored/factored_stimuli_thinking_v2/checkpoints/best.pt`（按 **val 内容增益 top1−zeroeeg** 选）
- 指标：`.../metrics/test_metrics.json`（headline 是 `content_gain`）+ 逐 epoch `history.csv` / `history.jsonl`
- 末尾会打印 `[verdict]`：若 `no_eeg_content_gain` → 内容没超过 zero-EEG 基线。

> 后台长跑：`nohup python scripts/factored_train.py --config configs/factored.yaml --stages stimuli,thinking --epochs 100 > train.log 2>&1 &`。

## 5. Stage-0：codec QC + 塌缩诊断（五路对照）

```bash
python scripts/factored_recon_eval.py \
    --config configs/factored.yaml \
    --checkpoint ../artifacts/outputs_factored/factored_stimuli_thinking_v2/checkpoints/best.pt \
    --split test_holdout --qc-cells 24 --save-wav 12
```

产物（带时间戳目录）：`audio_qc.json`（codec 是否健康）、`collapse_diagnostics.json`、`recon_pairs.csv`、
`listening_manifest.csv` + `wav/`（每格 5 路 wav：original/oracle/mean/pred_unscaled/pred_scaled）。
看 `audio_qc.json` 的 `verdict`：`CODEC OK` 表示病在模型不在 codec。

## 6. 重建全部（从 EEG 还原 wav）

```bash
python scripts/factored_synthesize.py \
    --config configs/factored.yaml \
    --checkpoint ../artifacts/outputs_factored/factored_stimuli_thinking_v2/checkpoints/best.pt \
    --split test_holdout --limit 100000
```

每个样本输出 5 路 wav（含 rep 序号，不再互相覆盖）+ `listening_manifest.csv`。
`--limit` 调大即"全部"（holdout≈394、seen≈600）；换 `--split test_seen` 听见过的组合。

## 7.（可选）嗓音插值 demo —— 证明学到跨被试音色

```bash
python scripts/factored_interpolate.py \
    --config configs/factored.yaml \
    --checkpoint ../artifacts/outputs_factored/factored_stimuli_thinking_v2/checkpoints/best.pt \
    --label f --subjects 01,10 --steps 5 \
    --out-dir ../artifacts/outputs_factored/factored_stimuli_thinking_v2/interp
```

固定内容 `f`，在 sub01 ↔ sub10 之间扫嗓音，输出「同一个音、嗓音连续渐变」的 5 段 wav。

---

## 备注

- 路径均相对 `app/`，务必在 `app` 目录下执行；各步 run 名一致，顺序复制即可。
- 训练只用缓存的 latent，不需要 encodec 权重；QC/重建/插值才加载 encodec decoder。
- **决定性数字是 `content_gain`（top1 − zeroeeg）**，不是 raw top1。跑完把 `content_probe_*.json` 和 `test_metrics.json` 贴回，用于填报告。
- 想堵掉 holdout 上 zero-EEG gaming：`configs/factored.yaml` 里设 `holdout_random: true` 重训。
