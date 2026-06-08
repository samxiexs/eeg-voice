# FEIS Local Run Guide

本地直接跑，建议只用统一 bundle，这样和你之后传服务器的目录完全一致。

目录是：

`server_bundle/feis_subject_aware_bundle`

## 1. 进入目录

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/server_bundle/feis_subject_aware_bundle/app
```

## 2. 创建并激活环境

如果你用 `conda`：

```bash
conda create -n feis_ssl python=3.12 -y
conda activate feis_ssl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果你用现有环境，也至少执行：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果 `torch` 在清华源装不上，直接临时切官方源：

```bash
env PIP_INDEX_URL=https://pypi.org/simple python -m pip install -r requirements.txt
```

## 3. 验证本地 HuBERT 路径

```bash
ls -lah ../models/hubert-base-ls960
```

你应该能看到至少这些文件：

```bash
config.json
preprocessor_config.json
pytorch_model.bin
```

## 4. 重新提取 HuBERT speech targets

这一步会生成真正的 `768` 维 SSL embedding target：

```bash
python scripts/extract_audio_targets.py --config configs/alignment_ssl_local.yaml
```

生成文件在：

```bash
../artifacts/audio_targets/feis_subject_templates_ssl.npz
```

## 5. 跑 subject-aware alignment 训练

`Protocol G`，多被试 pooled：

```bash
python scripts/train_alignment.py --config configs/alignment_ssl_local.yaml --protocol G
```

`Protocol S`，单被试：

```bash
python scripts/train_alignment.py --config configs/alignment_ssl_local.yaml --protocol S --subject 01
```

`Protocol U`，留一被试：

```bash
python scripts/train_alignment.py --config configs/alignment_ssl_local.yaml --protocol U --holdout-subject 21
```

## 6. 跑 alignment 评估

`Protocol G`：

```bash
python scripts/eval_alignment.py --config configs/alignment_ssl_local.yaml --protocol G --split test
```

`Protocol S`：

```bash
python scripts/eval_alignment.py --config configs/alignment_ssl_local.yaml --protocol S --subject 01 --split test
```

`Protocol U`：

```bash
python scripts/eval_alignment.py --config configs/alignment_ssl_local.yaml --protocol U --holdout-subject 21 --split test
```

## 7. 跑 raw-waveform baseline

单被试 baseline：

```bash
python scripts/train_waveform_protocol.py --config configs/waveform_protocol.yaml --protocol S --subject 01
python scripts/eval_waveform_protocol.py --config configs/waveform_protocol.yaml --protocol S --subject 01
```

pooled baseline：

```bash
python scripts/train_waveform_protocol.py --config configs/waveform_protocol.yaml --protocol G
python scripts/eval_waveform_protocol.py --config configs/waveform_protocol.yaml --protocol G
```

pooled + subject conditioning baseline：

```bash
python scripts/train_waveform_protocol.py --config configs/waveform_protocol.yaml --protocol G --subject-conditioning
python scripts/eval_waveform_protocol.py --config configs/waveform_protocol.yaml --protocol G
```

## 8. 跑 EEG ablation

随机 EEG：

```bash
python scripts/train_alignment.py --config configs/alignment_ssl_local.yaml --protocol G --ablation-mode random_noise
python scripts/eval_alignment.py --config configs/alignment_ssl_local.yaml --protocol G --ablation-mode random_noise --split test
```

shuffle EEG：

```bash
python scripts/train_alignment.py --config configs/alignment_ssl_local.yaml --protocol G --ablation-mode shuffle_eeg
python scripts/eval_alignment.py --config configs/alignment_ssl_local.yaml --protocol G --ablation-mode shuffle_eeg --split test
```

subject-average EEG：

```bash
python scripts/train_alignment.py --config configs/alignment_ssl_local.yaml --protocol G --ablation-mode subject_mean
python scripts/eval_alignment.py --config configs/alignment_ssl_local.yaml --protocol G --ablation-mode subject_mean --split test
```

## 9. 跑 pipeline audit

```bash
python scripts/audit_pipeline.py --output-path ../reports/pipeline_audit.json
```

## 10. 结果位置

alignment 结果默认在：

```bash
../artifacts/outputs_alignment/
```

waveform baseline 结果默认在：

```bash
../artifacts/outputs_waveform_protocol/
```

audio target cache 在：

```bash
../artifacts/audio_targets/
```

如果你要，下一步可以再补一份按顺序执行的完整脚本，比如 `run_local_feis_full.sh`。
