# Server runbook

Run all commands from:

```bash
cd ~/speech_decoding/eeg2wave_server_bundle/karaone_overt_recon_bundle/app
```

## 1. Environment

```bash
conda create -n karaone-eegvoice python=3.10 -y
conda activate karaone-eegvoice
pip install -r requirements.txt

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Check GPU:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## 2. Data audit

```bash
python scripts/analyze_karaone_data.py --config configs/karaone.yaml
```

Outputs:

```text
../reports/karaone_data_analysis.md
../reports/karaone_data_summary.json
```

## 3. Target cache

The bundle already includes:

```text
../artifacts/audio_targets/karaone_trial_encodec_latents.npz
```

Regenerate only if data or codec settings change:

```bash
python scripts/extract_karaone_targets.py --config configs/karaone.yaml
```

## 4. Baseline overt reconstruction

```bash
python scripts/train_karaone_recon.py \
  --config configs/karaone.yaml \
  --stages overt_like \
  --model baseline \
  --epochs 120
```

Output run:

```text
../artifacts/outputs_karaone/karaone_baseline_overt_like_v1/
```

## 5. MoE overt reconstruction

```bash
python scripts/train_karaone_recon.py \
  --config configs/karaone.yaml \
  --stages overt_like \
  --model moe \
  --epochs 120
```

## 6. Thinking fine-tune

Use an overt checkpoint as initialization:

```bash
python scripts/train_karaone_recon.py \
  --config configs/karaone.yaml \
  --stages thinking \
  --model moe \
  --init-from ../artifacts/outputs_karaone/karaone_moe_overt_like_v1/checkpoints/best.pt \
  --run-suffix thinking_ft_v1 \
  --epochs 80
```

## 7. Evaluation

```bash
python scripts/eval_karaone_recon.py \
  --config configs/karaone.yaml \
  --checkpoint ../artifacts/outputs_karaone/karaone_moe_overt_like_v1/checkpoints/best.pt \
  --split test

python scripts/eval_karaone_recon.py \
  --config configs/karaone.yaml \
  --checkpoint ../artifacts/outputs_karaone/karaone_moe_overt_like_v1/checkpoints/best.pt \
  --split subject_test
```

## 8. Listening samples

```bash
python scripts/synthesize_karaone.py \
  --config configs/karaone.yaml \
  --checkpoint ../artifacts/outputs_karaone/karaone_moe_overt_like_v1/checkpoints/best.pt \
  --split test \
  --limit 48
```

Each sample writes:

```text
original / oracle_codec / mean_latent / zeroeeg / pred / pred_scaled / zeroeeg_scaled
```

## 9. Optional second-stage refiner

```bash
python scripts/train_karaone_refiner.py \
  --config configs/karaone.yaml \
  --checkpoint ../artifacts/outputs_karaone/karaone_moe_overt_like_v1/checkpoints/best.pt \
  --epochs 60
```

The refiner is a residual denoising latent model. Treat it as an enhancement
branch; the baseline/MoE checkpoint remains the main result.

