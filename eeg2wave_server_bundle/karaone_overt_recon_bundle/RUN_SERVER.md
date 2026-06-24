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

Two acoustic targets are supported (switch via `target.kind` in the config or
`--target`). See [METHOD.md](METHOD.md) §4 and [DIFFUSION_PLAN.md](DIFFUSION_PLAN.md).

```bash
# mel target (DEFAULT) — log-mel + Griffin-Lim vocoder, offline / scipy-only
python scripts/extract_karaone_targets.py --target mel
# -> ../artifacts/audio_targets/karaone_trial_mel.npz

# EnCodec-latent target (best audio, EnCodec vocoder) — already shipped
python scripts/extract_karaone_targets.py --target encodec_latent
# -> ../artifacts/audio_targets/karaone_trial_encodec_latents.npz
```

### Pipeline switches (config `karaone.yaml`)

| switch | values | default |
|---|---|---|
| `target.kind` / `--target` | `mel`, `encodec_latent` | mel |
| `vocoder.kind` | `griffinlim`, `encodec` | griffinlim |
| `--model` (encoder channel-MoE) | `baseline`, `moe` | — |
| decoder head | `train_karaone_recon.py` (regression), `train_karaone_diffusion.py` (diffusion) | regression |
| `train.lambda_dtw` | 0 / >0 (DTW-aligned recon) | 1.0 |
| `train.lambda_gan` | 0 / >0 (adversarial anti-collapse) | 0.0 (set 0.1 to enable) |

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

`--model moe` turns on the channel-selecting/clustering MoE inside the EEG
encoder (see [METHOD.md](METHOD.md) §3.3).

```bash
python scripts/train_karaone_recon.py \
  --config configs/karaone.yaml \
  --stages overt_like \
  --model moe \
  --epochs 120
```

### Ablations worth running

The model is subject-agnostic and trained with a frame-wise regression **plus**
a cross-modal contrastive loss (Defossez 2022). To measure each piece, compare
`val_gain` / `pred_over_zero_cos_gain` across:

- `--model baseline` vs `--model moe` (encoder channel-MoE)
- contrastive on vs off: add `lambda_clip: 0.0` under `train:` in the config to disable

Only trust `pred_over_zero_cos_gain`, not raw `pred_recon_cos`.

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

## 9. Generative path: latent diffusion (escapes mean-collapse)

The regression model above collapses to the mean voice (low `std_ratio`, high
pairwise correlation). The diffusion model samples from `p(latent | EEG)` instead.
See [METHOD.md](METHOD.md) §4 and [DIFFUSION_PLAN.md](DIFFUSION_PLAN.md).

```bash
# train (writes metrics/training_curves.png with std_ratio / pairwise-corr panels)
python scripts/train_karaone_diffusion.py \
  --config configs/karaone.yaml \
  --model moe \
  --epochs 60

# sample reconstructions (multiple draws per trial show generative diversity)
python scripts/synthesize_karaone_diffusion.py \
  --config configs/karaone.yaml \
  --checkpoint ../artifacts/outputs_karaone/karaone_diffusion_moe_overt_like_v1/checkpoints/best.pt \
  --split test --limit 8 --num-samples 2
```

Judge it by: `pred_std_ratio_median` rising toward 1.0 and `pred_pairwise_corr_median`
well below the regression's ~0.94 (anti-collapse), alongside `pred_over_mean_cos_gain`.

## 10. Optional second-stage refiner (legacy)

```bash
python scripts/train_karaone_refiner.py \
  --config configs/karaone.yaml \
  --checkpoint ../artifacts/outputs_karaone/karaone_moe_overt_like_v1/checkpoints/best.pt \
  --epochs 60
```

The refiner is a residual denoising latent model. Treat it as an enhancement
branch; the baseline/MoE checkpoint remains the main result.

