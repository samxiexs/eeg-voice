# KaraOne Overt Reconstruction Bundle

This is the KaraOne-first EEG-to-waveform training bundle.

Primary task:

```text
KaraOne overt_like EEG -> same-trial overt wav
```

Secondary task:

```text
KaraOne thinking EEG -> same-trial overt wav
```

The bundle is intentionally separate from the FEIS factored bundle. KaraOne has
trial-synchronous real audio, so it is the right mainline for waveform fidelity
experiments.

## Contents

```text
app/
  configs/karaone.yaml
  scripts/analyze_karaone_data.py
  scripts/extract_karaone_targets.py
  scripts/train_karaone_recon.py
  scripts/eval_karaone_recon.py
  scripts/synthesize_karaone.py
  scripts/train_karaone_refiner.py
  src/karaone_recon/
data/karaone/
  processed KaraOne EEG/audio bundles
artifacts/audio_targets/
  karaone_trial_encodec_latents.npz
models/encodec_24khz/
  local EnCodec weights
reports/
  karaone_data_analysis.md
  karaone_data_summary.json
```

The raw 23GB KaraOne `.tar.bz2` archives are not included.

## Current data

- Subjects: 14
- Trials: 1913
- Labels: 11
- EEG channels: 62
- EEG stages: `clearing`, `stimulus_like`, `thinking`, `overt_like`
- Audio target: same-trial overt wav, normalized and encoded as EnCodec latents

## Mainline model

The first production baseline is:

```text
EEG [62 x L]
  -> stage/subject-conditioned spatial-temporal encoder
  -> latent generator
  -> EnCodec latent [T,128]
  -> frozen EnCodec decoder
  -> wav
```

Set `--model moe` to use a 4-expert mixture-of-experts generator. The second
stage `train_karaone_refiner.py` trains a residual denoising latent refiner on
top of a frozen baseline/MoE checkpoint.

## Metrics to trust

Always compare prediction against:

- `zeroeeg`: same subject/stage with EEG set to zero
- `mean_latent`: global target latent mean
- `oracle_codec`: true target latent decoded by EnCodec

The headline number is `pred_over_zero_cos_gain`, not raw reconstruction cosine.

