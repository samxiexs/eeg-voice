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
METHOD.md         # how reconstruction + EEG/audio alignment work, and the optimizations
DIFFUSION_PLAN.md # generative (latent-diffusion) path design — escapes mean-collapse
RUN_SERVER.md     # step-by-step runbook
```

## Switchable pipeline

`EEG -> encoder(opt. channel-MoE) -> decoder head -> acoustic target -> vocoder -> wav`.
Everything is a switch (see [METHOD.md](METHOD.md) and [DIFFUSION_PLAN.md](DIFFUSION_PLAN.md)):

- **Acoustic target** (`target.kind`): `mel` (log-mel + scipy Griffin-Lim vocoder,
  **default**, offline) or `encodec_latent` (EnCodec continuous latent + EnCodec decoder).
- **Decoder head**: `regression` (`train_karaone_recon.py`) or `diffusion`
  (`EEGLatentDiffusion`, `train_karaone_diffusion.py`).
- **Alignment / anti-collapse** (regression): `lambda_dtw` (DTW-aligned recon — handles
  the cross-trial onset jitter that naive frame-wise regression cannot) and `lambda_gan`
  (adversarial, fights mean-collapse). Defaults: dtw on, gan off (set `lambda_gan>0`).
- **Encoder channel-MoE** (`--model moe`): per-channel gate + channel clustering.
- **Subject-agnostic**: no subject-ID input anywhere (output is identical across subjects).

Default = `mel + regression + DTW + Griffin-Lim`, mirroring the EEG→speech literature
(NeuroTalk / Park 2025 / FESDE). The earlier EnCodec-latent cosine-regression path
collapses to the mean voice on this data (std-ratio ~0.15, sample pairwise-corr ~0.94)
and is kept only as an honest baseline switch.

The raw 23GB KaraOne `.tar.bz2` archives are not included.

## Current data

- Subjects: 14
- Trials: 1913
- Labels: 11
- EEG channels: 62
- EEG stages: `clearing`, `stimulus_like`, `thinking`, `overt_like`
- Audio target: same-trial overt wav, normalized and encoded as EnCodec latents

## Mainline model

```text
EEG [62 x L]
  -> channel-MoE front-end (selects/clusters channels)   # only with --model moe
  -> stage-conditioned spatial-temporal encoder           # NO subject ID
  -> valid-length masked time pooling -> utterance embedding
  -> latent head (content + EEG-derived global/voice)
  -> EnCodec latent [T,128]
  -> frozen EnCodec decoder
  -> wav (loudness rescaled by a predicted log-RMS head)
```

Key properties (see [METHOD.md](METHOD.md) for the full story):

- **Subject-agnostic.** The model takes *only* EEG (+ task `stage`). There is no
  per-subject lookup table; the output is bit-identical across subject ids. This
  is deliberate — we want generation *from the EEG*, not from an id prior.
- **Channel-MoE encoder** (`--model moe`): a learned per-channel gate + soft
  clustering of channels into experts, i.e. "not every channel is useful" made
  explicit. Plain spatial conv with `--model baseline`.
- **Two training signals for alignment**: frame-wise regression to the EnCodec
  latent *plus* a cross-modal contrastive loss (Defossez 2022) that pulls each
  trial's EEG embedding toward its own audio and away from other trials'.
- **Refiner is NOT diffusion.** `train_karaone_refiner.py` is an optional
  single-step residual post-filter on a frozen checkpoint. See `refiner.py`.

## Metrics to trust

Always compare prediction against:

- `zeroeeg`: EEG set to zero (now a global constant, since there is no subject id)
- `mean_latent`: global target latent mean
- `oracle_codec`: true target latent decoded by EnCodec (the quality ceiling)

The headline number is `pred_over_zero_cos_gain`, not raw reconstruction cosine.
Because the model is subject-agnostic, this gain is a clean measure of how much
the EEG actually contributes over a content-free baseline.

