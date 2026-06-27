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
  scripts/export_karaone_eeg_csv.py
  scripts/extract_karaone_targets.py
  scripts/train_karaone_recon.py
  scripts/eval_karaone_recon.py
  scripts/synthesize_karaone.py
  scripts/train_karaone_refiner.py
  src/karaone_recon/
data/karaone/
  processed KaraOne EEG/audio bundles
artifacts/audio_targets/
  karaone_trial_encodec_latents.npz  # must be the complete post-202606 format
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

- **Acoustic target** (`target.kind`): `encodec_latent` (**mainline**, EnCodec continuous latent + EnCodec decoder)
  or `mel` (log-mel + scipy Griffin-Lim baseline, offline).
- **Decoder head**: `flow`/`diffusion` (`train_karaone_diffusion.py`) or regression
  (`train_karaone_recon.py`). The mainline runner is `bash run_codec.sh`.
- **Alignment / anti-collapse** (regression): `lambda_dtw` (DTW-aligned recon — handles
  the cross-trial onset jitter that naive frame-wise regression cannot) and `lambda_gan`
  (adversarial, fights mean-collapse). Defaults: dtw on, gan off (set `lambda_gan>0`).
- **Encoder** (`model.encoder_kind`): `conformer` mainline, `transformer` ablation, `cnn` legacy.
- **Encoder channel-MoE** (`--model moe`): per-channel gate + channel clustering.
- **Subject-agnostic**: no subject-ID input anywhere (output is identical across subjects).

Default = `EnCodec latent + conditional flow matching + Conformer + EnCodec decoder`.
The mel + Griffin-Lim route is kept as an honest baseline via `run_mel.sh`.
The older EnCodec cache format is incomplete for synthesis; rebuild it with
`python app/scripts/extract_karaone_targets.py --target encodec_latent --force`, or
just run `bash run_codec.sh`, which checks and rebuilds it automatically.

The raw 23GB KaraOne `.tar.bz2` archives are not included.

## Current data

- Subjects: 14
- Trials: 1913
- Labels: 11
- EEG channels: 62
- EEG stages: `clearing`, `stimulus_like`, `thinking`, `overt_like`
- Audio target: same-trial overt wav, normalized and encoded as EnCodec latents

## EEG CSV audit export

The training pipeline reads compact `.npz` bundles, but you can export FEIS-style
wide CSV files for channel/time auditing and channel-MoE interpretation:

```bash
cd app

# Estimate full output rows without writing large CSV files.
python scripts/export_karaone_eeg_csv.py --dry-run

# Small sanity export: one subject, first two trials, thinking only.
python scripts/export_karaone_eeg_csv.py \
  --subjects MM05 \
  --stages thinking \
  --limit-trials-per-subject 2 \
  --overwrite

# Full per-subject CSV export. This can be several GB.
python scripts/export_karaone_eeg_csv.py --overwrite
```

Default output:

```text
data/karaone/eeg_csv/<SUBJECT>/full_eeg.csv
data/karaone/eeg_csv/manifest.csv
```

Each CSV row is one trial/stage/time sample, with metadata plus 62 channel
columns: `Time:256Hz, subject_id, trial_index, label, stage, sample_index,
valid, audio_path, <62 EEG channels>`. The exporter uses `channel_names` from
the subject bundle when available (for example `FP1`, `FPZ`, `AF3`); pass
`--generic-channel-names` to force `Ch001...Ch062`. Padding rows are preserved
by default and marked with `valid=false`; pass `--drop-padding` to keep only
valid samples.

## Mainline model

```text
EEG [62 x L]
  -> channel-MoE front-end (selects/clusters channels)   # only with --model moe
  -> conformer/transformer EEG encoder                    # NO subject ID
  -> valid-length masked time pooling -> utterance embedding
  -> latent head (content + EEG-derived global/voice)
  -> EnCodec latent [T,128] or conditional flow samples
  -> frozen EnCodec decoder
  -> wav (loudness/active energy monitored separately)
```

Key properties (see [METHOD.md](METHOD.md) for the full story):

- **Subject-agnostic.** The model takes *only* EEG (+ task `stage`). There is no
  per-subject lookup table; the output is bit-identical across subject ids. This
  is deliberate — we want generation *from the EEG*, not from an id prior.
- **Channel-MoE encoder** (`--model moe`): a learned per-channel gate + soft
  clustering of channels into experts, i.e. "not every channel is useful" made
  explicit. Plain spatial conv with `--model baseline`.
- **Two-stage supervision**: HuBERT/wav2vec-style semantic auxiliary targets and
  prompt-token CTC guide content, while EnCodec/flow handles acoustic realization.
- **Energy repair**: frame log-energy, active voiced-region RMS, decoder-scale, and
  envelope metrics directly target the low-volume/flat-waveform failure mode.
- **Refiner is NOT diffusion.** `train_karaone_refiner.py` is an optional
  single-step residual post-filter on a frozen checkpoint. See `refiner.py`.

## Metrics to trust

Always compare prediction against:

- `zeroeeg`: EEG set to zero (now a global constant, since there is no subject id)
- `mean_latent`: global target latent mean
- `oracle_encodec`: true target latent decoded by EnCodec (the quality ceiling)

The headline numbers are `pred_over_mean_cos_gain`, HuBERT/retrieval metrics, and
active-region synthesis metrics (`*_active_env_corr`, `*_voiced_rms_over_orig`), not
raw waveform Pearson.
Because the model is subject-agnostic, this gain is a clean measure of how much
the EEG actually contributes over a content-free baseline.
