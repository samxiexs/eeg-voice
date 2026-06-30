# KaraOne v9 Neural Semantic Transport Implementation

> Date: 2026-06-30  
> Scope: canonical v9 skeleton and smoke/audit implementation under `karaone_overt_recon_bundle`.

## Position

v9 is not a patch on v8.  The implemented code creates a separate package,
`app/src/karaone_v9`, whose canonical path is:

```text
raw 62ch EEG sequence
  -> subject-robust channel-time EEG token sequence
  -> speech semantic/prosody latent sequence
  -> conditional flow matching in codec latent space
  -> frozen or external neural codec decoder
```

Retrieval, prompt labels, and old active-core priors are diagnostics or weak
auxiliary supervision only.  They are not generation inputs for unknown EEG.

## Implemented Components

### 1. Canonical Data/Cache Interface

- `KaraOneV9TargetBank` wraps speech semantic, semantic-token, prosody, and codec caches:
  - HuBERT/wav2vec-style sequence cache as semantic target.
  - train-only k-means semantic token cache when available.
  - temporal-elastic active-core cache as prosody/event supervision.
  - EnCodec latent cache as conditional transport target.
- `KaraOneV9Dataset` defines explicit split names:
  - `subject_train`: all non-heldout subjects.
  - `subject_val`: default `P02`.
  - `subject_test`: default `MM21`.
  - `train/val/test`: within-train-subject trial split for diagnostics.
- `build_karaone_v9_canonical_cache.py` writes a JSON manifest with target shapes,
  coverage, split counts, valid EEG length stats, and split-overlap checks.
- `audit_karaone_v9_protocol.py` turns the manifest into pass/warn/fail checks.

### 2. EEG Representation Space

`KaraOneV9NeuralSemanticTransport` uses:

- Per-trial EEG normalization using valid EEG length.
- Channel reliability gates from channel log-variance.
- Channel-time convolutional patch tokenization.
- Stage embedding and learned positional embedding.
- Transformer token encoder.
- Masked token reconstruction path for Stage 1 EEG self-supervised pretraining.
- Separate content, prosody, uncertainty, and transport-condition streams.

The model forward path does not accept `subject_idx` or speaker identity input.
Subject labels are only used outside the model for adversarial leakage removal
and audit metrics.

### 3. Speech Semantic/Prosody Alignment

Implemented losses:

- Monotonic soft-OT alignment between EEG semantic sequence and speech SSL sequence.
- Framewise semantic cosine loss after sequence resizing.
- Symmetric EEG/speech InfoNCE.
- Speech-SSL soft-positive InfoNCE.
- Semantic-token CE.
- Prompt CTC and weak prompt CE.
- Prosody/event losses for active mask, energy, duration, and onset.
- Subject adversarial loss, CORAL subject alignment, group-DRO, and VICReg variance.

### 4. Conditional Transport Decoder

`ConditionalTransportDecoder` implements conditional flow matching in codec latent
space.  It learns a velocity field from Gaussian noise to codec latents,
conditioned on EEG-derived semantic/prosody tokens.  The training script supports
freezing the encoder during transport training.

### 5. Evaluation Protocol

`compute_v9_metrics` reports:

- semantic cosine vs zero-EEG and mean-query baselines;
- semantic label top-1/top-3/MRR;
- top-3 gain over mean baseline;
- same-label cross-subject gain;
- subject leakage classifier accuracy;
- prompt accuracy;
- collapse diagnostics: std ratio and pairwise correlation.

The v9 gate is explicit:

```text
subject_val:
  semantic_over_mean_gain > 0
  semantic_top3_gain_over_mean > 0
  same_label_cross_subject_gain >= 0
```

Waveform generation should be presented as diagnostic unless this gate passes on
subject validation and remains positive on subject test.

## Entrypoints

```bash
cd karaone_overt_recon_bundle

# Build canonical manifest and protocol audit.
bash run_karaone_v9_rebuild.sh audit

# Run synthetic module smoke plus a 1-epoch, 2-step real-data align smoke.
bash run_karaone_v9_rebuild.sh smoke

# Stage 1 EEG masked-token pretraining.
DEVICE=mps bash run_karaone_v9_rebuild.sh pretrain 20

# Stage 3 EEG-to-semantic/prosody alignment.
DEVICE=mps bash run_karaone_v9_rebuild.sh align 50

# Stage 4 codec-space conditional transport.
CKPT=artifacts/outputs_karaone/<v9_align_run>/checkpoints/best.pt \
DEVICE=mps bash run_karaone_v9_rebuild.sh transport 20
```

## Deliberate Non-Goals in This Pass

- No new external SSL/codec model downloads.  v9 uses local caches already
  produced by prior runs.
- No claim of intelligible EEG-to-speech generation.  The transport decoder is
  implemented, but waveform synthesis remains gated by semantic/prosody
  subject-holdout results.
- No reuse of v7/v8 model class as the v9 main path.
- No subject-specific generator or subject-id conditioning.

## Remaining Work

1. Run full Stage 1 pretraining and Stage 3 alignment on GPU/MPS long enough for
   meaningful subject_val selection.
2. Add teacher-forced `C/P -> codec` transport where the condition can come from
   speech teacher latents before scheduled sampling from EEG-predicted latents.
3. Wire a frozen codec decoder for waveform rendering and oracle-codec ceiling.
4. Add Whisper CER/WER and SSL perceptual metrics for rendered wavs.
5. Extend leave-one-subject cross-validation beyond the fixed `P02/MM21` split.
6. Add channel topology coordinates if reliable KaraOne electrode positions are
   recovered from source metadata.
