# KaraOne 0711v1

`0711v1` is an independent, strict subject-holdout KaraOne EEG-to-speech
pipeline. It does not import v10, v11, or v12.

The first supported experiment is:

```text
overt_like EEG -> adapted-HuBERT semantic representation -> EnCodec latent flow -> wav
```

`thinking` starts only after the overt-like validation gates have passed.

## Fixed split and leakage policy

- Train: `MM05, MM08, MM09, MM10, MM11, MM12, MM14, MM15, MM16, MM18, MM19, MM20`
- Validation/model selection: `P02`
- Final test: `MM21`

HuBERT domain adaptation, feature normalisation, semantic k-means codebook, and
all flow fitting use only train subjects. P02 is used solely for model selection;
MM21 is unread during training and can be evaluated only through an explicit,
one-time final-test command.

## Pipeline

1. `audio_ssl` fine-tunes only HuBERT's top two transformer blocks on real
   KaraOne overt wavs, then rebuilds adapted-HuBERT semantic targets and
   train-only semantic units.
2. `eeg_ssl` aligns raw `62×1280` EEG with a non-rendered time-frequency scalp
   tensor (`5 bands × 32 bins × 9×9 grid`).
3. `align_global` uses multi-positive CLIP: the matched trial is the main
   positive and same-label, different-subject trials are low-weight positives.
4. `align_token` is blocked until the global validation gate passes.
5. `flow` is blocked until the token validation gate passes, then learns a
   conditional probability-flow velocity field in continuous EnCodec latent
   space. EnCodec remains frozen.

The validation gate requires positive bootstrap lower bounds for cross-subject
semantic/token retrieval, positive EEG-over-zero gain, and anti-collapse
thresholds. A failed gate is a failed representation result, not a speech
reconstruction claim.

## Commands

All artefacts use `karaone_0711v1_<stage>_<phase>_s<seed>`; timestamps are
stored only in `run_manifest.json`.

```bash
cd app

bash run_karaone_0711v1.sh audit
bash run_karaone_0711v1.sh audio_ssl
bash run_karaone_0711v1.sh eeg_ssl

RESUME=../artifacts/outputs_karaone_0711v1/karaone_0711v1_overt_like_eeg_ssl_s11/checkpoints/best.pt \
  bash run_karaone_0711v1.sh align_global

RESUME=../artifacts/outputs_karaone_0711v1/karaone_0711v1_overt_like_align_global_s11/checkpoints/best.pt \
GATE=../artifacts/outputs_karaone_0711v1/karaone_0711v1_overt_like_align_global_s11/metrics/validation_gate.json \
  bash run_karaone_0711v1.sh align_token
```

After a passed token gate, use `flow` with the locked token-alignment checkpoint
and then `scripts/synthesize_karaone_0711v1.py` to decode EEG-only EnCodec
latents. The final MM21 evaluation/synthesis requires an explicit
`--allow-final-test`; it is intentionally absent from the `all` runner.

## Verification

```bash
cd app
python tests/test_karaone_0711v1_smoke.py
python scripts/train_karaone_0711v1.py --phase audit --smoke
```

The local environment must provide the packages in `app/requirements.txt`, the
local HuBERT checkpoint, and the local EnCodec checkpoint before running a full
audio/flow phase.
