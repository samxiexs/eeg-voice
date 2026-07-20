# FEIS + KARA ONE + ds004306 preprocessing bundle

This bundle creates a unified imagined-speech EEG dataset without modifying
anything under `data/`.

The output is written to `eeg_output/` and contains one compressed subject
bundle per recording, a trial manifest, audio cache, subject-disjoint split
manifest, and quality-control reports.

## What is harmonised

- EEG channel order: `F3 FC5 AF3 F7 T7 P7 O1 O2 P8 T8 F8 AF4 FC6 F4`
- EEG shape: 14 channels x 1280 samples (5 seconds at 256 Hz)
- FEIS: existing `thinking` stage, normalized against its `resting` stage
- KARA ONE: existing `thinking` stage, normalized against its `clearing` stage
- ds004306: raw 1024-Hz EEGLAB data, temporary `.set/.fdt` staging, 50-Hz
  notch, 1--40-Hz bandpass, average reference, 256-Hz resampling, then
  `Imagination_*` event epoching
- Audio: lossless mono 16-kHz WAV cache; variable durations are retained and
  recorded in `audio_valid_samples`

The default uses ds004306 **auditory-cued imagination** only.  Its published
audio files are stored by category rather than unambiguously by trial.  The
manifest therefore marks them `weak_category_level`; do not evaluate direct
waveform reconstruction on ds004306 as though every trial had a unique,
confirmed waveform target.

## Run

First validate paths without writing output:

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle
/opt/anaconda3/bin/python scripts/preprocess_combined_eeg.py --dry-run
```

Then launch preprocessing with visible per-subject progress:

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle
bash run_preprocess.sh
```

The default excludes FEIS subject `05`, which the source manifest marks as
anomalous.  To include it, or to also preprocess ds004306 text/image prompts:

```bash
/opt/anaconda3/bin/python scripts/preprocess_combined_eeg.py \
  --data-root data \
  --output-root eeg_output \
  --ds-modalities auditory text image \
  --include-feis-subject-05
```

The run processes ds004306 one continuous recording at a time.  It normally
needs several hours, roughly 4--8 GB peak RAM, and a few GB for outputs.  If a
run is interrupted, simply run the same command again: already completed
subject files are reused and only missing recordings are processed.  Use
`--overwrite` only when deliberately regenerating every output NPZ/CSV.

## Training invariants

- Mask all samples at or after `eeg_valid_samples`.
- Split only on `subject_group_id`; never randomly split trials.
- The **only authoritative training split** is
  `app/configs/split/combined_0715_v1_split.yaml`.
- `eeg_output/manifests/subject_holdout_splits.json` is generated only for
  preprocessing QC (`purpose=preprocessing_qc_only`,
  `authoritative_for_training=false`) and must never select training,
  validation, or locked-test rows.
- Preserve `dataset`, `modality`, `audio_pairing`, and `pairing_confidence` as
  model covariates/weights.  In particular, ds004306 is weaker audio
  supervision than KARA ONE.

## Combined 0715 V1 training

The first combined training implementation is documented in
[`reports/combined_0715_v1_training_plan.md`](reports/combined_0715_v1_training_plan.md).
It adapts the latest KaraOne 0715 codec-token method to 14-channel imagined EEG,
uses a locked cross-dataset subject split, and treats FEIS/ds004306 pairing as
weaker than KaraOne trial-level pairing.

Before training for the first time, rebuild the KaraOne normalized output with
valid clearing lengths:

```bash
bash run_preprocess.sh --datasets karaone --overwrite
```

After preprocessing exists, use the following order.  `--verify-only` scans
the existing manifest/NPZ files, validates the locked YAML split and writes a
real pass/fail QC record bound to the exact split, manifest, EEG NPZ and QC
hashes; it does not rebuild EEG data.  Any later change to those inputs makes
that verification stale and training stops until `--verify-only` is rerun.
The signal probe reads only train and validation EEG and no longer requires an
audio cache.

```bash
bash run_preprocess.sh --verify-only
bash app/run_combined_0715_v1.sh probe
bash app/run_combined_0715_v1.sh cache --rebuild
bash app/run_combined_0715_v1.sh audit-audio
bash app/run_combined_0715_v1.sh train-audio
bash app/run_combined_0715_v1.sh train-eeg
bash app/run_combined_0715_v1.sh validate
```

The cache command now writes `combined-0715-cache-v2`, including source audio
paths, valid sample counts and exact EnCodec scale metadata.  A legacy cache
must be rebuilt with `--rebuild`.  Checkpoints use the v2 checkpoint/lineage
contract and bind the config, locked split, unified manifest, all referenced
preprocessed EEG payloads, QC table and cache.  Legacy smoke checkpoints are
strictly rejected and must be retrained.

`audit-audio` performs a real EnCodec decode on a deterministic stratified
validation sample (at most four unique audio keys per dataset and label).  It
does not sample locked-test waveforms.  Each dataset passes only when median
waveform correlation is at least `0.65`, median SI-SDR is at least `0 dB`, and
median RMS-normalized log-spectrogram MAE is at most `12 dB`, in addition to
all cache structure checks passing.

The locked test phase requires explicit `--allow-final-test`, a passed
validation gate, the exact validation-report SHA, and exact checkpoint/lineage
hashes.  A metadata-only preauthorization happens before the manifest, cache,
or test EEG is opened; full current-lineage validation then runs again.
`--allow-failed-gate` is forbidden for test.  The default configuration uses the first 768 EEG
samples (3 seconds) to match KaraOne 0715; the original 1280-sample output
remains available for later ablations.

## Synthesis controls

The synthesis script exports one reference directory plus six generated
controls under `<output>/<dataset>/<split>/`:

```text
reference/
codec_oracle/
eeg_conditioned/
label_only/
zero_eeg/
shuffled_eeg/
dataset_only/
synthesis_manifest.json
```

`label_only` uses the same-trial EEG label-head probabilities, not the true
label. `shuffled_eeg` is a deterministic, same-dataset/same-label derangement
without self-loops. `dataset_only` uses the empirical training-label prior.
Validation export requires a passed round-trip audit unless
`--allow-failed-gate` is supplied, in which case the manifest is explicitly
marked exploratory.  Locked-test export cannot bypass either gate.

Example validation export:

```bash
/opt/anaconda3/bin/python app/scripts/synthesize_combined_0715.py \
  --cache artifacts/combined_0715_v1/cache/combined_0715_encodec_codes.npz \
  --audio-checkpoint artifacts/combined_0715_v1/audio/checkpoints/best.pt \
  --eeg-checkpoint artifacts/combined_0715_v1/eeg/checkpoints/best.pt \
  --dataset karaone --split validation \
  --output artifacts/combined_0715_v1/samples
```

ds004306 audio remains category-level candidate supervision; every synthesis
manifest therefore records `ds004306_trial_level_claim_allowed=false`.
FEIS permits only canonical-audio/coarse-code claims.  Only KaraOne can support
a trial-level acoustic claim, and only after its validation controls pass.

## Remaining formal-training blockers

The QC, cache-v2, round-trip, synthesis-control, multi-positive and lineage
work does **not** by itself make the current prototype ready for a formal long
run.  Before interpreting 60/80-epoch results, separately resolve and verify:

- dataset-slice masking/renormalisation for label distillation;
- enforcement of the ds004306 audio-audit boundary in all training losses;
- validation-based EEG checkpoint selection and an automatic validation gate;
- valid-audio-length handling inside the audio condition encoder.

Until those items are closed, runs are exploratory and must not be described
as confirmed reconstruction of hallucinated or imagined speech.
