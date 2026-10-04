# eeg-recon-0809

Experimental code for reconstructing speech from scalp EEG. The goal is imagined speech; perceived-speech EEG is the pre-training stage, and a listen/imagine music set measures how much of it transfers to imagery.

- **DS004940** (perceived sentences, 128-channel BioSemi): encoder training and the main validation/test protocol.
- **Broderick 2018** (continuous audiobook, same cap): additional perceived speech without a sentence-onset clock.
- **Marion 2021** (musicians listening to and imagining the same Bach melodies): the listen -> imagine transfer measurement.

```
EEG (128 × 1178, 256 Hz) ─► subject-free encoder ─► compact conditioning ─► conditional diffusion ─► MFCC-80 (80 × 251) ─► inverse DCT ─► log-mel ─► HiFi-GAN ─► 4 s waveform
                              ▲ CLIP to HuBERT L9, weighted MFCC-80 loss, envelope/onset head
audio ─► HuBERT layer 9 (fine-tuned) ─► AcousticDecoder ─► mel   (teacher target / decoder ceiling)
```

MFCC-80 is the orthonormal DCT of the native SpeechT5 log-mel (`app/mfcc.py`). With all 80 coefficients it is exactly invertible, so the audio ceiling is the log-mel one (oracle STOI 0.976, PESQ 3.80); truncating to 13 coefficients drops the pitch harmonics (STOI 0.715, PESQ 1.40), so nothing truncates. What changes against log-mel is only how a loss weights the coefficients.

## Layout

```
app/                      training, evaluation and analysis (table below)
app/src/eeg2speech/       shared modules: aligned model, dataset, losses, SpeechT5 front end
scripts/                  data download/preparation, target caching, run_mfcc_queue.sh
configs/                  data and experiment configs (aligned_* inherit training_data_v4 → v3 → v2)
tests/                    unit and regression tests
data/                     raw datasets (not in git)
artifacts/                preprocessed shards, target caches, splits (not in git)
outputs/                  checkpoints, evaluations, exported audio (not in git)
models/                   HuBERT / HiFi-GAN / SpeechT5 base weights (not in git)
logs/                     run logs (not in git)
```

| File | Purpose |
|---|---|
| `app/aligned_speech.py`, `app/aligned_recovery*.py`, `app/src/eeg2speech/*` | DS004940 dataset, the v3 encoder and its evaluation; hashed into every checkpoint, so left untouched |
| `app/universal_model.py`, `app/universal_train.py` | subject-free encoder (no participant index): MFCC-80 loss, continuous-speech domain, `--content-half` cross-fitting, `--evaluate-only` |
| `app/generative_recovery.py` | diffusion decoder: `cache`, `crossfit` (v3), `train` (`--space mfcc|mel`, `--coefficient-weights none|low|std`, `--conditioner v3|universal`), `export`, `compare`, `figures` |
| `app/mfcc.py` | MFCC-80 transform, per-coefficient scaler, MCD |
| `app/music.py` | continuous-window loader (`MusicWindows`, used for Broderick) and music utilities |
| `app/envelope_decoder.py`, `app/broderick_pretrain.py` | envelope decoder (v3 conditioning) and the Broderick envelope trunk used to initialise encoders |
| `app/audio_comparison.py` | STOI / PESQ / MCD / envelope metrics, 2AFC |
| `app/marion_imagery_gate.py` | listen -> imagine gate on Marion 2021 (raw EEG, or `--features encoder` for a trained encoder) |
| `app/feature_decodability.py` | which speech representation EEG can encode (Broderick; `--partial-envelope`) |
| `app/adapt_local_audio.py`, `app/run_aligned_local.sh` | audio side: HuBERT / HiFi-GAN fine-tuning, AcousticDecoder |
| `scripts/prepare_broderick_windows.py`, `scripts/cache_broderick_targets.py` | Broderick shards and HuBERT / mel / acoustic targets |
| `scripts/prepare_training_data.py`, `scripts/convert_marion.py` | DS004940 harmonisation (audit / splits / build), Marion release to HDF5 |
| `app/marion_transfer.py` | decoders trained on Marion listening, tested on imagery (scratch vs speech-pretrained encoders, linear reference) |
| `scripts/run_mfcc_queue.sh` | the subject-free generation route, one heavy process at a time |

## Environment

The venv sits on top of the conda environment `eegvoice` (Python 3.12; `requirements-preprocess.txt`):

```bash
bash app/run_aligned_local.sh setup      # creates .venv-aligned-local and pins transformers 4.57.6
```

All commands use `.venv-aligned-local/bin/python`; on Apple Silicon the scripts select MPS. On a 16 GB machine run one training process at a time (`run_mfcc_queue.sh` refuses to overlap).

## Data

Only preprocessed EEG is kept on disk (2026-09-27); the raw releases were deleted after every recording was harmonised and checked. Stimulus audio and small metadata stay in `data/` (about 0.9 GB), because targets are recomputed from audio whenever a teacher changes.

| Dataset | Preprocessed EEG | What it holds | Raw release (re-download to rebuild) |
|---|---|---|---|
| DS004940, N400Active, 17 participants | `artifacts/training_data/aligned_v1` (3.2 GB) | the training / validation / test data of every model; manifest hash-locked into the checkpoints | OpenNeuro ds004940 v1.0.1, `scripts/download_ds004940.sh` |
| DS004940, N400Passive × 22 + the 5 Active participants above 15 % bad channels | `artifacts/training_data/ds004940_complete` (5.5 GB) | same harmonisation (`configs/training_data_ds004940_complete.yaml`, bad-channel limit 1.0, masks stored per trial) | same |
| Broderick 2018, 19 × 20 runs | `artifacts/speech_continuous/broderick2018` (2.4 GB shards + 0.6 GB targets) | 128 Hz float16 µV (upsampled to 256 Hz per window by `MusicWindows`); all runs stored, 12 runs of Subject9 flagged in the manifest | Dryad doi:10.5061/dryad.070jc (CND); audio in `data/ds004408/stimuli` |
| Marion 2021 | `artifacts/marion2021/imagery.h5` (0.8 GB) | the authors' preprocessed release (64 Hz), float32, lossless (`scripts/convert_marion.py`) | Zenodo (Marion et al. 2021) |
| KaraOne, MUSIN-G | `artifacts/karaone`, `artifacts/music/musin_g` | kept, no longer used | — |

- DS004940: split by sentence 320 / 41 / 41 with all participants in every split; the test partition is read only under a pre-registered protocol. Rebuilding shards (`prepare_training_data.py audit|build`) needs the raw release again.
- Broderick: runs 19-20 are validation (never seen by the envelope trunk), 3 participants are held out. `scripts/prepare_broderick_windows.py` and `app/broderick_pretrain.py` need the raw CND; `app/feature_decodability.py` reads the shards.
- Marion: 21 musicians × {listen, imagine} × 4 Bach chorales × 11 repetitions, 64-channel BioSemi, with a metronome shared by all melodies.

## Pipelines

### 1. Audio side (done once)

```bash
bash app/run_aligned_local.sh download | prepare | references | bootstrap-cache | hubert | hifigan | cache | audio
```

### 2. Subject-free generation (MFCC-80)

```bash
bash scripts/run_mfcc_queue.sh tonight     # cache -> two cross-fitting encoders -> MFCC-80 diffusion -> export + compare (~8 h, throttled)
```

No participant index anywhere: not a model input, and no participant-keyed augmentation; participant ids only split training people from the 4 held-out people. The encoders (CLIP to HuBERT L9 + std-weighted MFCC-80 + envelope/onset head, DS004940 + Broderick) are trained on one half of the training sentences each, so every diffusion training trial is conditioned by an encoder that never saw its sentence. Evaluation is paired against controls on the same trial and noise seed (zero, duration-matched wrong-trial, time-block shuffle, pooled vs pooled-wrong). `python app/universal_train.py --evaluate-only A.pt B.pt --output DIR` compares two encoders on the same trials.

### 3. Analyses

```bash
python app/marion_imagery_gate.py [--features encoder --encoder CKPT] [--skip-subject-free]
python app/feature_decodability.py [--partial-envelope]
```

## Results so far (validation)

- **MFCC-80 vs log-mel diffusion** (same conditioning): plain per-coefficient standardisation loses (decoder ceiling STOI 0.58 vs 0.89) because c0, 96 % of the variance, gets 1/80 of the loss; weighting coefficients by their standard deviation (`--coefficient-weights std`, now the default) matches log-mel (ceiling 0.89, EEG-vs-wrong-trial envelope gain +0.062 vs +0.057). These runs used the participant-layer v3 conditioning and were deleted; the subject-free run is `scripts/run_mfcc_queue.sh tonight`.
- **Subject-free encoder** (`outputs/universal/mfcc_broderick_seed322`): Broderick held-out runs and people, 3 s retrieval MRR 0.170 vs chance 0.058; on DS004940 no better than the mel-loss baseline in EEG-specific gains (`outputs/universal/compare_mfcc_broderick_vs_speech`).
- **Listen -> imagine** (Marion, 4-way, chance 25 %): linear raw EEG 30-32 %; encoder features 28-29 %; decoders trained on listening and initialised from either speech encoder 27 % vs 27.5 % from scratch (`outputs/marion2021/transfer.json`) - perceived-speech pre-training does not help the transfer.
- **What EEG encodes** (Broderick, linear): mel / MFCC ~ the loudness envelope only; HuBERT L9 keeps 0.5 % of variance beyond it.

## Tests

```bash
.venv-aligned-local/bin/python -m unittest discover -s tests -p "test_*.py"
```

`tests/test_analysis_traps.py` pins down evaluation mistakes made earlier (oracle-duration shortcut, same-sentence foils, mean vs median template, output-side pooling, gains against a model's own zero-EEG output).

## History

- 2026-09-27: raw EEG replaced by preprocessed shards (DS004940 Active + Passive for all 22 participants, Broderick, Marion); 17,489 DS004940 trials, 380 Broderick runs and the Marion release checked before deletion (the Broderick decodability and the Marion gate reproduce).
- 2026-09-27: MFCC-80 route added (`app/mfcc.py`, MFCC space in the diffusion decoder and the subject-free encoder, Broderick continuous speech, subject-free conditioner, cross-fitting halves, Marion and feature-decodability analyses). Removed as superseded: KaraOne code, the music preparation route (MUSIN-G / Di Liberto) and `run_universal.sh`, the v3 drivers (`run_aligned_recovery.sh`, `recovery_reports.py`, `linear_envelope_check.py`, augmentation queue), the content analyses (`content_analysis.py`), the 2026-09-19 technical report and its figure script, the old outputs and logs, the MUSIN-G raw data and the ds004408 EEG. Everything tracked can be restored with `git restore <path>`.
- Earlier: the DS006104 line, the joint MFCC-renderer pipeline, unit-CTC decoding and the v2 data route were removed; the v3 encoder (`outputs/aligned_recovery_v3/full_seed322_positional`) stays as the conditioning of the mel baseline.
