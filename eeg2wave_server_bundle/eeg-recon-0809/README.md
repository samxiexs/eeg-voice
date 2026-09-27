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
| `scripts/run_mfcc_queue.sh` | the MFCC-80 route, one heavy process at a time |

## Environment

The venv sits on top of the conda environment `eegvoice` (Python 3.12; `requirements-preprocess.txt`):

```bash
bash app/run_aligned_local.sh setup      # creates .venv-aligned-local and pins transformers 4.57.6
```

All commands use `.venv-aligned-local/bin/python`; on Apple Silicon the scripts select MPS. On a 16 GB machine run one training process at a time (`run_mfcc_queue.sh` refuses to overlap).

## Data

- **DS004940** (OpenNeuro ds004940 v1.0.1, N400 sentence listening, N400Active, 17 participants with ≤15 % bad electrodes, 402 sentences, 6,641 trials): `SUBJECTS=all ./scripts/download_ds004940.sh`, `./scripts/verify_ds004940.sh`, `bash app/run_aligned_local.sh stimuli`. Split by sentence 320 / 41 / 41 with all participants in every split; the test partition is read only under a pre-registered protocol.
- **Broderick 2018** (Dryad "Natural Speech", CND, `data/broderick2018`): 19 participants × 20 runs × ~3 min. The CND release has no audio; the identical wavs are `data/ds004408/stimuli/audio<r>.wav` (the ds004408 EEG duplicates the CND and was removed). Runs 19-20 are validation (never seen by the envelope trunk), 3 participants are held out.
- **Marion 2021** (Zenodo, `data/marion2021`): 21 musicians × {listen, imagine} × 4 Bach chorales × 11 repetitions, 64-channel BioSemi, shipped preprocessed at 64 Hz with a metronome shared by all melodies.
- Kept but no longer used: `artifacts/karaone` (KaraOne continuous signal; imagined speech was at chance under block-wise CV) and `artifacts/music/musin_g` (MUSIN-G; envelope tracking r ≈ 0.012). Their preparation scripts were removed.

## Pipelines

### 1. Audio side (done once)

```bash
bash app/run_aligned_local.sh download | prepare | references | bootstrap-cache | hubert | hifigan | cache | audio
```

### 2. MFCC-80 route

```bash
bash scripts/run_mfcc_queue.sh a          # diffusion in standardised MFCC-80 on the existing v3 cross-fitted conditioning, vs the mel decoder
bash scripts/run_mfcc_queue.sh a2         # the same with the v-loss weighted by coefficient standard deviation
bash scripts/run_mfcc_queue.sh data       # Broderick shards + targets
bash scripts/run_mfcc_queue.sh encoder    # subject-free encoder: MFCC-80 loss + Broderick, selected on unseen DS004940 participants
bash scripts/run_mfcc_queue.sh folds      # the two --content-half encoders (final weights) for cross-fitting
bash scripts/run_mfcc_queue.sh diffusion  # MFCC-80 diffusion on the subject-free cross-fitted features; export + compare
bash scripts/run_mfcc_queue.sh marion     # listen -> imagine gate on the new and the old encoder's features
```

Every evaluation is paired against controls computed on the same trial with the same noise seed (zero EEG, duration-matched wrong-trial EEG, time-block shuffle; pooled vs pooled-wrong), next to the audio-only ceilings. Encoder selection uses the weighted MFCC-80 MAE among evaluations that pass the v3 gate and the MFCC controls. `python app/universal_train.py --evaluate-only A.pt B.pt --output DIR` scores two encoders on the same trials with crossed participant/sentence bootstrap CIs and paired differences.

### 3. Analyses

```bash
python app/marion_imagery_gate.py [--features encoder --encoder CKPT] [--skip-subject-free]
python app/feature_decodability.py [--partial-envelope]
```

## Results so far (validation)

**Diffusion decoder, mel vs MFCC-80 on identical conditioning** (240 trials, guidance 2; `outputs/generative_recovery/{crossfit,crossfit_mfcc}/export_validation/comparison.json`):

| | mel (global scale) | MFCC-80, standardised per coefficient |
|---|---:|---:|
| decoder ceiling (audio → teacher → diffusion): STOI / envelope r | 0.885 / 0.944 | 0.581 / 0.647 |
| real EEG: STOI / envelope r / MCD-80 | 0.339 / 0.240 / 78.9 dB | 0.294 / 0.203 / 81.2 dB |
| real minus wrong-trial EEG: STOI | +0.032 [+0.018, +0.046] | +0.025 [+0.010, +0.040] |
| real minus wrong-trial EEG: envelope r | +0.057 [+0.029, +0.085] | +0.047 [+0.021, +0.075] |
| real minus wrong-trial EEG: MCD-80 (dB, lower is better) | +1.04 [+0.46, +1.62] | +0.70 [+0.05, +1.38] |
| frame flux (real speech 0.222) | 0.215 | 0.351 |

Plain per-coefficient standardisation is worse: c0 (loudness) holds 96 % of the log-mel variance (silence padding) and carries weight 1/80 of the loss instead of ~77× the mean in the mel objective, so the samples' loudness jitters. The `a2` run weights coefficients by their standard deviation.

**Listen -> imagine (Marion 2021, linear, unseen melody, 4-way chance 25 %):** per participant, listening 93 %, imagery 33 % (p = .003), listen -> imagine 30 % (p = .03); subject-free, listening 84 % but listen -> imagine 27 % (n.s.). Imagery carries ~1/10 of the listening signal and needs same-person calibration.

**What EEG encodes (Broderick, linear, 19 participants):** log-mel and MFCC are both explained at ~2.2 %, ~92 % of it the loudness envelope (0.17-0.20 % beyond it); HuBERT L9 keeps 0.51 % beyond the envelope — align to self-supervised features, reconstruct through the mel/MFCC decoder.

## Tests

```bash
.venv-aligned-local/bin/python -m unittest discover -s tests -p "test_*.py"
```

`tests/test_analysis_traps.py` pins down evaluation mistakes made earlier (oracle-duration shortcut, same-sentence foils, mean vs median template, output-side pooling, gains against a model's own zero-EEG output).

## History

- 2026-09-27: MFCC-80 route added (`app/mfcc.py`, MFCC space in the diffusion decoder and the subject-free encoder, Broderick continuous speech, subject-free conditioner, cross-fitting halves, Marion and feature-decodability analyses). Removed as superseded: KaraOne code, the music preparation route (MUSIN-G / Di Liberto) and `run_universal.sh`, the v3 drivers (`run_aligned_recovery.sh`, `recovery_reports.py`, `linear_envelope_check.py`, augmentation queue), the content analyses (`content_analysis.py`), the 2026-09-19 technical report and its figure script, the old outputs and logs, the MUSIN-G raw data and the ds004408 EEG. Everything tracked can be restored with `git restore <path>`.
- Earlier: the DS006104 line, the joint MFCC-renderer pipeline, unit-CTC decoding and the v2 data route were removed; the v3 encoder (`outputs/aligned_recovery_v3/full_seed322_positional`) stays as the conditioning of the mel baseline.
