# eeg-recon-0809

Experimental code for reconstructing speech from scalp EEG. Two datasets:

- **DS004940** (perceived speech): 128-channel EEG recorded while participants listen to English sentences. Used to train the EEG encoder and the generative decoder; this is the pre-training stage of the project.
- **KaraOne** (imagined speech): EEG of imagined and spoken prompts, 11 classes. This is the research target. The current baselines sit at chance under block-wise cross-validation, and the DS004940 encoder does not transfer.

Current pipeline on DS004940:

```
EEG (128 × 1178, 256 Hz) ─► recovery-v3 encoder (frozen) ─► 18-channel conditioning ─► conditional diffusion decoder ─► mel (80 × 251) ─► HiFi-GAN ─► 4 s waveform
                                                                                              ▲
audio ─► HuBERT layer 9 (fine-tuned) ─► AcousticDecoder ─► mel ──── teacher target / decoder ceiling ──┘
```

The encoder is trained with a time-resolved in-batch CLIP loss that pulls EEG tokens into the HuBERT feature space, plus a mel loss through the frozen audio decoder. The diffusion decoder then samples from p(mel | EEG features); classifier-free guidance sets how strongly the EEG shapes the sample. Every evaluation is a paired comparison against three controls computed on the same trial with the same noise seed (zero EEG, wrong-trial EEG, time-shuffled EEG), alongside the audio-only ceilings. The full description of the data, splits, models and results is in `reports/updated-results_2026-09-19.md`.

## Layout

```
app/                       training, evaluation and analysis scripts (table below)
app/src/eeg2speech/        shared modules: aligned model, dataset, losses, SpeechT5 front end
scripts/                   data download, preprocessing, target caching, report figures
configs/                   data and experiment configs (aligned_* inherit training_data_v4 → v3 → v2)
tests/                     unit and regression tests
reports/                   technical report (updated-results_2026-09-19.md) and its figures
data/                      raw datasets (not in git)
artifacts/                 preprocessed shards, target caches, splits (not in git)
outputs/                   checkpoints, evaluations, exported audio (not in git)
models/                    HuBERT / HiFi-GAN / SpeechT5 base weights (not in git)
```

## Environment

The project venv is created on top of the conda environment `eegvoice` (Python 3.12; dependencies in `requirements-preprocess.txt`):

```bash
bash app/run_aligned_local.sh setup      # creates .venv-aligned-local and pins transformers 4.57.6
```

All commands below use `.venv-aligned-local/bin/python`. On Apple Silicon the scripts select MPS and set `PYTORCH_ENABLE_MPS_FALLBACK=1`. On a 16 GB machine run one training process at a time.

## Data

### DS004940

OpenNeuro ds004940 v1.0.1, an N400 sentence-listening paradigm recorded with a 128-channel BioSemi system. By default only the N400Active task is used: 17 participants with at most 15 % bad electrodes, 402 sentences, 6,641 trials.

```bash
SUBJECTS="001 002 003 004" ./scripts/download_ds004940.sh     # four participants to check the pipeline
SUBJECTS=all ./scripts/download_ds004940.sh                    # everyone
SUBJECTS=all ./scripts/verify_ds004940.sh
SUBJECTS=all MODE=active_passive ./scripts/download_ds004940.sh   # optional: Passive task as well (v2 data route)
bash app/run_aligned_local.sh stimuli                          # sentence reference transcripts
```

### KaraOne

Zhao & Rudzicz 2015, 14 participants, about 24.8 GB. The script resumes interrupted downloads, verifies the archives and extracts to `data/karaone/<participant>/`.

```bash
bash scripts/download_karaone.sh                     # all participants
SUBJECTS="MM05 P02" bash scripts/download_karaone.sh  # a subset
bash scripts/download_karaone.sh verify
```

### Broderick 2018 (optional)

Audiobook-listening EEG from 19 participants, used only to pre-train the envelope-tracking trunk (`scripts/prepare_broderick.py`, `app/broderick_pretrain.py`). Pre-training shortens convergence but does not improve the final metrics, so the main pipeline does not depend on it.

## Pipelines

### 1. Data preparation and audio-side models

`app/run_aligned_local.sh` drives `app/aligned_speech.py` and `app/adapt_local_audio.py` stage by stage:

```bash
bash app/run_aligned_local.sh download          # HuBERT-base and SpeechT5 HiFi-GAN base weights -> models/
bash app/run_aligned_local.sh prepare           # audit, EEG shards, manifest, sentence-level split (artifacts/aligned_speech_local_v1)
bash app/run_aligned_local.sh references        # verify reference transcripts
bash app/run_aligned_local.sh bootstrap-cache   # teacher features from the base HuBERT
bash app/run_aligned_local.sh hubert            # fine-tune HuBERT layers 0-8 on train-fold audio
bash app/run_aligned_local.sh hifigan           # fine-tune HiFi-GAN on train-fold audio
bash app/run_aligned_local.sh cache             # rebuild the target cache (targets_adapted.h5) with the fine-tuned HuBERT
bash app/run_aligned_local.sh audio             # train the AcousticDecoder (HuBERT -> mel) and export the audio-side ceilings
```

The split is by sentence: 320 train / 41 validation / 41 test sentences, with all 17 participants in every split. The test partition is read only under a pre-registered protocol; day-to-day work uses validation.

### 2. EEG encoder (recovery v3)

```bash
bash app/run_aligned_recovery.sh linear     # linear envelope gate: ridge regression against prior / wrong-trial / circular-shift nulls
bash app/run_aligned_recovery.sh m0         # closed-loop fit on 50 training trials, used only as a gate
bash app/run_aligned_recovery.sh full       # 4000-update training; best_passed.pt is written only when every validation control passes
bash app/run_aligned_recovery.sh evaluate   # validation report with bootstrap CIs and waveform export
bash app/run_aligned_recovery.sh sweep      # three seeds and a seed-level summary
```

Current checkpoint: `outputs/aligned_recovery_v3/full_seed322_positional/best_passed.pt` (update 1800). All metrics are computed on presented-speech frames only; the `full_*` keys cover the whole 4 s window for comparison with older runs.

### 3. Generative decoder (diffusion)

```bash
PY=.venv-aligned-local/bin/python
$PY app/generative_recovery.py cache        # EEG / mel / teacher features as npy (about 4 min)
$PY app/generative_recovery.py crossfit     # one encoder and one envelope decoder per half of the training sentences (about 25 min)
$PY app/generative_recovery.py train --run crossfit --crossfit --updates 8000 --eval-every 500 \
    --condition-noise 0.1 --condition-dropout 0.05                                   # about 45 min, resumable
$PY app/generative_recovery.py export --crossfit --checkpoint outputs/generative_recovery/crossfit/best.pt \
    --export-output outputs/generative_recovery/crossfit/export_validation --limit 240 --guidance 2 --steps 50
$PY app/generative_recovery.py compare --export-output outputs/generative_recovery/crossfit/export_validation
$PY app/generative_recovery.py figures --export-output outputs/generative_recovery/crossfit/export_validation
```

- `--conditioning compact` (default): the encoder head output projected on the top 16 principal components of the teacher space, the predicted duration and the envelope-decoder output, 18 channels in total. Conditioning on the raw 128-d trunk lets the decoder memorise trial fingerprints and is no longer used.
- `--crossfit`: every training trial is conditioned by fold models that never saw its sentence, matching the validation situation. Without it the features come from the seed-322 encoder (`--run compact`).
- `export` writes, per validation trial, `original`, `native_mel_oracle` (vocoder ceiling), `teacher_oracle` (audio → teacher → diffusion, the decoder ceiling), `regression` (the deterministic v3 route) and the diffusion samples `correct` / `zero` / `wrong_trial` / `time_block_shuffle` / `pooled` / `pooled_wrong`. All samples of a trial share one noise seed; nothing uses the true duration.
- `compare` computes STOI, PESQ, MCD, envelope and modulation-spectrum correlation, a 2AFC against a duration-matched foil sentence, speech-likeness measures of the mel, and paired differences with bootstrap CIs, written to `comparison.json` and `comparison.png`.
- `figures` writes `features_stacked.png` (per condition: mel with pitch contour, MFCC, energy) and `features_overlay.png` (all conditions on shared axes) into every trial folder.
- `outputs/generative_recovery/listening/` is a copy of a few trials with sentence names and a `README.txt`.

### 4. Analysis scripts

| Script | Purpose |
|---|---|
| `app/linear_envelope_check.py` | Linear envelope gate (feasibility check before encoder training) |
| `app/envelope_decoder.py` | Network that regresses the speech envelope directly; band-split, per-subject and Broderick-initialised variants |
| `app/group_decoding.py` | Reconstruction and retrieval with repeated presentations averaged at the encoder input |
| `app/group_content.py` | Pooled sentence identification fusing the embedding and envelope evidence channels |
| `app/content_pipeline.py` | Closed-set content decisions with a confidence gate and fixed-voice synthesis |
| `app/congruency_probe.py` | N400 semantic-congruency probe |
| `app/unit_ctc.py` | CTC decoding of discrete speech units |
| `app/broderick_pretrain.py` | Envelope-tracking pre-training on Broderick 2018 |
| `app/audio_comparison.py`, `app/plot_audio_comparison.py` | Waveform-level metrics, 2AFC, listening-test bundles and spectrogram figures |
| `app/synthesize_text.py` | SpeechT5 TTS synthesis (output end of the content-first route) |
| `app/aggregate_recovery_runs.py` | Multi-seed summary |
| `scripts/technical_report_figures.py` | Figures for the technical report |

### 5. v2 data route

`app/run_aligned_v2_data.sh` materialises both tasks (Active + Passive) for 22 participants with the sentence split pinned to v1 (`configs/aligned_speech_local_v2.yaml`), reusing the v1 fine-tuned HuBERT and HiFi-GAN and rebuilding only the target cache and the AcousticDecoder. Train with `ALIGNED_CONFIG=configs/aligned_speech_local_v2.yaml ALIGNED_RUN_ROOT=aligned_recovery_v3_data_v2 bash app/run_aligned_recovery.sh ...`. 2.6 times the data lowers the mel error slightly and leaves the EEG-specific gains unchanged.

### 6. KaraOne

```bash
bash app/run_karaone.sh prepare     # .cnt -> artifacts/karaone/shards
bash app/run_karaone.sh baselines   # 11-way prompt decoding: block-wise CV, permutation tests, artifact controls
bash app/run_karaone.sh transfer    # DS004940 encoder on KaraOne and heard -> imagined generalisation
```

## Results at a glance

Validation, 240 trials, guidance 2 (details and confidence intervals in the report, section 3):

| Condition | STOI | Envelope r | 2AFC (envelope) |
|---|---:|---:|---:|
| audio → teacher → diffusion (decoder ceiling) | 0.885 | 0.944 | 1.00 |
| v3 deterministic regression | 0.476 | 0.384 | 0.59 |
| diffusion, real EEG | 0.339 | 0.240 | 0.59 |
| diffusion, zero EEG | 0.312 | 0.179 | 0.50 |
| diffusion, wrong-trial EEG | 0.306 | 0.183 | 0.54 |
| diffusion, EEG pooled over 17 presentations | 0.385 | 0.341 | 0.66 |

The diffusion samples have the spectral texture of real speech, and the real-EEG sample is closer to its own sentence than every control in paired comparisons (against wrong-trial EEG: STOI +0.032 [+0.018, +0.047], envelope r +0.057 [+0.029, +0.083]). The pooled gain is sentence-specific: pooling the presentations of a different sentence scores at chance. What the EEG determines is rhythm, duration and envelope; the words are not recovered. The deterministic route scores higher on STOI because a blurred conditional mean wins on distortion metrics. All numbers are validation results.

## Tests

```bash
.venv-aligned-local/bin/python -m unittest discover -s tests -p "test_*.py"
```

`tests/test_analysis_traps.py` pins down mistakes made earlier in the analyses (oracle-duration shortcut, foils that must be a different sentence, the level at which permutation tests shuffle, the median-template baseline); run it after changing any evaluation code. `tests/test_generative_recovery.py` checks the diffusion sampler identities and initialisation.

## Report

`reports/updated-results_2026-09-19.md` covers the data and splits, model architecture and tensor shapes, fine-tuning and alignment, training curves, result tables and their interpretation. Figures are in `reports/figures/technical_report/` and are produced by `scripts/technical_report_figures.py`. KaraOne results are written by `app/run_karaone.sh` to `outputs/karaone/`.

## History

- The DS006104 (TMS phoneme/word perception) data line was removed.
- The earlier joint / `train_joint` pipeline (MFCC renderer + Griffin-Lim) was removed from this directory; its code snapshot and checkpoint contracts are kept in `../eeg-recon-0809_explore_8h_v1_backup/`.
- The deterministic route that preceded the diffusion decoder (recovery v3) is kept in full as the encoder and as a baseline.
