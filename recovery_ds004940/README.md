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

The raw `.cnt` recordings were deleted on 2026-09-23; `artifacts/karaone/` (2.0 GB) keeps everything the analyses need — the **continuous** 256 Hz signal of all 14 participants (average reference, 0.5-45 Hz, 60 Hz notch), electrode positions, the auxiliary M1/M2/VEO/HEO/EKG/EMG channels used as artifact controls, all four stage boundaries of the 1,913 trials, the BioSemi-128 interpolation map and the normalizer statistics. Because the signal is stored continuously rather than as cut epochs, windows and stages can be re-chosen freely. Only a change to the preprocessing itself (a wider band than 45 Hz, or a rate above 256 Hz) needs the 27 GB re-downloaded with the script above.

### Broderick 2018 (optional)

Audiobook-listening EEG from 19 participants, used only to pre-train the envelope-tracking trunk (`scripts/prepare_broderick.py`, `app/broderick_pretrain.py`). Pre-training shortens convergence but does not improve the final metrics, so the main pipeline does not depend on it.

### Di Liberto 2020 music EEG (for the subject-free speech + music route)

"Cortical encoding of melodic expectations in human temporal cortex" (eLife 2020), CC0: 20 participants (1-10 non-musicians, 11-20 pianists), 64-channel BioSemi at 512 Hz, 10 monophonic Bach pieces (~150 s) each heard 3 times, in CND format. The DOI is [10.5061/dryad.g1jwstqmh](https://doi.org/10.5061/dryad.g1jwstqmh); Dryad's API now needs a login token, so the script uses the identical [Zenodo record 5083410](https://zenodo.org/records/5083410) with md5 checks. The audio that was played is not included, only the MIDI files; `scripts/render_music_audio.py` re-renders them and aligns each rendering to the envelope vector shipped with the data.

```bash
bash scripts/download_music.sh data        # 5.98 GB zip + MIDI + README -> data/diliberto2020/ (needs ~14 GB free while extracting)
bash scripts/download_music.sh models      # MERT-v1-95M, EnCodec 24 kHz, BigVGAN-v2 24 kHz 100-band -> models/
bash scripts/download_music.sh soundfont   # MuseScore_General.sf2 (216 MB); FluidSynth itself: brew install fluid-synth
```

### MUSIN-G music EEG (second music set)

OpenNeuro [ds003774](https://openneuro.org/datasets/ds003774/versions/1.0.2), CC0: 20 participants, 128-channel EGI HydroCel at 250 Hz, 12 songs of different genres (~2 min each). Unlike Di Liberto, the presented audio ships with the data (`Code/ESongs`), but at **8 kHz** (nothing above 4 kHz). Stored in `data/ds003774/` (per-song BIDS files only, 10.9 GB; the continuous recordings in `sourcedata/` are the same samples and are not needed).

Checked on 2026-09-23 against the continuous recording: session / run N is song N, and every per-song file is an exact cut from 10.00 s before the song to its end. The annotations inside those files are copied unshifted from the start of the session and are wrong, so `scripts/prepare_musin_g.py` never reads them; it takes the onset at 10 s and checks each file's length against its song (all 24 files of sub-001/002 within 0.06 s). E129 is the flat Cz reference and is kept as zeros through the average reference. The 20 face/neck/eye electrodes further than 15 deg from every DS004940 electrode are dropped (109 kept), which leaves the EGI net on the same scalp region as the BioSemi caps.

```bash
bash scripts/download_music.sh musin-g                       # needs the aws CLI (no credentials)
MUSIC_DATASET=musin_g bash app/run_universal.sh prepare      # -> artifacts/music/musin_g (about 4 min)
```

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

Training-time augmentation beyond the v3 defaults (10 % channel dropout, two ≤250 ms time masks, ±20 % gain, σ = 0.1 white noise) is off unless requested; the ablation queue below runs the variants one at a time, throttled for temperature, against the existing three-seed baseline:

```bash
bash scripts/run_augmentation_queue.sh                    # robust, pool, both × seeds 322 323 324; skips finished runs
ALIGNED_TAG=robust ALIGNED_THROTTLE=0.5 \
  ALIGNED_EXTRA="--shift-max 10 --channel-gain 0.15 --background-mix 0.5 --background-alpha 0.2 0.8" \
  bash app/run_aligned_recovery.sh full                   # one variant by hand
```

| variant  | flags | idea |
|----------|-------|------|
| `robust` | `--shift-max 10 --channel-gain 0.15 --background-mix 0.5 --background-alpha 0.2 0.8` | ±40 ms response-latency jitter (zero filled, the audio clock is fixed), per-channel gain jitter, and with p = 0.5 a 0.2–0.8 fraction of the participant's own EEG from another sentence added as real background noise |
| `pool`   | `--mix-partners 3 --mix-start 0.8 --mix-anneal-epochs 12` with `ALIGNED_MIX=0.2` | SNR curriculum: with probability 0.8 → 0.2 (linear over 12 of ~24 epochs) the trial is averaged with up to three other presentations of its sentence; test inputs stay single-trial |
| `both`   | union of the two | |

`--throttle f` sleeps `f` × the wall time of every update (and evaluation), a duty cycle for laptops. Partner and background trials come from an in-memory float16 copy of the training EEG (`EEGBank`, 1.6 GB), because the gzip-chunked shards make random single-trial reads cost more than the update.

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
| `app/content_analysis.py` | The content route in four subcommands: `congruency` (N400 probe), `sentence` (closed-set decisions with a confidence gate), `pooled` (pooled sentence identification fusing embedding and envelope evidence) and `speak` (fixed-voice SpeechT5 synthesis) |
| `app/broderick_pretrain.py` | Envelope-tracking pre-training on Broderick 2018 |
| `app/audio_comparison.py` | Waveform-level metrics, 2AFC, listening-test bundles; `--figure` draws the spectrogram panels |
| `app/recovery_reports.py` | `evaluate` (one checkpoint, bootstrap CIs, waveform export) and `aggregate` (multi-seed summary) |
| `scripts/technical_report_figures.py` | Figures for the technical report |

### 5. Subject-free speech + music route

`app/universal_model.py` has no participant index at all (v3's per-participant 128 x 128 mixing is gone), so one set of weights serves participants never seen in training; a spatial-attention front end over electrode positions accepts any montage (128-channel DS004940, 64-channel Di Liberto) and is invariant to channel order; per-domain heads keep the v3 alignment chain (speech: HuBERT L9 -> 80-band mel -> SpeechT5 HiFi-GAN; music: MERT or EnCodec -> 100-band mel -> BigVGAN-v2) and a shared head predicts the envelope and onset strength. `app/universal_train.py` holds out 4 DS004940 participants entirely (hash-chosen: sub-004, sub-009, sub-010, sub-020) and selects checkpoints on validation sentences x those unseen participants, with the v3 controls. Spatial augmentation is part 2 of `app/universal_model.py` (cap rotation/shift/jitter by spherical splines, montage dropout, random re-reference, volume-conduction smoothing/sharpening, cross-participant covariance recolouring, two-view consistency); `app/music.py` holds the music file formats, the windowed dataset and the music decoder.

```bash
bash app/run_universal.sh prepare         # CND -> shards (same harmonisation as DS004940), manifest, piece/participant splits
bash app/run_universal.sh audio           # MIDI -> 24 kHz audio aligned to the EEG clock (fails below r = 0.5)
bash app/run_universal.sh targets         # MERT layer 6 (MUSIC_TEACHER=encodec as fallback), BigVGAN mel, envelope/onset
bash app/run_universal.sh music-decoder   # teacher -> 100-band mel decoder
bash app/run_universal.sh speech          # A1: DS004940 only, trial + spatial augmentation
bash app/run_universal.sh pretrain        # stage 1: speech + music, acoustic objectives
bash app/run_universal.sh joint           # stage 2 (from stage 1)
bash app/run_universal.sh speech-ft       # stage-2 control without music
```

Set `MUSIC_DATASET=musin_g` (default `diliberto2020`) to run the music stages and training on MUSIN-G; outputs are kept apart per dataset. All stages are throttled (`UNIVERSAL_THROTTLE=0.5`) and refuse to start while another training process runs.

### 6. KaraOne

```bash
bash app/run_karaone.sh prepare     # .cnt -> artifacts/karaone/shards
bash app/run_karaone.sh baselines   # app/karaone.py baselines: 11-way prompt decoding: block-wise CV, permutation tests, artifact controls
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
- 2026-09-23 consolidation: 25 modules in `app/` became 16. `content_analysis.py` = congruency_probe + synthesize_text + content_pipeline + group_content (subcommands `congruency|speak|sentence|pooled`); `karaone.py` = karaone_baselines + karaone_transfer (`baselines|transfer`); `recovery_reports.py` = evaluate_aligned_recovery + aggregate_recovery_runs (`evaluate|aggregate`); `music.py` = music_io + music_data + music_decoder; `universal_model.py` absorbed `spatial_augment.py`; `audio_comparison.py` absorbed `plot_audio_comparison.py` (`--figure`). Colliding names were renamed rather than allowed to shadow each other: `karaone.transfer_participant` / `transfer_summary`, `music.DECODER_CONTRACT`, `content_analysis.SENTENCE_CONTRACT` / `POOLED_CONTRACT` / `AUDIO_RATE`. `app/aligned_speech.py`, `app/src/eeg2speech/*.py` and the three `aligned_recovery*.py` files were left untouched because they are hashed into every checkpoint.
- 2026-09-23 script cleanup (restore any file with `git restore <path>`): `app/run_aligned_speech_v1.sh` and `scripts/download_aligned_audio.py` (the v1 AlignedEEGModel EEG route and its LibriSpeech pre-training, superseded by recovery v3; the audio-side stages of `app/run_aligned_local.sh` are unchanged, its v1 EEG stages were removed); `app/group_decoding.py` (encoder-input pooling, superseded by the pooled route that produced the pre-registered test result); `app/unit_ctc.py` and `scripts/build_speech_units.py` (unit-CTC decoding: on 2026-09-16 the same head fed zero EEG matched it, i.e. it learned only the corpus' unit prior). The v1 EEG code inside `app/aligned_speech.py` stays, because that file and every `app/src/eeg2speech/*.py` are part of the checkpoint runtime hash. Also removed: the v2 data route (`app/aligned_prepare_v2.py`, `app/run_aligned_v2_data.sh`, `configs/aligned_speech_local_v2.yaml`, `configs/training_data_aligned_v2.yaml`) and the abandoned joint/OOD configs, with their artifacts and outputs — 2.6 times the data had left every EEG-specific gain unchanged, and the subject-free route reaches more participants through `--heldout-subjects` instead.
