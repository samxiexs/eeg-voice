# Physiology-Guided EEG-to-Speech via Conditional Flow Matching

Project root: `speech_decoding/`. Two generations of the pipeline live here.

- **v1 (implemented, unit-tested):** ds004408 perceived-speech EEG → TCN predicting WavLM temporal features → CFM → Vocos native Mel → 24 kHz waveform. Entry point `scripts/perceived_flow.py`, code in `eegspeech/*.py`.
- **v2.1 NeuroSpeech (current research plan):** four English perceived-speech sources plus KaraOne for the later imagined-speech transfer, REVE adaptation, physiology-guided alignment and neural-consistent CFM. Entry point `scripts/neurospeech.py`, code in `eegspeech/neurospeech/`. Data ingest is implemented; **no v2.1 model has been trained or evaluated.**

[Current research plan: NeuroSpeech v2.1](report/neurospeech_research_plan_2026-10-13.md) · [Progress report: data stage](report/neurospeech_progress_2026-10-13.md) ([中文](report/neurospeech_progress_2026-10-13.zh.md)) · [Chinese original of the plan](report/neurospeech_research_plan_2026-10-13.zh.md) · [Report index](report/README.md) · Chinese originals of this file and the report index are kept as `README.zh.md`.

## NeuroSpeech v2.1

### Data: processed-only store

Every source is ingested one batch at a time: download → check against the pinned checksum → extract scalp EEG in µV → 0.5–40 Hz, 200 Hz (plus a 15 Hz low-pass, 64 Hz copy for TRFs) → reopen and verify → delete the raw batch. Only `data/processed/neurospeech/{dataset}/` is kept; `data/staging/neurospeech/` is a transit area. Audio monitor, EOG/ECG, trigger and MEG channels never enter a record.

```bash
python scripts/neurospeech.py status
```

```bash
python scripts/neurospeech.py ingest kymata_english
```

```bash
python scripts/neurospeech.py verify-data
```

`ingest` accepts `ds004408`, `brennan2019`, `etard`, `kymata_english`, `karaone` or `all`, resumes where it stopped, and refuses to start a batch without the batch size plus 5 GB of free disk. Logs go to `logs/neurospeech_v2/`.

| Dataset | Subjects with usable pairing | Pairing evidence | Notes |
|---|---|---|---|
| ds004408 | 19 | source-aligned (official README) | 20 stimuli; word and phoneme TextGrids kept; annotated bad channels are masked (see data decisions) |
| Brennan2019 | 31 of 45 | 28 measured on the AUD monitor per segment, 3 source-aligned | the 31 are the sample the authors analyzed; the other 14 are processed but not used (see data decisions). Segments whose `events.tsv` labels are missing or one segment late are located by a whole-recording AUD search |
| Etard (English clean, lb, mb, hb) | 18 | source-aligned (authors' HDF5) | fetched by HTTP range into RAM, no archive on disk; all 16 story parts are different speech, so no clean↔noise content is shared |
| Kymata English | 20 | measured on the MISC audio monitor, per repetition | 64 EEG channels with digitized positions; audio starts about 19 ms after trigger 3 and runs about 0.054 % slower than the acquisition clock (≈ 0.2 s over the 400 s stimulus), so EEG is resampled onto the audio clock; 60 s content blocks; `sub-E3b` is stored under `sub-E3` |
| KaraOne | 14 | state ranges from `epoch_inds.mat`; articulation audio is not synchronous truth | later imagined/articulated stage only; prompt and articulation ranges are placed by position because `speaking_inds` is out of order in some sessions; empty slots are withheld |

Retry passes for the readers: `python -m eegspeech.neurospeech.sources.brennan --retry-quarantined`, `... .kymata --retry-quarantined`, `... .karaone --retry-withheld`.

### Annotations the sources do not ship

```bash
python scripts/neurospeech.py annotate etard
```

`annotate` takes `brennan2019`, `etard` or `kymata_english`. Forced alignment uses Montreal Forced Aligner 3.4.3 with `english_us_arpa` (acoustic, dictionary, G2P) in the project-local conda environment `.venv-mfa`; models live in `artifacts/neurospeech_v2/mfa`. Tool version and file checksums are written into each content's metadata.

| Dataset | What was missing | How it is produced | Automatic check |
|---|---|---|---|
| Brennan2019 | phonemes | MFA on the official word table | MFA word onsets against the official word times: median 12–21 ms, 95th percentile 45–58 ms |
| Etard | transcripts, word times, phonemes | The stories are chapters of *The Children of Odin* (Project Gutenberg 24737; the original Zenodo 7086209 stimulus files are named `odin_<chapter>_<part>`). The reference words are the book's; a Whisper draft only locates each passage and its pauses; MFA gives word and phoneme times | 95–99.6 % of the book words in each passage are confirmed by the draft; MFA and draft word onsets differ by a median of 40–50 ms |
| Kymata English | speaker-overlap annotation (three speakers, none released) | `configs/neurospeech_v2/content_masks.json`: speech-level sound inside pauses of the released word tier (15 intervals, 11.4 s of 401 s); windows overlapping them are dropped | none available. Speech that overlaps a transcribed word is not detected; the file is marked `human_review: false` |

The plan (§6.1) asks for a manual sample of 200 boundaries per source; the automatic checks above stand in for it and no manual review has been done.

### Data decisions (2026-10-13)

1. **Annotated bad channels do not reject windows.** ds004408 ships per-run bad-channel annotations (0–34 of 128 channels). The original studies kept all 19 subjects and spline-interpolated noisy channels (Broderick et al. 2018, Methods). Here annotated bad channels are masked out of the model input instead of interpolated. The 10 % limit (`max_bad_fraction`) now counts only flat or over-range channels among those not annotated bad, and `min_valid_fraction: 0.5` requires half of the cap to remain. Counting annotated channels toward the 10 % had removed sub-002 entirely (13–15 bad channels, 10–12 %) and most of sub-013, while sub-010 with 12 stayed in.
2. **Brennan2019 uses the authors' analyzed sample.** `participants.tsv` of the release marks 31 subjects as analyzed in Brennan & Hale (2019) and 14 as excluded for low comprehension and/or high noise. `subject_policy.brennan2019: authors_analyzed` in `configs/neurospeech_v2/plan.yaml` keeps the 31; removing that line uses every paired subject again.
3. **Three analyzed subjects without a working audio monitor are source-aligned.** sub-005, sub-022 and sub-038 use the authors' trial table (`events.tsv`): all twelve stimuli are listed back to back in the right order, and onsets are shifted by the offset measured on the monitor for 305 stimuli of the other subjects (median −5.0 ms, MAD 2.1 ms, range −13 to +27 ms). Their records carry `status: source-aligned`. sub-009 and sub-023 have the same dead monitor and were excluded by the authors for low comprehension, so they stay out.

### Open items before any v2.1 training

1. **Etard noise evaluation** (`evaluate --mode noise`, plan §3.3) scores lb / mb / hb for the held-out Etard subjects of a P1 fold. Only the clean attended-speech track of each noisy story is stored; the release also has the babble track (`part_k_distractor.wav`), which is not ingested yet, so ASR on the noisy mixture that was presented is not available.
2. **Not implemented yet:** command-line variants for ablations A1–A6, a matrix runner, and the audit of REVE pretraining-corpus overlap (plan §3.2). The training and evaluation code has only run on synthetic fixtures.

### Stages (plan §11)

```bash
python scripts/neurospeech.py prepare
```

`prepare` builds the content registry, splits and window list from the processed store and trains nothing. The remaining commands train or evaluate and have **not been run**: `copy-gate`, `oracle`, `trf`, `encoder`, `reliability`, `flow`, `evaluate [--mode zero-shot|few-shot|noise]`, `freeze`, `report`; each takes `--protocol P1|P2 --fold F0..F4|external --seed 11|22|33 --system S0..S3 --device cuda`.

### Imagined speech as generation (transfer from the listened-speech model)

Code in `eegspeech/neurospeech/imagined.py`; **not run**, because it needs a trained listened-speech model first.

- **Path:** imagined EEG → frozen listened-speech encoder → re-timer → the same continuous condition → frozen flow generator → waveform. The re-timer is the only trained part and starts as an identity, so before transfer training the model is exactly the listened-speech model applied to imagined EEG. There is no classifier; prompt identity is used only in the measures and the split.
- **Target:** imagination has no audio, so the regression target is the first 2 s of the same trial's spoken recording.
- **Window:** 4.5 s of EEG inside the 5 s imagined state gives 2 s of audio (KaraOne: 1,913 trials, 14 people, 11 prompts).
- **Two tests:** unseen people (`--split subject --held 0..3`: three held-out people per fold, two development people), and an unseen prompt (`--split prompt --held pat`: one prompt never used for transfer). The second is the one that separates generation from reproducing memorized sounds.
- **Controls:** imagined EEG of another prompt from the same person, rest EEG of the same trial, and an empty condition; plus the zero-shot model without the re-timer (`--zero-shot`).
- **Measures on the generated audio:** similarity to the person's own spoken recording in the speech-model feature space, Mel error, and the rank of the true prompt among the 11 using other people's recordings. A fixed listening set is written for every control.
- **Limits:** eleven short prompts cannot show free-form imagined speech, and the spoken recording is a stand-in for what was imagined, with unknown timing.

```bash
python scripts/neurospeech.py imagined-transfer --split subject --held 0 --device cuda
```

```bash
python scripts/neurospeech.py imagined-evaluate --split subject --held 0 --device cuda
```

## v1 pipeline (single dataset)

The v1 commands read `data/raw/ds004408`, which has been migrated to the processed store and deleted. Running v1 again needs `python scripts/perceived_flow.py download` first (about 20 GB).

### What v1 implements

- EEG / WAV / TextGrid download with resume and checksum comparison against the pinned OpenNeuro git-annex snapshot.
- BrainVision reading with calibrated units, 128 channels, bad channels / CAR, FIR, time context and a strict double hold-out split.
- Frozen WavLM-Base layer 6 frame-level teacher; native Vocos 100-bin natural-log magnitude Mel.
- TCN, scalp grouping, positive response-lag pooling, local multi-positive contrast and envelope / onset supervision.
- A single CFM: oracle pretraining, predicted-condition training, joint optimization, 32-step Heun inference.
- Training-side cross-fitted TRF, circular-shift / FDR channel selection, gate on unseen audio, differentiable Neural Consistency.
- Three systems B0 / B1 / B2; wrong / null controls; free-ASR WER; subject-level statistics, result tables and figures.
- 60-second labeled adapter, with filtering and model context counted in the calibration budget.
- Resumable checkpoints, model / data / statistics version checks, a freeze lock for the formal test and an EEG-only inference interface.

Whether training works can only be judged after the formal experiments. Unit checks and real-file interface checks do not replace the oracle, TRF or EEG-conditioning gates.

### Environment

Python 3.12. The local environment is `.venv-aligned-local/`:

```bash
source .venv-aligned-local/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

On a CUDA server install a matching torch / torchaudio first, then the rest. Ordinary training supports BF16 and microbatch gradient accumulation; Vocos, the acoustic physiology features and the TRF use FP32. Apple platforms default to CPU so that the STFT / ISTFT gradient path works. Formal training should use a CUDA GPU with at least 24 GB; the local CPU is for code checks and small interface checks.

### Data and weights

Official source: [ds004408](https://github.com/OpenNeuroDatasets/ds004408), pinned snapshot `b2d91784a90c18e1d76bd77dcb2b29b0ce27a0be`, about 20 GB: BrainVision EEG of 19 subjects, 20 real WAVs and 20 TextGrids. `--data-only` / `--models-only` download one side; `--subjects sub-007` downloads one person's EEG but still all 20 audio / TextGrid pairs.

`models/manifest.json` (v1) and `configs/neurospeech_v2/models.json` (v2.1) pin WavLM, Vocos, Whisper small.en, REVE-Base and the REVE position bank to full revisions and file SHA256. Training and inference only read the local pinned weights; nothing is fetched implicitly at model load.

### Unit correction of the pinned ds004408 snapshot

Multiplying the nominal BrainVision / MNE volts by 1e6 makes this snapshot a million times larger than the original authors' CND data. The config therefore multiplies by a fixed 1e−6 to restore the authors' numeric scale (channel-median correlation of the trajectories about 0.99995 against the same source); no per-recording scale is estimated. [The unit audit](report/dataset_unit_audit.json) stores the evidence. **The numeric scale is verified directly; reading it as absolute µV is an inference from the BioSemi device and source amplitudes, not an erratum published by the authors.** This correction must not be copied to any other dataset.

### Run order

A — preparation and speech-side validation:

```bash
python scripts/perceived_flow.py prepare --device cuda
```

```bash
python scripts/perceived_flow.py oracle --seed 11 --device cuda
```

```bash
python scripts/perceived_flow.py fit-trf --fold F0
```

B — F0 / seed 11 development: `train-encoder --fold F0 --seed 11`, then `train-flow --fold F0 --seed 11 --system B0|B1|B2`. Development models select checkpoints only on audio15–17 and sub-001–003. If the B0 EEG condition fails the development controls, the formal expansion stops.

C — fixed formal matrix and test: `matrix --resume`, `freeze`, `evaluate --split test`, `neural-diagnostic --fold all --seed 11`, `adapt --seconds 60`, `evaluate --split test --system B2 --adapted`, `report --split test`. The matrix is 3 systems × 5 folds × 3 training seeds, not an architecture search. The final test requires the complete matrix, the gate records and an unmodified protocol lock.

### EEG-only inference

```bash
python scripts/perceived_flow.py infer --fold F0 --seed 11 --system B2 --input data/my_processed_eeg.npy --output outputs/my_reconstruction.wav
```

The input array is `[832,128]`: 128 Hz EEG after the protocol filter, CAR and resampling, in µV, channel order A1–D32, covering 1 s before and 1.5 s after the 4 s core. `--mask` gives 128 valid-channel flags. The output is always 4 s; the interface takes no target audio, text, embedding or true length.

## Tests

```bash
python -m pytest tests -q
```

Unit tests download nothing and train nothing. `python scripts/verify_pipeline.py --asr --device cpu` (v1) and `python scripts/verify_neurospeech_models.py` (v2.1) check the real frozen-model interfaces; their TRF weights are synthetic fixtures and they report no scientific result.

## Directories

| Directory | Content | Git |
|---|---|---|
| eegspeech / scripts / tests | v1 pipeline, `eegspeech/neurospeech/` v2.1, entry points and tests | tracked |
| configs | fixed parameters, splits, pinned model specs, content masks | tracked |
| report | plan, small audit reports and final tables | tracked |
| data/processed/neurospeech | processed EEG (200 Hz + 64 Hz), canonical 24 kHz audio, labels, provenance | ignored |
| data/staging/neurospeech | transit area for one raw batch at a time | ignored |
| artifacts | statistics, caches, TRF / OOF, checkpoints, protocol locks | ignored |
| models / outputs / logs | frozen weights, generated audio / figures / per-sample metrics, logs | ignored |

The old imagined-speech code, configs, retired data and failed outputs were deleted; see [the cleanup record](report/cleanup_2026-10-13.json). Reference PDFs / BibTeX stay in `Paper-ECoG-EEG`.
