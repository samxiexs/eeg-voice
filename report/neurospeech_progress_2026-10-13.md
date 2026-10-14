# NeuroSpeech v2.1 Progress Report: Data Stage (A0)

Date: 2026-10-13 · covers plan version `neurospeech_plan_v2.1` · Chinese version: [neurospeech_progress_2026-10-13.zh.md](neurospeech_progress_2026-10-13.zh.md)

## Status

- **Scope of this report:** what was done on 2026-10-13 after the [research plan](neurospeech_research_plan_2026-10-13.md) was written: data ingest, synchronization, annotations, data decisions and the registry.
- **Verification status:** all five datasets are ingested and checksum-verified; the registry and window list are built. **No model has been trained or evaluated.** The training and evaluation code has only run on synthetic test fixtures.
- **Where this sits in the plan:** stage A0 of plan §11 is complete. Stages A1–D (copy / oracle, TRF, encoder, reliability, flow, evaluation) have not started.
- **Storage:** only processed data is kept (27 GB). Every raw batch was deleted after its processed output was verified; about 184 GB of raw files passed through and were removed, and Etard (about 18 GB) was streamed through memory without touching the disk.

## 1. Datasets

| Dataset | Subjects acquired | Subjects used | Paired EEG–audio (used) | EEG channels | How EEG and audio are paired | Words / phonemes |
|---|---|---|---|---|---|---|
| ds004408 | 19 | 19 | 19.2 h | 128 | source-aligned (official README) | released TextGrids: 11,419 words, 35,248 phonemes |
| Brennan2019 | 45 | 31 | 6.0 h (8.7 h processed) | 60 | 28 subjects measured on the audio-monitor channel, 3 from the authors' event table | official word table (2,125 words); 6,781 phonemes from forced alignment |
| Etard, English | 18 | 18 | 2.9 h clean + 11.3 h in babble noise | 63 | source-aligned (authors' aligned HDF5) | book-text transcript: 8,395 words, 27,677 phonemes from forced alignment |
| Kymata English | 20 | 20 | 18.1 h | 64 | measured on the audio-monitor channel for every repetition | released tiers: 1,396 words, 4,837 phonemes |
| KaraOne (later stage) | 14 | 14 | 1,913 trials | 62 | trial states from `epoch_inds.mat` | prompt label per trial |

Processed EEG is stored at 200 Hz (0.5–40 Hz) plus a 64 Hz copy low-passed at 15 Hz, in µV, with one canonical 24 kHz audio file per stimulus. Audio-monitor, EOG / ECG, trigger and MEG channels never enter a record.

## 2. Findings and fixes per dataset

### 2.1 Brennan2019

- **Segment labels were wrong or missing for 14 subjects.** In the re-release, `events.tsv` is missing for 6 subjects and one story segment late for 8. A search of the whole recording on the audio-monitor channel (AUD), followed by the same three-point check used everywhere else, located these segments. Subjects with a usable pairing went from 26 to 40.
- **Five subjects have no working audio monitor** (sub-005, sub-009, sub-022, sub-023, sub-038). See data decision 3.
- **Four word tokens had a corrupted apostrophe** in the official table (now "what's", "she'd", "who'd", "it's").
- **The release carries the authors' quality control.** `participants.tsv` marks 31 subjects as analyzed in Brennan & Hale (2019) and 14 as excluded for low comprehension and/or high noise. See data decision 2.

### 2.2 Kymata English

- **The reader did not exist** and was written: one mixed EEG/MEG FIF at a time is downloaded, checked against the audited SHA256, reduced to EEG and deleted.
- **The files have 64 EEG channels**, not the 70 stated in the plan and the paper. Electrode positions are the individual digitized ones stored in the FIF.
- **The audio runs on a different clock.** It starts about 19 ms after trigger 3 and plays about 0.054 % slower than the acquisition clock, which is about 0.2 s over the 400 s stimulus. Onset and clock ratio are measured on the audio-monitor channel for every repetition and the EEG is resampled onto the audio clock; the remaining error is about 1 ms.
- **One repetition had a 14 ms playback slip** (sub-E12, run 1, repetition 2). It is aligned block by block and the single 60 s block containing the slip is withheld.
- **One recording had no repetition trigger** (sub-E6, run 2). The onsets were found on the audio monitor and both repetitions were recovered.
- **Result:** 81 of 81 recordings, 162 repetitions aligned, none quarantined. `sub-E3b` (second session) is stored under `sub-E3`. The 400 s stimulus is split into seven fixed content blocks of 60 s.

### 2.3 Etard

- The ingest was resumed from 4 to 18 subjects; files are fetched by HTTP range into memory, so no archive is stored. A stalled connection is now retried instead of ending the run.
- **All 16 English story parts are different speech** (4 clean, 12 in noise). The plan's open question is answered: no content is shared between clean and noise, so "noise transfer on content seen in training" does not exist for this dataset.

### 2.4 KaraOne

- The plan schedules KaraOne for the later imagined-speech stage. It was already half ingested, so it was finished.
- The reader crashed on MM15. In some sessions the `speaking_inds` entries are out of order, duplicated, or repeat other ranges. Prompt and articulation ranges are now placed by position between the rest and imagined ranges, which are consistent in every session. A slot without a range is withheld, never guessed: 123 of 3,826 prompt / articulation ranges are withheld (MM15 42, MM16 1, MM20 80). All rest and imagined ranges are published.

### 2.5 ds004408

- Already complete. The only change is the bad-channel rule in data decision 1.

## 3. Annotations the sources do not ship

| Dataset | Missing | How it was produced | Automatic check |
|---|---|---|---|
| Brennan2019 | phonemes | Montreal Forced Aligner 3.4.3 (`english_us_arpa`) on the official word table | aligner word onsets against the official word times: median 13–20 ms, 95th percentile 45–61 ms |
| Etard | transcripts, word times, phonemes | The stories are chapters of *The Children of Odin* (Project Gutenberg eBook 24737); the original Zenodo archive names the stimulus files `odin_<chapter>_<part>`. The reference words are the book's. A Whisper draft only locates each passage and its pauses. The aligner gives word and phoneme times | 95–99.6 % of the book words in each passage are confirmed by the draft; aligner and draft word onsets differ by a median of 40–50 ms |
| Kymata English | speaker-overlap annotation (three speakers; none is released) | speech-level sound inside pauses of the released word tier; 15 intervals totalling 11.4 s of 401 s (about 7 s of sound plus 0.1 s padding on each side), in `configs/neurospeech_v2/content_masks.json` | none available |

- The plan (§6.1) asks for a manual sample of 200 boundaries per source. The automatic checks above stand in for it; **no manual review has been done.**
- The Kymata mask finds untranscribed sound between words (backchannels, laughter, cross-talk). Speech that overlaps a transcribed word is not detected. The file is marked `human_review: false`, and windows that overlap a masked interval are dropped (about 10 % of Kymata windows).

## 4. Data decisions

1. **Annotated bad channels do not reject windows.** ds004408 ships per-run bad-channel annotations (0–34 of 128 channels). The original studies kept all 19 subjects and interpolated noisy channels. Here annotated bad channels are masked out of the model input. The 10 % limit now counts only flat or over-range channels among those not annotated bad, and at least half of the cap must remain valid. Before this change sub-002 lost every window and sub-013 most of them; ds004408 window rejection went from 10.2 % to 0.2 %.
2. **Brennan2019 uses the authors' analyzed sample.** The 31 subjects marked `analyzed` in the release are used; the 14 the authors excluded are processed but not used. This is one line in `configs/neurospeech_v2/plan.yaml` (`subject_policy`).
3. **Three analyzed Brennan subjects without an audio monitor are source-aligned.** sub-005, sub-022 and sub-038 use the authors' event table: all twelve segments are listed back to back in the right order, and onsets are shifted by the offset measured on the other subjects (median −5.0 ms over 305 segments, range −13 to +27 ms). Their records carry `status: source-aligned`. sub-009 and sub-023 have the same problem and were excluded by the authors for low comprehension, so they stay out.

## 5. Registry and splits

`python scripts/neurospeech.py prepare` builds these from the processed store. All numbers are for the subjects in the registry.

| Dataset | Subjects | Content groups (train / validation / test) | Windows (train / validation / test) |
|---|---|---|---|
| ds004408 | 19 | 14 / 3 / 3 stimuli | 20,534 / 2,298 / 2,240 |
| Brennan2019 | 31 | 8 / 2 / 2 segments | 3,809 / 487 / 500 |
| Etard clean | 18 | 60 s blocks | 1,422 / 360 / 324 |
| Etard in noise (evaluation only) | 18 | 60 s blocks | 9,648 in total |
| Kymata English | 20 | 5 / 1 / 1 blocks of 60 s | 11,297 / 1,132 / 1,296 |

- **Total:** 55,347 windows of 4 s (training stride 2 s, evaluation stride 4 s).
- **Protocol P1** (four sources): 88 subjects; 13 development subjects (ds004408 sub-001–003, 5 Brennan, 2 Etard, 3 Kymata); five test folds of 18 / 15 / 14 / 14 / 14 subjects.
- **Protocol P2** (Brennan held out): 57 source subjects, 8 of them for development; 31 Brennan subjects as the external test set.

## 6. Code added or changed

| File | Change |
|---|---|
| `eegspeech/neurospeech/sources/kymata.py` | new reader: whitelist download, monitor-based clock measurement, per-block fallback, trigger-free onset search |
| `eegspeech/neurospeech/sources/brennan.py` | whole-recording AUD search, retry option, source-aligned fallback, apostrophe repair, keeps added phoneme tiers on rerun |
| `eegspeech/neurospeech/sources/karaone.py` | position-based trial states, retry option |
| `eegspeech/neurospeech/sources/etard.py` | retry of stalled range downloads |
| `eegspeech/neurospeech/annotation.py` | new: book-text transcripts, forced alignment, Kymata mask |
| `eegspeech/neurospeech/data.py` | content masks, bad-channel rule, subject policy, plan-fixed ds004408 development subjects, cache key includes the phoneme tier |
| `eegspeech/neurospeech/evaluation.py`, `scripts/neurospeech.py` | `annotate` command; `evaluate --mode zero-shot / few-shot / noise` |
| `configs/neurospeech_v2/plan.yaml`, `content_masks.json` | `min_valid_fraction`, `subject_policy`, the Kymata mask |
| `tests/test_neurospeech_sources.py` | new tests for the readers and annotation; 60 unit tests pass |
| `README.md`, `report/` | English documentation; Chinese originals kept as `*.zh.md` |

The forced aligner lives in the project-local conda environment `.venv-mfa` (1.3 GB); its models are in `artifacts/neurospeech_v2/mfa`.

## 7. Differences from the plan

- KaraOne was ingested now instead of after the perceived-speech validation.
- Kymata has 64 EEG channels in the files, not 70.
- Etard clean and noise conditions share no content, so the "seen content" noise comparison of plan §3.3 cannot be made.
- The manual 200-boundary phoneme check (§6.1) is replaced by automatic comparisons.
- The Kymata overlap mask (§2.1, §3.1) is derived automatically and is not human-reviewed.
- Brennan2019 is limited to the authors' analyzed sample (31 instead of 45 subjects), three of them source-aligned.
- The window QC rule treats annotated bad channels as masked rather than as rejections.

## 8. Not done

1. **All training and evaluation** (plan stages A1–D). Nothing has been run by instruction. Expect debugging on the first real run, because this code has only seen synthetic fixtures.
2. **Etard babble tracks.** The release has `part_k_distractor.wav` for the noisy stories; they are not ingested, so ASR on the noisy mixture that was presented (§3.3) is not available.
3. **Ablation variants A1–A6** as command-line options, and a runner for the 72-run matrix.
4. **Audit of REVE pretraining-corpus overlap** (§3.2).
5. **Manual review** of the phoneme boundaries and of the Kymata mask, if wanted.

## 9. How to check

```bash
python scripts/neurospeech.py status
```

```bash
python scripts/neurospeech.py verify-data
```

```bash
python scripts/neurospeech.py prepare
```

```bash
python -m pytest tests -q
```

Per-dataset evidence (source receipts, synchronization measurements, annotation checks, cleanup logs) is stored next to the data in `data/processed/neurospeech/{dataset}/manifest.json`, `metadata/` and `cleanup.jsonl`. The current commands and data decisions are also in [the project README](../README.md).

## 10. Addendum: imagined speech as generation

Written after the data stage, at the request to treat imagined speech as generation rather than classification. **Code only; nothing has been trained.**

- `eegspeech/neurospeech/imagined.py` transfers the listened-speech model to KaraOne: imagined EEG → frozen listened-speech encoder → a small re-timing module (the only trained part; it starts as an identity) → the same continuous condition → frozen flow generator → 2 s waveform. There is no classifier in the path.
- The regression target is the same trial's spoken recording, because imagination has no audio. The EEG window (4.5 s) lies inside the 5 s imagined state.
- Two tests: unseen people (four folds of three people) and an unseen prompt (one of the eleven prompts is never used for transfer). Controls: another prompt's imagined EEG from the same person, the trial's rest EEG, an empty condition, and the zero-shot model without the re-timer.
- Measures are taken on the generated audio: similarity to the person's own spoken recording, Mel error, and the rank of the true prompt among the eleven. Prompt identity appears only there and in the split.
- Checked on real KaraOne trials with untrained weights: shapes, the gradient path (only the re-timer receives gradients) and the writing of audio for every control. This replaces the classification-first validation of plan §14.
- Limits: eleven short prompts cannot show free-form imagined speech; the spoken recording is a stand-in with unknown timing.

## 11. Mac feasibility probe (2026-10-14)

Small encoder (S0, no REVE), protocol P2 source side (ds004408, Etard, Kymata; 28,179 training windows from 49 people), 3,000 steps on the laptop CPU, scored on 8 development people. Configurations: `configs/neurospeech_v2/probe_mac.yaml` and `probe_mac_contrastive.yaml`.

| Run | Objective | Correct vs wrong-time EEG | Audio identification (mean rank percentile, chance 0.5) |
|---|---|---|---|
| 1 | regression to speech features (plan weights) | no difference (p = 0.96); the output collapsed to the average feature | not measured |
| 2 | contrastive first (contrast 1.0, regression 0.1) | no difference | 0.486 at the end (best 0.481): ds004408 0.498, Etard 0.499, Kymata 0.461 (7 candidates); ds004408 sits exactly at chance, the sign of an EEG-independent output |
| check | linear envelope reconstruction, per person, unseen content (`scripts/probe_envelope.py`) | — | median r 0.08–0.12 per dataset; 75 of 88 people above their shuffled control |

Reading: EEG and audio are paired correctly and the EEG carries acoustic (envelope) information, but the small encoder learned no speech content in 3,000 steps with either objective. This does not test the full REVE model or longer training.
