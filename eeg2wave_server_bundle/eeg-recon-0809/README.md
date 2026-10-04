# eeg-recon-0809

Decoding **imagined speech** from scalp EEG. Listening EEG, which is plentiful, is the stepping stone:
an encoder is pretrained on people listening to speech, then trained on items that people hear, speak,
mouth and imagine, and evaluated on the **imagined trials of people it has never seen**. The output
waveform is the decoded item rendered as speech.

Earlier work here regressed 4-s spectrograms from listening EEG and generated audio with diffusion.
Retrieval stayed at chance and the decoder learned to ignore the EEG, while linear analyses showed that
listening EEG carries little beyond the loudness envelope. That pipeline is in the git history before
this rewrite.

## Plan

| Step | What | Script |
|---|---|---|
| 0. Gates | Linear floor per dataset (alignment + band log-variance + logistic regression). Then a check of whether imagery has stimulus-locked activity that a listening decoder could transfer: inter-subject and split-half correlation, with item-common activity removed | `baseline.py`, `isc.py` |
| 1. Listen | Match-mismatch (EEG vs 1 matched + 4 mismatched speech segments) plus agreement between two people hearing the same stimulus window. Data: SparrKULee, Broderick, DS004940 | `train.py listen` |
| 2. Imagery | Item cross-entropy per dataset (non-imagined trials at half weight) plus supervised contrast across modalities: heard, spoken, mouthed and imagined trials of the same item are positives | `train.py imagery --fold k` |
| 3. Personalisation | No subject ids anywhere. Euclidean alignment uses the person's own unlabelled EEG (always on). Optional person vector from unlabelled calibration windows (gated FiLM, Zhang et al. 2026). Few-shot prototypes from k labelled trials of the new person | `--set person_dim=32`, evaluation |
| 4. Speech | Decoded item rendered with a macOS `say` voice | `render.py` |

Hypotheses tested by `scripts/run_plan.sh` (paired over the same held-out people):
- **H1**: listening pretraining helps imagery (`pretrained` vs `scratch`).
- **H2**: spoken and heard trials help imagery (`scratch` vs `imagined_only`).
- **P1**: a person vector helps (`person` vs `scratch`).

The encoder is montage-agnostic: spatial attention over electrode positions. It has a phase branch for
stimulus-locked listening responses and a band-power branch for the non-phase-locked activity of
imagery (`eegspeech/model.py`).

## Data

All datasets are converted to one HDF5 layout in `artifacts/store/` (`eegspeech/store.py`).
Raw downloads are deleted after conversion. Rebuild them with `scripts/download.py` and `scripts/prepare.py`.

| Store | Role | People | Content | Size |
|---|---|---|---|---|
| `sparrkulee` | listen | 85 | Dutch audiobooks/podcasts, 64-ch, 159 h (Accou et al. 2024) | 4.8 GB |
| `broderick2018` | listen | 19 | English audiobook, 128-ch, 19 h | 2.3 GB |
| `ds004940` | listen | 22 | English sentences (N400), 128-ch | 2.7 GB |
| `marion2021` | listen + imagine | 21 | 4 Bach melodies heard and imagined with a metronome, 64-ch | 0.4 GB |
| `thinking_out_loud` | spoken / inner / visualised | 10 | 4 Spanish words, 128-ch (Nieto et al. 2022) | 0.5 GB |
| `cpseed` | spoken / mouthed / imagined | 13 | 10 Mandarin Pinyin syllables, 32-ch (Ma et al. 2025) | 0.4 GB |
| `karaone` | cue / spoken / imagined | 14 | 7 phonemes + 4 words, 62-ch | 0.4 GB |
| `bci2020` | imagined | 15 | 5 English phrases, 64-ch (BCI Competition 2020, Track 3) | 0.2 GB |
| `chisco` | imagined | 5 | ~12,600 imagined sentences each, 39 semantic categories, 122-ch (Zhang et al. 2024) | ~8 GB |

Stimulus audio for re-computing listening features stays in `data/ds004940`, `data/ds004408` and `data/marion2021`.

Excluded subjects:
- cpseed: sub-02 (duplicated session files) and sub-10 (no epoched data).
- cpseed: sub-16..20. Their files hold 32 unnamed channels whose order matches no known layout.
- thinking_out_loud: inner and visualised trials that the authors flag for EMG.
- chisco: 7 of 122 channels (P11/P12, PO11/PO12, POO11h/POO12h, TPP5h) are masked; they have no standard position.

## Usage

```bash
.venv-aligned-local/bin/python -m unittest discover -s tests     # synthetic tests
python scripts/download.py <dataset>                              # thinking_out_loud | cpseed | bci2020 | sparrkulee | chisco
python scripts/prepare.py <dataset>                               # -> artifacts/store/<dataset>.h5
bash scripts/run_plan.sh gates | listen | imagery | report | all  # the plan, finished steps skipped
python scripts/render.py items thinking_out_loud
python scripts/render.py decode thinking_out_loud --run outputs/imagery/pretrained_f0 --subject sub-03 --shots 5
```

`configs/plan.yaml` holds windows, weights, folds, steps and the TTS vocabularies. Use `python` from
`.venv-aligned-local`, which runs on MPS. Run one heavy process at a time on a 16 GB machine.

## Layout

```
eegspeech/   store.py signal.py data.py model.py losses.py evaluation.py metrics.py
scripts/     download.py prepare.py baseline.py isc.py train.py report.py render.py run_plan.sh
configs/     plan.yaml
tests/       test_core.py
```

## Results so far

Linear floor (accuracy on imagined trials; cross = held-out people, 5 subject folds):

| Dataset | Chance | Within person | Across people |
|---|---|---|---|
| thinking_out_loud | 0.250 | 0.276 | 0.264 |
| cpseed | 0.100 | 0.127 | 0.109 |
| karaone | 0.091 | 0.120 | 0.118 |
| bci2020 | 0.200 | 0.302 | 0.198 |
| marion2021 (music) | 0.250 | 0.402 | 0.273 (0.301 when listening trials join training) |

Stimulus-locking gate on Marion (1-8 Hz, `outputs/isc_marion2021.json`):
- **Listening**: melody-specific activity is shared across people (ISC 0.28, p = 0.005).
- **Imagery**: shared activity is entirely melody-common, i.e. metronome or task (ISC 0.19 raw, 0.00 melody-specific).
- **Imagery within a person**: melody-specific split-half reliability is weak but positive (0.012, above the null in 15/21 people).

Imagery therefore offers no shared stimulus clock for a listening decoder to lock onto. Transfer has to go
through representations (spatial and spectral front end, cross-modal item codes) and through per-person calibration.
