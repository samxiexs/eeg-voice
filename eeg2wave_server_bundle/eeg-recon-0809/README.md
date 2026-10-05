# eeg-recon-0809

Reconstructing **imagined speech** from scalp EEG as audible speech. Listening EEG was the planned
stepping stone. The evidence below moved the work to per-person calibration on the full EEG band, with
a CLIP speech space and a diffusion decoder producing the audio.

## What the data say

| Finding | Evidence |
|---|---|
| Across people, imagined items are not decodable | Linear and deep models, with or without listening pretraining, spoken/heard trials or person vectors: at chance on held-out people (KaraOne +1-2 points) |
| Within a person, there is a real but small signal | Linear, 5 contiguous folds per person: BCI2020 0.359 (chance 0.20), Thinking Out Loud 0.290 (0.25) |
| It sits mostly above 55 Hz (likely covert-articulation EMG) | Same decoder with bands <= 45 Hz only: BCI2020 0.320, Thinking Out Loud 0.272. So the stores keep the full band (0.5-120 Hz, 256 Hz) |
| Simple beats deep within a person | BCI2020: aligned band log-power + linear 0.36; time-resolved band power 0.30; tangent-space covariances 0.33; deep encoder trained on everyone 0.24 (fold 0) |
| Listening EEG does not transfer to imagery | Listening is decodable (match-mismatch 41-56 %), but imagery has no shared stimulus clock (Marion: melody-specific ISC 0.28 listening, 0.00 imagery) |
| High published accuracies use leaky splits | Random trial splits put back-to-back repetitions of a cue into both train and test. Every split here is contiguous in recording order |

## Reconstruction (`scripts/reconstruct.py`)

| Step | What |
|---|---|
| Targets | Each vocabulary item is spoken by 8-12 macOS voices at 3 rates. Each rendering gets a SpeechT5 log-mel and HuBERT embeddings |
| CLIP space | Anchors `a_k` are the items' HuBERT embeddings (best layer by held-out-voice identification), centred, PCA to K-1 dimensions |
| Encoder | Per person: Euclidean alignment (from the person's unlabelled EEG), full-band log-power of every channel in 7 bands, and a linear map into the CLIP space. Trained with InfoNCE in both directions (EEG <-> speech) and optional supervised contrast across modalities. No identity input anywhere: personalisation comes from the person's own calibration trials. Settings come from inner cross-validation |
| Decoder | Mel diffusion (v-prediction, adaLN-zero blocks, DDIM) conditioned on `e = sum_k p_k a_k`. It is trained on speech alone with synthetic posteriors (Dirichlet; the spoken item is drawn from `p`), with classifier-free guidance. EEG never enters its training |
| Generation | Held-out trial -> calibrated `p(k \| EEG)` -> `e` -> mel -> SpeechT5 HiFi-GAN -> 16 kHz. Posterior sharpness and guidance are chosen on cross-fitted training trials |
| Measures | What the speech is heard as: Whisper-small forced choice among the vocabulary (listener), smallest DTW mel-cepstral distance (MCD), nearest HuBERT centroid. Also the MCD to the true item |
| Controls | Another trial of the same person (wrong), no condition (prior), the true item's anchor (oracle = decoder ceiling). EEG vs wrong is tested per person (Wilcoxon) |
| Listening | `outputs/reconstruct/index.html` holds the numbers and links pages with reference, reconstruction and controls for each person and fold |

Protocol: within person. Fold k holds out the k-th contiguous fifth of every person's trials in every
modality. Vocabularies: Thinking Out Loud (4 Spanish words), BCI2020 (5 English phrases), KaraOne
(7 phonemes + 4 words). Chisco sentences and CPSEED syllables have no speech targets yet.

## Data

One HDF5 layout for every dataset in `artifacts/store/` (`eegspeech/store.py`). Raw downloads are
deleted after conversion. Rebuild them with `scripts/download.py` and `scripts/prepare.py`. About 37 GB in all.

| Store | Role | People | Content | Band | Size |
|---|---|---|---|---|---|
| `thinking_out_loud` | spoken / inner / visualised | 10 | 4 Spanish words, 128-ch (Nieto et al. 2022) | 0.5-100 Hz | 1.0 GB |
| `bci2020` | imagined | 15 | 5 English phrases, 64-ch (BCI Competition 2020, Track 3) | 0.5-120 Hz | 0.4 GB |
| `karaone` | cue / spoken / imagined | 14 | 7 phonemes + 4 words, 62-ch (Zhao & Rudzicz 2015) | 0.5-120 Hz | 0.9 GB |
| `chisco` | imagined | 5 | ~11,700 imagined sentences each, 39 categories, 122-ch (Zhang et al. 2024) | 1-120 Hz | 15.3 GB |
| `cpseed` | spoken / mouthed / imagined | 18 | 10 Mandarin syllables, 32-ch (Ma et al. 2025) | 4-45 Hz (authors) | 0.5 GB |
| `marion2021` | listen + imagine | 21 | 4 Bach melodies, 64-ch | 0.1-30 Hz | 0.4 GB |
| `sparrkulee` | listen | 85 | Dutch speech, 64-ch, 159 h (Accou et al. 2024) | 0.5-32 Hz | 4.8 GB |
| `broderick2018` | listen | 19 | English audiobook, 128-ch | 0.5-45 Hz | 2.3 GB |
| `ds004940` | listen | 22 | English sentences, 128-ch | 0.5-45 Hz | 2.7 GB |

Notes:
- Thinking Out Loud keeps the trials the authors flag for mouth EMG, since covert articulation is information.
- CPSEED: sub-02 (duplicated files) and sub-10 (no epochs) are excluded. Channel order comes from the EDF headers.
- KaraOne Cb1/Cb2 and 7 Chisco channels have no standard position and are masked.
- ds004940, broderick2018 and marion2021 were converted from the previous project's caches. Those caches
  are gone, so these three stores cannot be rebuilt from this repository.
- Models in `models/`: SpeechT5 HiFi-GAN, HuBERT-base, Whisper-small.

## Usage

```bash
.venv-aligned-local/bin/python -m unittest discover -s tests       # synthetic tests
python scripts/download.py <dataset> && python scripts/prepare.py <dataset>
python scripts/download.py models        # SpeechT5 HiFi-GAN, HuBERT-base, Whisper-small -> models/
bash scripts/run_plan.sh reconstruct     # targets -> decoders -> 5 folds -> report (RECON_DATASETS, FOLDS)
bash scripts/status.sh                   # what is running, last steps, errors
python scripts/reconstruct.py run --datasets bci2020 --fold 0      # one fold of one dataset
python scripts/baseline.py [--max-hz 45]                           # linear floor
bash scripts/run_plan.sh gates | listen | imagery | report         # cross-person plan (listening pretraining)
```

`configs/plan.yaml` holds every setting. `targets` needs macOS `say`. Copy `artifacts/audio` to run
the rest elsewhere (CUDA, MPS or CPU). Run one heavy process at a time on a 16 GB machine.

## Layout

```
eegspeech/   store signal data model losses evaluation metrics      (stores, deep encoder, cross-person plan)
             features clip diffusion audio                           (reconstruction)
scripts/     download prepare baseline isc train report run_plan.sh status.sh reconstruct
configs/     plan.yaml
tests/       test_core.py
```

## Results

Cross-person imagery (5 subject folds, 4 variants, without Chisco): every dataset stays at or near
chance. Listening pretraining (H1), spoken/heard trials (H2) and person vectors (P1) show no reliable
effect. Details are in `outputs/imagery/summary.json`.

Personal CLIP encoder vs logistic regression, same within-person folds (imagined trials):

| Dataset (chance) | Logistic regression | CLIP, cosine (Adam) | CLIP, dot product (L-BFGS) | + supervised contrast 0.1 |
|---|---|---|---|---|
| bci2020 (0.20) | 0.367 | 0.331 | 0.370 | 0.358 |
| thinking_out_loud (0.25) | 0.296 | 0.282 | 0.290 | 0.293 |

The encoder therefore uses dot-product logits; inner cross-validation decides on supervised contrast and
on the person's non-imagined trials.

Reconstruction: run `bash scripts/run_plan.sh reconstruct`. Numbers land in
`outputs/reconstruct/summary.json`. Expect the EEG reconstructions to be identified about as often as
the encoder is right; the oracle row is the decoder's ceiling.
