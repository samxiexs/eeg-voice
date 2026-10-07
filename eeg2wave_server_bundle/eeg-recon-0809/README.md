# eeg-recon-0809

Reconstructing **imagined speech** from scalp EEG as audible speech, within a person. There are two
pipelines:
- **Items** (`scripts/reconstruct.py`): BCI2020 (5 English phrases) and Thinking Out Loud (4 Spanish
  words). A personal CLIP encoder picks the item, and a mel diffusion decoder says it in the person's
  voice.
- **Sentences** (`scripts/sentences.py`): Chisco, about 6,500 Mandarin sentences. The encoder retrieves
  sentences never seen in training through a CLIP speech space, and the retrieved sentence is spoken in
  the person's voice.

Every evaluation holds out whole blocks of a person's recording (see Protocol) and reports chance-level
controls.

## What the data say

| Finding | Evidence |
|---|---|
| Within a person there is a real but small signal | Items: BCI2020 0.230 (chance 0.20, 12/15 people above), Thinking Out Loud 0.294 (0.25). Sentences (Chisco): rank percentile of the true sentence among ~130 candidates 0.512 (chance 0.5, 5/5 people above; robust in 3) |
| The speech side is not the bottleneck | Decoder ceiling (true item rendered and judged): 97-100 % identification. When the item encoder is right, the speech is heard as that item 97-100 % of the time. Chisco: the true sentence, rendered and transcribed by Whisper, is identified among the run's candidates 91 % of the time |
| In Chisco the information is in the time course of band power, and it is coarse | Whole-trial log-power: 0.503 (null). Log-power of 3 consecutive parts of the 3.3 s imagery epoch as deviations from the whole trial: 0.512. Every fold chose the strongest regularisation and the coarsest speech space tried. Topic (6.9 % vs 6.8 %) and exact sentence (top-1 0.84 % vs 0.76 %) are not recovered |
| Chisco's released EEG still holds its muscle activity | The authors' pipeline can remove ICLabel "muscle" components by ICA. The stored EEG has full rank except the one null eigenvalue of the average reference, so no components or interpolated channels were removed. Its 55-115 Hz power relative to alpha matches Thinking Out Loud and BCI2020. Rebuilding from the raw EDF would not add muscle signal back |
| BCI2020's file order is not the recording order, and the repetitions of a cue are near-duplicates | Each cue was imagined 4 times back to back. Neighbouring trials in the files are no more alike than any two (feature cosine 0.02 at lag 1, vs 0.15 in Thinking Out Loud). On >= 55 Hz power, with no learning, a trial's nearest neighbour is the same item 70 % of the time (chance 19 %), falling to 59 / 48 / 35 / 31 % for the 2nd-5th neighbour: 3 siblings per trial. The official split is leaky too: a test trial's nearest training trial is the same item 67 % of the time, but within the test file (one repetition per cue) only 20 %. The folds hold out whole reconstructed episodes (`features.episodes`); with them BCI2020 falls from 0.36 to 0.23 |
| Chisco's runs present topics in blocks | The next trial shares the topic 20 % of the time, against 3 % if shuffled, so trial-level splits let time stand in for content. The published Chisco results use random 8:2 trial splits. The folds hold out whole runs |
| Chisco's reading epochs were stored as imagined | Each run had a reading (5 s, sentence on screen) and an imagery (3.3 s) epoch file. Both were labelled imagined until 2026-10-06, so half of the "imagined" trials were reading |
| Above 55 Hz | Thinking Out Loud with bands <= 45 Hz only: 0.272 vs 0.290 full band. Chisco >= 55 Hz alone: 0.503 vs 0.507 with all bands (inner CV). The stores keep the full band (0.5-120 Hz at 256 Hz) |
| Across people, imagined items are not decodable | Linear and deep models, with or without listening pretraining, spoken/heard trials or person vectors: at chance on held-out people (2026-10-04; that code was removed on 2026-10-06, results in `outputs/imagery/`) |
| Listening EEG does not transfer to imagery | Listening is decodable (match-mismatch 41-56 %), but imagery has no shared stimulus clock (Marion: melody-specific ISC 0.28 listening, 0.00 imagery) |

## What to do next, and why (2026-10-06)

1. **The encoder is the only lever.** Decoders and renderers are at their ceilings, so effort goes into
   reading more from the EEG.
2. **Chisco: learn the time course.** The signal is in the time course of band power. The linear
   encoder now uses it by default, and its search extends past the edges where the last run's choices sat
   (weight decay 30-1000, speech space 4-64 dimensions, duration weight 1-3). The deep encoder learns the
   same chain end to end: a shared learned filter bank, personal spatial filters and the log-power time
   course, trained with the CLIP loss. This is the deep counterpart of filter-bank log-power
   (ShallowConvNet, Schirrmeister et al. 2017; depthwise spatial filters as in EEGNet, Lawhern et al.
   2018; CLIP loss with a per-person layer as in Défossez et al. 2023). With about 6,000 trials per
   person, Chisco is the first dataset here large enough for it; with 200-400 trials, deep lost to linear.
3. **Items: the same time course** replaces the 75 % sub-window augmentation, which was chosen in 7 of 30
   folds and changed the log-loss by 0.001.
4. **Positive control.** The same pipeline on Chisco's reading epochs (`run_plan.sh control`) measures how
   far it reaches when the content is surely in the EEG.
5. **Not now.** Rebuilding Chisco from raw EDF (see above). An open-vocabulary diffusion renderer: with
   top-1 at chance it would only render wrong sentences better; it follows once top-1 is clearly above
   chance.

## Items (`scripts/reconstruct.py`)

The output has two parts: the content (what is said, decoded from EEG) and the voice (who says it: sex,
timbre, pitch). The voice comes from the person's speech, never from EEG or an identity input.

| Step | What |
|---|---|
| Targets | Each vocabulary item is spoken by 8-12 macOS voices at 3 rates. Each clip gets a SpeechT5 log-mel, HuBERT embeddings, a WavLM speaker embedding and its median F0 |
| Content (CLIP) | Anchors `a_k` are the items' HuBERT embeddings (best layer by held-out-voice identification), centred, PCA to K-1 dimensions |
| Encoder | Per person: Euclidean alignment from the person's unlabelled EEG (never the fold's held-out trials; per session, or per person for BCI2020). Log-power of every channel in 8 bands, whole trial and optionally its time course in 3 parts. Optional causal drift removal (minus the mean of the 20 trials before). A linear map into the CLIP space, trained with InfoNCE in both directions and optional supervised contrast across modalities. A coordinate search scored by inner cross-validation (same fold protocol, log-loss) chooses bands, drift removal, other modalities, time course, weight decay and supervised contrast. The calibrated logits of the best 3 are averaged |
| Voice | The traits of a voice are its mean speaker embedding and median log F0. BCI2020 and Thinking Out Loud recorded neither voices nor sexes, so each person is assigned a synthetic voice in turn |
| Decoder | Mel diffusion (v-prediction, adaLN-zero blocks, DDIM) conditioned on the content (whitened anchors) and on the voice traits. Trained on speech alone with synthetic posteriors. Classifier-free guidance acts on the content only |
| Generation | Held-out trial -> the encoder's decision (the Bayes choice for identification) + the person's voice -> mel -> SpeechT5 HiFi-GAN, at the guidance the decoder check rates best on oracle speech. Soft posteriors, chosen on 150 training trials, once cost 9.5 points in a fold, so the decision is always rendered |
| Measures | Content: Whisper-small forced choice among the vocabulary, smallest DTW mel-cepstral distance (MCD), nearest HuBERT centroid. Voice: speaker identification, sex (F0 above 145 Hz = female), pitch error |
| Controls | Another trial of the same person from another cue episode (wrong), no content (prior), the true item (oracle = decoder ceiling). EEG vs wrong is tested per person (Wilcoxon) |

## Sentences (`scripts/sentences.py`, Chisco)

| Step | What |
|---|---|
| Targets | Every imagined sentence (6,566) spoken by 6 synthetic Mandarin voices (3 F, 3 M; macOS `say`; 39,396 clips). Each clip gets a SpeechT5 log-mel (for rendering), a Mandarin HuBERT embedding (`TencentGameMate/chinese-hubert-base`, layer 9, which identifies a sentence across voices 88 % of the time) and its duration |
| CLIP space | Anchor of a sentence: the HuBERT embeddings averaged over voices, centred, PCA, whitened, plus the log duration, unit length. Synthetic speech only, no EEG |
| Linear encoder (default) | Per person: aligned log-power of the 3.3 s imagery epoch, whole and its time course in 3 or 6 parts, causal drift removal, PCA to 512 on the training trials. Then a linear map to the CLIP space by ridge or by symmetric InfoNCE between a run's trials and its sentences. A coordinate search scored by inner cross-validation over runs; the 3 best averaged as calibrated log-probabilities |
| Deep encoder (`--encoder deep`) | `eegspeech/deep.py`: epochs at 128 Hz, a filter bank of 32 learned band-passes shared by everyone, 2 spatial filters per band per person, log-power every 0.125 s over 0.25 s, a personal readout to the CLIP space. Trained with cross-entropy against every training sentence's anchor (CLIP with a frozen speech tower) on random 3 s crops. Early-stopped on each person's last inner block of training runs, by the test measure |
| Retrieval | A held-out trial ranks the sentences imagined in its run (~130, none seen in training) |
| Speech | The top sentence in the person's voice (a synthetic voice of the person's sex, from `participants.tsv`) -> HiFi-GAN |
| Measures | Top-1/5/10 against chance, rank percentile of the true sentence (chance 0.5), topic of the retrieved sentence against its chance in the run, character accuracy of the retrieved text. Per person, a circular-shift test: each run's EEG sequence is shifted against its sentences, keeping the autocorrelation of both. On 20 trials per person and fold: Whisper transcript, character accuracy, which candidate it sounds like, voice, F0. Controls: a trial half a run away (wrong), no EEG (prior), the true sentence (oracle) |

## Protocol

Within person. Fold k holds out the k-th fifth of every person's recording:
- contiguous trials in recording order (Thinking Out Loud);
- whole cue episodes per item (BCI2020);
- whole runs in run order (Chisco). Sentences imagined in a held-out run are dropped from training,
  whoever imagined them.

Inner folds follow the same protocol. Nothing of the held-out trials enters training or alignment, and
the drift baseline of a trial uses only the trials before it.

## Results

Items, 2026-10-06, after the audit (`outputs/reconstruct/`, all 5 folds). The time-course encoder has
not been run yet. Content: share identified as the true item by Whisper / MCD / HuBERT.

| Dataset (chance) | Encoder | From EEG | Wrong trial | Prior | Oracle (decoder ceiling) | Voice of EEG reconstructions | EEG > wrong trial |
|---|---|---|---|---|---|---|---|
| bci2020 (0.20) | 0.230 | **0.229 / 0.229 / 0.229** | 0.190 / 0.189 / 0.189 | 0.192 / 0.192 / 0.193 | 0.992 / 0.993 / 0.992 | sex 95 %, 1.6 semitones | 14/15 people, p = 0.002 |
| thinking_out_loud (0.25) | 0.294 | **0.290 / 0.294 / 0.291** | 0.237 / 0.235 / 0.239 | 0.247 / 0.255 / 0.258 | 0.975 / 1.000 / 0.945 | sex 96 %, 0.9 semitones | 8/10 people, p = 0.005 (Whisper) |

Sentences, 2026-10-06 (5 folds, 29,336 held-out imagined trials). Whole-trial power is in
`outputs/sentences_v1_whole_trial/` and the time course (3 parts, ridge, weight decay 100, 16-64
dimensions) in `outputs/sentences_v2_time/`. The extended search and the deep encoder have not been run yet.

| Person | Whole-trial power | Time course | Wrong trial | Circular-shift p (time course), folds 0-4 |
|---|---|---|---|---|
| sub-01 | 0.506 | **0.527** | 0.500 | 0.016 0.011 0.001 0.001 0.002 |
| sub-02 | 0.512 | **0.515** | 0.501 | 0.008 0.020 0.008 0.015 0.18 |
| sub-03 | 0.501 | **0.513** | 0.502 | 0.002 0.017 0.043 0.57 0.39 |
| sub-04 | 0.499 | 0.504 | 0.499 | 0.12 0.41 0.30 0.23 0.58 |
| sub-05 | 0.499 | 0.503 | 0.505 | 0.07 0.29 0.72 0.58 0.46 |
| mean (rank percentile, chance 0.5) | 0.503 | **0.512** (5/5 above, p = 0.031) | 0.502 | |

Time course: top-10 8.3 % (chance 7.6 %), top-1 0.84 % (0.76 %), topic 6.9 % (6.8 %).

Earlier comparison, the personal CLIP encoder vs logistic regression (BCI2020 on folds later found
leaky): dot-product logits solved by L-BFGS match logistic regression; the cosine form trained with Adam
loses 3-4 points.

## Data

One HDF5 layout for every dataset in `artifacts/store/` (`eegspeech/store.py`). Raw downloads are deleted
after conversion; rebuild with `scripts/download.py` and `scripts/prepare.py`. About 16.6 GB in all.

| Store | Role | People | Content | Band | Size |
|---|---|---|---|---|---|
| `thinking_out_loud` | spoken / inner / visualised | 10 | 4 Spanish words, 128-ch (Nieto et al. 2022) | 0.5-100 Hz | 1.0 GB |
| `bci2020` | imagined | 15 | 5 English phrases, 64-ch (BCI Competition 2020, Track 3) | 0.5-120 Hz | 0.4 GB |
| `chisco` | imagined (+ reading) | 5 | 5,100-6,400 imagined sentences each (3.3 s) and the reading epoch before each (5 s), ~6,500 sentences, 39 topics, 122-ch (Zhang et al. 2024) | 1-120 Hz | 15.3 GB |

Notes:
- Thinking Out Loud keeps the trials the authors flag for mouth EMG: covert articulation is information.
- Chisco: `prepare.py chisco --relabel` fixed the reading epochs in place (original tables in
  `artifacts/store/chisco_segments_before_relabel.npz`). The store lists runs in name order; the folds use
  the run number. 7 channels have no standard position and are masked.
- Speech targets: `artifacts/audio/<dataset>.npz` (items) and `artifacts/audio/chisco_sentences.h5`
  (sentences); both need macOS `say` to rebuild.
- Removed 2026-10-06: the `cpseed` and `marion2021` stores, the listening stores and the cross-person
  plan's code (`train.py`, `model.py`, `baseline.py`, `isc.py`; still in git history).
- Models in `models/`: SpeechT5 HiFi-GAN, HuBERT-base (English), Chinese HuBERT-base, Whisper-small,
  WavLM-base-plus-sv.

## Usage

```bash
.venv-aligned-local/bin/python -m unittest discover -s tests       # synthetic tests
python scripts/download.py <dataset> && python scripts/prepare.py <dataset>
python scripts/download.py models        # HiFi-GAN, HuBERT (en, zh), Whisper-small, WavLM-SV -> models/
bash scripts/run_plan.sh reconstruct     # items: targets -> decoders -> 5 folds -> report
bash scripts/run_plan.sh sentences       # Chisco, linear encoders -> outputs/sentences
bash scripts/run_plan.sh deep            # Chisco, deep encoder -> outputs/sentences_deep
bash scripts/run_plan.sh control         # Chisco reading epochs, fold 0 (positive control)
bash scripts/status.sh                   # what is running, last steps, errors
```

`configs/plan.yaml` holds every setting. Building targets needs macOS `say`; copy `artifacts/audio` to run
the rest elsewhere (CUDA, MPS or CPU). Run one heavy process at a time on a 16 GB machine.

## Layout

```
eegspeech/   store signal features metrics                (stores, alignment, features and folds, statistics)
             clip deep diffusion audio                     (CLIP encoders, deep encoder, mel decoder, speech)
scripts/     download prepare reconstruct sentences run_plan.sh status.sh
configs/     plan.yaml
tests/       test_core.py
```
