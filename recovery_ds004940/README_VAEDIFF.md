# vaediff: VAE + diffusion EEG-to-speech on DS004940

A new route written on 2026-10-16 from two proposals, placed next to the restored recovery code and reusing its
diffusion block and sampler (`generative_recovery.py`), its EEG encoder trunk, augmentation and per-sentence batching
(`aligned_recovery_model.py`), and its audio measures (`audio_comparison.py`). Chinese version:
[README_VAEDIFF.zh.md](README_VAEDIFF.zh.md).

**Status: code written, unit tests and a synthetic end-to-end smoke run pass, DS004940 re-downloaded and processed
(raw files deleted one subject at a time). No model has been trained on real data; there are no results below.**

## How the two proposals map to code

**Proposal 1: three weighted losses with hard constraints, weights adjusted from results.** The stage-2 loss is

```
L = w1 · L_voice + w2 · L_text + w3 · L_diff + β · KL
```

| Term | What it is | Scale |
|---|---|---|
| `L_voice` (timbre / content, w1) | (a) chunk contrast between EEG and audio codes (0.4 s chunks, same position, other sentences of the batch as negatives, the NPC/NPI partner of the same sentence frame excluded); plus (b) the timbre distance of the decoder's one-step estimate: level-free long-term spectrum over speech frames vs the truth, in dB | (a) divided by log #candidates, chance = 1; (b) divided by the timbre threshold |
| `L_text` (text, w2) | attention-pooled text head on the EEG code, in-batch contrast against the frozen MiniLM sentence embedding | divided by log batch, chance = 1 |
| `L_diff` (diffusion, w3) | v-prediction loss of the mel diffusion over samples conditioned 60 % on EEG codes, 30 % on audio codes, 10 % null (for classifier-free guidance) | raw |
| `KL` (VAE, fixed β) | see proposal 2 | nats per dimension and frame, free bits |

Initial values by intuition: `w1 = 0.5, w2 = 0.2, w3 = 1.0, β = 0.05`. Every 50 updates the controller measures, on a
fixed set of 256 dev trials (dev sentences never appear in training):

- `voice_rank`: rank percentile at which the EEG code retrieves its own sentence's audio code, chunk by chunk
  (0 best, 0.5 chance); constraint ≤ 0.42;
- `timbre_db`: timbre distance (dB) of the decoder's one-step estimate at a fixed noise step (t = 200) under EEG
  conditioning; constraint ≤ threshold (default 1.5 × the same quantity on the audio route of the stage-1 checkpoint);
- `text_rank`: rank percentile at which the text head retrieves its own sentence's text; constraint ≤ 0.45.

Rule: `w1` follows `max(voice_rank / 0.42, timbre_db / threshold)`, `w2` follows `text_rank / 0.45`; a ratio above 1
(violated) multiplies the weight by `exp(0.5 · (ratio − 1))`, below 1 lowers it; one change is capped at ±0.5 in the
exponent and the weight stays within 1/4–4× its base; `w3` is the anchor and changes only by hand. Every round is one
line in `outputs/vaediff/<run>/tune.jsonl`.

The hook for "Claude guesses every 50 results": a `control.json` in the run folder, checked every 50 updates and read
whenever it changes:

```json
{"weights": {"voice": 0.8, "text": 0.3}, "thresholds": {"voice": {"voice_rank": 0.40}}, "mode": "hold", "note": "guess 3: text never met, hold"}
```

Weights given there become the new base; `"mode": "hold"` stops the automatic rule; `"auto"` resumes it. Reading
`bash scripts/run_vaediff.sh status` and writing the next `control.json` is one round of guessing; the change and its
note are logged in `tune.jsonl`. Someone (or a `/loop` session) has to be watching for that; training never calls Claude.

**Proposal 2: different people's EEG for the same audio differs, so the code must be a distribution (VAE) with a
diffusion decoder.**

```
audio ─ WavLM-base layers 3/6/9/12 (frozen) ─ small conv ─► q_a(z | audio) = N(μ_a, σ_a²)          stage 1, then frozen
EEG   ─ subject channel mixing + shared conv trunk + subject FiLM ─► q_e(z | EEG, subject) = N(μ_e, σ_e²)   stage 2
z (32 dims, 25 Hz) ─► conditional mel diffusion (v-prediction, adaLN, CFG) ─► Vocos log-mel ─► Vocos ─► 24 kHz audio
```

- Both posteriors share one latent space; the decoder sees only `z`, not whether it came from audio or EEG.
- `KL(q_e ‖ N(μ_a, σ_a² + s²))`: the EEG code is pulled toward the sentence's audio code, but the prior is widened by a
  person spread `s` (default 0.3), so each person's code can sit about `s` away from the sentence code at little cost
  instead of being forced onto one point. `s` is the explicit parameter of "different people, different EEG".
- A person never seen in training (subject −1) takes the shared path, without subject mixing or FiLM.
- Pooling several presentations is a product of experts: precision-weighted mean, the single-trial mean precision kept
  (the decoder was trained only at single-trial noise).

## Data: DS004940 re-downloaded and processed

`app/vaediff_data.py`, one subject at a time: download both tasks' BDF files → preprocess → cut epochs, store float16 →
read back and verify → delete that subject's raw files → next subject. The stimulus WAVs were deleted after processing;
the 24 kHz waveforms are kept in the processed table.

- Preprocessing as in September: 128 BioSemi channels, declared-bad channels interpolated by spherical splines (a
  recording above 15 % bad is skipped and recorded), average reference, 0.5–45 Hz band-pass. Two differences: resampled
  to 128 Hz (nothing above 45 Hz survives the filter); each continuous run is scaled per channel by its own median/MAD
  (unsupervised, identical for every split).
- Both tasks (Active, Passive), at most 2 × 400 trials per person; the two sentences the authors eliminated
  (`(LD5_Eliminated)…`) are left out.
- Stored epochs span [−0.5, 4.5) s; the model uses EEG [−0.25, 3.6) s and audio [0, 3.6) s by default. Reason: in the
  Active task the prompt "Did this sentence end correctly" starts **exactly 2.0 s** after every sentence, the shortest
  sentence is 1.6 s, so EEG after 3.6 s may contain the response to the prompt, whose timing gives away the sentence
  duration. The September 4.35 s window included it. Only the longest sentences lose their ends (1.6–3.84 s).
- Splits: the 402 sentences are 201 pairs (one frame, congruent vs incongruent ending); **whole pairs** go to one split:
  161 train, 20 dev, 20 test. Three people, chosen by hash among those whose both recordings survived quality control,
  appear only in the test (unseen people × unseen sentences).
- Result (2026-10-16 01:19–02:05): 35 of 44 recordings kept, 20 of 22 people have data; sub-006 and sub-019 exceed 15 %
  bad channels in both tasks; sub-007, 011, 012, 014, 017 keep one task. 13,807 epochs, 2.1 GB. Unseen people: sub-004,
  010, 022. Trials per role: train 9,224, dev 1,156, test with the 17 seen people 1,151, test with the 3 unseen people 234;
  the unseen people's 2,042 trials of train/dev sentences are not used. 473 trials (3.4 %) peak above 30 MAD units and are
  dropped by default (`--max-peak`).
- Split recordings (Active of sub-004, 007, 012; Passive of sub-019) checked: the event clock is the sum of the runs'
  sample counts, without gaps at the joins. sub-004 completed only 278 sentences.

Two observations to keep in mind (sub-001 only, not conclusions): the largest negativity of the average ERP after the
sentence trigger is at 0.35–0.40 s rather than the usual auditory N1 at 0.1 s, and a lag scan of a backward envelope
model peaks at +0.25 s (single-trial r 0.079 vs 0.048 at 0 s). There may be a playback delay after the trigger. The
encoder is non-causal (field of about ±2 s) and absorbs a fixed delay, but fine timing measures should be read with it
in mind.

## Running

```bash
cd recovery_ds004940
bash scripts/run_vaediff.sh test        # unit tests + synthetic smoke run, about 15 s
bash scripts/run_vaediff.sh audio       # stage 1, no EEG, about 40 min
bash scripts/run_vaediff.sh eeg         # stage 2, about 60 min, weights tuned every 50 updates
bash scripts/run_vaediff.sh status      # latest tuning rounds
bash scripts/run_vaediff.sh evaluate    # test-set evaluation
```

`RUN=name` names the stage-2 run, `EXTRA="..."` adds flags (e.g. `EXTRA="--spread 0.5 --w-text 0.4"`). Logs:
`logs/vaediff_<stage>_<run>.log`.

Checkpoint selection: every 500 updates the EMA weights really sample 48 dev trials, and `best.pt` is the one whose
generated mel best identifies its own sentence among the dev sentences (rank percentile, frame partners not
candidates); never by L1 or diffusion loss.

## Evaluation

`evaluate` reports seen and unseen people separately, on test sentences only:

1. Latent retrieval (chunked voice/content, text): null = sentence labels shuffled within each person × task, 1000
   times; intervals = bootstrap over people and sentences.
2. Gate: audio is generated only if at least one retrieval beats its null (`--force-audio` overrides; the report says so).
3. Generation, same trial and same noise for every condition: real EEG; another sentence's EEG from the same person
   and task, closest in duration; zeroed EEG; unconditional; product of experts over all presentations of the sentence
   and its control (another sentence's product); the audio-code ceiling; the true mel through Vocos. Measures:
   identification rank, STOI, MCD, envelope r, timbre distance; paired differences with two-way bootstrap intervals.
4. Whisper small.en on 40 trials: WER, and the sentence-embedding distance between transcript and sentence.
5. Six listening examples per group.

Report: `outputs/vaediff/<run>/evaluate_best/report.md`.

## Differences from the September numbers

September: Active only, 17 people, sentence split (320/41/41, pairs not kept together), 4.35 s window (with the
prompt response), 256 Hz, train-fold normaliser, 16 kHz HiFi-GAN; the best envelope r 0.341 came from averaging 17
presentations. Now: both tasks, pair-wise split, 3.6 s window, held-out people, 24 kHz Vocos. The numbers are not
directly comparable; the paired "real EEG − mismatched EEG" differences inside one evaluation are.

## Not done, not verified

- Neither stage has been trained on real data, and the time per update has not been measured; 40 / 60 min are guesses.
- The thresholds (0.42, 0.45, 1.5× timbre) are intuition; reset them from the dev levels after the first run.
- Stage 1 has only 322 training sentences (about 15 min of audio); the decoder may memorise them, so read the audio
  ceiling on dev and test sentences.
- The text encoder MiniLM (`sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243`) was downloaded for the text
  loss, into `../models/all-MiniLM-L6-v2`.
