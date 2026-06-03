# thinking_waveform_pairs Data Description

## Overview

This document describes the processed dataset under:

`/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/data/processed/thinking_waveform_pairs`

The folder contains stage-level paired EEG segments and waveform targets for two datasets:

- `feis`
- `karaone`

The main purpose of this processed dataset is to support simple end-to-end experiments such as:

- `thinking EEG -> discrete EEG token -> waveform reconstruction`
- `thinking EEG -> prompt/audio prototype retrieval`
- `thinking vs overt/stimulus transfer learning`

## Folder Layout

```text
thinking_waveform_pairs/
  summary.json
  feis/
    manifest.json
    trials.csv
    segments.csv
    subjects/
      *.npz
    audio/
      ...
  karaone/
    manifest.json
    trials.csv
    segments.csv
    subjects/
      *.npz
    audio/
      ...
```

## Dataset-Level Summary

### FEIS

- subject count: `21`
- trial count: `3312`
- segment count: `16560`
- exported stages:
  - `stimuli`
  - `articulators`
  - `thinking`
  - `speaking`
  - `resting`
- hearing-equivalent stage: `stimuli`
- audio pairing: `subject-level canonical wav for the same prompt label`
- target EEG sampling rate: `256 Hz`
- target audio sampling rate: `16000 Hz`
- preprocessing:
  - bandpass: `1-40 Hz`
  - notch: `50 Hz`
  - reference: common average reference
  - baseline normalization: each phase is normalized against the same trial's `resting` window

### KaraOne

- subject count: `14`
- trial count: `1913`
- segment count: `7652`
- exported stages:
  - `clearing`
  - `stimulus_like`
  - `thinking`
  - `overt_like`
- hearing-equivalent stage: `stimulus_like`
- audio pairing: `same-trial overt wav`
- target EEG sampling rate: `256 Hz`
- target audio sampling rate: `16000 Hz`
- preprocessing:
  - bandpass: `1-40 Hz`
  - notch: `60 Hz`
  - reference: common average reference
  - baseline normalization: each stage is normalized against the same trial's `clearing` window

## EEG Data

## Storage Format

EEG is stored subject-wise in:

- `feis/subjects/*.npz`
- `karaone/subjects/*.npz`

Each `.npz` file contains:

- `trial_indices`
- `labels`
- `audio_relpaths`
- `channel_names`
- `eeg_sfreq_hz`
- `stage_names`
- stage arrays such as:
  - `stage__thinking`
  - `stage__speaking`
  - `stage__stimuli`
  - `stage__resting`
  - `stage__clearing`
  - `stage__stimulus_like`
  - `stage__overt_like`
- per-stage valid lengths:
  - `stage__thinking__valid_lengths`
  - etc.
- KaraOne additionally stores source ranges:
  - `stage__thinking__src_ranges`
  - etc.

## EEG Tensor Meaning

For each stage array:

- shape is approximately `[num_trials, num_channels, num_samples]`
- values are processed EEG after filtering, rereferencing, resampling, and baseline normalization

### FEIS EEG Shape

FEIS is the more regular dataset.

- channel count: `14`
- sampling rate: `256 Hz`
- `thinking` length: `1280` samples
- duration per `thinking` segment: `5.0 s`

Other FEIS stages are also stored as fixed-length arrays.

## What The FEIS `5 s` Data Means

The FEIS `5 s` quantity refers to the EEG stage window, not to the wav duration.

More specifically:

- `stage__thinking` in FEIS is a fixed `5.0 s` imagined-speech EEG segment
- at `256 Hz`, this becomes `1280` EEG samples per trial
- this window is the neural segment used to represent the subject's imagined pronunciation of the prompt

So the FEIS pair should be interpreted as:

- EEG side: a fixed `5 s` imagined-speech window
- wav side: a short canonical prompt recording for the same `subject + label`

This processed FEIS set is therefore not:

- `5 s EEG -> 5 s matched same-trial audio`

It is closer to:

- `5 s imagined EEG epoch -> prompt-level reference wav`

This distinction is important for modeling:

- the EEG window is long and regular, which is good for batching and tokenization
- the wav target is usually short, which is good for a first reconstruction demo
- the task is better viewed as prompt-conditioned waveform prototype reconstruction than as full natural speech synthesis

### KaraOne EEG Shape

KaraOne is less uniform than FEIS.

- channel count: typically `62` after dropping non-EEG channels
- sampling rate after processing: `256 Hz`
- stage lengths are padded/cropped to a robust target length per stage
- source timing information is preserved in `__src_ranges`

This means KaraOne is still easy to load, but its stage definitions are more heterogeneous than FEIS.

## WAV Data

Waveform targets are stored under:

- `feis/audio/`
- `karaone/audio/`

All stored target audio is:

- mono
- resampled to `16 kHz`
- saved as `.wav`

## WAV Duration Summary

The stored wav files are not fixed-length at this preprocessing stage.

### FEIS wav duration

- minimum: `1.00 s`
- maximum: `5.00 s`
- median: `1.00 s`
- mean: `1.19 s`

Interpretation:

- most FEIS wav files are short prompt recordings
- many are around `1 second`
- the wav duration should not be confused with the `5 s` EEG thinking window

### KaraOne wav duration

- minimum: `0.768 s`
- maximum: `7.52 s`
- median: `1.44 s`
- mean: `1.471 s`

Interpretation:

- KaraOne wav duration is more variable than FEIS
- later modeling will usually need an additional fixed-length crop/pad step

## FEIS WAV Meaning

In FEIS, the waveform target is not a trial-unique recording for each EEG segment.

Instead:

- the target wav is the canonical waveform for the same `subject + label`
- multiple trials with the same label can point to the same wav file

Example from `trials.csv`:

- `feis,01,0,goose,audio/01/goose.wav,subjects/01.npz`

So FEIS is best understood as:

- `thinking EEG -> prompt-level canonical waveform target`

This is well suited for a simple first demo, because the supervision is stable and regular.

## KaraOne WAV Meaning

In KaraOne, the waveform target is closer to a trial-level paired recording.

- each trial points to `audio/{subject}/{trial_index}.wav`
- this wav is the same-trial overt speech recording

Example from `trials.csv`:

- `karaone,MM05,0,/uw/,audio/MM05/000.wav,subjects/MM05.npz`

So KaraOne is better understood as:

- `thinking EEG -> same-trial overt waveform target`

This is conceptually stronger pairing, but the EEG and timing structure are less uniform than FEIS.

## Metadata Tables

## `trials.csv`

Each row represents one trial and provides the main EEG-to-audio pairing.

Columns:

- `dataset`
- `subject_id`
- `trial_index`
- `label`
- `audio_path`
- `eeg_subject_bundle`

This table is the simplest entry point if you want one EEG trial paired with one target wav.

## `segments.csv`

Each row represents one `trial x stage` segment.

Important columns:

- `dataset`
- `subject_id`
- `trial_index`
- `segment_stage`
- `segment_role`
- `segment_array_key`
- `label`
- `audio_path`
- `eeg_subject_bundle`
- `baseline_window`
- `eeg_num_channels`
- `eeg_num_samples`
- `eeg_valid_num_samples`
- `eeg_sfreq_hz`
- `audio_sfreq_hz`

KaraOne also includes:

- `segment_start_sample_src`
- `segment_end_sample_src`
- `baseline_start_sample_src`
- `baseline_end_sample_src`

This table is the best entry point if you want to filter only:

- `thinking`
- `speaking`
- `stimuli`
- `stimulus_like`

## Recommended Usage for a Minimal EEG-to-WAV Demo

If the goal is the simplest first reconstruction demo, the recommended subset is:

- dataset: `FEIS`
- stage: `thinking`

Reasons:

- fixed `14 x 1280` EEG shape
- fixed `256 Hz` EEG sampling rate
- regular stage segmentation
- smaller prompt vocabulary
- stable waveform supervision via canonical wavs

Recommended loading logic:

1. Read `feis/segments.csv`
2. Filter rows where `segment_stage == thinking`
3. Open the corresponding `subjects/{subject}.npz`
4. Read EEG from `stage__thinking[trial_index]`
5. Read wav from `audio_path`

## Important Interpretation Notes

- FEIS and KaraOne do not provide the same type of EEG-to-audio supervision.
- FEIS uses label-level canonical wav pairing.
- KaraOne uses same-trial overt wav pairing.
- FEIS `5 s` refers to the imagined-speech EEG window, not to a `5 s` wav target.
- Therefore, results from the two datasets should not be interpreted identically.
- FEIS is better for a simple prompt-conditioned waveform reconstruction demo.
- KaraOne is better for later validation of tighter trial-level EEG-audio pairing.

## Suggested Citation / Internal Description

If you need a short description in a proposal or note, you can describe this processed dataset as:

> A stage-structured imagined-speech EEG and waveform pairing resource derived from FEIS and KaraOne, with subject-wise EEG bundles, stage-level indexing tables, and 16 kHz waveform targets for simple end-to-end EEG-to-waveform reconstruction experiments.
