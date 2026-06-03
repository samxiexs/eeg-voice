# EEG2Wave Demo Bundle

Minimal FEIS imagined-speech demo:

- input: `FEIS thinking` EEG
- bottleneck: discrete VQ tokens
- output: fixed-length waveform

This bundle is aligned to the processed local dataset described in:

- `/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_modeling_prompt.md`
- [reports/preprocessing/thinking_waveform_pairs_data_description.md](/Users/samxie/.codex/worktrees/02d3/speech_decoding/reports/preprocessing/thinking_waveform_pairs_data_description.md)

## What This Demo Assumes

- dataset: `FEIS`
- stage: `thinking`
- per-subject training
- EEG shape: `[14, 1280]`
- EEG duration: `5 s`
- audio target: canonical prompt wav
- audio output length: `1.5 s` at `16 kHz` = `24000` samples

Important:

- the `5 s` in FEIS refers to the imagined-speech EEG window
- it does not mean the target wav is `5 s`
- FEIS target wavs are prompt-level canonical references, usually around `1 s`

## Layout

```text
eeg2wave_demo_bundle/
  README.md
  requirements.txt
  config.yaml
  dataset.py
  model.py
  losses.py
  utils.py
  train.py
  infer.py
  prepare_local_bundle.sh
```

This folder is intended to become a self-contained upload bundle for cloud/server training.

If you want a single folder with data included, run:

```bash
bash eeg2wave_demo_bundle/prepare_local_bundle.sh
```

That copies the real processed FEIS data into:

```text
eeg2wave_demo_bundle/data/feis/
```

## Install

```bash
python -m pip install -r eeg2wave_demo_bundle/requirements.txt
```

## Train One Subject

```bash
python eeg2wave_demo_bundle/train.py --subject 01
```

## Run Inference

```bash
python eeg2wave_demo_bundle/infer.py --subject 01
```

## Default Data Root

The code defaults to:

`eeg2wave_demo_bundle/data/feis`

So after data is copied in, this bundle is self-contained.

## Output Location

Training and inference outputs are written to:

`eeg2wave_demo_bundle/outputs/`

## Expected Cloud Upload State

Before uploading to a server, this bundle should contain:

```text
eeg2wave_demo_bundle/
  data/
    feis/
      manifest.json
      trials.csv
      segments.csv
      subjects/
      audio/
  outputs/
  README.md
  config.yaml
  dataset.py
  infer.py
  losses.py
  model.py
  prepare_local_bundle.sh
  requirements.txt
  train.py
  utils.py
```
