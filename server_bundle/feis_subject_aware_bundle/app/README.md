# EEG2Wave Demo Bundle

最小化 FEIS EEG-to-waveform 对照 demo，加上新的 subject-aware EEG-to-speech alignment 路径：

- input: `FEIS thinking` or `FEIS stimuli` EEG
- waveform baseline: discrete VQ tokens -> fixed-length waveform
- alignment path: EEG -> speech embedding + prosody -> downstream retrieval/demo

For the full run guide, tuning notes, and training-objective explanation, see:

- `RUN_GUIDE.md`

This bundle is aligned to the processed local dataset described in:

- `/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_modeling_prompt.md`
- [reports/preprocessing/thinking_waveform_pairs_data_description.md](/Users/samxie/.codex/worktrees/02d3/speech_decoding/reports/preprocessing/thinking_waveform_pairs_data_description.md)

## What This Demo Assumes

- dataset: `FEIS`
- stage: `thinking` by default, with `stimuli` also supported
- per-subject training
- EEG shape: `[14, 1280]`
- EEG duration: `5 s`
- audio target: canonical prompt wav
- audio output length: `1.0 s` at `16 kHz` = `16000` samples

Important:

- the `5 s` in FEIS refers to the imagined-speech EEG window
- it does not mean the target wav is `5 s`
- FEIS target wavs are prompt-level canonical references, usually around `1 s`

## Layout

```text
eeg2wave_demo_bundle/
  README.md
  RUN_GUIDE.md
  requirements.txt
  configs/
    config.yaml
    alignment.yaml
    alignment_ssl_local.yaml
    waveform_protocol.yaml
  src/
    dataset.py
    model.py
    losses.py
    utils.py
  scripts/
    train.py
    infer.py
    extract_audio_targets.py
    train_alignment.py
    eval_alignment.py
    eval_alignment_retrieval.py
    analyze_alignment_space.py
    report_phase2.py
    train_waveform_protocol.py
    eval_waveform_protocol.py
    prepare_server_bundle.py
    prepare_local_bundle.sh
```

This folder is intended to become a self-contained upload bundle for cloud/server training.

If you want a single folder with data included, run:

```bash
bash eeg2wave_demo_bundle/scripts/prepare_local_bundle.sh
```

That copies the real processed FEIS data into:

```text
eeg2wave_demo_bundle/data/feis/
```

## Install

```bash
python -m pip install -r eeg2wave_demo_bundle/requirements.txt
```

## Local HuBERT / Wav2Vec2

The alignment path supports any local Hugging Face-compatible HuBERT or Wav2Vec2 directory.

Point the config field below at a local snapshot directory:

```yaml
targets:
  backend: ssl_local
  ssl_model_name_or_path: /absolute/path/to/local/model_dir
  local_files_only: true
```

That local model directory should contain at least:

- `config.json`
- `preprocessor_config.json`
- one of `model.safetensors` or `pytorch_model.bin`

Then extract SSL targets with:

```bash
python eeg2wave_demo_bundle/scripts/extract_audio_targets.py --config eeg2wave_demo_bundle/configs/alignment.yaml --backend ssl_local
```

## Phase 2 Retrieval Evaluation

Run the new retrieval-waveform evaluation with the SSL alignment config:

```bash
python server_bundle/feis_subject_aware_bundle/app/scripts/eval_alignment_retrieval.py \
  --config server_bundle/feis_subject_aware_bundle/app/configs/alignment_ssl_local.yaml \
  --protocol G \
  --split test
```

This writes:

- per-trial retrieved wavs
- top-5 retrieval JSON
- predicted embedding cache
- protocol-aware metrics including exact/label top-k and waveform-space NTA

For template-space structure analysis:

```bash
python server_bundle/feis_subject_aware_bundle/app/scripts/analyze_alignment_space.py \
  --config server_bundle/feis_subject_aware_bundle/app/configs/alignment_ssl_local.yaml
```

To consolidate audit + retrieval + space analysis into one markdown report:

```bash
python server_bundle/feis_subject_aware_bundle/app/scripts/report_phase2.py \
  --config server_bundle/feis_subject_aware_bundle/app/configs/alignment_ssl_local.yaml \
  --protocol G \
  --split test
```

## Train One Subject

```bash
python eeg2wave_demo_bundle/scripts/train.py --subject 01
```

## Compare `thinking` And `stimuli`

```bash
bash eeg2wave_demo_bundle/scripts/run_stage_compare.sh
```

## Run Inference

```bash
python eeg2wave_demo_bundle/scripts/infer.py --subject 01
```

## Prepare A Single Upload Folder

To build one server-transfer folder containing code, processed FEIS data, reports, outputs, and an optional local HuBERT/Wav2Vec2 model:

```bash
python eeg2wave_demo_bundle/scripts/prepare_server_bundle.py --clean
```

If you already have a local SSL model snapshot:

```bash
python eeg2wave_demo_bundle/scripts/prepare_server_bundle.py \
  --clean \
  --ssl-model-dir /absolute/path/to/local/hubert_or_wav2vec2_dir \
  --ssl-model-name hubert-base-ls960
```

This creates:

```text
server_bundle/feis_subject_aware_bundle/
  app/
  data/feis/
  models/
  artifacts/
  reports/
```

## Default Data Root

The code defaults to:

`eeg2wave_demo_bundle/data/feis`

So after data is copied in, this bundle is self-contained.

## Output Location

Training and inference outputs are written to:

`eeg2wave_demo_bundle/outputs-thinking/`

or

`eeg2wave_demo_bundle/outputs-stimuli/`

The code now separates outputs by stage by default, so `thinking` and `stimuli` do not share the same output folder.

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
  RUN_GUIDE.md
  configs/
    config.yaml
  src/
  scripts/
  requirements.txt
```
