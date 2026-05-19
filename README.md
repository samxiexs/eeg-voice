# EEG-Voice Speech Decoding

This repository is a research codebase for building an EEG-to-voice token foundation model. The current V1 scope is not waveform generation. The first objective is to learn discrete EEG tokens that align with speech and voice attributes, then use those tokens for voice or speaker retrieval.

```text
EEG -> grouped discrete token
    -> content / pitch / timbre / speaker / style / mode alignment
    -> voice / speaker retrieval
```

The project is organized around a multi-dataset EEG and speech catalog. Public datasets are treated as selected research data if they are included in the catalog and are publicly obtainable or requestable. Local download status is tracked separately and does not define whether a dataset belongs to the research pool.

## Current Status

The V1 model skeleton has been implemented. It can run synthetic batches through the tokenizer, grouped RVQ, alignment heads, speaking-mode head, retrieval head, and reconstruction losses.

The real-data training system is not finished yet. Dataset registry, real-data collators, target extraction, samplers, training scripts, and evaluation scripts are the next engineering layer.

| Area | Status |
| --- | --- |
| V1 model design | Implemented in documentation |
| `EEGVoiceTokenV1` code skeleton | Implemented |
| Hierarchical grouped RVQ | Implemented |
| q7 weak residual policy | Implemented |
| Speaking-mode dataset adapter | Implemented |
| Acquisition device context | Implemented |
| Memory queue retrieval negatives | Implemented |
| Synthetic model tests | Passing |
| Real selected-dataset training | Not connected yet |

## Research Boundary

V1 is designed to answer whether EEG can support a stable speech and voice token interface:

- Can continuous EEG be compressed into discrete tokens with reasonable codebook usage?
- Do the tokens carry readable content, pitch, prosody, timbre, speaker, style, and mode information?
- Can EEG-derived voice tokens retrieve matching audio, speaker, or stream embeddings?

V1 deliberately does not claim personalized subjective voice-image reconstruction. That target would require a unified voice bank, same-subject subjective similarity ratings, and controlled F0/formant/style manipulation. The current public data pool is appropriate for token learning, attribute alignment, and retrieval, not for final personalized perceptual voice reconstruction.

## English-First Data Policy

The first experimental chain is English-first. Cross-lingual datasets are kept for transfer and robustness analysis, rather than mixed into the first main conclusion.

| Data layer | Role |
| --- | --- |
| English-first core | Main tokenizer, attribute alignment, and retrieval training |
| English / near-English retrieval expansion | Attention stream and speaker retrieval robustness |
| Cross-lingual reserved | Later transfer tests for Mandarin, Cantonese, Spanish, Dutch, Danish, and related data |
| Auditory proxy | Auxiliary auditory pretraining and ablation |

The detailed selected dataset catalog is in [`docs/multi_dataset_voice_eeg_catalog_0518.md`](docs/multi_dataset_voice_eeg_catalog_0518.md).

## Model V1

The V1 model is named `EEGVoiceTokenV1`.

```text
EEG
-> preprocessing / montage normalization
-> acquisition device context
-> sensor-aware temporal encoder
-> latent token former
-> hierarchical grouped RVQ
-> alignment heads
-> retrieval embedding space
```

The grouped RVQ uses eight quantizer levels:

| Quantizer | Group | Role |
| --- | --- | --- |
| q0-q1 | `base` | onset, envelope, shared auditory response |
| q2-q3 | `content` | phoneme, syllable, word, speech unit |
| q4 | `prosody` | F0, intensity, rhythm, prosody |
| q5-q6 | `voice` | timbre, speaker, style, stream identity |
| q7 | `residual` | weak reconstruction residual and dataset nuisance |

Head routing is fixed:

| Head | Token groups |
| --- | --- |
| Content / phoneme | `base + content` |
| Pitch / prosody | `base + prosody` |
| Timbre / style / retrieval | `base + voice` |
| Speaking mode | `base + content + prosody + voice` |
| Aligned reconstruction | q0-q6 |
| Full reconstruction | q0-q7 |

q7 does not enter alignment, retrieval, or speaking-mode heads. It only participates in low-weight full reconstruction.

Device information is handled separately from q7. `acquisition_device_id`, `montage_id`, `reference_id`, `sampling_rate_hz`, and `native_channel_count` are embedded as recording-level acquisition context. This context conditions the sensor representation and latent token former, but it is not used as a retrieval target or an attribute label.

## Repository Layout

```text
configs/
  model_v1.yaml                  # V1 default model and data-layer config

docs/
  multi_dataset_voice_eeg_catalog_0518.md
  model_v1_design_0518.md
  model_v1_development_status_0518.md
  assets/                        # V1 architecture and routing figures

paper-ref/
  Reference papers used for model and dataset design

scripts/
  Dataset probing, sample download, derivative building, and visualization scripts

src/eeg_voice_model/
  tokenizer.py                   # EEGVoiceTokenizerV1 and grouped RVQ
  voice_model.py                  # EEGVoiceTokenV1 and batch/target schemas
  heads.py                        # Alignment, mode, and retrieval heads
  losses.py                       # Reconstruction, retrieval, and token metrics
  builders.py                     # Config-to-model construction
  modules.py                      # Encoder, latent aggregator, decoder blocks

tests/
  test_model_v1_synthetic.py      # Synthetic V1 forward and builder tests
```

Local raw data, derived arrays, checkpoints, and downloaded audio or EEG files are ignored by git.

## Quick Start

The repository currently has no package installer or pinned environment file. For the synthetic V1 tests, the minimum practical dependencies are Python, PyTorch, and pytest. The real-data path will additionally need MNE-Python, NumPy, pandas, SciPy, and torchaudio.

Run the current verification:

```bash
python3 -m py_compile src/eeg_voice_model/*.py
PYTHONPATH=. python3 -m pytest -q
git diff --check
```

Build the V1 model from config:

```python
from src.eeg_voice_model.builders import build_eeg_voice_token_v1

model = build_eeg_voice_token_v1("configs/model_v1.yaml")
print(type(model).__name__)
```

Expected model name:

```text
EEGVoiceTokenV1
```

## Main Documents

| Document | Purpose |
| --- | --- |
| [`docs/multi_dataset_voice_eeg_catalog_0518.md`](docs/multi_dataset_voice_eeg_catalog_0518.md) | Selected EEG-voice dataset catalog and availability interpretation |
| [`docs/model_v1_design_0518.md`](docs/model_v1_design_0518.md) | Full V1 model design, data-to-loss mapping, RVQ policy, and future interfaces |
| [`docs/model_v1_development_status_0518.md`](docs/model_v1_development_status_0518.md) | Current implementation status and next engineering steps |

## Current Engineering Gaps

The next phase should turn the model skeleton into a real selected-dataset training system:

1. Build a `DatasetRegistry` for the English-first core datasets.
2. Implement a real `EEGVoiceBatch` collator for local sample folders and derived EEG/audio files, including device, montage, reference, sampling-rate, and channel-count metadata.
3. Extract content, phoneme, F0, prosody, style, speaker, and audio embeddings into a unified target schema.
4. Add English-first mixed batching and retrieval hard-negative sampling.
5. Add smoke training on one or two real examples.
6. Add evaluation scripts for Recall@K, phoneme accuracy, pitch correlation, token usage, q7 ablation, and q7 dataset predictability.
7. Add seen-device and held-out-device splits to test whether device context improves cross-device transfer instead of creating a shortcut.

## Data Handling

Large EEG, audio, archive, model, and derivative files should stay outside git. The repository keeps catalog and metadata reports in text form, while raw data and local samples are expected under ignored directories such as:

```text
data/
datasets/
downloads/
openneuro/
zenodo/
derived/
checkpoints/
```

For selected datasets, the catalog is the source of truth for research inclusion. Local folders only indicate download or conversion progress.
