# EEG-Voice

Research workspace for decoding speech from scalp EEG. The goal is imagined speech; the current work establishes what perceived-speech EEG supports first, because that is where paired EEG and audio exist at scale.

## What is here

```
eeg2wave_server_bundle/
  eeg-recon-0809/                        active project: EEG → speech on DS004940, KaraOne baselines
  eeg-recon-0809_explore_8h_v1_backup/   read-only snapshot of the earlier MFCC / Griffin-Lim pipeline
  generate_all_waveform_comparisons.py   standalone waveform-comparison helpers from an earlier bundle
  waveform_compare_utils.py
paper-ref/                               literature notes, reading lists, BibTeX and manifests (PDFs stay local)
docs/                                    local working documents: reviews, protocols, design notes, talk notes (not tracked)
requirements.txt                         server-side Python requirements
```

The code lives in `eeg2wave_server_bundle/eeg-recon-0809`; start with its [README](eeg2wave_server_bundle/eeg-recon-0809/README.md). Raw EEG, audio, model weights, caches and generated outputs are kept outside git.

## Current pipeline

```
EEG (128 ch, 256 Hz, 4.6 s window)
  → recovery-v3 encoder, trained with a time-resolved CLIP loss against HuBERT features and a mel loss through a frozen audio decoder
  → 18-channel conditioning (encoder head output on the teacher PCA basis, predicted duration, predicted envelope)
  → conditional diffusion decoder over the 80 × 251 SpeechT5 mel, classifier-free guidance
  → HiFi-GAN → 16 kHz waveform
```

Audio-side models (HuBERT teacher, mel decoder, vocoder) are fine-tuned on the training-fold audio only and provide the reconstruction ceilings. Every result is a paired comparison against zero-EEG, wrong-trial and time-shuffled controls on the same trial and noise seed.

## Datasets

| Dataset | Content | Use |
|---|---|---|
| OpenNeuro ds004940 | 128-channel EEG, English sentence listening (N400 paradigm); 17 participants, 402 sentences, 6,641 trials in the default configuration | encoder and decoder training, all reported numbers |
| KaraOne (Zhao & Rudzicz 2015) | imagined and spoken prompts, 11 classes, 14 participants | the target task; baselines and transfer tests |
| Broderick et al. 2018 | audiobook listening, 19 participants | optional envelope-tracking pre-training |

## Status

On held-out sentences the diffusion decoder produces samples with the spectral texture of speech, and the real-EEG sample is measurably closer to its own sentence than every control (STOI +0.032 [+0.018, +0.047] and envelope correlation +0.057 [+0.029, +0.083] against wrong-trial EEG; 2AFC 0.59 versus 0.45–0.54 for the controls). Pooling the EEG of all participants who heard a sentence raises the 2AFC to 0.66–0.84, and pooling a different sentence scores at chance. The EEG determines rhythm, duration and envelope; word content is not recovered. On KaraOne, imagined-speech decoding is at chance under block-wise cross-validation and the DS004940 encoder does not transfer.

The full account (data, splits, architectures, tensor shapes, training curves, result tables) is in [`eeg2wave_server_bundle/eeg-recon-0809/reports/updated-results_2026-09-19.md`](eeg2wave_server_bundle/eeg-recon-0809/reports/updated-results_2026-09-19.md).

## Environment

The project is developed on macOS (Apple Silicon, MPS) in a conda environment `eegvoice` (Python 3.12) with a project venv created by `bash app/run_aligned_local.sh setup` inside `eeg-recon-0809`. `requirements.txt` at this level lists the equivalent packages for a CUDA server; install the CUDA-matched PyTorch build first.

## Literature

`paper-ref/` collects the reading behind the design: EEG-to-speech decoding (CCF-A venues), audio decoders and vocoders, factorised speech representations, visual decoding from EEG embeddings, and the dataset papers. Each folder has a README or manifest; `deep-research-report.md` and `eeg_speech_factorized_decoding_literature_review_20260726.md` are the two summaries.
