# EEG-to-Speech CCF-A + Key arXiv Paper References

This folder collects papers for the current EEG-to-speech / waveform reconstruction work in this repository.
The selection prioritizes CCF-A venues and keeps a small number of direct EEG-to-speech or audio-token foundation papers that are essential even when they are arXiv-only.

## Files

- `pdf/`: local PDF copies.
- `papers.csv`: structured index with venue/status, CCF-A flag, URL, local PDF path, and relevance note.
- `papers.bib`: BibTeX entries pointing at the local PDF files.
- `metadata/download_log.txt`: download and local-reuse log.
- `metadata/search_queries.md`: search directions, inclusion rule, and PDF source preference.

## Reading Order

### P0 direct EEG-to-speech

| Year | Venue | CCF-A | Paper | Why it matters |
| --- | --- | --- | --- | --- |
| 2026 | arXiv | No | [NeuroSonic: Conditional Flow Matching for EEG-to-Speech Reconstruction](pdf/2026_arXiv_NeuroSonic_Conditional_Flow_Matching_EEG_to_Speech.pdf) | Most direct current reference for EEG-conditioned speech reconstruction using conditional flow matching. |
| 2023 | arXiv | No | [Towards Voice Reconstruction from EEG during Imagined Speech](pdf/2023_arXiv_NeuroTalk_Voice_Reconstruction_from_EEG.pdf) | NeuroTalk baseline for imagined-speech EEG-to-voice reconstruction and domain adaptation from spoken EEG. |
| 2022 | arXiv / Nature Machine Intelligence-adjacent line | No | [Decoding speech perception from non-invasive brain recordings](pdf/2022_arXiv_Decoding_Speech_Perception_Non_Invasive_Brain_Recordings.pdf) | Core non-invasive M/EEG speech decoding reference; supports contrastive speech-representation retrieval baselines. |
| 2025 | arXiv | No | [EEG-to-Voice Decoding of Spoken and Imagined speech Using Non-Invasive EEG](pdf/2025_arXiv_EEG_to_Voice_Spoken_Imagined_Speech.pdf) | Direct open-loop EEG-to-mel/vocoder reconstruction for spoken and imagined speech; closest to the current FEIS/KaraOne evaluation style. |

### P0 audio codec/token target

| Year | Venue | CCF-A | Paper | Why it matters |
| --- | --- | --- | --- | --- |
| 2020 | NeurIPS | Yes | [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](pdf/2020_NeurIPS_wav2vec_2_0_Speech_Representations.pdf) | Canonical self-supervised speech representation target for EEG-to-speech semantic alignment. |
| 2023 | ICML | Yes | [Robust Speech Recognition via Large-Scale Weak Supervision](pdf/2023_ICML_Whisper_Robust_Speech_Recognition.pdf) | Robust speech encoder and ASR sanity-check reference for reconstructed wav intelligibility. |
| 2023 | ICML / arXiv | Yes | [BEATs: Audio Pre-Training with Acoustic Tokenizers](pdf/2023_ICML_BEATs_Audio_Pretraining_Acoustic_Tokenizers.pdf) | Acoustic tokenizer reference for non-speech and speech auditory token targets. |
| 2023 | NeurIPS | Yes | [High-Fidelity Audio Compression with Improved RVQGAN](pdf/2023_NeurIPS_Improved_RVQGAN_Audio_Compression.pdf) | High-fidelity neural audio codec baseline for codec-token waveform rendering. |
| 2024 | ICML | Yes | [NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models](pdf/2024_ICML_NaturalSpeech_3_Factorized_Codec_Diffusion.pdf) | Best factorized content/prosody/timbre/acoustic codec reference for EEG token decomposition. |
| 2025 | AAAI | Yes | [Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Model](pdf/2025_AAAI_X_Codec_Semantic_Audio_Codec.pdf) | Semantic-enhanced codec reference; useful for bridging EEG semantic targets and acoustic tokens. |

### P1 generative speech decoder

| Year | Venue | CCF-A | Paper | Why it matters |
| --- | --- | --- | --- | --- |
| 2023 | NeurIPS | Yes | [Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale](pdf/2023_NeurIPS_Voicebox_Universal_Speech_Generation.pdf) | Flow-matching speech generation reference for using partial EEG-derived conditions to realize speech. |
| 2023 | NeurIPS | Yes | [StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models](pdf/2023_NeurIPS_StyleTTS_2_Style_Diffusion.pdf) | Style/prosody diffusion decoder reference for voice realization from high-level conditions. |
| 2023 | NeurIPS | Yes | [DASpeech: Directed Acyclic Transformer for Fast and High-quality Speech-to-Speech Translation](pdf/2023_NeurIPS_DASpeech_Two_Stage_Speech_Decoder.pdf) | Two-stage linguistic-to-acoustic decoding design, useful when EEG provides incomplete high-level speech structure. |
| 2023 | NeurIPS | Yes | [P-Flow: A Fast and Data-Efficient Zero-Shot TTS through Speech Prompting](pdf/2023_NeurIPS_P_Flow_Zero_Shot_TTS.pdf) | Prompt-conditioned flow decoder reference for fast voice-conditioned waveform realization. |
| 2023 | ACM MM | Yes | [CoMoSpeech: One-Step Speech and Singing Voice Synthesis via Consistency Model](pdf/2023_ACMMM_CoMoSpeech_Consistency_Speech_Synthesis.pdf) | Consistency-model speech synthesis reference for low-step generation from acoustic conditions. |
| 2024 | ICML | Yes | [UniAudio: Towards Universal Audio Generation with Large Language Models](pdf/2024_ICML_UniAudio_Universal_Audio_Generation.pdf) | Universal audio-token generation reference for downstream audio decoder design. |
| 2024 | NeurIPS | Yes | [UniAudio 1.5: Large Language Model-driven Audio Codec is A Few-shot Audio Task Learner](pdf/2024_NeurIPS_UniAudio_1_5_LLM_Driven_Audio_Codec.pdf) | LLM-driven audio codec reference for treating audio tokens as a language interface. |

### P2 background/foundation

| Year | Venue | CCF-A | Paper | Why it matters |
| --- | --- | --- | --- | --- |
| 2022 | arXiv | No | [AudioLM: a Language Modeling Approach to Audio Generation](pdf/2022_arXiv_AudioLM_Language_Modeling_Audio_Generation.pdf) | Foundational semantic-to-acoustic token LM; useful for EEG-derived token completion. |
| 2023 | arXiv | No | [SoundStorm: Efficient Parallel Audio Generation](pdf/2023_arXiv_SoundStorm_Parallel_Audio_Generation.pdf) | Parallel codec-token generation reference for faster decoder completion. |
| 2023 | arXiv | No | [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](pdf/2023_arXiv_VALL_E_Neural_Codec_Language_Model.pdf) | Canonical neural codec LM for prompt-based voice generation. |
| 2024 | arXiv | No | [VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers](pdf/2024_arXiv_VALL_E_2_Human_Parity_Zero_Shot_TTS.pdf) | Improved grouped codec modeling reference for long sequence stability. |
| 2022 | OpenReview / arXiv | No | [High Fidelity Neural Audio Compression](pdf/2022_arXiv_EnCodec_High_Fidelity_Neural_Audio_Compression.pdf) | Practical neural codec backend used by VALL-E/VoiceCraft-style token decoders. |
| 2023 | arXiv | No | [SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models](pdf/2023_arXiv_SpeechTokenizer_Unified_Speech_Tokenizer.pdf) | Hierarchical semantic/acoustic speech tokenization; close to grouped EEG-token alignment. |
| 2024 | arXiv | No | [MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer](pdf/2024_arXiv_MaskGCT_Masked_Generative_Codec_Transformer.pdf) | Two-stage semantic-to-acoustic token generation reference for incomplete EEG-derived conditions. |
| 2024 | arXiv | No | [Moshi/Mimi real-time speech foundation model](pdf/2024_arXiv_Moshi_Mimi_Real_Time_Speech_Foundation_Model.pdf) | Streaming speech-text/audio-token foundation reference for future real-time EEG voice decoding. |

## Notes

- `CCF-A=Yes` means the paper is in the planned CCF-A core set or attached to a CCF-A venue/status used by the project reading list.
- Direct EEG-to-speech papers are retained even when arXiv-only because they are the closest methodological references for KaraOne/FEIS reconstruction.
- If this ignored folder needs to be committed, use `git add -f paper-ref/eeg-to-speech-ccf-a`.
