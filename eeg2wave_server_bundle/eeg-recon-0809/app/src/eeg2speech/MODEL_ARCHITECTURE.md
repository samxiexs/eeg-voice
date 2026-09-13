# DS004940 + DS006104 Joint EEG-to-Speech Model

This folder implements a **non-autoregressive EEG-to-speech-content model**.
Its primary output is a fixed-length MFCC content representation, not raw
waveform samples or codec tokens.  The architecture is deliberately designed
so that DS004940 (128-channel BioSemi EEG) and DS006104 (61-channel 10--20
EEG) share a content model without pretending that their channel names or
trials are identical.

## Main modules

| Module | Actual implementation | Simple role | Shared across datasets? |
|---|---|---|---|
| Dataset adapter | `DatasetInputAdapter`: dataset-indexed scalar gain and bias | Adjusts overall EEG scale/offset differences before common processing. It does not inject subject, label, or audio content. | No; two small dataset-specific affine parameter sets. |
| Temporal stem | Three multiscale 1D CNN branches | Extracts short-, medium-, and longer-timescale EEG patterns independently for every channel. | Yes |
| Coordinate encoder | xyz-coordinate MLP | Turns each electrode's 3D location into a learnable spatial representation. | Yes |
| Channel fusion | Learned channel-attention score + coordinate/temporal fusion | Combines the available channels without requiring a fixed channel-name intersection. | Yes |
| Sequence backbone | Conformer stack | Models temporal relationships between the fused EEG tokens. | Yes |
| Local content head | Second Conformer stack | Produces local EEG content tokens used for time-resolved speech alignment. | Yes |
| Global content head | pooled token MLP with L2 normalization | Produces one utterance-level EEG embedding for content retrieval/contrastive learning. | Yes |
| V3 frozen content teacher | train-fold HuBERT-global whitening/PCA plus a complete content bank | Supplies a fixed 64-D target and negatives for EEG content retrieval. Audio teacher weights and validation/test content are never fitted. | DS004940 v3 |
| MFCC head | linear projection to `39 × 161` | Predicts the main speech-content target: 39 MFCC coefficients over 161 relative-time frames. | Yes |
| V3 duration head | standardized log-duration residual | Predicts native SpeechT5 mel length relative to train-fold duration statistics; zero EEG returns the train-fold mean. | DS004940 v3 |
| Phoneme auxiliary head | linear classifier from pooled EEG | Lets DS006104 single-phoneme label-only trials provide a low-weight auxiliary signal. | Yes |
| Audio teacher | cached frozen HuBERT local/global representations | Provides speech-content alignment targets; HuBERT is not updated by EEG training. | Shared target space |
| Native acoustic renderer | `DurationConditionedNativeRenderer`: 1D CNN | Expands relative-time MFCC plus predicted duration into native-duration SpeechT5 mel. | Audio-only, train-fold fitted |
| Optional diffusion refiner | `ConditionalMelDiffusion`: conditional 1D DDPM with DDIM sampling | Legacy mode is audio-only. V3 qualitative mode may receive a frozen EEG global latent, but never labels/text/audio IDs; random listening samples are excluded from EEG metrics. | Switchable |
| Neural vocoder | frozen local SpeechT5 HiFi-GAN | Converts native SpeechT5 mel to a waveform without Griffin--Lim phase reconstruction. | Post-training only |

## Training losses

| Loss / signal | Meaning | When used |
|---|---|---|
| MFCC Smooth-L1 | Direct error between EEG-predicted and audio-target MFCC. | Audio-content-supervised pairs. |
| MFCC delta loss | Preserves changes over relative time rather than only average MFCC level. | Audio-content-supervised pairs. |
| Local alignment | Aligns EEG local tokens with cached HuBERT local content tokens. | Audio-content-supervised pairs with HuBERT targets. |
| Global CLIP-style contrastive loss | Makes matching EEG/audio utterance embeddings more similar than mismatched pairs. | Audio-content-supervised pairs. |
| V3 teacher-bank InfoNCE | Compares each EEG global latent with every train-fold content teacher, even when the physical batch has only two contents. | DS004940 v3 only. |
| V3 standardized residual / covariance | Matches target-relative MFCC residual direction and variation instead of rewarding the dataset-mean template. | DS004940 v3 only. |
| V3 duration Huber | Learns standardized log-duration rather than an unconstrained positive frame count. | DS004940 v3 only. |
| Phoneme auxiliary loss | Cross-entropy phoneme label prediction. | DS006104 `single-phoneme` label-only trials; it has no MFCC/HuBERT loss. |

`global_clip` in training logs means an **InfoNCE / CLIP-style contrastive
objective**, not an imported OpenAI CLIP image-text model.

## What is *not* in this model

| Component | Used? | Explanation |
|---|---:|---|
| Autoregressive decoder | No | The system predicts all 161 MFCC frames in parallel. It does not generate one token/sample after another. |
| EEG-to-waveform regression | No | Waveform samples are never the primary supervised target. |
| EnCodec / RVQ code prediction | No | Codec-token inversion from the reference research route is intentionally excluded from this pilot. |
| Autoregressive waveform decoder | No | The SpeechT5 HiFi-GAN and optional diffusion refiner are non-autoregressive post-processing modules. |
| Waveform-domain diffusion | No | The optional diffusion module refines native mel; it does not directly denoise or generate waveform samples. |
| Fixed channel intersection | No | DS004940 and DS006104 retain their native EEG channel spaces and use xyz plus masks. |

## Joint training in one sentence

`joint` alternates DS004940 and DS006104 **dataset-homogeneous batches** through
the same coordinate-aware CNN/Conformer/MFCC model, with dataset-specific input
normalization and supervision masks; it does not concatenate incompatible EEG
channels or cross-pair one dataset's EEG with the other dataset's audio.

## Compact data flow

```text
EEG + channel xyz + channel/time masks
→ dataset input affine
→ multiscale channel-wise CNN
→ xyz fusion + channel attention
→ shared Conformer
→ local/global EEG content representations
→ MFCC prediction (39 × 161)
→ duration-conditioned native SpeechT5-mel renderer
→ optional random qualitative diffusion mel refinement
→ frozen SpeechT5 HiFi-GAN
→ diagnostic native-duration WAV export
```

## V3 output contract

`metric/eeg_deterministic.wav` is the deterministic EEG-only reconstruction
used for retrieval and counterfactual comparisons.  When enabled,
`listening/eeg_random_sample-XX.wav` uses fresh diffusion noise on every
export; its seed is stored in `export_manifest.json`.  The latter is useful for
listening diversity, but it is never evidence that EEG carried the content.

For source details, see [`model.py`](model.py), [`losses.py`](losses.py), and
[`data.py`](data.py) in this directory.
