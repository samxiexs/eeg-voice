# FEIS EEG-Only Bundle

This bundle now contains only the identity-free direct route.

- Model input: `EEG + stage_idx`
- Training target: speech latent and energy targets
- Optional auxiliary supervision: content label CE
- EEG encoder: optional channel-cluster MoE routes/gates EEG channels before the temporal trunk
- External identity fields are not accepted by model, batch, checkpoint, manifest, or metrics schema
- Generation method: EEG-conditioned EnCodec-latent generation. By default this uses
  latent diffusion trained with denoising noise prediction; the direct latent head
  remains as an auxiliary coarse predictor and non-diffusion ablation.

Run from `app/`:

```bash
bash run_direct_eeg_only.sh
```

The active Python package is `src/direct_eeg2speech`. Older factorized,
alignment, and waveform routes were removed from the runnable code surface.

Current training emphasizes audio-specific temporal behavior rather than image-style
spatial reconstruction: latent frame cosine/SmoothL1, first- and second-order
temporal dynamics, latent energy envelope, log-RMS prediction, diversity, and
collapse diagnostics. MoE regularizers are limited to EEG channel routing:
load balance, sparse channel gates, confident routing, and cohesion for channels
with similar temporal signals.

Diffusion is used only in normalized EnCodec latent space. Training samples a
noise step, adds Gaussian noise to the paired speech latent, and predicts that
noise conditioned on EEG encoder features. Synthesis runs DDIM denoising to
produce the latent sequence before the frozen EnCodec decoder converts it to wav.
