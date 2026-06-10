"""EEG -> Speech v3 package.

Implements the v3 design (see NEW_DESIGN_eeg2speech_v3.md):

- Decoupled recognition + frozen-vocoder synthesis. The network never
  regresses raw waveform samples; it predicts EnCodec latent frames, which a
  frozen EnCodec decoder turns into a natural waveform.
- Spatial-temporal EEG encoder with FiLM subject conditioning, enabling
  cross-subject pooling.
- Training objective = InfoNCE retrieval + latent (cosine/MSE) + 16-way
  classification, which structurally avoids the mode collapse of the old
  waveform-regression path. Optional cross-stage knowledge distillation
  (speaking teacher -> thinking student).
"""

from .model import (
    DatasetHead,
    EEG2SpeechMD,
    EEG2SpeechMDConfig,
    EEG2SpeechV3,
    EEG2SpeechV3Config,
)

__all__ = [
    "EEG2SpeechV3",
    "EEG2SpeechV3Config",
    "EEG2SpeechMD",
    "EEG2SpeechMDConfig",
    "DatasetHead",
]
