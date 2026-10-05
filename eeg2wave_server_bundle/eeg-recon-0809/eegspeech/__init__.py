"""Imagined speech from scalp EEG: decoding and reconstruction as speech.

Data: ``store`` (one HDF5 per dataset), ``signal`` (filters, alignment, speech features,
electrode positions).  Reconstruction: ``features`` (aligned full-band log-power), ``clip``
(personal CLIP encoders into a speech space), ``diffusion`` (mel diffusion decoder), ``audio``
(speech targets, vocoder, HuBERT, Whisper listener, mel-cepstral distance).  Cross-person
plan: ``data`` (samplers), ``model`` (montage-agnostic deep encoder), ``losses``,
``evaluation``.  Shared: ``metrics``.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'artifacts' / 'store'
