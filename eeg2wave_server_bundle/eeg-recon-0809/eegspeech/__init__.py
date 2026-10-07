"""Imagined speech from scalp EEG: decoding and reconstruction as speech.

Data: ``store`` (one HDF5 per dataset), ``signal`` (filters, alignment, electrode positions),
``features`` (aligned filter-bank log-power and its time course; the within-person folds).
Content: ``clip`` (personal linear CLIP encoders into a speech space, for items and open-vocabulary
sentences), ``deep`` (end-to-end CLIP encoder of sentences).  Speech: ``audio`` (speech targets,
vocoder, HuBERT, Whisper listener, speaker and pitch, mel-cepstral distance), ``diffusion`` (mel
diffusion decoder of items in a voice).  ``metrics``: per-person summaries and tests.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'artifacts' / 'store'
