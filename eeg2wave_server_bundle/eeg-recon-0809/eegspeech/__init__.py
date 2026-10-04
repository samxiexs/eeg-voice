"""Imagined-speech decoding from scalp EEG, bootstrapped from listening EEG.

Modules: ``store`` (one HDF5 per dataset), ``signal`` (filters, alignment, speech
features, electrode positions), ``data`` (samplers), ``model`` (montage-agnostic
encoder), ``losses``, ``metrics``.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'artifacts' / 'store'
