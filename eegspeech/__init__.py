"""Physiology-guided perceived EEG-to-speech reconstruction."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "artifacts/perceived_flow/numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/perceived_flow/matplotlib"))
__version__ = "1.0.0"
