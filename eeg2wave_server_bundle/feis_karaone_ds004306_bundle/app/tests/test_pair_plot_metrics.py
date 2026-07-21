from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


APP_DIR = Path(__file__).resolve().parents[1]
SCRIPT = APP_DIR / "scripts" / "plot_combined_0715_pairs.py"
SPEC = importlib.util.spec_from_file_location("combined_pair_plot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["combined_pair_plot"] = MODULE
SPEC.loader.exec_module(MODULE)


def test_display_time_axis_preserves_original_duration_after_downsampling() -> None:
    axis = MODULE.display_time_axis(original_samples=48000, plotted_samples=4000, sample_rate=24000)
    assert len(axis) == 4000
    assert axis[0] == 0.0
    assert axis[-1] == pytest.approx(47999 / 24000)


def test_display_envelope_tracks_amplitude_not_carrier_phase() -> None:
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    amplitude = np.exp(-0.5 * ((time - 0.5) / 0.1) ** 2)
    first = amplitude * np.sin(2.0 * np.pi * 220.0 * time)
    second = amplitude * np.sin(2.0 * np.pi * 220.0 * time + np.pi / 2.0)
    first_envelope = MODULE.display_envelope(first, sample_rate)
    second_envelope = MODULE.display_envelope(second, sample_rate)
    assert np.corrcoef(first_envelope, second_envelope)[0, 1] > 0.99
