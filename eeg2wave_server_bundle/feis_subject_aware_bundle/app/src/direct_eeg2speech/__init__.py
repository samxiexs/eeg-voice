"""EEG-only speech reconstruction modules.

This package deliberately excludes subject-id conditioning. Subject labels may
be used by datasets and metrics as metadata, but never as model input.
"""

from .model import DirectEEG2Speech, DirectEEG2SpeechConfig

__all__ = ["DirectEEG2Speech", "DirectEEG2SpeechConfig"]
