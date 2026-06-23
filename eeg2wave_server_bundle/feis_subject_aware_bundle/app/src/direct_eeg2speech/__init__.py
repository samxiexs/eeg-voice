"""EEG-only speech reconstruction modules.

This package deliberately excludes external identity conditioning. Only EEG and
stage are valid model inputs.
"""

from .model import DirectEEG2Speech, DirectEEG2SpeechConfig

__all__ = ["DirectEEG2Speech", "DirectEEG2SpeechConfig"]
