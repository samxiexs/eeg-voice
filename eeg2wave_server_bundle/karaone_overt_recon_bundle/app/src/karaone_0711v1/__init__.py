"""Independent KaraOne 0711v1 cross-subject EEG-to-speech pipeline.

This package deliberately has no dependency on the retired v10/v11/v12 code.
"""

from .data import KaraOne0711Dataset, SplitManifest, TopographicProjector
from .model import EEG0711Encoder, ConditionalFlowDecoder

__all__ = [
    "ConditionalFlowDecoder",
    "EEG0711Encoder",
    "KaraOne0711Dataset",
    "SplitManifest",
    "TopographicProjector",
]
