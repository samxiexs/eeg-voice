"""BrainOmni-style EEG voice model v0.

Heavy torch modules are loaded lazily so lightweight helpers such as
`audio_features` can still be used before PyTorch is installed.
"""

__all__ = [
    "AudioContrastiveHead",
    "BrainStyleEEGTokenizerConfig",
    "BrainStyleEEGTokenizerV0",
    "PhonemeSequenceHead",
    "ProbeHead",
    "SegmentContrastiveHead",
    "TokenCentricEEGVoiceConfig",
    "TokenCentricEEGVoiceModelV01",
    "TokenMetrics",
    "VoiceAttributeHead",
    "VoiceProfileHead",
]


def __getattr__(name: str):
    if name in {"BrainStyleEEGTokenizerV0", "BrainStyleEEGTokenizerConfig"}:
        from .tokenizer import BrainStyleEEGTokenizerConfig, BrainStyleEEGTokenizerV0

        return {
            "BrainStyleEEGTokenizerConfig": BrainStyleEEGTokenizerConfig,
            "BrainStyleEEGTokenizerV0": BrainStyleEEGTokenizerV0,
        }[name]
    if name in {
        "AudioContrastiveHead",
        "PhonemeSequenceHead",
        "ProbeHead",
        "SegmentContrastiveHead",
        "TokenMetrics",
        "VoiceAttributeHead",
        "VoiceProfileHead",
    }:
        from .heads import (
            AudioContrastiveHead,
            PhonemeSequenceHead,
            ProbeHead,
            SegmentContrastiveHead,
            TokenMetrics,
            VoiceAttributeHead,
            VoiceProfileHead,
        )

        return {
            "AudioContrastiveHead": AudioContrastiveHead,
            "PhonemeSequenceHead": PhonemeSequenceHead,
            "ProbeHead": ProbeHead,
            "SegmentContrastiveHead": SegmentContrastiveHead,
            "TokenMetrics": TokenMetrics,
            "VoiceAttributeHead": VoiceAttributeHead,
            "VoiceProfileHead": VoiceProfileHead,
        }[name]
    if name in {"TokenCentricEEGVoiceConfig", "TokenCentricEEGVoiceModelV01"}:
        from .voice_model import TokenCentricEEGVoiceConfig, TokenCentricEEGVoiceModelV01

        return {
            "TokenCentricEEGVoiceConfig": TokenCentricEEGVoiceConfig,
            "TokenCentricEEGVoiceModelV01": TokenCentricEEGVoiceModelV01,
        }[name]
    raise AttributeError(name)
