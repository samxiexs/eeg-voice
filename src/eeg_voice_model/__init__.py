"""EEGVoiceTokenV1 public package exports."""

__all__ = [
    "EEGVoiceBatch",
    "EEGVoiceTokenV1",
    "EEGVoiceTokenizerV1",
    "EEGVoiceV1Config",
    "GroupedRVQOutput",
    "VoiceAlignmentTargets",
    "build_eeg_voice_token_v1",
    "build_model_v1_bundle",
]


def __getattr__(name: str):
    if name in {"EEGVoiceTokenizerV1", "EEGVoiceV1Config", "GroupedRVQOutput"}:
        from .tokenizer import EEGVoiceTokenizerV1, EEGVoiceV1Config, GroupedRVQOutput

        return {
            "EEGVoiceTokenizerV1": EEGVoiceTokenizerV1,
            "EEGVoiceV1Config": EEGVoiceV1Config,
            "GroupedRVQOutput": GroupedRVQOutput,
        }[name]
    if name in {"EEGVoiceBatch", "EEGVoiceTokenV1", "VoiceAlignmentTargets"}:
        from .voice_model import EEGVoiceBatch, EEGVoiceTokenV1, VoiceAlignmentTargets

        return {
            "EEGVoiceBatch": EEGVoiceBatch,
            "EEGVoiceTokenV1": EEGVoiceTokenV1,
            "VoiceAlignmentTargets": VoiceAlignmentTargets,
        }[name]
    if name in {"build_eeg_voice_token_v1", "build_model_v1_bundle"}:
        from .builders import build_eeg_voice_token_v1, build_model_v1_bundle

        return {
            "build_eeg_voice_token_v1": build_eeg_voice_token_v1,
            "build_model_v1_bundle": build_model_v1_bundle,
        }[name]
    raise AttributeError(name)
