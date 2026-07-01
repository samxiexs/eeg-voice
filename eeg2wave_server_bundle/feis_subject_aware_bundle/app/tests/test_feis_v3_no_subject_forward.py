from __future__ import annotations

import inspect
import sys
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.feis_v3.data import assert_v3_model_forward_keys
from src.feis_v3.model import FEISV3ModelConfig, FEISV3TokenGenerator


def test_feis_v3_forward_signature_has_no_subject_or_speaker():
    params = set(inspect.signature(FEISV3TokenGenerator.forward).parameters)
    forbidden = {"subject_id", "subject_idx", "speaker_id", "audio_source_subject"}
    assert not (params & forbidden)
    assert {"eeg", "stage_idx", "eeg_valid_len", "channel_cluster_id"} <= params


def test_feis_v3_rejects_subject_forward_config_and_keys():
    try:
        FEISV3TokenGenerator(FEISV3ModelConfig(use_subject_id_in_forward=True))
    except ValueError:
        pass
    else:
        raise AssertionError("FEIS v3 accepted subject_id forwarding")
    assert_v3_model_forward_keys(("eeg", "stage_idx", "channel_cluster_id"))
    try:
        assert_v3_model_forward_keys(("eeg", "subject_id"))
    except ValueError:
        return
    raise AssertionError("FEIS v3 forward guard accepted subject_id")


if __name__ == "__main__":
    test_feis_v3_forward_signature_has_no_subject_or_speaker()
    test_feis_v3_rejects_subject_forward_config_and_keys()
    print("FEIS v3 subject-forward guard passed")
