"""Split-integrity test: train / val_seen / test_seen / test_holdout do not leak,
and held-out cells never appear in train (V2_PLAN Stage-3)."""
from __future__ import annotations

import sys
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import load_simple_yaml, resolve_bundle_path
from src.feis_factored.data import FactoredFEISDataset
from src.feis_factored.targets import FactoredTargets

CFG = BUNDLE_DIR / "configs" / "factored.yaml"


def _datasets():
    cfg = load_simple_yaml(CFG)
    cache = resolve_bundle_path(cfg["data"]["target_cache"], BUNDLE_DIR)
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    if not Path(cache).exists() or not Path(root).exists():
        return None
    targets = FactoredTargets(cache)
    common = dict(data_root=root, targets=targets, stages=("stimuli", "thinking"),
                  include_anomalous=False, holdout_offset=0)
    return {s: FactoredFEISDataset(split=s, **common)
            for s in ("train", "val_seen", "test_seen", "test_holdout")}


def _keys(ds):
    return {(e.subject, e.label, e.stage, e.trial_index) for e in ds.entries}


def test_no_trial_leak_between_splits():
    dss = _datasets()
    if dss is None:
        print("[skip] data not present"); return
    keys = {s: _keys(d) for s, d in dss.items()}
    names = list(keys)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = keys[names[i]] & keys[names[j]]
            assert not inter, f"leak between {names[i]} and {names[j]}: {list(inter)[:3]}"


def test_holdout_cells_absent_from_train():
    dss = _datasets()
    if dss is None:
        print("[skip] data not present"); return
    ho = dss["train"].holdout_cells
    train_cells = {(e.subject, e.label) for e in dss["train"].entries}
    assert not (train_cells & ho), "held-out cells leaked into train"
    # every holdout entry is a held-out cell
    for e in dss["test_holdout"].entries:
        assert (e.subject, e.label) in ho


if __name__ == "__main__":
    test_no_trial_leak_between_splits()
    test_holdout_cells_absent_from_train()
    print("factored splits test passed")
