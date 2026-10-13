from __future__ import annotations
from pathlib import Path
import yaml
from . import ROOT
from .utils import fingerprint, read_json


class Config:
    def __init__(self, path: str | Path = ROOT / "configs/perceived_flow.yaml"):
        self.path = Path(path).resolve()
        self.root = self.path.parent.parent
        self.values = yaml.safe_load(self.path.read_text())
        self.splits = read_json(self.path.parent / "perceived_flow_splits.json")
        self.models = read_json(self.path.parent / "perceived_flow_models.json")
        self.validate()
        self.hash = fingerprint({"config": self.values, "splits": self.splits, "model_specs": self.models})

    def __getitem__(self, key):
        return self.values[key]

    def dir(self, key: str) -> Path:
        return self.root / self.values["paths"][key]

    def fold_subjects(self, fold: str, split: str) -> list[str]:
        held = self.splits["test_folds"][fold]
        if split == "test": return held
        if split == "validation": return self.splits["development_subjects"]
        if split != "train": raise ValueError(split)
        excluded = set(held + self.splits["development_subjects"])
        return [s for s in self["dataset"]["subjects"] if s not in excluded]

    def validate(self) -> None:
        sets = [set(v) for v in self.splits["audio"].values()]
        if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
            raise ValueError("Audio groups overlap across splits")
        dev = set(self.splits["development_subjects"])
        held = [s for group in self.splits["test_folds"].values() for s in group]
        if len(set(held)) != len(held) or dev & set(held):
            raise ValueError("Subject groups overlap")
        if set(held) | dev != set(self["dataset"]["subjects"]):
            raise ValueError("Subject split is incomplete")
        p, w, a = self["preprocessing"], self["window"], self["audio"]
        if p["eeg_rate"] != 128 or p["neural_rate"] != 64:
            raise ValueError("The locked protocol uses EEG128 / neural64 Hz")
        if (a["sample_rate"], a["mel_bins"], a["n_fft"], a["hop_length"]) != (24000, 100, 1024, 256):
            raise ValueError("Incompatible Vocos frontend")
        if w["seconds"] != 4 or a["teacher_dim"] != 768:
            raise ValueError("This protocol uses 4 seconds and WavLM-Base768")

    def run_dir(self, fold: str, seed: int, system: str) -> Path:
        return self.dir("artifacts") / "runs" / fold / str(seed) / system

    def model_dir(self, kind: str) -> Path:
        return self.dir("models") / self.models[kind]["local_dir"]
