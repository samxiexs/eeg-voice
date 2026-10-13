"""Audio preparation, immutable provenance and publication protocol locking."""
from __future__ import annotations
import importlib.metadata
from pathlib import Path
import platform
import numpy as np
import torch
from .perceived_audio import acoustic_features, interpolate_features, read_wave
from .utils import Moments, atomic_json, digest, read_json


def audio_window_rows(cfg, split):
    import soundfile as sf
    stride = cfg["window"]["train_stride"] if split == "train" else cfg["window"]["evaluation_stride"]
    guard = cfg["preprocessing"]["guard_seconds"]
    rows = []
    for clip in cfg.splits["audio"][split]:
        duration = sf.info(cfg.dir("raw") / "stimuli" / f"{clip}.wav").duration
        for start in np.arange(guard, duration - guard - 4 + 1e-6, stride):
            rows.append({"audio": clip, "start": float(start), "key": f"{clip}_{start:.3f}"})
    return rows


def prepare_audio_targets(cfg, audio):
    from .perceived_data import textgrid_words
    folder = cfg.dir("artifacts") / "audio_targets"
    full = cfg.dir("artifacts") / "audio_full"
    folder.mkdir(parents=True, exist_ok=True); full.mkdir(parents=True, exist_ok=True)
    moments = {"teacher": Moments(768), "mel": Moments(100), "phi": Moments(2)}
    model_manifest_hash = digest(cfg.dir("models") / "manifest.json")
    provenance = cfg.dir("artifacts") / "audio_provenance.json"
    if provenance.exists():
        old = read_json(provenance)
        if old["config_hash"] != cfg.hash or old["model_manifest_hash"] != model_manifest_hash:
            raise ValueError("Cached speech features have incompatible model/config revisions")
    atomic_json(provenance, {"config_hash": cfg.hash, "model_manifest_hash": model_manifest_hash, "state": "extracting"})
    for split in ["train", "validation", "test"]:
        rows = audio_window_rows(cfg, split)
        for clip in cfg.splits["audio"][split]:
            wave24 = read_wave(cfg.dir("raw") / "stimuli" / f"{clip}.wav", 24000)
            wave16 = read_wave(cfg.dir("raw") / "stimuli" / f"{clip}.wav", 16000)
            phi = acoustic_features(torch.from_numpy(wave24)[None]).squeeze(0)
            np.savez_compressed(full / f"{clip}.npz", phi=phi.numpy())
            if split == "train": moments["phi"].add(phi.numpy())
            for row in [r for r in rows if r["audio"] == clip]:
                path = folder / f"{row['key']}.npz"
                if not path.exists():
                    a = row["start"]
                    crop = wave24[round(a*24000):round((a+4)*24000)]
                    teacher_crop = wave16[round((a-1)*16000):round((a+5)*16000)]
                    with torch.no_grad():
                        mel = audio.vocos.features(torch.from_numpy(crop)[None].to(audio.device)).squeeze(0).cpu()
                        teacher, times = audio.wavlm(teacher_crop)
                        teacher, times = teacher.squeeze(0).cpu(), times.cpu()
                        phi_crop = phi[round(a*64):round((a+4)*64)]
                        phi_time = torch.arange(len(phi_crop)) / 64
                        phi_aligned = interpolate_features(phi_crop[None], phi_time, times).squeeze(0)
                    np.savez_compressed(path, teacher=teacher.numpy().astype(np.float16), times=times.numpy(), mel=mel.numpy(), phi=phi_aligned.numpy())
                if split == "train":
                    with np.load(path) as data:
                        moments["teacher"].add(data["teacher"])
                        moments["mel"].add(data["mel"].T)
            print(f"Prepared speech targets {clip}", flush=True)
    stats = {key: value.result() for key, value in moments.items()}
    stats.update(config_hash=cfg.hash, training_audio=cfg.splits["audio"]["train"], model_manifest_hash=model_manifest_hash)
    atomic_json(cfg.dir("artifacts") / "stats/audio.json", stats)
    atomic_json(provenance, {"config_hash": cfg.hash, "model_manifest_hash": model_manifest_hash, "training_statistics": cfg.splits["audio"]["train"]})


def environment_manifest(cfg):
    packages = {}
    for name in ["torch", "torchaudio", "numpy", "scipy", "mne", "transformers", "huggingface_hub", "vocos", "praatio", "openai-whisper"]:
        packages[name] = importlib.metadata.version(name)
    value = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages, "config_hash": cfg.hash}
    atomic_json(cfg.dir("artifacts") / "environment.json", value)
    return value


def provenance(cfg, fold=None):
    value = {"config_hash": cfg.hash, "dataset_revision": cfg["dataset"]["revision"], "model_manifest_hash": digest(cfg.dir("models") / "manifest.json"), "audio_stats_hash": digest(cfg.dir("artifacts") / "stats/audio.json")}
    value["code_hash"] = __import__("hashlib").sha256("".join(digest(p) for p in sorted((cfg.root / "eegspeech").glob("*.py"))).encode()).hexdigest()
    value["unit_calibration_hash"] = digest(cfg.root / cfg["dataset"]["unit_calibration_report"])
    if fold: value["data_manifest_hash"] = digest(cfg.dir("artifacts") / "manifest.json")
    if fold: value["fold_stats_hash"] = digest(cfg.dir("artifacts") / "stats" / f"{fold}.json")
    return value


def require_provenance(checkpoint, cfg, fold=None):
    expected = provenance(cfg, fold)
    if checkpoint.get("provenance") != expected:
        raise ValueError("Checkpoint's config/models/training statistics differ from current protocol")


def require_full_data(cfg):
    manifest = read_json(cfg.dir("artifacts") / "manifest.json")
    if not manifest["complete_subjects"] or manifest["config_hash"] != cfg.hash:
        raise ValueError("Formal training requires all 19 declared subjects and the locked preprocessing")


def freeze_protocol(cfg):
    from .utils import load_checkpoint
    require_full_data(cfg)
    required, checkpoints = [cfg.dir("artifacts") / "manifest.json", cfg.dir("artifacts") / "stats/audio.json"], {}
    for seed in cfg["training"]["seeds"]:
        gate = cfg.dir("artifacts") / "oracle" / str(seed) / "gate.json"
        if not read_json(gate)["passed"]: raise ValueError("Oracle gate did not pass")
        required.extend([gate, cfg.dir("artifacts") / "oracle" / str(seed) / "best.pt"])
        for fold in cfg.splits["test_folds"]:
            if not read_json(cfg.dir("artifacts") / "trf" / fold / "gate.json")["passed"]:
                raise ValueError(f"TRF gate failed for {fold}")
            required.extend([cfg.dir("artifacts") / "stats" / f"{fold}.json", cfg.dir("artifacts") / "trf" / fold / "gate.json"])
            for system in ["B0", "B1", "B2"]:
                path = cfg.run_dir(fold, seed, system) / "best.pt"
                saved = load_checkpoint(path)
                require_provenance(saved, cfg, fold)
                if not saved.get("completed") or saved["trained_updates"] != cfg["training"]["flow_steps"]:
                    raise ValueError("Formal matrix is incomplete")
                required.append(path)
    pilot = read_json(cfg.run_dir("F0", 11, "B0") / "gate.json")
    if not pilot["passed"]: raise ValueError("Pilot conditioning gate failed")
    for path in sorted(set(required)):
        checkpoints[str(path.relative_to(cfg.root))] = digest(path)
    lock = {"config_hash": cfg.hash, "model_manifest_hash": digest(cfg.dir("models") / "manifest.json"), "audio_stats_hash": digest(cfg.dir("artifacts") / "stats/audio.json"), "checkpoints": checkpoints, "test_opened": False}
    atomic_json(cfg.dir("artifacts") / "protocol.lock.json", lock)
    return lock


def require_test_lock(cfg, paths=()):
    lock_path = cfg.dir("artifacts") / "protocol.lock.json"
    if not lock_path.exists(): raise ValueError("Final test is locked; complete formal matrix and run freeze first")
    lock = read_json(lock_path)
    if lock["config_hash"] != cfg.hash or lock["model_manifest_hash"] != digest(cfg.dir("models") / "manifest.json") or lock["audio_stats_hash"] != digest(cfg.dir("artifacts") / "stats/audio.json"):
        raise ValueError("Protocol changed after freeze")
    for path in paths:
        key = str(path.relative_to(cfg.root))
        if key not in lock["checkpoints"] or digest(path) != lock["checkpoints"][key]:
            raise ValueError(f"Checkpoint changed after freeze: {key}")
    for path in [cfg.dir("artifacts") / "manifest.json", cfg.dir("artifacts") / "stats/audio.json"]:
        key = str(path.relative_to(cfg.root))
        if key not in lock["checkpoints"] or digest(path) != lock["checkpoints"][key]:
            raise ValueError("Data manifest changed after freeze")
    lock["test_opened"] = True
    atomic_json(lock_path, lock)


def decode_mel(normalized, stats, vocos):
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=normalized.device)[None, :, None]
    std = torch.as_tensor(stats["std"], dtype=torch.float32, device=normalized.device)[None, :, None]
    return vocos(normalized.float() * std + mean)
