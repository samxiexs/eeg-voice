#!/usr/bin/env python3
"""Staged, resumable runner for aligned_speech_v1. Run --help for commands."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
DS004940_STIMULUS_PARAMETERS = ROOT / "data/external_metadata/N400PvsA_stimuli_parameters.tsv"
sys.path.insert(0, str(ROOT / "app/src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eeg2speech.aligned import (CONTRACT, SAMPLE_RATE, WAVE_SAMPLES, LAGS_MS, AcousticDecoder,
                               AlignedEEGModel, acoustic_loss, alignment_loss, atomic_json,
                               convolution_times, fixed_wave, sha256, tree_hash, two_way_bootstrap)
from eeg2speech.aligned_data import (AlignedDataset, content_split, fit_eeg_normalizer, fixed_bank,
                                    local_path, truth, validate_split, wrong_trial_indices)
from eeg2speech.losses import counterfactual_eeg
from eeg2speech.speecht5 import native_speecht5_mel, SpeechT5HiFiGan
from eeg2speech.model import DurationConditionedNativeRenderer


class MFCCBaseline(torch.nn.Module):
    """Legacy relative-MFCC renderer refitted on the NEW training fold only."""
    def __init__(self, frames):
        super().__init__()
        self.frames = frames
        self.renderer = DurationConditionedNativeRenderer(dropout=0.)

    def forward(self, mfcc, oracle_duration_frames):
        # This AUDIO-ONLY oracle is allowed the original duration. It must not
        # understate the old route's upper bound by withholding its input.
        if bool((oracle_duration_frames < 1).any()) or bool((oracle_duration_frames > self.frames).any()):
            raise ValueError("oracle duration exceeds the fixed acoustic window")
        mel, mask = self.renderer(mfcc, oracle_duration_frames)
        mel = mel.masked_fill(~mask[:, None], -10.)
        return F.pad(mel, (0, self.frames - mel.shape[-1]), value=-10.)


def audio_forward(model, batch):
    if isinstance(model, MFCCBaseline):
        return model(batch["mfcc"], batch["oracle_duration_frames"])
    return model(batch["teacher"])


def config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("schema_version") != CONTRACT:
        raise ValueError("wrong experiment config")
    return cfg


def path_of(cfg, key):
    return local_path(ROOT, os.path.expandvars(str(cfg[key]))).resolve()


def artifact_paths(cfg):
    base = path_of(cfg, "artifact_root")
    return base, base / "manifest.csv", base / "targets.h5", base / "eeg_normalizer.json"


def device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this Python process; select --device cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu for engineering checks")
    return torch.device(name)


def runtime_hash():
    from importlib.metadata import version
    files = [Path(__file__), *sorted((ROOT / "app/src/eeg2speech").glob("*.py"))]
    libraries = {name: version(name) for name in ("torch", "transformers", "numpy", "scipy", "h5py")}
    return hashlib.sha256(json.dumps({"files": [(p.name, sha256(p)) for p in files], "libraries": libraries}, sort_keys=True).encode()).hexdigest()


def atomic_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp); temp.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move(batch, target):
    return {k: v.to(target) if torch.is_tensor(v) else v for k, v in batch.items()}


def decoder_from(payload):
    spec = payload["decoder_spec"]
    if spec.get("kind") == "mfcc":
        model = MFCCBaseline(spec["frames"])
        model.load_state_dict(payload["decoder"])
        return model
    model = AcousticDecoder(torch.tensor(spec["speech_times"]), torch.tensor(spec["mel_times"]),
                            speech_dimension=spec["speech_dimension"], hidden=spec["hidden"], layers=spec["layers"])
    model.load_state_dict(payload["decoder"])
    return model


def load_payload(path):
    # Only locally created training checkpoints; never accept untrusted .pt files.
    result = torch.load(path, map_location="cpu", weights_only=False)
    if result.get("contract") != CONTRACT or result.get("runtime_hash") != runtime_hash():
        raise RuntimeError("checkpoint contract/runtime mismatch; do not silently reuse old experiments")
    return result


def audit_active_sources(data_cfg):
    """Reuse the DS004 event adapter without requiring DS006 audio/auxiliary files."""
    from collections import Counter
    from prepare_training_data import (_ds004_trial_rows, inventory_sha256, source_lock_entry,
                                       output_root, write_frame, stable_json, sha256_bytes)
    root = output_root(data_cfg); root.mkdir(parents=True, exist_ok=True)
    spec = data_cfg["sources"]["ds004940"]
    source = ROOT / spec["data_root"]
    description = source / "dataset_description.json"
    if sha256(description) != spec["dataset_description_sha256"]:
        raise RuntimeError("DS004940 dataset description differs from the pinned version")
    if inventory_sha256(source / "stimuli") != spec["stimulus_inventory_sha256"]:
        raise RuntimeError("DS004940 stimulus inventory differs from the pinned version")
    qc = {"actual_subjects": {}, "exclusions": Counter(), "warnings": []}
    lock = {"schema_version": data_cfg["schema_version"], "config_sha256": data_cfg["_config_sha256"],
            "files": [source_lock_entry(description, "dataset_description")], "official_aux": {},
            "scope": "ds004940_N400Active_only"}
    all_rows = _ds004_trial_rows(data_cfg, lock, qc)
    if len(all_rows) != spec["expected_trials"] or qc["actual_subjects"].get("ds004940") != spec["expected_subjects"]:
        raise RuntimeError("DS004940 event/subject count differs from pinned inventory")
    rows = [row for row in all_rows if row["task"] == "N400Active"]
    paths = sorted({row[k] for row in rows for k in ("source_eeg_path", "source_channels_path", "source_event_path", "audio_path") if row.get(k)})
    digests = {}
    for i, relative in enumerate(paths):
        entry = source_lock_entry(ROOT / relative, "source")
        lock["files"].append(entry); digests[relative] = entry["sha256"]
        if i % 25 == 0:
            print(json.dumps({"audited_sources": i + 1, "total": len(paths)}), flush=True)
    lock["files"].sort(key=lambda x: x["path"])
    lock["source_lock_sha256"] = sha256_bytes(stable_json(lock))
    for row in rows:
        row.update(source_eeg_sha256=digests.get(row["source_eeg_path"], ""), source_lock_sha256=lock["source_lock_sha256"],
                   preprocess_config_sha256=data_cfg["_config_sha256"], code_commit="aligned-runtime-hash",
                   code_diff_hash=runtime_hash(), audit_timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_frame(pd.DataFrame(rows), root / "manifests/manifest_all", pd)
    atomic_json(root / "source_lock.json", lock)
    qc.update(status="warning" if qc["warnings"] else "pass", scope="ds004940_N400Active_only",
              source_lock_sha256=lock["source_lock_sha256"], trial_counts={"ds004940": len(rows)})
    qc["exclusions"] = dict(qc["exclusions"])
    atomic_json(root / "qc/audit.json", qc)
    if qc["warnings"]:
        raise RuntimeError("source audit warnings; inspect the isolated audit report")


def prepare(args, cfg):
    from prepare_training_data import make_splits, load_config, output_root, build
    from prepare_m0_artifacts import _buildable_rows
    data_cfg, _ = load_config(path_of(cfg, "data_config"))
    data_root = output_root(data_cfg)
    base, manifest, _, normalizer = artifact_paths(cfg)
    base.mkdir(parents=True, exist_ok=True)
    if not (data_root / "manifests/manifest_all.csv").exists():
        audit_active_sources(data_cfg)
    source_lock = json.loads((data_root / "source_lock.json").read_text())
    if source_lock["config_sha256"] != data_cfg["_config_sha256"]:
        raise ValueError("preprocessing config changed after audit; choose a new artifact directory")
    if not (data_root / "splits/assignment.json").exists():
        make_splits(data_cfg)
    original = pd.read_csv(data_root / "manifests/manifest_all.csv", keep_default_na=False, low_memory=False)
    eligible = original[(original.dataset == "ds004940") & (original.task == "N400Active") &
                        (original.pairing_level == "verified_exact") & (original.build_status == "included") & truth(original.qc_pass)]
    eligible = _buildable_rows(eligible, data_cfg)
    joint_protocol = cfg.get("protocol", "known_subject_unseen_content") == "joint_ood"
    if joint_protocol:
        import itertools
        selected = content_split(eligible, seed=cfg["split_seed"], joint_ood=True)
        train = selected[selected.role == "train"]
        m0_ids = set()
        for subjects in itertools.combinations(sorted(set(train.subject)), 5):
            subset = train[train.subject.isin(subjects)]
            coverage = subset.groupby("content_group").subject.nunique()
            contents = sorted(coverage[coverage == 5].index)[:10]
            if len(contents) == 10:
                m0_ids = set(subset[subset.content_group.isin(contents)].sort_values("trial_id")
                             .drop_duplicates(["subject", "content_group"]).trial_id)
                break
        if len(m0_ids) != 50:
            raise ValueError("no 5-subject × 10-content M0 inside joint training fold")
    else:
        m0_path = path_of(cfg, "legacy_m0_manifest")
        m0 = pd.read_csv(m0_path, keep_default_na=False)
        m0_ids = set(m0.loc[(m0.build_status == "included") & (m0.dataset == "ds004940"), "trial_id"])
        if len(m0_ids) != 50 or not m0_ids <= set(eligible.trial_id):
            raise RuntimeError("the existing 50-pair M0 is missing or fails current QC")
        selected = content_split(eligible, seed=cfg["split_seed"], reserved_train_ids=m0_ids)
    selected["is_m0"] = selected.trial_id.isin(m0_ids)
    selected["audio_key"] = selected.audio_sha256
    split_path = data_root / "splits/aligned_v1_fold-0.csv"
    cols = ["trial_id", "role", "fold", "content_group", "is_m0", "audio_key"]
    csv_text = selected[cols].sort_values("trial_id").to_csv(index=False)
    if split_path.exists() and split_path.read_text() != csv_text:
        raise RuntimeError("existing split differs; use a new version directory")
    split_path.write_text(csv_text)
    # Future joint OOD uses a separate manifest and freshly fitted normalizer.
    joint = content_split(eligible, seed=cfg["split_seed"], joint_ood=True)
    joint[cols[:3] + ["content_group"]].to_csv(base / "joint_ood_assignment.csv", index=False)
    if args.materialize:
        for entry in source_lock["files"]:
            if sha256(local_path(ROOT, entry["path"])) != entry["sha256"]:
                raise ValueError(f"source changed since audit: {entry['path']}")
        # The legacy builder stores one shard per subject/task, not per role.
        # Select all physical trials in ONE build pass. This build-only role
        # is never used for training; the actual scientific roles are merged
        # below and are the only input to fit_eeg_normalizer/AlignedDataset.
        transport = selected.loc[selected.role != "excluded", cols].copy()
        transport["role"] = "materialize"
        transport_path = data_root / "splits/aligned_v1_materialize_fold-0.csv"
        transport_text = transport.sort_values("trial_id").to_csv(index=False)
        if transport_path.exists() and transport_path.read_text() != transport_text:
            raise RuntimeError("build selection changed; use a new version directory")
        transport_path.write_text(transport_text)
        build(data_cfg, "ds004940", "all", "N400Active", None, None, None,
              "any", "materialize", "aligned_v1_materialize", 0, True, False, "aligned_v1")
        built = pd.read_csv(data_root / "manifests/manifest_aligned_v1.csv", keep_default_na=False)
        built = built[built.build_status == "included"]
        absent = set(selected.loc[selected.role != "excluded", "trial_id"]) - set(built.trial_id)
        if absent:
            atomic_json(base / "build_failures.json", {"missing_trials": sorted(absent)})
            raise RuntimeError("selected trials failed materialization; inspect build_failures.json")
        built = built.drop(columns=[c for c in cols[1:] if c in built], errors="ignore")
        selected = built.merge(selected[cols], on="trial_id", validate="one_to_one")
        # Legacy builder manifest rows can retain the generic assignment hash
        # while the actual named shard pins the selected build CSV. Verify it
        # rather than weakening the new loader's provenance checks.
        for name, indices in selected.groupby("shard_path").groups.items():
            with h5py.File(local_path(ROOT, name), "r") as h5:
                actual = str(h5.attrs["split_index_sha256"])
                if actual != sha256(transport_path):
                    raise ValueError("built shard pins an unexpected materialization selection")
            selected.loc[indices, "split_index_sha256"] = actual
        selected = selected.sort_values("trial_id")
        text = selected.to_csv(index=False)
        if manifest.exists() and manifest.read_text() != text:
            raise RuntimeError("materialized manifest changed; preserve existing experiment")
        manifest.write_text(text)
        if not normalizer.exists():
            fit_eeg_normalizer(ROOT, manifest, normalizer)
    selected[cols].to_csv(base / "assignment.csv", index=False)
    atomic_json(base / "split_report.json", {
        "contract": CONTRACT, "protocol": "joint_ood" if joint_protocol else "known_subject_unseen_content", "role_counts": selected.role.value_counts().to_dict(),
        "unique_contents": selected.groupby("role").content_group.nunique().to_dict(),
        "subjects": selected.groupby("role").subject.nunique().to_dict(), "m0_pairs": int(selected.is_m0.sum()),
        "materialized": bool(args.materialize), "historical_test_status": "exploratory_previously_inspected_dataset",
        "split_sha256": sha256(split_path)})
    print((base / "split_report.json").read_text())


def corpus_manifest(args, cfg):
    import soundfile as sf
    corpus = Path(args.corpus_root).resolve(); rows = []
    for subset, role in (("train-clean-100", "train"), ("dev-clean", "validation")):
        files = sorted((corpus / subset).rglob("*.flac"))
        if not files:
            raise RuntimeError(f"missing LibriSpeech {subset}: {corpus}")
        for path in files:
            info = sf.info(path)
            if info.samplerate != SAMPLE_RATE or info.channels != 1:
                raise ValueError(f"unexpected corpus audio format: {path}")
            digest = sha256(path)
            for start in range(0, info.frames, WAVE_SAMPLES):
                if info.frames - start < SAMPLE_RATE // 2:
                    continue
                key = hashlib.sha256(f"{digest}:{start}".encode()).hexdigest()
                rows.append({"trial_id": key, "audio_key": key, "audio_sha256": digest, "audio_path": str(path),
                             "offset_samples": start, "role": role, "subject": path.stem.split("-")[0],
                             "content_group": path.stem, "linguistic_content_id": path.stem})
    frame = pd.DataFrame(rows)
    for column in ("audio_sha256", "subject"):
        if frame.groupby(column).role.nunique().max() != 1:
            raise RuntimeError(f"corpus train/dev overlap: {column}")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    value = frame.to_csv(index=False)
    if output.exists() and output.read_text() != value:
        raise RuntimeError("corpus manifest changed; select a new output")
    output.write_text(value)
    print(json.dumps({"segments": len(frame), "manifest": str(output)}))


def cache_targets(args, cfg):
    import soundfile as sf
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    from scipy.signal import resample_poly
    from cache_speech_targets import content_features
    _, manifest, target, _ = artifact_paths(cfg)
    manifest = Path(args.manifest) if args.manifest else manifest
    target = Path(args.output) if args.output else target
    frame = pd.read_csv(manifest, keep_default_na=False).drop_duplicates("audio_key")
    teacher_path = Path(args.hubert or os.environ.get("HUBERT_LOCAL_PATH", ""))
    if not (teacher_path / "config.json").is_file():
        raise RuntimeError("supply --hubert or HUBERT_LOCAL_PATH with local frozen HuBERT weights")
    signature = {"contract": CONTRACT, "manifest_sha256": sha256(manifest), "teacher_sha256": tree_hash(teacher_path),
                 "teacher_layer": 9, "wave_samples": WAVE_SAMPLES, "runtime_hash": runtime_hash()}
    target.parent.mkdir(parents=True, exist_ok=True)
    working = target.with_suffix(".partial.h5")
    if target.exists():
        with h5py.File(target, "r") as h5:
            if not all(str(h5.attrs.get(k)) == str(v) for k, v in signature.items()):
                raise RuntimeError("completed target cache provenance differs")
        print("target cache already complete"); return
    model = HubertModel.from_pretrained(teacher_path, local_files_only=True).eval().requires_grad_(False).to(device(args.device))
    processor = Wav2Vec2FeatureExtractor.from_pretrained(teacher_path, local_files_only=True)
    times = convolution_times(WAVE_SAMPLES, model.config.conv_kernel, model.config.conv_stride)
    with h5py.File(working, "a") as h5:
        if len(h5.attrs) and not all(str(h5.attrs.get(k)) == str(v) for k, v in signature.items()):
            raise RuntimeError("partial cache provenance differs")
        h5.attrs.update(signature)
        groups = h5.require_group("targets")
        if "speech_times" not in h5:
            h5.create_dataset("speech_times", data=times.numpy())
        for count, row in enumerate(frame.itertuples()):
            if row.audio_key in groups and groups[row.audio_key].attrs.get("complete", False):
                continue
            if row.audio_key in groups:
                del groups[row.audio_key]
            path = local_path(ROOT, row.audio_path)
            if sha256(path) != row.audio_sha256:
                raise RuntimeError(f"source audio changed: {path}")
            offset = int(getattr(row, "offset_samples", 0))
            if hasattr(row, "offset_samples"):
                wave, rate = sf.read(path, start=offset, frames=WAVE_SAMPLES, dtype="float32")
            else:
                wave, rate = sf.read(path, dtype="float32")
            if wave.ndim == 2:
                wave = wave.mean(1)
            if rate != SAMPLE_RATE:
                divisor = math.gcd(rate, SAMPLE_RATE)
                wave = resample_poly(wave, SAMPLE_RATE // divisor, rate // divisor)
            original = wave.copy(); wave = fixed_wave(wave)
            inputs = processor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.inference_mode():
                embedding = model(**{k: v.to(model.device) for k, v in inputs.items()}, output_hidden_states=True).hidden_states[9][0].cpu()
                mel = native_speecht5_mel(torch.from_numpy(wave))[0].cpu()
            if embedding.shape != (len(times), model.config.hidden_size):
                raise RuntimeError("teacher time axis does not match output")
            if "mel_times" not in h5:
                # SpeechT5 uses centered STFT, hop=256 samples (16 ms).
                h5.create_dataset("mel_times", data=np.arange(mel.shape[-1], dtype="float32") * 256 / SAMPLE_RATE)
            if mel.shape[-1] != len(h5["mel_times"]):
                raise RuntimeError("mel frontend changed")
            group = groups.create_group(row.audio_key)
            group.attrs["source_samples_16k"] = len(original)
            for key, value in {"teacher": embedding.numpy(), "mel": mel.numpy(), "wave": wave,
                               "mfcc": content_features(original)[0]}.items():
                if not np.isfinite(value).all():
                    raise ValueError(f"nonfinite {key}")
                group.create_dataset(key, data=value.astype("float32"), compression="gzip")
            group.attrs["complete"] = True
            h5.flush()
            if count % 25 == 0:
                print(json.dumps({"cached": count + 1, "total": len(frame)}), flush=True)
    working.replace(target)


def dataset_for(cfg, role, *, audio=False, m0=False, manifest=None, cache=None):
    _, default_manifest, default_cache, normalizer = artifact_paths(cfg)
    return AlignedDataset(ROOT, Path(manifest) if manifest else default_manifest,
                          Path(cache) if cache else default_cache, role,
                          None if audio else normalizer, audio_only=audio, m0=m0)


def fit_speech_stats(decoder, dataset):
    total = 0; sums = None; squares = None
    with h5py.File(dataset.cache, "r") as h5:
        for key in sorted(set(dataset.frame.audio_key)):
            value = torch.from_numpy(h5["targets"][key]["teacher"][:]).double()
            if sums is None:
                sums = value.sum(0); squares = value.square().sum(0)
            else:
                sums += value.sum(0); squares += value.square().sum(0)
            total += len(value)
    decoder.normalizer.mean.copy_((sums / total).float())
    decoder.normalizer.scale.copy_((squares / total - (sums / total).square()).clamp_min(1e-8).sqrt().float())


def make_decoder(cfg, dataset):
    with h5py.File(dataset.cache, "r") as h5:
        dimension = h5["targets"][dataset.frame.iloc[0].audio_key]["teacher"].shape[-1]
    spec = dict(speech_times=dataset.speech_times.tolist(), mel_times=dataset.mel_times.tolist(),
                speech_dimension=dimension, **cfg["decoder"])
    model = AcousticDecoder(dataset.speech_times, dataset.mel_times, speech_dimension=dimension, **cfg["decoder"])
    fit_speech_stats(model, dataset)
    return model, spec


def validate_audio(model, dataset, target, batch_size):
    model.eval(); errors = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size):
            batch = move(batch, target)
            errors.extend((audio_forward(model, batch) - batch["mel"]).abs().mean((1, 2)).cpu().tolist())
    return float(np.mean(errors))


def train_audio(args, cfg):
    seed_all(args.seed); target = device(args.device)
    external = args.stage == "pretrain"
    if not external and (args.manifest or args.cache):
        raise ValueError("adapt/MFCC must use this experiment's exact train/validation split")
    if external and (not args.manifest or not args.cache):
        raise ValueError("pretrain requires LibriSpeech --manifest and --cache")
    train = dataset_for(cfg, "train", audio=True, manifest=args.manifest, cache=args.cache)
    validation = dataset_for(cfg, "validation", audio=True, manifest=args.manifest, cache=args.cache)
    if args.stage == "mfcc":
        model, spec = MFCCBaseline(len(train.mel_times)), {"kind": "mfcc", "frames": len(train.mel_times)}
    elif external:
        if "offset_samples" not in train.frame:
            raise ValueError("pretrain requires the external corpus manifest")
        experiment_manifest = artifact_paths(cfg)[1]
        if experiment_manifest.exists():
            experiment = pd.read_csv(experiment_manifest, keep_default_na=False)
            if set(experiment.audio_sha256) & set(train.frame.audio_sha256):
                raise ValueError("external pretraining corpus overlaps experimental audio")
        model, spec = make_decoder(cfg, train)
    else:
        if not args.initialize and cfg.get("audio_initialization") == "project_train_only":
            model, spec = make_decoder(cfg, train)
        else:
            if not args.initialize:
                raise ValueError("adapt requires --initialize or project_train_only config")
            initial = load_payload(Path(args.initialize))
            if initial["stage"] != "pretrain" or initial["teacher_sha256"] != train.teacher_sha256:
                raise ValueError("adaptation requires matching frozen teacher and pretraining stage")
            model, spec = decoder_from(initial), initial["decoder_spec"]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    model.to(target)
    train_loop(args, cfg, model, train, validation, output, target, spec, audio=True)


def train_loop(args, cfg, model, train, validation, output, target, decoder_spec, *, audio):
    """Resume at exact minibatch offsets; selection never evaluates test data."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    decoder = model if audio else model.decoder
    labels, bank = (None, None) if audio else fixed_bank(train, decoder.normalizer.cpu())
    model.to(target)
    bank = bank.to(target) if bank is not None else None
    lookup = {label: i for i, label in enumerate(labels)} if labels else {}
    signature = {"config": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                 "manifest": sha256(Path(args.manifest) if args.manifest else artifact_paths(cfg)[1]),
                 "cache": sha256(train.cache), "initialize": sha256(Path(args.initialize)) if args.initialize else None,
                 "stage": args.stage, "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size,
                 "epochs": args.epochs, "max_steps": args.max_steps, "lag_ms": getattr(args, "lag_ms", 0),
                 "m0": bool(getattr(args, "m0", False))}
    progress = output / "training_state.pt"
    step = start_epoch = start_batch = 0; best = float("inf"); stale = 0; history = []
    if progress.exists():
        saved = load_payload(progress)
        if saved["signature"] != signature:
            raise RuntimeError("resume inputs differ; choose a new output directory")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(target)
        step, start_epoch, start_batch = saved["step"], saved["epoch"], saved["next_batch"]
        best, stale, history = saved["best"], saved["stale"], saved["history"]
        torch.set_rng_state(saved["torch_rng"]); np.random.set_state(saved["numpy_rng"]); random.setstate(saved["python_rng"])
        if target.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if target.type == "mps" and saved.get("mps_rng") is not None:
            torch.mps.set_rng_state(saved["mps_rng"])
        if saved["complete"]:
            print("training already complete"); return

    def payload(epoch, next_batch, complete=False):
        return {"contract": CONTRACT, "runtime_hash": runtime_hash(), "signature": signature,
                "stage": args.stage, "teacher_sha256": train.teacher_sha256, "decoder_spec": decoder_spec,
                "decoder": decoder.state_dict(), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "eeg_spec": cfg["encoder"] if not audio else None, "lag_ms": getattr(args, "lag_ms", 0),
                "decoder_origin_sha256": getattr(args, "decoder_origin", None),
                "step": step, "epoch": epoch, "next_batch": next_batch, "best": best, "stale": stale,
                "history": history, "complete": complete, "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if target.type == "cuda" else None,
                "mps_rng": torch.mps.get_rng_state() if target.type == "mps" else None}

    epoch, next_batch = start_epoch, start_batch
    stop_requested = [False]
    previous_handler = signal.getsignal(signal.SIGINT)
    def stop_after_step(signum, frame):
        stop_requested[0] = True
        print("interrupt requested; finishing optimizer step before saving", flush=True)
    signal.signal(signal.SIGINT, stop_after_step)
    try:
        for epoch in range(start_epoch, args.epochs):
            generator = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(len(train), generator=generator).tolist()
            batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
            model.train()
            if not audio:
                decoder.eval()
            for index, ids in enumerate(batches):
                if epoch == start_epoch and index < start_batch:
                    continue
                batch = move(torch.utils.data.default_collate([train[i] for i in ids]), target)
                optimizer.zero_grad(set_to_none=True)
                if audio:
                    loss = acoustic_loss(audio_forward(model, batch), batch["mel"])
                else:
                    state = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"])
                    indices = torch.tensor([lookup[c] for c in batch["content"]], device=target)
                    loss, _ = alignment_loss(state, batch["teacher"], batch["mel"], decoder.normalizer,
                                             bank, indices, acoustic=args.stage != "align")
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite loss")
                loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step(); step += 1; next_batch = index + 1
                if stop_requested[0]:
                    atomic_save(progress, payload(epoch, next_batch))
                    print("resumable state saved", flush=True)
                    return
                if step % args.checkpoint_every == 0:
                    atomic_save(progress, payload(epoch, next_batch))
                    print(json.dumps({"step": step, "epoch": epoch, "loss": float(loss.detach())}), flush=True)
                if args.max_steps and step >= args.max_steps:
                    break
            if audio:
                metric = validate_audio(model, validation, target, args.batch_size)
                report = {"native_mel_mae": metric}; eligible = True
            else:
                report = evaluate_model(model, validation, train, target, args.batch_size, bootstrap=False)
                metric = report["native_mel_mae"]
                # Alignment-stage selection uses representation error; acoustic
                # eligibility becomes mandatory for M0 and fine-tuning stages.
                eligible = report["beats_template"] and report["beats_wrong_trial"]
                if args.stage == "align":
                    metric = report["sequence_mae"]; eligible = True
            history.append({"epoch": epoch + 1, "step": step, "eligible": bool(eligible), **report})
            if eligible and metric < best - 1e-5:
                best = metric; stale = 0
                atomic_save(output / "best_checkpoint.pt", payload(epoch + 1, 0))
            else:
                stale += 1
            atomic_json(output / "metrics.json", {"stage": args.stage, "history": history,
                                                    "best_metric": best if np.isfinite(best) else None})
            print(json.dumps({"stage": args.stage, "epoch": epoch + 1, "metric": metric, "eligible": bool(eligible)}), flush=True)
            complete = ((epoch + 1 >= args.epochs) or (args.max_steps and step >= args.max_steps) or
                        (epoch + 1 >= cfg["minimum_epochs"] and stale >= cfg["patience"]))
            atomic_save(progress, payload(epoch + 1, 0, complete=bool(complete)))
            atomic_save(output / "last_checkpoint.pt", payload(epoch + 1, 0, complete=bool(complete)))
            if complete:
                break
    except KeyboardInterrupt:
        atomic_save(progress, payload(epoch, next_batch))
        print("interrupted; resumable training state saved", flush=True)
        raise
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def model_from(payload):
    model = AlignedEEGModel(decoder_from(payload), lag_ms=payload["lag_ms"], **payload["eeg_spec"])
    model.load_state_dict(payload["model"])
    return model


def train_eeg(args, cfg):
    seed_all(args.seed); target = device(args.device)
    initial = load_payload(Path(args.initialize))
    train = dataset_for(cfg, "train", m0=args.m0)
    validation = train if args.m0 else dataset_for(cfg, "validation")
    if initial["teacher_sha256"] != train.teacher_sha256:
        raise ValueError("teacher changed between audio and EEG training")
    check_eeg_artifacts(initial, cfg)
    if args.stage == "align":
        if initial["stage"] != "adapt":
            raise ValueError("alignment must start from train-fold adapted acoustic decoder")
        if initial["signature"]["manifest"] != sha256(artifact_paths(cfg)[1]):
            raise ValueError("acoustic decoder was adapted on another split")
        model = AlignedEEGModel(decoder_from(initial), lag_ms=args.lag_ms, **cfg["encoder"])
        args.decoder_origin = sha256(Path(args.initialize))
    else:
        if initial["stage"] != "align" or initial["signature"]["m0"] != args.m0 or initial["lag_ms"] != args.lag_ms:
            raise ValueError("fine-tuning requires corresponding alignment checkpoint, M0 mode and lag")
        model = model_from(initial)
        args.decoder_origin = initial["decoder_origin_sha256"]
    if not args.m0:
        if getattr(args, "objective_only", False):
            if not args.m0_report:
                raise ValueError("objective-only EEG training still requires --m0-report")
            gate = json.loads(Path(args.m0_report).read_text())
            if not gate.get("m0_passed") or gate.get("role") != "train" or not gate.get("m0") or gate.get("stage") != "finetune":
                raise RuntimeError("M0 reconstruction gate has not passed")
            if gate.get("manifest_sha256") != sha256(artifact_paths(cfg)[1]) or gate.get("decoder_origin_sha256") != args.decoder_origin:
                raise RuntimeError("M0 report uses a different split or decoder")
        elif not args.audio_review or not args.m0_report:
            raise ValueError("full EEG training requires --audio-review and --m0-report; use --m0 for engineering closure")
        else:
            review = json.loads(Path(args.audio_review).read_text())
            gate = json.loads(Path(args.m0_report).read_text())
            decoder_origin = args.decoder_origin
            if not review.get("passed") or review.get("kind") != "audio" or review.get("checkpoint_sha256") != decoder_origin:
                raise RuntimeError("audio intelligibility review must pass for this exact adapted decoder")
            if not gate.get("m0_passed") or gate.get("role") != "train" or not gate.get("m0") or gate.get("stage") != "finetune":
                raise RuntimeError("M0 reconstruction gate has not passed")
            if gate.get("manifest_sha256") != sha256(artifact_paths(cfg)[1]) or gate.get("decoder_origin_sha256") != decoder_origin:
                raise RuntimeError("M0 report uses a different split or decoder")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    train_loop(args, cfg, model.to(target), train, validation, output, target, initial["decoder_spec"], audio=False)


def correlation(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    a = a - a.mean(); b = b - b.mean()
    denominator = a.norm() * b.norm()
    return float((a @ b / denominator).clamp(-1, 1)) if denominator > 1e-12 else 0.0


def evaluate_model(model, dataset, train_dataset, target, batch_size, *, bootstrap=True, include_records=False):
    model.eval(); template = torch.zeros(80, len(dataset.mel_times)); count = 0
    with h5py.File(train_dataset.cache, "r") as h5:
        for key in sorted(set(train_dataset.frame.audio_key)):
            template += torch.from_numpy(h5["targets"][key]["mel"][:]); count += 1
    template = (template / count).to(target)
    wrong = wrong_trial_indices(dataset.frame)
    predictions, targets, embeddings, teachers, records = [], [], [], [], []
    with torch.inference_mode():
        for offset in range(0, len(dataset), batch_size):
            ids = list(range(offset, min(len(dataset), offset + batch_size)))
            batch = move(torch.utils.data.default_collate([dataset[i] for i in ids]), target)
            swapped = move(torch.utils.data.default_collate([dataset[wrong[i]] for i in ids]), target)
            states = {}
            for control in ("correct", "zero", "wrong_trial", "time_block_shuffle", "channel_shuffle"):
                if control == "correct":
                    eeg = batch["eeg"]
                elif control == "wrong_trial":
                    eeg = swapped["eeg"]
                else:
                    eeg = counterfactual_eeg(batch["eeg"], control, time_mask=batch["time_mask"], channel_mask=batch["channel_mask"])
                states[control] = model(eeg, batch["channel_xyz"], batch["channel_mask"], batch["time_mask"])
            state = states["correct"]
            predictions.append(state.native_mel.cpu()); targets.append(batch["mel"].cpu())
            embeddings.append(state.global_embedding.cpu())
            teachers.append(F.normalize(model.decoder.normalizer(batch["teacher"]).mean(1), dim=-1).cpu())
            for i in range(len(ids)):
                errors = {name: float((value.native_mel[i] - batch["mel"][i]).abs().mean()) for name, value in states.items()}
                baseline = float((template - batch["mel"][i]).abs().mean())
                record = {"trial_id": batch["trial_id"][i], "subject": batch["subject"][i], "content": batch["content"][i],
                          "native_mel_mae": errors["correct"], "template_mae": baseline,
                          "sequence_mae": float((model.decoder.normalizer(state.aligned_sequence[i]) -
                                                   model.decoder.normalizer(batch["teacher"][i])).abs().mean()),
                          "mel_temporal_profile_correlation": correlation(state.native_mel[i].mean(0), batch["mel"][i].mean(0))}
                record.update({name + "_gain": value - errors["correct"] for name, value in errors.items() if name != "correct"})
                records.append(record)
    pred, truth_mel = torch.cat(predictions), torch.cat(targets)
    emb, teacher = torch.cat(embeddings), torch.cat(teachers)
    labels = [r["content"] for r in records]; unique = sorted(set(labels))
    prototypes = torch.stack([teacher[torch.tensor([x == label for x in labels])].mean(0) for label in unique])
    order = (emb @ F.normalize(prototypes, dim=-1).T).argsort(1, descending=True)
    indices = torch.tensor([unique.index(label) for label in labels])
    ranks = (order == indices[:, None]).nonzero()[:, 1].float() + 1
    variance = float(pred.var(0, unbiased=False).mean() / truth_mel.var(0, unbiased=False).mean().clamp_min(1e-8))
    result = {"pairs": len(records), "unique_contents": len(unique), "retrieval_r1": float((ranks == 1).float().mean()),
              "retrieval_mrr": float((1 / ranks).mean()), "chance_r1": 1 / len(unique), "prediction_variance_ratio": variance}
    for key in records[0]:
        if key not in ("trial_id", "subject", "content"):
            result[key] = float(np.mean([r[key] for r in records]))
    result["template_improvement"] = 1 - result["native_mel_mae"] / max(result["template_mae"], 1e-8)
    result["beats_template"] = result["template_improvement"] > 0
    result["beats_wrong_trial"] = result["wrong_trial_gain"] > 0
    result["m0_passed"] = bool(result["retrieval_r1"] >= .9 - 1e-6 and result["template_improvement"] >= .1 and
                                variance >= .25 and all(result[c + "_gain"] > 0 for c in ("zero", "wrong_trial", "time_block_shuffle")))
    if bootstrap:
        result["subject_content_bootstrap"] = {key: two_way_bootstrap(records, key) for key in
                                               ("native_mel_mae", "wrong_trial_gain", "zero_gain", "time_block_shuffle_gain")}
    if include_records:
        result["records"] = records
    return result


def check_eeg_artifacts(payload, cfg):
    _, manifest, cache, _ = artifact_paths(cfg)
    if payload["signature"]["manifest"] != sha256(manifest) or payload["signature"]["cache"] != sha256(cache):
        raise RuntimeError("evaluation artifacts differ from training")


def require_selection(args, payload):
    if args.role == "test":
        if not args.selection:
            raise ValueError("test export/evaluation requires a validation-only --selection file")
        selected = json.loads(Path(args.selection).read_text())
        if not selected.get("passed") or selected.get("selected_sha256") != sha256(Path(args.checkpoint)) or selected.get("selection_role") != "validation":
            raise ValueError("test checkpoint was not selected on validation")
    if args.m0 and args.role != "train":
        raise ValueError("M0 is training-only")
    if bool(payload["signature"]["m0"]) != bool(args.m0):
        raise ValueError("checkpoint/M0 evaluation mismatch")


def evaluate(args, cfg):
    payload = load_payload(Path(args.checkpoint)); check_eeg_artifacts(payload, cfg); require_selection(args, payload)
    model = model_from(payload).to(device(args.device))
    train = dataset_for(cfg, "train", m0=args.m0)
    data = dataset_for(cfg, args.role, m0=args.m0)
    result = evaluate_model(model, data, train, device(args.device), args.batch_size, include_records=True)
    decoder_origin = payload["decoder_origin_sha256"]
    result.update(contract=CONTRACT, role=args.role, m0=args.m0, lag_ms=payload["lag_ms"], stage=payload["stage"],
                  checkpoint_sha256=sha256(Path(args.checkpoint)), manifest_sha256=sha256(artifact_paths(cfg)[1]),
                  decoder_origin_sha256=decoder_origin, interpretation="exploratory_previously_inspected_dataset")
    if not args.m0:
        result["m0_passed"] = False
    atomic_json(Path(args.output), result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False))


def select_lag(args, cfg):
    candidates = []
    for path in args.checkpoints:
        payload = load_payload(Path(path)); check_eeg_artifacts(payload, cfg)
        if payload["stage"] != "finetune" or payload["signature"]["m0"]:
            raise ValueError("lag selection requires full fine-tuned EEG candidates")
        model = model_from(payload).to(device(args.device))
        report = evaluate_model(model, dataset_for(cfg, "validation"), dataset_for(cfg, "train"),
                                device(args.device), args.batch_size, bootstrap=False)
        candidates.append({"checkpoint": str(Path(path).resolve()), "sha256": sha256(Path(path)),
                           "seed": payload["signature"]["seed"], "lag_ms": payload["lag_ms"], **report})
        del model
    if sorted(c["lag_ms"] for c in candidates) != list(LAGS_MS) or len({c["seed"] for c in candidates}) != 1:
        raise ValueError("select all five lags for exactly one seed")
    eligible = [c for c in candidates if c["beats_template"] and c["beats_wrong_trial"]]
    if not eligible:
        atomic_json(Path(args.output), {"selection_role": "validation", "passed": False, "candidates": candidates})
        raise RuntimeError("no candidate beats both template and wrong-trial controls")
    best = min(eligible, key=lambda c: (c["native_mel_mae"], c["lag_ms"]))
    atomic_json(Path(args.output), {"selection_role": "validation", "passed": True,
                                   "selected_checkpoint": best["checkpoint"], "selected_sha256": best["sha256"],
                                   "selected_lag_ms": best["lag_ms"], "candidates": candidates})


def probe(args, cfg):
    target = device(args.device); seed_all(31)
    speech_times = convolution_times(WAVE_SAMPLES, [10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2])
    decoder = AcousticDecoder(speech_times, torch.arange(251).float() * .016, **cfg["decoder"])
    model = AlignedEEGModel(decoder, **cfg["encoder"]).to(target)
    started = time.monotonic()
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    eeg = torch.randn(args.batch_size, args.channels, 1178, device=target)
    xyz = torch.randn(args.batch_size, args.channels, 3, device=target)
    channels = torch.ones(args.batch_size, args.channels, dtype=torch.bool, device=target)
    mask = torch.ones(args.batch_size, 1178, dtype=torch.bool, device=target)
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        state = model(eeg, xyz, channels, mask); state.native_mel.square().mean().backward()
    if target.type == "cuda":
        torch.cuda.synchronize()
    if target.type == "mps":
        torch.mps.synchronize()
    result = {"device": str(target), "gpu": torch.cuda.get_device_name() if target.type == "cuda" else None,
              "batch_size": args.batch_size, "seconds_per_forward_backward": (time.monotonic() - started) / 3,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if target.type == "cuda" else None,
              "parameters": sum(p.numel() for p in model.parameters()), "synthetic_only": True}
    atomic_json(Path(args.output), result); print(json.dumps(result))


def readiness(args, cfg):
    from importlib.util import find_spec
    base, manifest, cache, normalizer = artifact_paths(cfg)
    result = {"cuda_available": torch.cuda.is_available(), "split_prepared": (base / "split_report.json").exists(),
              "mps_available": torch.backends.mps.is_available(),
              "manifest": manifest.exists(), "cache": cache.exists(),
              "eeg_normalizer": normalizer.exists(), "hubert": (Path(os.environ.get("HUBERT_LOCAL_PATH", "/missing")) / "config.json").exists(),
              "hifigan": (Path(os.environ.get("HIFIGAN_LOCAL_PATH", "/missing")) / "config.json").exists(),
              "official_stimulus_parameters": DS004940_STIMULUS_PARAMETERS.exists(),
              "optional_audio_io_soundfile": find_spec("soundfile") is not None,
              "training_status": "inspect_output_reports", "human_review": "not_assumed"}
    print(json.dumps(result, indent=2))


def summarize(args, cfg):
    root = Path(args.output_root)
    rows = []
    split = artifact_paths(cfg)[0] / "split_report.json"
    if split.exists():
        value = json.loads(split.read_text())
        rows.append({"stage": "data", "status": "materialized" if value["materialized"] else "split_prepared",
                     "role_counts": value["role_counts"], "m0_pairs": value["m0_pairs"]})
    local_audio = cfg.get("audio_initialization") == "project_train_only"
    if local_audio:
        for stage in ("hubert", "hifigan"):
            path = root / stage / "adaptation_report.json"
            value = json.loads(path.read_text()) if path.exists() else {}
            rows.append({"stage": stage, "status": value.get("status", "not_run_or_incomplete"),
                         "epochs_evaluated": len(value.get("history", []))})
    for stage in (("adapt", "mfcc", "m0_align", "m0_finetune") if local_audio else ("pretrain", "adapt", "mfcc", "m0_align", "m0_finetune")):
        path = root / stage / "metrics.json"
        if path.exists():
            value = json.loads(path.read_text())
            rows.append({"stage": stage, "status": "metrics_available", "best_metric": value["best_metric"],
                         "epochs_evaluated": len(value["history"])})
        else:
            rows.append({"stage": stage, "status": "not_run_or_incomplete"})
    for seed in (31, 47, 73):
        path = root / f"seed-{seed}/test_evaluation.json"
        if path.exists():
            value = json.loads(path.read_text())
            rows.append({"stage": f"test_seed_{seed}", "status": "evaluated",
                         **{k: value[k] for k in ("native_mel_mae", "retrieval_r1", "prediction_variance_ratio", "wrong_trial_gain")}})
        else:
            rows.append({"stage": f"test_seed_{seed}", "status": "not_evaluated"})
    audio_review = root / "audio_review.json"
    review = json.loads(audio_review.read_text()) if audio_review.exists() else {"passed": None, "status": "not_reviewed"}
    atomic_json(root / "stage_report.json", {"stages": rows, "audio_review": review,
                                            "interpretation": "engineering_and_exploratory_results; no inferred intelligibility"})
    lines = ["# Aligned speech v1 阶段报告", "", "未运行/未填写的阶段不视为通过。", "",
             "| Stage | Status | Metrics |", "|---|---|---|"]
    for row in rows:
        metrics = {k: v for k, v in row.items() if k not in ("stage", "status")}
        lines.append(f"| {row['stage']} | {row['status']} | {json.dumps(metrics)} |")
    lines.extend(["", f"音频人工评价通过：{review.get('passed')}", "",
                  "测试集属于已有研究数据的探索性再分析。可懂度必须查阅各 seed 的人工评价结果。"])
    (root / "stage_report.md").write_text("\n".join(lines) + "\n")
    print(str(root / "stage_report.md"))


def export_audio(args, cfg):
    from scipy.io import wavfile
    target = device(args.device); payload = load_payload(Path(args.checkpoint))
    check_eeg_artifacts(payload, cfg)
    audio_only = args.kind == "audio"
    if audio_only:
        if payload["stage"] != "adapt" or args.role != "validation":
            raise ValueError("audio oracle listening uses the adapted decoder and validation fold only")
        model = decoder_from(payload).to(target).eval()
        if not args.mfcc_checkpoint:
            raise ValueError("three-path oracle requires --mfcc-checkpoint refitted on the new training fold")
        mfcc_payload = load_payload(Path(args.mfcc_checkpoint)); check_eeg_artifacts(mfcc_payload, cfg)
        if mfcc_payload["stage"] != "mfcc":
            raise ValueError("not a new-fold MFCC baseline checkpoint")
        mfcc_model = decoder_from(mfcc_payload).to(target).eval()
    else:
        require_selection(args, payload)
        model = model_from(payload).to(target).eval()
    vocoder_path = Path(args.hifigan or os.environ.get("HIFIGAN_LOCAL_PATH", "/missing"))
    vocoder = SpeechT5HiFiGan(vocoder_path, device=target)
    data = dataset_for(cfg, args.role, audio=audio_only, m0=args.m0)
    wrong = None if audio_only else wrong_trial_indices(data.frame)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    signature = {"kind": args.kind, "role": args.role, "checkpoint_sha256": sha256(Path(args.checkpoint)),
                 "manifest_sha256": sha256(artifact_paths(cfg)[1]), "vocoder_sha256": tree_hash(vocoder_path),
                 "mfcc_checkpoint_sha256": sha256(Path(args.mfcc_checkpoint)) if audio_only else None}
    passport = output / "export_manifest.json"
    if passport.exists():
        existing = json.loads(passport.read_text())
        if existing["signature"] != signature:
            raise ValueError("export directory belongs to a different experiment")
    content_rows = data.frame.drop_duplicates("content_group").copy()
    content_rows["order"] = content_rows.content_group.map(lambda c: hashlib.sha256(f"listening:31:{c}".encode()).hexdigest())
    selected = set(content_rows.sort_values("order").head(40).index)
    if len(selected) < 40 and not args.m0:
        raise ValueError("listening study needs at least 40 distinct held-out contents")
    blind_rows, private_rows, metric_rows, references = [], [], [], []
    official_references, reference_source = official_reference_transcripts(data.frame)
    salt = signature["checkpoint_sha256"]
    with torch.inference_mode():
        for i in range(len(data)):
            batch = move(torch.utils.data.default_collate([data[i]]), target)
            trial = batch["trial_id"][0]
            bundle = output / "bundles" / trial; bundle.mkdir(parents=True, exist_ok=True)
            waves = {"source": batch["wave"][0].cpu().numpy(),
                     "mel_oracle": vocoder.synthesize(batch["mel"])[0].cpu().numpy()[:WAVE_SAMPLES]}
            if audio_only:
                waves["teacher_oracle"] = vocoder.synthesize(model(batch["teacher"]))[0].cpu().numpy()[:WAVE_SAMPLES]
                waves["mfcc_oracle"] = vocoder.synthesize(audio_forward(mfcc_model, batch))[0].cpu().numpy()[:WAVE_SAMPLES]
            else:
                swapped = data[wrong[i]]["eeg"].unsqueeze(0).to(target)
                for control in ("correct", "wrong_trial", "zero", "time_block_shuffle", "channel_shuffle"):
                    eeg = (batch["eeg"] if control == "correct" else swapped if control == "wrong_trial" else
                           counterfactual_eeg(batch["eeg"], control, time_mask=batch["time_mask"], channel_mask=batch["channel_mask"]))
                    state = model(eeg, batch["channel_xyz"], batch["channel_mask"], batch["time_mask"])
                    waves[control] = vocoder.synthesize(state.native_mel)[0].cpu().numpy()[:WAVE_SAMPLES]
            source_envelope = np.sqrt(np.mean(waves["source"].reshape(-1, 160) ** 2, axis=1))
            row = {"trial_id": trial, "subject": batch["subject"][0], "content": batch["content"][0]}
            for name, wave in waves.items():
                if len(wave) != WAVE_SAMPLES or not np.isfinite(wave).all():
                    raise RuntimeError("vocoder did not produce finite fixed-four-second audio")
                # Float WAV avoids per-file normalization or hidden clipping.
                wavfile.write(bundle / f"{name}.wav", SAMPLE_RATE, wave.astype(np.float32))
                if name != "source":
                    envelope = np.sqrt(np.mean(wave.reshape(-1, 160) ** 2, axis=1))
                    row[name + "_envelope_correlation"] = correlation(torch.from_numpy(envelope), torch.from_numpy(source_envelope))
                if i in selected and name not in ("source", "mel_oracle", "zero", "time_block_shuffle", "channel_shuffle"):
                    sample = hashlib.sha256(f"{salt}:{trial}:{name}".encode()).hexdigest()[:20]
                    public = output / "blind"; public.mkdir(exist_ok=True)
                    wavfile.write(public / f"{sample}.wav", SAMPLE_RATE, wave.astype(np.float32))
                    for listener in ("listener-1", "listener-2", "listener-3"):
                        blind_rows.append({"listener_id": listener, "sample_id": sample, "transcript": ""})
                    private_rows.append({"sample_id": sample, "trial_id": trial, "condition": name})
            if i in selected:
                references.append({"trial_id": trial, **official_references[trial]})
            metric_rows.append(row)
            if i % 25 == 0:
                print(json.dumps({"exported": i + 1, "total": len(data)}), flush=True)
    # Never replace transcriptions entered after a previous export.
    def create_csv(path, records):
        if not path.exists():
            pd.DataFrame(records).to_csv(path, index=False)
    blind_rows.sort(key=lambda r: hashlib.sha256(f"{r['listener_id']}:{r['sample_id']}".encode()).hexdigest())
    create_csv(output / "blind/transcriptions.csv", blind_rows)
    create_csv(output / "private_key.csv", private_rows)
    create_csv(output / "reference_transcripts.csv", references)
    pd.DataFrame(metric_rows).to_csv(output / "waveform_metrics.csv", index=False)
    atomic_json(passport, {"contract": CONTRACT, "signature": signature, "pairs": len(data),
                           "listening_contents": len(selected), "sample_rate": SAMPLE_RATE, "wave_samples": WAVE_SAMPLES,
                           "blind_csv_sha256": sha256(output / "private_key.csv"),
                           "reference_source": reference_source,
                           "interpretation": "objective_metrics_are_not_a_human_intelligibility_pass"})


def official_reference_transcripts(frame: pd.DataFrame) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Map DS004940 BIDS ``stim_file`` values to the authors' written stimuli."""
    if not DS004940_STIMULUS_PARAMETERS.is_file():
        raise FileNotFoundError(
            "DS004940 official stimulus table is missing: "
            f"{DS004940_STIMULUS_PARAMETERS}. Restore it before creating a listening export."
        )
    table = pd.read_csv(DS004940_STIMULUS_PARAMETERS, sep="\t", keep_default_na=False)
    word_columns = [str(i) for i in range(1, 9)]
    if not {"stim_file", *word_columns} <= set(table):
        raise ValueError("official DS004940 stimulus table has an unexpected schema")
    table["stim_file"] = table.stim_file.map(lambda x: Path(str(x)).name)
    if table.stim_file.duplicated().any():
        raise ValueError("official DS004940 stimulus table duplicates a file name")
    sentences = {}
    for _, row in table.iterrows():
        tokens = [str(row[column]).strip() for column in word_columns
                  if str(row[column]).strip().lower() not in {"", "n/a", "na", "nan", "none"}]
        sentence = " ".join(tokens)
        if not sentence:
            raise ValueError(f"official DS004940 stimulus is textless: {row.stim_file}")
        sentences[row.stim_file] = sentence
    result = {}
    for row in frame.itertuples(index=False):
        stimulus = Path(str(row.stim_file)).name
        if stimulus not in sentences:
            raise ValueError(f"no official sentence reference for {stimulus}")
        result[str(row.trial_id)] = {"reference_transcript": sentences[stimulus], "verified": True}
    return result, {"kind": "official_ds004940_stimulus_parameters",
                    "path": str(DS004940_STIMULUS_PARAMETERS),
                    "sha256": sha256(DS004940_STIMULUS_PARAMETERS)}


def write_references(args, cfg):
    """Materialize all official sentence references before any model training."""
    _, manifest, _, _ = artifact_paths(cfg)
    if not manifest.is_file():
        raise FileNotFoundError("run prepare first so the versioned manifest exists")
    frame = pd.read_csv(manifest, keep_default_na=False).drop_duplicates("audio_key")
    values, source = official_reference_transcripts(frame)
    output = Path(args.output) if args.output else path_of(cfg, "artifact_root") / "official_reference_transcripts.csv"
    rows = []
    for row in frame.itertuples(index=False):
        rows.append({"trial_id": row.trial_id, "stim_file": row.stim_file, **values[str(row.trial_id)]})
    pd.DataFrame(rows).sort_values("stim_file").to_csv(output, index=False)
    atomic_json(output.with_suffix(".provenance.json"), {"source": source, "manifest_sha256": sha256(manifest), "rows": len(rows)})
    print(json.dumps({"output": str(output), "rows": len(rows), "source": source}, ensure_ascii=False))


def words(text):
    import re
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower().replace("’", "'"))


def word_accuracy(reference, hypothesis):
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        raise ValueError("empty reference transcription")
    previous = list(range(len(hyp) + 1))
    for i, token in enumerate(ref, 1):
        current = [i]
        for j, other in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (token != other)))
        previous = current
    return max(0., 1 - previous[-1] / len(ref))


def score_review(args, cfg):
    directory = Path(args.export_root)
    passport = json.loads((directory / "export_manifest.json").read_text())
    key = pd.read_csv(directory / "private_key.csv", keep_default_na=False)
    if sha256(directory / "private_key.csv") != passport["blind_csv_sha256"]:
        raise ValueError("listening assignment changed")
    answers = pd.read_csv(args.transcriptions, keep_default_na=False)
    references = pd.read_csv(args.references, keep_default_na=False)
    if references.trial_id.duplicated().any() or not truth(references.verified).all():
        raise ValueError("reference transcripts must be individually verified and unique")
    if answers.duplicated(["listener_id", "sample_id"]).any() or answers.listener_id.nunique() < 3:
        raise ValueError("need at least three independent listeners, without duplicate responses")
    if (answers.listener_id == "").any() or set(answers.sample_id) != set(key.sample_id):
        raise ValueError("missing or unknown blinded samples/listeners")
    if (answers.transcript.str.strip() == "").any():
        raise ValueError("unfinished transcriptions; use [unintelligible] for a completed but unrecognized sample")
    for _, group in answers.groupby("listener_id"):
        if set(group.sample_id) != set(key.sample_id):
            raise ValueError("each listener must transcribe every assigned sample")
    frame = answers.merge(key, on="sample_id", validate="many_to_one").merge(references, on="trial_id", validate="many_to_one")
    if len(frame) != len(answers) or frame.trial_id.nunique() < 40:
        raise ValueError("need verified references for forty distinct sentences")
    frame["word_accuracy"] = [word_accuracy(r.reference_transcript, "" if r.transcript == "[unintelligible]" else r.transcript)
                              for r in frame.itertuples()]
    means = frame.groupby("condition").word_accuracy.mean().to_dict()
    kind = passport["signature"]["kind"]
    if kind == "audio":
        passed = means.get("teacher_oracle", 0) >= .9
    else:
        passed = means.get("correct", 0) >= .7 and means.get("correct", 0) - means.get("wrong_trial", 1) >= .2
    atomic_json(Path(args.output), {"kind": kind, "passed": bool(passed), "word_accuracy": means,
                                   "listeners": int(frame.listener_id.nunique()), "sentences": int(frame.trial_id.nunique()),
                                   "checkpoint_sha256": passport["signature"]["checkpoint_sha256"],
                                   "transcriptions_sha256": sha256(Path(args.transcriptions)), "references_sha256": sha256(Path(args.references)),
                                   "note": "listener independence must be verified by the study organizer"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/aligned_speech_v1.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("readiness")
    p = sub.add_parser("report"); p.add_argument("--output-root", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--materialize", action="store_true")
    p = sub.add_parser("references"); p.add_argument("--output")
    p = sub.add_parser("corpus-manifest"); p.add_argument("--corpus-root", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("cache"); p.add_argument("--manifest"); p.add_argument("--output"); p.add_argument("--hubert"); p.add_argument("--device", default="auto")
    for name in ("train-audio", "train-eeg"):
        p = sub.add_parser(name)
        p.add_argument("--stage", choices=("pretrain", "adapt", "mfcc") if name == "train-audio" else ("align", "finetune"), required=True)
        p.add_argument("--initialize", required=name == "train-eeg")
        p.add_argument("--manifest"); p.add_argument("--cache")
        p.add_argument("--output", required=True); p.add_argument("--device", default="auto")
        p.add_argument("--seed", type=int, default=31); p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--epochs", type=int, default=100); p.add_argument("--max-steps", type=int, default=0)
        p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--checkpoint-every", type=int, default=50)
        if name == "train-eeg":
            p.add_argument("--lag-ms", type=int, choices=LAGS_MS, default=0); p.add_argument("--m0", action="store_true")
            p.add_argument("--objective-only", action="store_true")
            p.add_argument("--audio-review"); p.add_argument("--m0-report")
    for name in ("evaluate", "export"):
        p = sub.add_parser(name); p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
        p.add_argument("--role", choices=("train", "validation", "test"), default="validation")
        p.add_argument("--device", default="auto"); p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--m0", action="store_true"); p.add_argument("--selection")
        if name == "export":
            p.add_argument("--kind", choices=("audio", "eeg"), required=True); p.add_argument("--hifigan"); p.add_argument("--mfcc-checkpoint")
    p = sub.add_parser("select-lag"); p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--device", default="auto"); p.add_argument("--batch-size", type=int, default=4); p.add_argument("--output", required=True)
    p = sub.add_parser("probe"); p.add_argument("--device", default="auto"); p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--channels", type=int, default=128); p.add_argument("--output", required=True)
    p = sub.add_parser("score-review"); p.add_argument("--export-root", required=True)
    p.add_argument("--transcriptions", required=True); p.add_argument("--references", required=True); p.add_argument("--output", required=True)
    args = parser.parse_args(); cfg = config(args.config)
    for name in ("batch_size", "epochs", "checkpoint_every", "channels"):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if hasattr(args, "max_steps") and args.max_steps < 0:
        parser.error("max_steps cannot be negative")
    if args.command == "train-eeg" and (args.manifest or args.cache):
        parser.error("EEG training uses the versioned experiment manifest/cache only")
    commands = {"prepare": prepare, "references": write_references, "corpus-manifest": corpus_manifest, "cache": cache_targets,
                "train-audio": train_audio, "train-eeg": train_eeg, "evaluate": evaluate,
                "select-lag": select_lag, "probe": probe, "readiness": readiness,
                "export": export_audio, "score-review": score_review, "report": summarize}
    commands[args.command](args, cfg)


if __name__ == "__main__":
    main()
