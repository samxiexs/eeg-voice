#!/usr/bin/env python3
"""Evaluate content prediction, retrieval and EEG counterfactual controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(APP / "src"))

from eeg2speech.data import (JointManifestDataset, homogeneous_collate,
                             phoneme_vocabulary_from_manifest, pilot_indices)
from eeg2speech.losses import counterfactual_eeg
from eeg2speech.model import AudioMFCCRenderer, JointEEGContentModel


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


def _resolve(path: str | Path, base: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (base / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_code_sha256() -> str:
    digest = hashlib.sha256()
    paths = sorted([APP / "train_joint.py", *list((APP / "src").rglob("*.py"))], key=lambda path: str(path))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode()); digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _model_code_sha256() -> str:
    digest = hashlib.sha256()
    paths = sorted((APP / "src").rglob("*.py"), key=lambda path: str(path))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode()); digest.update(_sha256(path).encode())
    return digest.hexdigest()


def bootstrap_mean(values: list[float], seed: int = 31, repetitions: int = 2000) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    boot = np.asarray([rng.choice(array, len(array), replace=True).mean() for _ in range(repetitions)])
    return {"mean": float(array.mean()), "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975))}


def compare_results(first: Path, second: Path) -> dict:
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    common = sorted(set(a["subject_mfcc_l1"]) & set(b["subject_mfcc_l1"]))
    # Positive means the second/joint run reduced error.
    gains = [float(a["subject_mfcc_l1"][key]) - float(b["subject_mfcc_l1"][key]) for key in common]
    return {"subjects": common, "single_minus_joint_error_gain": bootstrap_mean(gains)}


def content_retrieval(prediction: torch.Tensor, target: torch.Tensor, labels: list[str]) -> dict[str, float]:
    """Multi-positive retrieval: any trial with the same content is correct."""
    left = torch.nn.functional.normalize(prediction.flatten(1), dim=-1)
    right = torch.nn.functional.normalize(target.flatten(1), dim=-1)
    order = (left @ right.T).argsort(1, descending=True)
    positive = torch.tensor([[a == b for b in labels] for a in labels], dtype=torch.bool)
    ranked_positive = positive.gather(1, order)
    first = ranked_positive.float().argmax(1) + 1
    return {
        "r1": float((first == 1).float().mean()),
        "mrr": float((1.0 / first.float()).mean()),
        # Expected R@1 for a uniformly selected candidate under the same
        # multi-positive label structure. This avoids hard-coded 1/N lines in
        # downstream comparison figures when contents repeat across subjects.
        "chance_r1": float(positive.float().mean()),
        "unique_contents": int(len(set(labels))),
    }


def template_metrics(prediction: torch.Tensor, target: torch.Tensor, labels: list[str],
                     audio_ids: list[str], global_reference: torch.Tensor,
                     global_reference_source: str = "train_fold_mean") -> dict[str, float | bool | int | str | None]:
    if global_reference.shape != target.shape[1:]:
        raise ValueError(f"global template shape {tuple(global_reference.shape)} != target {tuple(target.shape[1:])}")
    global_template = global_reference.unsqueeze(0).expand_as(target)
    error = float((prediction - target).abs().mean())
    global_error = float((global_template - target).abs().mean())
    # Same-content is a diagnostic only when another independently recorded
    # realization exists.  Never include the current waveform in its own
    # template, even when that waveform is repeated across subjects.
    same_errors = []
    same_prediction_errors = []
    for index, (label, audio_id) in enumerate(zip(labels, audio_ids)):
        candidates = [other for other, (other_label, other_audio) in enumerate(zip(labels, audio_ids))
                      if other_label == label and other_audio != audio_id]
        if not candidates:
            continue
        reference = target[candidates].mean(0)
        same_errors.append(float((reference - target[index]).abs().mean()))
        same_prediction_errors.append(float((prediction[index] - target[index]).abs().mean()))
    same_error = float(np.mean(same_errors)) if same_errors else None
    same_prediction_error = float(np.mean(same_prediction_errors)) if same_prediction_errors else None
    return {
        "dataset_mean_template_mfcc_l1": global_error,
        "dataset_mean_template_improvement": float(1.0 - error / max(global_error, 1e-8)),
        "dataset_mean_template_source": global_reference_source,
        "same_content_template_mfcc_l1": same_error,
        "same_content_template_pairs": len(same_errors),
        "same_content_template_gate_applicable": bool(same_error is not None and same_error > 1e-8),
        "same_content_template_improvement": (
            float(1.0 - same_prediction_error / same_error)
            if same_error is not None and same_error > 1e-8 else None
        ),
    }


def training_target_reference(manifest: Path, split: Path, dataset_name: str, targets: Path,
                              normalizer: Path, cfg: dict, vocabulary: dict[str, int],
                              stage: str) -> tuple[torch.Tensor, int]:
    dataset = JointManifestDataset(manifest, split, "train", dataset_name, targets, normalizer,
                                   float(cfg["loss"]["weak_content_weight"]),
                                   supervision_types={"paired_audio", "weak_audio"},
                                   phoneme_vocabulary=vocabulary)
    indices = pilot_indices(dataset, cfg, stage, "train")
    values = []
    for index in indices:
        row = dataset.frame.iloc[index]
        audio_id = dataset._audio_id(row)
        if dataset.targets is None or audio_id not in dataset.targets:
            dataset.close()
            raise RuntimeError(f"train-fold baseline is missing target {audio_id}")
        values.append(torch.from_numpy(dataset.targets[audio_id]["content_mfcc"][:].astype("float32")))
    dataset.close()
    if not values:
        raise RuntimeError("train-fold baseline has zero targets")
    return torch.stack(values).mean(0), len(values)


def registered_collapse_check(templates: dict, gate: dict) -> tuple[str, bool]:
    baseline = str(gate.get("collapse_baseline", "dataset_mean"))
    threshold = float(gate["template_improvement_min"])
    if baseline == "dataset_mean":
        return "registered_dataset_mean_collapse_baseline", templates["dataset_mean_template_improvement"] >= threshold
    if baseline == "same_content_leave_one_realization_out":
        passed = bool(templates["same_content_template_gate_applicable"]) and templates["same_content_template_improvement"] >= threshold
        return "registered_same_content_template", passed
    raise RuntimeError(f"unknown registered collapse baseline {baseline}")


def stratified_error(errors: list[float], metadata: dict[str, list[str]]) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    for key, values in metadata.items():
        groups = defaultdict(list)
        for label, error in zip(values, errors):
            groups[str(label)].append(float(error))
        result[key] = {label: {"pairs": len(group), "mfcc_l1": float(np.mean(group))}
                       for label, group in sorted(groups.items())}
    return result


def pairwise_diversity(values: torch.Tensor, labels: list[str]) -> dict[str, float | int]:
    """Report content-sensitive diversity without treating random audio as EEG evidence."""
    if len(values) < 2:
        return {"pairs": 0, "mean_cosine": float("nan"), "within_content_distance": float("nan"),
                "between_content_distance": float("nan"), "between_over_within": float("nan")}
    flat = values.flatten(1).float()
    normalized = torch.nn.functional.normalize(flat, dim=-1, eps=1e-6)
    cosine = normalized @ normalized.T
    distance = torch.cdist(flat, flat, p=2) / max(flat.shape[1] ** 0.5, 1.0)
    upper = torch.triu(torch.ones(len(values), len(values), dtype=torch.bool), diagonal=1)
    same = torch.tensor([[left == right for right in labels] for left in labels], dtype=torch.bool)
    within = distance[upper & same]; between = distance[upper & ~same]
    within_value = float(within.mean()) if len(within) else float("nan")
    between_value = float(between.mean()) if len(between) else float("nan")
    return {"pairs": int(upper.sum()), "mean_cosine": float(cosine[upper].mean()),
            "within_content_distance": within_value, "between_content_distance": between_value,
            "between_over_within": float(between_value / max(within_value, 1e-8)) if np.isfinite(within_value) else float("nan")}


def wrong_trial_order(labels: list[str], subjects: list[str], device: torch.device) -> torch.Tensor:
    """Deterministically select another-content EEG trial for every target."""
    if len(labels) < 2 or len(set(labels)) < 2:
        raise RuntimeError("wrong_trial control requires at least two distinct contents per batch")
    order = []
    for index, label in enumerate(labels):
        candidate = next((other for other, other_label in enumerate(labels)
                          if other_label != label and subjects[other] == subjects[index]), None)
        if candidate is None:
            candidate = next(other for other, other_label in enumerate(labels) if other_label != label)
        order.append(candidate)
    return torch.tensor(order, dtype=torch.long, device=device)


def leave_one_out_subject_probe(embeddings: torch.Tensor, subjects: list[str]) -> dict[str, float | int]:
    """Nearest-centroid diagnostic; high accuracy signals subject leakage."""
    if len(embeddings) < 2 or len(set(subjects)) < 2:
        return {"accuracy": float("nan"), "chance": float("nan"), "subjects": len(set(subjects)), "evaluated": 0}
    value = torch.nn.functional.normalize(embeddings, dim=-1)
    unique = sorted(set(subjects)); correct = 0; evaluated = 0
    for index, expected in enumerate(subjects):
        centroids = []; labels = []
        for label in unique:
            selected = [other for other, subject in enumerate(subjects) if subject == label and other != index]
            if selected:
                centroids.append(torch.nn.functional.normalize(value[selected].mean(0), dim=0)); labels.append(label)
        if expected not in labels or len(labels) < 2:
            continue
        predicted = labels[int((value[index] @ torch.stack(centroids).T).argmax())]
        correct += int(predicted == expected); evaluated += 1
    return {"accuracy": float(correct / evaluated) if evaluated else float("nan"),
            "chance": float(1.0 / len(unique)), "subjects": len(unique), "evaluated": evaluated}


def acoustic_reconstruction(renderer_checkpoint: Path, prediction: torch.Tensor, target_mel: torch.Tensor,
                            target_rms: torch.Tensor, target_activity: torch.Tensor,
                            exact: torch.Tensor, device: torch.device) -> dict:
    payload = torch.load(renderer_checkpoint, map_location="cpu", weights_only=False)
    if not all(payload.get("gate", {}).values()):
        raise RuntimeError("audio renderer checkpoint has not passed its audio-only oracle gate")
    if not exact.any():
        return {"status": "not_applicable_no_verified_exact_pairs"}
    renderer = AudioMFCCRenderer(**payload["model_config"]).to(device)
    renderer.load_state_dict(payload["model"]); renderer.eval()
    with torch.no_grad(): state = renderer(prediction[exact].to(device))
    mel = target_mel[exact].to(device); rms = target_rms[exact].to(device); activity = target_activity[exact].to(device).bool()
    predicted_activity = state.activity_logits >= 0
    tp = (predicted_activity & activity).sum().float(); fp = (predicted_activity & ~activity).sum().float(); fn = (~predicted_activity & activity).sum().float()
    rms_left = state.rms.flatten() - state.rms.mean(); rms_right = rms.flatten() - rms.mean()
    correlation = float((rms_left @ rms_right) / (rms_left.norm() * rms_right.norm()).clamp_min(1e-8))
    return {"status": "diagnostic_acoustic_renderer", "pairs": int(exact.sum()),
            "log_mel_mae": float((state.log_mel - mel).abs().mean()),
            "rms_mae": float((state.rms - rms).abs().mean()), "rms_correlation": correlation,
            "activity_f1": float(2 * tp / (2 * tp + fp + fn).clamp_min(1)),
            "waveform_status": "not_generated_no_validated_vocoder"}


def evaluate_label_only(model, manifest: Path, split: Path, role: str, targets: Path, normalizer: Path,
                        cfg: dict, vocabulary: dict[str, int], device: torch.device) -> dict:
    try:
        dataset = JointManifestDataset(manifest, split, role, "ds006104", targets, normalizer,
                                       float(cfg["loss"]["weak_content_weight"]),
                                       supervision_types={"label_only"}, phoneme_vocabulary=vocabulary)
    except ValueError:
        return {"status": "not_built_for_role", "pairs": 0}
    maximum = int(cfg["pilot"]["label_only_max_generalization_pairs"])
    loader = DataLoader(Subset(dataset, list(range(min(len(dataset), maximum)))),
                        batch_size=int(cfg["training"]["batch_size"]), shuffle=False, collate_fn=homogeneous_collate)
    correct = total = 0; by_task = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for batch in loader:
            tensor = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            state = model(tensor["eeg"], tensor["channel_xyz"], tensor["channel_mask"], tensor.get("model_time_mask", tensor["time_mask"]), tensor["dataset_id"])
            valid = tensor["phoneme_index"] >= 0; predicted = state.phoneme_logits.argmax(1)
            match = predicted[valid] == tensor["phoneme_index"][valid]
            correct += int(match.sum()); total += int(valid.sum())
            for index in valid.nonzero(as_tuple=False).flatten().tolist():
                key = f"{batch['task'][index]}|{'TMS1' if bool(batch['tms_applied'][index]) else 'TMS0'}"
                by_task[key][0] += int(predicted[index] == tensor["phoneme_index"][index]); by_task[key][1] += 1
    result = {"status": "evaluated", "pairs": total, "accuracy": float(correct / total) if total else float("nan"),
              "vocabulary_size": len(vocabulary),
              "strata": {key: {"correct": value[0], "pairs": value[1], "accuracy": value[0] / value[1]}
                          for key, value in sorted(by_task.items())}}
    dataset.close(); return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", choices=["ds004940", "ds006104"])
    parser.add_argument("--role", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-single", type=Path)
    parser.add_argument("--compare-joint", type=Path)
    parser.add_argument("--renderer-checkpoint", type=Path)
    args = parser.parse_args()
    if args.compare_single and args.compare_joint:
        result = compare_results(args.compare_single, args.compare_joint)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0
    if not args.checkpoint or not args.dataset:
        parser.error("--checkpoint and --dataset are required unless comparison files are provided")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["pilot_config"]
    data_cfg_path = _resolve(cfg["data_config"], ROOT / "configs")
    data_cfg = yaml.safe_load(data_cfg_path.read_text())
    artifact_root = ROOT / data_cfg["output_root"]
    split_protocol = payload.get("split_protocol", cfg["split"]["protocol"] if payload.get("stage") == "overfit" else "stage2_joint_ood")
    split = artifact_root / "splits" / f"{split_protocol}_fold-{cfg['split']['fold']}.csv"
    artifact_set = payload.get("artifact_set") or ("built" if payload.get("stage") == "overfit" else "stage2")
    manifest = artifact_root / "manifests" / f"manifest_{artifact_set}.csv"
    target_name = payload.get("target_name") or ("speech_targets" if artifact_set == "built" else "speech_targets_stage2")
    targets = artifact_root / "speech_targets" / f"{target_name}.h5"
    normalizer_name = payload.get("normalizer_name") or split.stem
    normalizer = artifact_root / "normalizers" / f"{normalizer_name}.json"
    sources = artifact_root / "source_lock.json"
    expected_hashes = payload.get("artifact_hashes", {})
    current_hashes = {path.name: _sha256(path) for path in (sources, split, manifest, targets, normalizer)}
    if expected_hashes and current_hashes != expected_hashes:
        changed = sorted(key for key in current_hashes if current_hashes.get(key) != expected_hashes.get(key))
        raise RuntimeError(f"checkpoint artifact provenance mismatch: {changed}")
    if payload.get("model_code_sha256"):
        if payload["model_code_sha256"] != _model_code_sha256():
            raise RuntimeError("checkpoint model code provenance mismatch")
    elif payload.get("runtime_code_sha256") and payload["runtime_code_sha256"] != _runtime_code_sha256():
        if payload.get("run_kind") != "explore":
            raise RuntimeError("legacy checkpoint model/runtime code provenance mismatch")
        print(json.dumps({
            "warning": "legacy_explore_checkpoint_control_plane_upgrade",
            "detail": "artifact/model configuration provenance matched; train control-script hash changed",
        }))
    vocabulary = payload.get("phoneme_vocabulary") or phoneme_vocabulary_from_manifest(manifest)
    dataset = JointManifestDataset(manifest, split, args.role,
                                   args.dataset, targets, normalizer,
                                   float(cfg["loss"]["weak_content_weight"]),
                                   supervision_types={"paired_audio", "weak_audio"},
                                   phoneme_vocabulary=vocabulary)
    selection_stage = payload.get("stage", "generalization") if args.role == "train" else "generalization"
    indices = pilot_indices(dataset, cfg, selection_stage, args.role)
    loader = DataLoader(Subset(dataset, indices), batch_size=int(cfg["training"]["batch_size"]), shuffle=False,
                        collate_fn=homogeneous_collate)
    target_device = _device()
    model = JointEEGContentModel(**payload["model_config"]).to(target_device)
    model.load_state_dict(payload["model"], strict=False)
    templates = payload.get("target_templates")
    if bool(payload["model_config"].get("zero_centered", False)):
        if not templates:
            raise RuntimeError("zero-centered checkpoint is missing its train-fold template")
        model.set_target_templates(templates["mfcc_mean"], templates["mfcc_scale"], templates.get("hubert_mean"))
        if bool(payload["model_config"].get("duration_standardized", False)):
            duration_keys = ("duration_log_mean", "duration_log_scale", "duration_min_frames", "duration_max_frames")
            if not all(key in templates for key in duration_keys):
                raise RuntimeError("v3 checkpoint is missing train-fold duration statistics")
            model.set_duration_statistics(*(templates[key] for key in duration_keys))
    if int(payload["model_config"].get("teacher_dimension", 0)) > 0:
        teacher = payload.get("speech_teacher")
        if not teacher:
            raise RuntimeError("v3 checkpoint is missing frozen speech teacher")
        model.set_speech_teacher(teacher["mean"], teacher["components"], teacher["scale"],
                                 teacher["bank"], teacher["labels"])
    model.eval()
    label_only_metrics = (evaluate_label_only(model, manifest, split, args.role, targets, normalizer, cfg, vocabulary, target_device)
                          if args.dataset == "ds006104" else {"status": "not_applicable", "pairs": 0})
    predictions=[]; target_values=[]; subjects=[]; contents=[]; audio_ids=[]; tasks=[]; conditions=[]; tms_conditions=[]
    target_mels=[]; target_rms=[]; target_activities=[]; exact_flags=[]; predicted_durations=[]; target_durations=[]
    eeg_globals=[]; hubert_eeg_local=[]; hubert_audio_local=[]; hubert_eeg_global=[]; hubert_audio_global=[]; hubert_labels=[]
    configured_controls = tuple(control for control in cfg.get("controls", ("zero", "time_shuffle", "channel_shuffle"))
                                if control in {"zero", "time_shuffle", "time_block_shuffle", "channel_shuffle", "wrong_trial"})
    if not configured_controls:
        configured_controls = ("zero", "time_shuffle", "channel_shuffle")
    control_errors=defaultdict(list); residual_values=[]; target_residual_values=[]; zero_template_errors=[]
    with torch.no_grad():
        for batch in loader:
            tensor = {key: value.to(target_device) if torch.is_tensor(value) else value for key, value in batch.items()}
            model_time_mask = tensor.get("model_time_mask", tensor["time_mask"])
            state = model(tensor["eeg"], tensor["channel_xyz"], tensor["channel_mask"], model_time_mask, tensor["dataset_id"])
            eligible = tensor["pairing_weight"] > 0
            if eligible.any():
                predictions.append(state.mfcc[eligible].cpu()); target_values.append(tensor["content_mfcc"][eligible].cpu())
                selected_indices = eligible.nonzero(as_tuple=False).flatten().tolist()
                subjects.extend(batch["subject"][i] for i in selected_indices); contents.extend(batch["linguistic_content_id"][i] for i in selected_indices)
                audio_ids.extend(batch["audio_id"][i] for i in selected_indices)
                tasks.extend(batch["task"][i] for i in selected_indices); conditions.extend(batch["condition"][i] for i in selected_indices)
                tms_conditions.extend("TMS1" if bool(batch["tms_applied"][i]) else "TMS0" for i in selected_indices)
                target_mels.append(tensor["acoustic_log_mel"][eligible].cpu()); target_rms.append(tensor["acoustic_rms"][eligible].cpu())
                target_activities.append(tensor["acoustic_activity"][eligible].cpu()); exact_flags.append(tensor["acoustic_supervision"][eligible].cpu())
                eeg_globals.append(state.global_embedding[eligible].cpu())
                if state.predicted_duration is not None:
                    predicted_durations.append(state.predicted_duration[eligible].cpu())
                    target_durations.append(tensor["audio_duration_frames"][eligible].float().cpu())
                hubert_eligible = eligible & tensor["hubert_mask"].any(1)
                if hubert_eligible.any():
                    audio_local = model.centered_audio(tensor["hubert_local"][hubert_eligible])
                    hubert_eeg_local.append(state.local[hubert_eligible].cpu()); hubert_audio_local.append(audio_local.cpu())
                    hubert_eeg_global.append(state.global_embedding[hubert_eligible].cpu())
                    if model.teacher_dimension > 0 and "hubert_global" in tensor:
                        hubert_audio_global.append(torch.nn.functional.normalize(
                            model.teacher_target(tensor["hubert_global"][hubert_eligible]), dim=-1,
                        ).cpu())
                    else:
                        hubert_audio_global.append(torch.nn.functional.normalize(audio_local.mean(1), dim=-1).cpu())
                    hubert_indices = hubert_eligible.nonzero(as_tuple=False).flatten().tolist()
                    hubert_labels.extend(batch["linguistic_content_id"][i] for i in hubert_indices)
                correct = (state.mfcc[eligible] - tensor["content_mfcc"][eligible]).abs().mean((1,2))
                control_errors["correct"].extend(correct.cpu().tolist())
                if state.residual_mfcc is not None and state.baseline_mfcc is not None:
                    residual_values.append(state.residual_mfcc[eligible].cpu())
                    target_residual_values.append((tensor["content_mfcc"][eligible] - state.baseline_mfcc[eligible]).cpu())
                for control in configured_controls:
                    if control == "wrong_trial":
                        order = wrong_trial_order(batch["linguistic_content_id"], batch["subject"], target_device)
                        eeg = tensor["eeg"][order]
                    else:
                        eeg = counterfactual_eeg(tensor["eeg"], control, time_mask=model_time_mask, channel_mask=tensor["channel_mask"])
                    output = model(eeg, tensor["channel_xyz"], tensor["channel_mask"], model_time_mask, tensor["dataset_id"])
                    error = (output.mfcc[eligible] - tensor["content_mfcc"][eligible]).abs().mean((1,2))
                    control_errors[control].extend(error.cpu().tolist())
                    if control == "zero" and output.baseline_mfcc is not None:
                        zero_template_errors.extend((output.mfcc[eligible] - output.baseline_mfcc[eligible]).abs().mean((1,2)).cpu().tolist())
    prediction = torch.cat(predictions) if predictions else torch.empty(0,39,161)
    target = torch.cat(target_values) if target_values else torch.empty(0,39,161)
    if len(prediction):
        retrieval = content_retrieval(prediction, target, contents)
        wrong_indices = []
        for index, label in enumerate(contents):
            candidate = next((other for other, other_label in enumerate(contents) if other_label != label), None)
            wrong_indices.append(candidate if candidate is not None else index)
        wrong = float((prediction - target[wrong_indices]).abs().mean()) if len(set(contents)) > 1 else float("nan")
        delta = float(((prediction[...,1:]-prediction[...,:-1])-(target[...,1:]-target[...,:-1])).abs().mean())
    else:
        retrieval={"r1":float("nan"),"mrr":float("nan"),"chance_r1":float("nan"),"unique_contents":0}; wrong=delta=float("nan")
    subject_values=defaultdict(list)
    subject_control_values={control: defaultdict(list) for control in control_errors}
    for control, errors in control_errors.items():
        if len(errors) != len(subjects):
            raise RuntimeError(f"subject/control alignment mismatch for {control}: {len(errors)} != {len(subjects)}")
        for subject, value in zip(subjects, errors):
            subject_control_values[control][subject].append(value)
    for subject, value in zip(subjects, control_errors["correct"]): subject_values[subject].append(value)
    if len(prediction):
        train_reference, train_reference_pairs = training_target_reference(
            manifest, split, args.dataset, targets, normalizer, cfg, vocabulary,
            payload.get("stage", "generalization"),
        )
        templates = template_metrics(prediction, target, contents, audio_ids, train_reference)
        templates["dataset_mean_template_train_pairs"] = train_reference_pairs
    else:
        templates = {}
    control_means = {key:float(np.mean(value)) for key,value in control_errors.items()}
    subject_control_gains = {}
    for control in configured_controls:
        common_subjects = sorted(set(subject_control_values["correct"]) & set(subject_control_values[control]))
        gains = [float(np.mean(subject_control_values[control][subject])) -
                 float(np.mean(subject_control_values["correct"][subject])) for subject in common_subjects]
        subject_control_gains[control] = bootstrap_mean(gains)
    residual_variance_ratio = float("nan")
    if residual_values and target_residual_values:
        predicted_residual = torch.cat(residual_values).flatten(1)
        target_residual = torch.cat(target_residual_values).flatten(1)
        residual_variance_ratio = float(predicted_residual.var(0, unbiased=False).mean() /
                                        target_residual.var(0, unbiased=False).mean().clamp_min(1e-8))
    duration_metrics = {"pairs": 0, "mae_frames": float("nan"), "pearson": float("nan"),
                        "predicted_std": float("nan"), "target_std": float("nan"),
                        "std_retention": float("nan"), "unique_predicted_frames": 0}
    if predicted_durations:
        predicted_duration = torch.cat(predicted_durations).float()
        target_duration = torch.cat(target_durations).float()
        centered_left = predicted_duration - predicted_duration.mean()
        centered_right = target_duration - target_duration.mean()
        pearson = (centered_left @ centered_right) / (centered_left.norm() * centered_right.norm()).clamp_min(1e-8)
        predicted_std = predicted_duration.std(unbiased=False)
        target_std = target_duration.std(unbiased=False)
        duration_metrics = {"pairs": int(len(predicted_duration)),
                            "mae_frames": float((predicted_duration - target_duration).abs().mean()),
                            "pearson": float(pearson), "predicted_std": float(predicted_std),
                            "target_std": float(target_std),
                            "std_retention": float(predicted_std / target_std.clamp_min(1e-8)),
                            "unique_predicted_frames": int(predicted_duration.round().unique().numel())}
    hubert_metrics = {"pairs": 0, "local_cosine": float("nan"), "global_retrieval": {
        "r1": float("nan"), "mrr": float("nan"), "chance_r1": float("nan"), "unique_contents": 0,
    }}
    if hubert_eeg_local:
        eeg_local, audio_local = torch.cat(hubert_eeg_local), torch.cat(hubert_audio_local)
        local_cosine = torch.nn.functional.cosine_similarity(eeg_local, audio_local, dim=-1).mean()
        eeg_global, audio_global = torch.cat(hubert_eeg_global), torch.cat(hubert_audio_global)
        hubert_metrics = {"pairs": len(eeg_local), "local_cosine": float(local_cosine),
                          "global_retrieval": content_retrieval(eeg_global, audio_global, hubert_labels)}
    leakage = leave_one_out_subject_probe(torch.cat(eeg_globals) if eeg_globals else torch.empty(0, 1), subjects)
    strata = stratified_error(control_errors["correct"], {"task": tasks, "condition": conditions, "tms": tms_conditions})
    diversity = pairwise_diversity(prediction, contents) if len(prediction) else pairwise_diversity(torch.empty(0, 1), [])
    if args.renderer_checkpoint:
        reconstruction = acoustic_reconstruction(
            args.renderer_checkpoint, prediction, torch.cat(target_mels), torch.cat(target_rms),
            torch.cat(target_activities), torch.cat(exact_flags).bool(), target_device,
        )
    else:
        reconstruction = {"status": "not_run_audio_renderer_checkpoint_not_supplied",
                          "waveform_status": "not_generated_no_validated_vocoder"}
    checks = {}
    if args.role == "train" and payload.get("stage") == "overfit" and len(prediction):
        gate = cfg["gates"]["overfit"]
        checks["content_retrieval"] = retrieval["r1"] >= float(gate["content_retrieval_r1_min"])
        baseline_name, baseline_passed = registered_collapse_check(templates, gate)
        checks[baseline_name] = baseline_passed
        for control in gate["correct_must_beat"]:
            checks[f"correct_beats_{control}"] = control_means["correct"] < control_means[control]
        if "hubert_global_retrieval_r1_min" in gate:
            checks["hubert_global_retrieval"] = hubert_metrics["global_retrieval"]["r1"] >= float(gate["hubert_global_retrieval_r1_min"])
        if "duration_pearson_min" in gate:
            checks["duration_correlation"] = duration_metrics["pearson"] >= float(gate["duration_pearson_min"])
        if "residual_variance_retention_min" in gate:
            checks["residual_variance_retained"] = residual_variance_ratio >= float(gate["residual_variance_retention_min"])
    elif payload.get("stage") == "generalization" and len(prediction) and bool(payload["model_config"].get("zero_centered", False)):
        threshold = float(cfg["gates"]["generalization"].get("paired_control_ci_low_min", 0.0))
        for control in configured_controls:
            checks[f"correct_beats_{control}"] = control_means["correct"] < control_means[control]
            checks[f"subject_bootstrap_{control}_ci_low"] = subject_control_gains[control]["ci_low"] > threshold
        checks["mfcc_residual_variance_retained"] = residual_variance_ratio >= float(cfg["loss"].get("variance_floor_fraction", 0.25))
        checks["zero_equals_train_template"] = bool(zero_template_errors) and max(zero_template_errors) <= 1e-7
        global_retrieval = hubert_metrics["global_retrieval"]
        checks["hubert_global_above_chance"] = bool(np.isfinite(global_retrieval["r1"])) and global_retrieval["r1"] > global_retrieval["chance_r1"]
    result={"dataset":args.dataset,"role":args.role,"pairs":len(prediction),"mfcc_l1":float((prediction-target).abs().mean()) if len(prediction) else float("nan"),
            "delta_l1":delta,"wrong_pair_mfcc_l1":wrong,"retrieval":retrieval,
            "controls":control_means,
            "subject_control_error_gain": subject_control_gains,
            "residual_variance_ratio": residual_variance_ratio,
            "duration": duration_metrics,
            "trial_diversity": diversity,
            "zero_template_max_abs_error": max(zero_template_errors) if zero_template_errors else float("nan"),
            "subject_mfcc_l1":{key:float(np.mean(value)) for key,value in subject_values.items()},
            "subject_control_mfcc_l1": {
                control: {key: float(np.mean(value)) for key, value in values.items()}
                for control, values in subject_control_values.items()
            },
            "stratified_mfcc_l1": strata,
            "hubert_similarity": hubert_metrics,
            "subject_leakage_probe": leakage,
            "speaker_leakage_probe": {"status":"not_available_no_speaker_identity_in_dataset_metadata"},
            "label_only_phoneme": label_only_metrics,
            "templates":templates,
            "gate":{"checks":checks,"passed":bool(checks) and all(checks.values()),
                    "registered_collapse_baseline": cfg["gates"]["overfit"].get("collapse_baseline", "dataset_mean"),
                    "note":"same-content is diagnostic only and uses leave-one-realization-out when independent audio IDs exist"},
            "run_kind":payload.get("run_kind","unknown"),
            "scientific_interpretation":(
                "engineering_only" if payload.get("run_kind") == "smoke"
                else ("exploratory_only_not_registered" if payload.get("run_kind") == "explore" else "registered_pilot")
            ),
            "reconstruction": reconstruction}
    output = args.output or args.checkpoint.parent / f"evaluation_{args.dataset}_{args.role}.json"
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2)); dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
