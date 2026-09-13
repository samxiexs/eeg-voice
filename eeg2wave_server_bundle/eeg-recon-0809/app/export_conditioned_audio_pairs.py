#!/usr/bin/env python3
"""Export native-duration DS004940 v2 qualitative audio/control bundles.

Generated files are diagnostic until the M1 control gate passes.  Unlike the
legacy exporter, this path never uses Griffin-Lim or independent peak gain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.io import wavfile
from tqdm.auto import tqdm

APP = Path(__file__).resolve().parent; ROOT = APP.parent
sys.path.insert(0, str(APP / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cache_speech_targets import load_wave
from eeg2speech.data import JointManifestDataset, homogeneous_collate, phoneme_vocabulary_from_manifest, pilot_indices
from eeg2speech.diffusion import ConditionalMelDiffusion, denormalize_mel, normalize_mel
from eeg2speech.losses import counterfactual_eeg
from eeg2speech.model import DurationConditionedNativeRenderer, JointEEGContentModel
from eeg2speech.speecht5 import SAMPLE_RATE, SpeechT5HiFiGan


def dev() -> torch.device: return torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
def write(path: Path, wave: torch.Tensor | np.ndarray, gain: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source = wave.detach().cpu().numpy() if torch.is_tensor(wave) else np.asarray(wave)
    value = np.clip(np.nan_to_num(source.squeeze()) * float(gain), -1, 1)
    wavfile.write(path, SAMPLE_RATE, np.round(value * 32767).astype(np.int16))
def load_eeg(path: Path, device: torch.device) -> tuple[dict, JointEEGContentModel, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False); model = JointEEGContentModel(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model"], strict=False)
    templates = payload.get("target_templates")
    if bool(payload["model_config"].get("zero_centered", False)):
        if not templates: raise RuntimeError("zero-centered EEG checkpoint lacks target_templates")
        model.set_target_templates(templates["mfcc_mean"], templates["mfcc_scale"], templates.get("hubert_mean"))
        if bool(payload["model_config"].get("duration_standardized", False)):
            keys = ("duration_log_mean", "duration_log_scale", "duration_min_frames", "duration_max_frames")
            if not all(key in templates for key in keys):
                raise RuntimeError("v3 EEG checkpoint lacks duration statistics")
            model.set_duration_statistics(*(templates[key] for key in keys))
    if int(payload["model_config"].get("teacher_dimension", 0)) > 0:
        teacher = payload.get("speech_teacher")
        if not teacher: raise RuntimeError("v3 EEG checkpoint lacks frozen speech teacher")
        model.set_speech_teacher(teacher["mean"], teacher["components"], teacher["scale"], teacher["bank"], teacher["labels"])
    model.eval(); return payload, model, payload["pilot_config"]
def plot(path: Path, rows: dict[str, torch.Tensor]) -> None:
    fig, axes = plt.subplots(len(rows), 1, figsize=(8, 1.7 * len(rows)), sharex=True)
    if len(rows) == 1: axes = [axes]
    for axis, (name, mel) in zip(axes, rows.items()):
        axis.imshow(mel.detach().cpu().numpy(), origin="lower", aspect="auto", interpolation="nearest", cmap="magma")
        axis.set_ylabel(name)
    axes[-1].set_xlabel("native frames (10 ms)"); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_energy(path: Path, rows: dict[str, np.ndarray]) -> None:
    figure, axis = plt.subplots(1, 1, figsize=(8, 3.8))
    for name, waveform in rows.items():
        value = np.asarray(waveform, dtype=np.float32)
        if len(value) < 400: value = np.pad(value, (0, 400 - len(value)))
        frames = np.lib.stride_tricks.sliding_window_view(value, 400)[::160]
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
        axis.plot(np.arange(len(rms)) * 0.01, rms, label=name, linewidth=1.1)
    axis.set_xlabel("seconds"); axis.set_ylabel("RMS energy"); axis.legend(frameon=False, ncol=2)
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--role", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--diffusion-mode", choices=["off", "on"])
    parser.add_argument("--diffusion-checkpoint", type=Path)
    parser.add_argument("--sampling-mode", choices=["deterministic", "random"], default="deterministic",
                        help="metric output remains deterministic; random mode writes qualitative listening samples")
    parser.add_argument("--sampling-seed", type=int,
                        help="optional root seed for reproducible qualitative random samples")
    parser.add_argument("--samples-per-trial", type=int, default=1)
    args = parser.parse_args(); device = dev(); payload, model, cfg = load_eeg(args.checkpoint, device)
    if args.samples_per_trial < 1:
        raise ValueError("--samples-per-trial must be positive")
    renderer_payload = torch.load(args.renderer, map_location="cpu", weights_only=False)
    if not renderer_payload.get("gate") or not all(renderer_payload["gate"].values()):
        raise RuntimeError("native audio renderer has not passed its validation-content gate")
    renderer = DurationConditionedNativeRenderer(**renderer_payload.get("model_config", {})).to(device)
    renderer.load_state_dict(renderer_payload["model"]); renderer.eval()
    diffusion_spec = cfg.get("native_audio", {}).get("diffusion", {})
    diffusion_mode = args.diffusion_mode or ("on" if bool(diffusion_spec.get("enabled", False)) else "off")
    if args.sampling_mode == "random" and diffusion_mode != "on":
        raise RuntimeError("random qualitative sampling requires --diffusion-mode on")
    diffusion = None; diffusion_mean = None; diffusion_scale = None; diffusion_steps = None
    if diffusion_mode == "on":
        if args.diffusion_checkpoint is None or not args.diffusion_checkpoint.is_file():
            raise RuntimeError("diffusion-mode=on requires an existing --diffusion-checkpoint")
        diffusion_payload = torch.load(args.diffusion_checkpoint, map_location="cpu", weights_only=False)
        if not diffusion_payload.get("gate") or not all(diffusion_payload["gate"].values()):
            raise RuntimeError("diffusion refiner has not passed its validation-content gate")
        expected_renderer = diffusion_payload.get("artifact_hashes", {}).get(args.renderer.name)
        actual_renderer = hashlib.sha256(args.renderer.read_bytes()).hexdigest()
        if expected_renderer != actual_renderer:
            raise RuntimeError("diffusion checkpoint was trained with a different native renderer")
        diffusion = ConditionalMelDiffusion(**diffusion_payload["model_config"]).to(device)
        diffusion.load_state_dict(diffusion_payload["model"]); diffusion.eval()
        diffusion_mean = diffusion_payload["mel_mean"].to(device)
        diffusion_scale = diffusion_payload["mel_scale"].to(device)
        diffusion_steps = int(diffusion_payload.get("sampling_steps", diffusion_spec.get("sampling_steps", 20)))
        if diffusion.conditioning_dimension not in (0, model.teacher_dimension):
            raise RuntimeError("diffusion EEG conditioning dimension does not match the EEG checkpoint")
    # Checkpoints preserve exactly which isolated artifacts they were trained on.
    data_path = (ROOT / "configs" / Path(str(cfg["data_config"])).name).resolve()
    source_cfg = yaml.safe_load(data_path.read_text()); root = ROOT / source_cfg["output_root"]
    stage = str(payload.get("stage", "generalization")); spec = cfg.get("stage2", {})
    protocol = str(payload.get("split_protocol", spec.get("protocol", cfg["split"]["protocol"])))
    artifact_set = str(payload.get("artifact_set", "explore_stage2_ds004940_conditioned_v2"))
    targets_name = str(payload.get("target_name", spec.get("explore_target_name")))
    normalizer_name = str(payload.get("normalizer_name", spec.get("explore_normalizer_name")))
    manifest = root / "manifests" / f"manifest_{artifact_set}.csv"; split = root / "splits" / f"{protocol}_fold-0.csv"
    targets = root / "speech_targets" / f"{targets_name}.h5"; normalizer = root / "normalizers" / f"{normalizer_name}.json"
    dataset = JointManifestDataset(manifest, split, args.role, "ds004940", targets, normalizer,
                                   float(cfg["loss"]["weak_content_weight"]), supervision_types={"paired_audio"},
                                   phoneme_vocabulary=phoneme_vocabulary_from_manifest(manifest))
    selected_indices = pilot_indices(dataset, cfg, stage if args.role == "train" else "generalization", args.role)
    indices = selected_indices[:args.max_pairs]
    hifigan_root = (Path(cfg["native_audio"]["local_hifigan_path"]) if Path(cfg["native_audio"]["local_hifigan_path"]).is_absolute()
                    else (ROOT / cfg["native_audio"]["local_hifigan_path"]).resolve())
    vocoder = SpeechT5HiFiGan(hifigan_root, device=device); records = []
    for index in tqdm(indices, desc="native audio pairs", unit="pair"):
        raw = homogeneous_collate([dataset[index]]); batch = {k: v.to(device) if torch.is_tensor(v) else v for k,v in raw.items()}
        current_content = raw["linguistic_content_id"][0]
        current_subject = raw["subject"][0]
        wrong_index = next(other for other in selected_indices
                           if dataset.frame.iloc[other].linguistic_content_id != current_content and
                           dataset.frame.iloc[other].subject == current_subject)
        wrong_raw = homogeneous_collate([dataset[wrong_index]])
        wrong_batch = {k: v.to(device) if torch.is_tensor(v) else v for k,v in wrong_raw.items()}
        mask = batch.get("model_time_mask", batch["time_mask"])
        with torch.no_grad():
            predictions = {"eeg": model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], mask, batch["dataset_id"])}
            for name in ("zero", "time_block_shuffle", "channel_shuffle"):
                eeg = counterfactual_eeg(batch["eeg"], name, time_mask=mask, channel_mask=batch["channel_mask"])
                predictions[name] = model(eeg, batch["channel_xyz"], batch["channel_mask"], mask, batch["dataset_id"])
            predictions["wrong_trial"] = model(wrong_batch["eeg"], batch["channel_xyz"], batch["channel_mask"],
                                                 mask, batch["dataset_id"])
            mel_rows = {"target_native": batch["native_speecht5_mel"][0, :, batch["native_audio_mask"][0]]}
            folder = args.output / str(raw["trial_id"][0]); metric = folder / "metric"; listening = folder / "listening"
            source_wave, _, _ = load_wave(ROOT / str(dataset.frame.iloc[index].audio_path))
            oracle_wave = vocoder.synthesize(batch["native_speecht5_mel"]).squeeze(0)
            generated_waves = {}; refined_waves = {}; refined_mel_rows = {}; random_waves: dict[str, list[torch.Tensor]] = {}
            random_sample_seeds: list[int] = []
            trial_id = str(raw["trial_id"][0])
            deterministic_seed = int(hashlib.sha256(trial_id.encode()).hexdigest()[:8], 16)
            root_random_seed = (int(args.sampling_seed) if args.sampling_seed is not None else secrets.randbits(63))
            if args.sampling_mode == "random":
                random_sample_seeds = [
                    int(hashlib.sha256(f"{root_random_seed}|{trial_id}|{sample_index}".encode()).hexdigest()[:16], 16) % (2**63 - 1)
                    for sample_index in range(args.samples_per_trial)
                ]
            for name, state in predictions.items():
                frames = state.predicted_duration.round().long().clamp(1, 1000)
                mel, mel_mask = renderer(state.mfcc, frames)
                generated = vocoder.synthesize(mel[:, :, :int(mel_mask.sum(1).max())])
                generated_waves[name] = generated.squeeze(0); mel_rows[name] = mel[0, :, mel_mask[0]]
                if diffusion is not None:
                    normalized = normalize_mel(mel, diffusion_mean, diffusion_scale)
                    context = state.global_embedding if diffusion.conditioning_dimension else None
                    if args.sampling_mode == "deterministic":
                        # This fixed stream is used only for an optional
                        # deterministic diagnostic refinement.  It remains
                        # separate from the random listening samples below.
                        generator = torch.Generator(device="cpu").manual_seed(deterministic_seed)
                        noise = torch.randn(normalized.shape, generator=generator).to(device)
                        refined = denormalize_mel(
                            diffusion.refine(normalized, mel_mask, steps=diffusion_steps, noise=noise, context=context),
                            diffusion_mean, diffusion_scale,
                        )
                        refined_generated = vocoder.synthesize(refined[:, :, :int(mel_mask.sum(1).max())])
                        refined_waves[name] = refined_generated.squeeze(0)
                        refined_mel_rows[name] = refined[0, :, mel_mask[0]]
                    else:
                        samples: list[torch.Tensor] = []
                        for sample_seed in random_sample_seeds:
                            generator = torch.Generator(device="cpu").manual_seed(sample_seed)
                            noise = torch.randn(normalized.shape, generator=generator).to(device)
                            refined = denormalize_mel(
                                diffusion.refine(normalized, mel_mask, steps=diffusion_steps, noise=noise, context=context),
                                diffusion_mean, diffusion_scale,
                            )
                            samples.append(vocoder.synthesize(refined[:, :, :int(mel_mask.sum(1).max())]).squeeze(0))
                            if name not in refined_mel_rows:
                                refined_mel_rows[name] = refined[0, :, mel_mask[0]]
                        random_waves[name] = samples
            wave_rows = {"source": np.asarray(source_wave), "oracle": oracle_wave.detach().cpu().numpy().squeeze(),
                         **{f"{name}_raw": value.detach().cpu().numpy().squeeze() for name, value in generated_waves.items()},
                         **{f"{name}_diffusion": value.detach().cpu().numpy().squeeze() for name, value in refined_waves.items()}}
            write(metric / "00_source.wav", source_wave); write(metric / "01_target_native_hifigan_oracle.wav", oracle_wave)
            for name, value in generated_waves.items(): write(metric / f"{name}_native_hifigan.wav", value)
            # v3 keeps a named deterministic EEG-only output for scientific
            # comparison; it is intentionally not the random diffusion sample.
            write(metric / "eeg_deterministic.wav", generated_waves["eeg"])
            for name, value in refined_waves.items(): write(metric / f"{name}_native_hifigan_diffusion.wav", value)
            common_peak = max(float(np.max(np.abs(value))) if len(value) else 0.0 for value in wave_rows.values())
            for values in random_waves.values():
                for value in values:
                    common_peak = max(common_peak, float(value.detach().abs().max()))
            common_gain = 0.95 / max(common_peak, 1e-8)
            for name, value in wave_rows.items(): write(listening / f"{name}.wav", value, gain=common_gain)
            for name, values in random_waves.items():
                for sample_index, value in enumerate(values):
                    write(listening / f"{name}_random_sample-{sample_index:02d}.wav", value, gain=common_gain)
            plot(folder / "native_mel_comparison.png", mel_rows)
            if refined_mel_rows:
                plot(folder / "native_mel_diffusion_comparison.png",
                     {"target_native": mel_rows["target_native"], **refined_mel_rows})
            plot_energy(folder / "energy_comparison.png", wave_rows)
            records.append({"trial_id": raw["trial_id"][0], "role": args.role,
                            "wrong_trial_id": wrong_raw["trial_id"][0], "bundle_listening_gain": common_gain,
                            "diffusion_mode": diffusion_mode,
                            "diffusion_checkpoint": str(args.diffusion_checkpoint) if args.diffusion_checkpoint else "",
                            "diffusion_sampling_steps": diffusion_steps,
                            "sampling_mode": args.sampling_mode,
                            "random_root_seed": root_random_seed if args.sampling_mode == "random" else None,
                            "resolved_random_sample_seeds": random_sample_seeds,
                            "eeg_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                            "renderer_checkpoint_sha256": hashlib.sha256(args.renderer.read_bytes()).hexdigest(),
                            "files": sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()),
                            "warning": "Metric WAVs are unnormalized; listening WAVs share one bundle-level gain. Diffusion is qualitative and excluded from EEG efficacy metrics."})
    args.output.mkdir(parents=True, exist_ok=True); (args.output / "export_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    dataset.close(); print(json.dumps({"output": str(args.output), "bundles": len(records)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
