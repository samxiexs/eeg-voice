from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from scripts.train_karaone_v10 import make_model_config
from src.audio_features import AudioFeatureConfig, load_codec_backend
from src.karaone_v9.data import KaraOneV9TargetBank
from src.karaone_v10.data import KaraOneV10ClusterBank, KaraOneV10ClusteredDataset
from src.karaone_v10.model import KaraOneV10ClusteredChannelMoEFlow
from src.utils import (
    ensure_dir,
    load_simple_yaml,
    load_wav_fixed,
    normalize_rms,
    pad_or_crop_audio,
    resolve_bundle_path,
    save_wav,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate v10 reference, retrieval-diagnostic, and generated-codec wavs.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_v10.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="subject_test", choices=["subject_train", "subject_val", "subject_test", "all"])
    parser.add_argument("--stages", default=None)
    parser.add_argument("--cluster-bank", default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--no-retrieval", dest="include_retrieval", action="store_false", help="do not write retrieval-diagnostic recon wavs")
    parser.add_argument("--include-generated-codec", action="store_true", help="also sample v10 codec flow and decode with local EnCodec")
    parser.add_argument("--generated-steps", type=int, default=32)
    parser.add_argument("--codec-model", default="../models/encodec_24khz")
    parser.add_argument("--codec-bandwidth", type=float, default=6.0)
    parser.set_defaults(include_retrieval=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = synthesize(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def synthesize(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_simple_yaml(args.config)
    synth_cfg = cfg.get("synthesis", {})
    device = torch.device(args.device or default_device())
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    stages = tuple(item.strip() for item in str(args.stages or cfg["data"].get("stages", "overt_like")).split(",") if item.strip())
    subject_val = str(cfg["data"].get("subject_val", "P02"))
    subject_test = str(cfg["data"].get("subject_test", "MM21"))
    eeg_len = int(cfg["data"].get("eeg_len", 1280))
    cache_cfg = cfg.get("cache", {})
    targets = KaraOneV9TargetBank(
        resolve_bundle_path(cache_cfg["semantic"], BUNDLE_DIR),
        codec_cache=resolve_bundle_path(cache_cfg["codec"], BUNDLE_DIR) if cache_cfg.get("codec") else None,
        prosody_cache=resolve_bundle_path(cache_cfg["prosody"], BUNDLE_DIR) if cache_cfg.get("prosody") else None,
        semantic_token_cache=resolve_bundle_path(cache_cfg["semantic_tokens"], BUNDLE_DIR) if cache_cfg.get("semantic_tokens") else None,
        data_root=root,
    )
    cluster_path = args.cluster_bank or cache_cfg.get("cluster_bank", "")
    cluster_bank = KaraOneV10ClusterBank(resolve_bundle_path(cluster_path, BUNDLE_DIR) if cluster_path else None)
    train_ds = KaraOneV10ClusteredDataset(
        root,
        targets,
        "subject_train",
        cluster_bank=cluster_bank,
        stages=stages,
        subject_val=subject_val,
        subject_test=subject_test,
        eeg_len=eeg_len,
    )
    eval_datasets = make_eval_datasets(
        root=root,
        targets=targets,
        split=args.split,
        cluster_bank=cluster_bank,
        stages=stages,
        subject_val=subject_val,
        subject_test=subject_test,
        eeg_len=eeg_len,
    )
    model = KaraOneV10ClusteredChannelMoEFlow(make_model_config(cfg, targets, train_ds, eeg_len=eeg_len)).to(device)
    checkpoint = torch.load(resolve_bundle_path(args.checkpoint, BUNDLE_DIR), map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    train_bank = build_train_audio_bank(root, targets, train_ds)
    codec_backend = build_codec_decoder(args, cfg) if args.include_generated_codec else None
    out_dir = ensure_dir(args.out_dir.expanduser().resolve())
    sample_rate = int(args.sample_rate or synth_cfg.get("sample_rate", 16000))
    duration_sec = float(args.duration_sec or synth_cfg.get("duration_sec", 2.0))
    n_samples = int(round(sample_rate * duration_sec))
    limit = int(args.limit if args.limit is not None else synth_cfg.get("limit", 12))
    if limit <= 0:
        limit = 10**12
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    total_seen = 0

    with torch.no_grad():
        for split_name, eval_ds in eval_datasets:
            for idx in range(len(eval_ds)):
                if total_seen >= limit:
                    break
                total_seen += 1
                item = eval_ds[idx]
                out = model(
                    item["eeg"].unsqueeze(0).to(device),
                    item["stage_idx"].view(1).to(device),
                    item["eeg_valid_len"].view(1).to(device),
                    mask_ratio=0.0,
                    lambda_subject_adv=0.0,
                )
                pred = out["pred_semantic_summary"].detach().cpu().numpy()[0]
                prompt_idx = int(out["prompt_logits"].detach().cpu().argmax(dim=-1).item())
                query_subject = str(item["subject"])
                query_label = str(item["label"])
                query_stage = str(item["stage"])
                query_trial = int(item["trial_index"])
                sample_key = safe_name(f"{split_name}_{query_subject}_{query_label}_{query_stage}_t{query_trial:03d}")
                reference_path = resolve_audio_path(root, targets.semantic.audio_path(query_subject, query_trial))
                if not reference_path.exists():
                    continue

                reference_audio = load_wav_fixed(reference_path, sample_rate, n_samples, normalize="rms", target_rms=0.08)
                reference_file = f"reference_{sample_key}.wav"
                save_wav(out_dir / reference_file, reference_audio, sample_rate)
                retrieved = retrieve_nearest(pred, train_bank, exclude_key=f"{query_subject}:{query_trial}")
                base_meta = {
                    "sample_key": sample_key,
                    "subject": query_subject,
                    "label": query_label,
                    "stage": query_stage,
                    "trial_index": query_trial,
                    "split": split_name,
                    "retrieved_subject": retrieved["subject"],
                    "retrieved_label": retrieved["label"],
                    "retrieved_trial_index": retrieved["trial_index"],
                    "retrieval_cosine": retrieved["cosine"],
                    "pred_prompt": label_from_idx(eval_ds.label_vocab, prompt_idx),
                    "generated_steps": int(args.generated_steps) if args.include_generated_codec else "",
                    "generated_status": "codec_flow_attempt" if args.include_generated_codec else "",
                }
                manifest_rows.append({**base_meta, "wav_type": "original", "file": reference_file, "rms": rms(reference_audio)})
                row_payload: dict[str, Any] = {**base_meta, "reference_wav": str(out_dir / reference_file)}

                if args.include_retrieval:
                    recon_source = Path(str(retrieved["audio_path"]))
                    if recon_source.exists():
                        recon_audio = load_wav_fixed(recon_source, sample_rate, n_samples, normalize="rms", target_rms=0.08)
                        recon_file = f"recon_{sample_key}.wav"
                        save_wav(out_dir / recon_file, recon_audio, sample_rate)
                        manifest_rows.append({**base_meta, "wav_type": "pred_env_scaled", "file": recon_file, "rms": rms(recon_audio)})
                        row_payload["recon_wav"] = str(out_dir / recon_file)

                if args.include_generated_codec:
                    if codec_backend is None or targets.codec is None:
                        raise RuntimeError("Generated codec wavs require codec cache and local EnCodec backend")
                    generated_audio, generated_sr, latent_path = generate_codec_wav(
                        model=model,
                        codec_backend=codec_backend,
                        targets=targets,
                        item=item,
                        out_dir=out_dir,
                        sample_key=sample_key,
                        device=device,
                        generated_steps=int(args.generated_steps),
                        duration_sec=duration_sec,
                    )
                    generated_file = f"generated_codec_{sample_key}.wav"
                    save_wav(out_dir / generated_file, generated_audio, generated_sr)
                    manifest_rows.append({**base_meta, "wav_type": "generated_codec", "file": generated_file, "rms": rms(generated_audio)})
                    row_payload["generated_codec_wav"] = str(out_dir / generated_file)
                    row_payload["generated_codec_latent"] = str(latent_path)
                rows.append(row_payload)
            if total_seen >= limit:
                break

    if not manifest_rows:
        raise RuntimeError("v10 synthesis wrote no wavs; check audio paths and split/stage selection")

    write_manifest(out_dir / "listening_manifest.csv", manifest_rows)
    metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
    gate_pass = bool(metrics.get("subject_val_v10_research_gate_pass", False)) and bool(metrics.get("subject_test_v10_research_gate_pass", False))
    if args.include_generated_codec and args.include_retrieval:
        waveform_status = "diagnostic_retrieval_plus_generated_codec_attempt"
    elif args.include_generated_codec:
        waveform_status = "generated_codec_attempt_unverified"
    else:
        waveform_status = "semantic_gate_pass_retrieval_backend" if gate_pass else "diagnostic_retrieval_gate_not_passed"
    summary = {
        "event": "karaone_v10_synthesis",
        "checkpoint": str(resolve_bundle_path(args.checkpoint, BUNDLE_DIR)),
        "out_dir": str(out_dir),
        "split": args.split,
        "stages": list(stages),
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "n_pairs": len(rows),
        "include_retrieval": bool(args.include_retrieval),
        "include_generated_codec": bool(args.include_generated_codec),
        "generated_steps": int(args.generated_steps) if args.include_generated_codec else None,
        "waveform_status": waveform_status,
        "claim": "retrieval wavs are diagnostic; generated_codec wavs are codec-flow attempts and are not evidence of EEG-to-waveform success unless transport is trained and semantic/prosody gates pass",
        "checkpoint_missing_keys": missing,
        "checkpoint_unexpected_keys": unexpected,
        "rows": rows,
    }
    write_json(out_dir / "v10_synthesis_summary.json", summary)
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# KaraOne v10 Wavs",
                "",
                f"- waveform_status: `{waveform_status}`",
                f"- include_retrieval: `{bool(args.include_retrieval)}`",
                f"- include_generated_codec: `{bool(args.include_generated_codec)}`",
                "- `pred_env_scaled`: EEG semantic summary -> nearest train-only speech audio retrieval.",
                "- `generated_codec`: EEG condition -> v10 codec-flow sample -> local EnCodec decoder.",
                "- caution: generated codec wavs require a trained transport checkpoint before they can be interpreted.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def make_eval_datasets(
    *,
    root: Path,
    targets: KaraOneV9TargetBank,
    split: str,
    cluster_bank: KaraOneV10ClusterBank,
    stages: tuple[str, ...],
    subject_val: str,
    subject_test: str,
    eeg_len: int,
) -> list[tuple[str, KaraOneV10ClusteredDataset]]:
    splits = ["subject_train", "subject_val", "subject_test"] if split == "all" else [split]
    out = []
    for name in splits:
        out.append(
            (
                name,
                KaraOneV10ClusteredDataset(
                    root,
                    targets,
                    name,
                    cluster_bank=cluster_bank,
                    stages=stages,
                    subject_val=subject_val,
                    subject_test=subject_test,
                    eeg_len=eeg_len,
                ),
            )
        )
    return out


def build_train_audio_bank(root: Path, targets: KaraOneV9TargetBank, dataset: KaraOneV10ClusteredDataset) -> dict[str, Any]:
    summaries = []
    rows = []
    for entry in dataset.entries:
        audio_path = resolve_audio_path(root, targets.semantic.audio_path(entry.subject, entry.trial_index))
        if not audio_path.exists():
            continue
        summaries.append(targets.semantic_summary(entry.subject, entry.trial_index))
        rows.append(
            {
                "subject": entry.subject,
                "label": entry.label,
                "trial_index": int(entry.trial_index),
                "key": f"{entry.subject}:{int(entry.trial_index)}",
                "audio_path": str(audio_path),
            }
        )
    if not rows:
        raise RuntimeError("No train audio files found for v10 semantic retrieval")
    matrix = np.stack(summaries, axis=0).astype(np.float32)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)
    return {"summary": matrix, "rows": rows}


def retrieve_nearest(query: np.ndarray, bank: dict[str, Any], *, exclude_key: str | None = None) -> dict[str, Any]:
    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-8)
    scores = (q @ bank["summary"].T).reshape(-1)
    if exclude_key:
        for idx, row in enumerate(bank["rows"]):
            if str(row.get("key")) == str(exclude_key):
                scores[idx] = -np.inf
    if not np.isfinite(scores).any():
        scores = (q @ bank["summary"].T).reshape(-1)
    idx = int(np.argmax(scores))
    row = dict(bank["rows"][idx])
    row["cosine"] = float(scores[idx])
    return row


def build_codec_decoder(args: argparse.Namespace, cfg: dict):
    synth_cfg = cfg.get("synthesis", {})
    codec_model = resolve_bundle_path(args.codec_model, BUNDLE_DIR)
    return load_codec_backend(
        AudioFeatureConfig(
            duration_sec=float(args.duration_sec or synth_cfg.get("duration_sec", 2.0)),
            target_kind="encodec_latent",
            backend="encodec_latent",
            codec_model_name_or_path=str(codec_model),
            codec_bandwidth=float(args.codec_bandwidth),
            local_files_only=True,
        )
    )


def generate_codec_wav(
    *,
    model: KaraOneV10ClusteredChannelMoEFlow,
    codec_backend,
    targets: KaraOneV9TargetBank,
    item: dict[str, Any],
    out_dir: Path,
    sample_key: str,
    device: torch.device,
    generated_steps: int,
    duration_sec: float,
) -> tuple[np.ndarray, int, Path]:
    if targets.codec is None:
        raise RuntimeError("targets.codec is required for generated codec wavs")
    generated = model.generate_codec(
        item["eeg"].unsqueeze(0).to(device),
        item["stage_idx"].view(1).to(device),
        item["eeg_valid_len"].view(1).to(device),
        steps=int(generated_steps),
        codec_steps=int(targets.codec_steps),
    )
    pred_norm = generated["pred_codec_seq"].detach().cpu().numpy()[0].astype(np.float32)
    pred_raw = pred_norm * targets.codec.target_std.reshape(1, -1) + targets.codec.target_mean.reshape(1, -1)
    latent_dir = ensure_dir(out_dir / "generated_codec_latents")
    latent_path = latent_dir / f"generated_codec_{sample_key}.npz"
    np.savez_compressed(latent_path, pred_codec_norm=pred_norm, pred_codec_raw=pred_raw)
    audio = codec_backend.decode(pred_raw.astype(np.float32), decoder_scales=targets.codec.default_decoder_scales)
    sample_rate = int(codec_backend.sample_rate)
    target_len = int(round(sample_rate * float(duration_sec)))
    audio = pad_or_crop_audio(np.asarray(audio, dtype=np.float32), target_len)
    audio = normalize_rms(audio, target_rms=0.08, max_gain=8.0)
    return audio.astype(np.float32), sample_rate, latent_path


def resolve_audio_path(root: Path, audio_path: str | Path) -> Path:
    path = Path(audio_path)
    return path if path.is_absolute() else root / path


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_") or "sample"


def rms(audio: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(np.square(audio), dtype=np.float64)) + 1e-12))


def label_from_idx(labels: list[str], idx: int) -> str:
    return str(labels[idx]) if 0 <= int(idx) < len(labels) else str(idx)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    main()

