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

from scripts.organize_karaone_wavs import organize_wavs  # noqa: E402
from scripts.synthesize_karaone_v11 import (  # noqa: E402
    build_codec_decoder,
    build_train_audio_bank,
    default_device,
    label_from_idx,
    retrieve_nearest,
    rms,
    safe_name,
    write_manifest,
)
from scripts.train_karaone_v12 import make_model_config  # noqa: E402
from src.karaone_v9.data import KaraOneV9TargetBank  # noqa: E402
from src.karaone_v12.data import KaraOneV10ClusterBank, KaraOneV11TokenBank, KaraOneV12Dataset, KaraOneV12TimeAnchorBank  # noqa: E402
from src.karaone_v12.model import KaraOneV12TokenGenerator  # noqa: E402
from src.karaone_v12.time_anchor import shift_audio  # noqa: E402
from src.utils import ensure_dir, load_simple_yaml, load_wav_fixed, normalize_rms, pad_or_crop_audio, resolve_bundle_path, save_wav, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate v12 reference, retrieval, raw generated codec, and predicted-lag generated wavs.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_v12.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="subject_test", choices=["subject_train", "subject_val", "subject_test", "all"])
    parser.add_argument("--stages", default=None)
    parser.add_argument("--cluster-bank", default=None)
    parser.add_argument("--token-bank", default=None)
    parser.add_argument("--time-anchor-bank", default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--no-retrieval", dest="include_retrieval", action="store_false")
    parser.add_argument("--include-generated-codec", action="store_true")
    parser.add_argument("--codec-model", default="../models/encodec_24khz")
    parser.add_argument("--codec-bandwidth", type=float, default=6.0)
    parser.set_defaults(include_retrieval=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(synthesize(args), ensure_ascii=False, indent=2))


def synthesize(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_simple_yaml(args.config)
    synth_cfg = cfg.get("synthesis", {})
    device = torch.device(args.device or default_device())
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    stages = tuple(item.strip() for item in str(args.stages or cfg["data"].get("stages", "overt_like")).split(",") if item.strip())
    cache_cfg = cfg.get("cache", {})
    subject_val = str(cfg["data"].get("subject_val", "P02"))
    subject_test = str(cfg["data"].get("subject_test", "MM21"))
    eeg_len = int(cfg["data"].get("eeg_len", 1280))
    targets = KaraOneV9TargetBank(
        resolve_bundle_path(cache_cfg["semantic"], BUNDLE_DIR),
        codec_cache=resolve_bundle_path(cache_cfg["codec"], BUNDLE_DIR) if cache_cfg.get("codec") else None,
        prosody_cache=resolve_bundle_path(cache_cfg["prosody"], BUNDLE_DIR) if cache_cfg.get("prosody") else None,
        semantic_token_cache=resolve_bundle_path(cache_cfg["semantic_tokens"], BUNDLE_DIR) if cache_cfg.get("semantic_tokens") else None,
        data_root=root,
    )
    cluster_bank = KaraOneV10ClusterBank(resolve_bundle_path(args.cluster_bank or cache_cfg.get("cluster_bank", ""), BUNDLE_DIR))
    token_bank = KaraOneV11TokenBank(resolve_bundle_path(args.token_bank or cache_cfg.get("v11_token_bank", ""), BUNDLE_DIR))
    time_bank = KaraOneV12TimeAnchorBank(resolve_bundle_path(args.time_anchor_bank or cache_cfg.get("v12_time_anchor_bank", ""), BUNDLE_DIR))
    train_ds = KaraOneV12Dataset(root, targets, "subject_train", cluster_bank=cluster_bank, token_bank=token_bank, time_anchor_bank=time_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len)
    eval_datasets = [
        (split, KaraOneV12Dataset(root, targets, split, cluster_bank=cluster_bank, token_bank=token_bank, time_anchor_bank=time_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len))
        for split in (["subject_train", "subject_val", "subject_test"] if args.split == "all" else [args.split])
    ]
    model = KaraOneV12TokenGenerator(make_model_config(cfg, targets, train_ds, token_bank, time_bank, eeg_len=eeg_len, stages=stages), codec_codebook=torch.from_numpy(token_bank.codec_codebook)).to(device)
    checkpoint = torch.load(resolve_bundle_path(args.checkpoint, BUNDLE_DIR), map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    train_bank = build_train_audio_bank(root, targets, train_ds)
    codec_backend = build_codec_decoder(args, cfg) if args.include_generated_codec else None
    out_dir = ensure_dir(args.out_dir.expanduser().resolve())
    for sub in ["reference", "retrieval_diagnostic", "generated_codec", "generated_codec_pred_lag", "generated_codec_latents", "time_aligned_diagnostic"]:
        ensure_dir(out_dir / sub)
    sample_rate = int(args.sample_rate or synth_cfg.get("sample_rate", 16000))
    duration_sec = float(args.duration_sec or synth_cfg.get("duration_sec", 2.0))
    n_samples = int(round(sample_rate * duration_sec))
    limit = int(args.limit if args.limit is not None else synth_cfg.get("limit", 12))
    if limit <= 0:
        limit = 10**12
    rows, manifest_rows, trace_rows = [], [], []
    total_seen = 0
    with torch.no_grad():
        for split_name, eval_ds in eval_datasets:
            for idx in range(len(eval_ds)):
                if total_seen >= limit:
                    break
                total_seen += 1
                item = eval_ds[idx]
                channel_clusters = item["channel_cluster_id"].unsqueeze(0).to(device)
                out = model.generate_codec_tokens(
                    item["eeg"].unsqueeze(0).to(device),
                    item["stage_idx"].view(1).to(device),
                    item["eeg_valid_len"].view(1).to(device),
                    channel_cluster_id=channel_clusters,
                )
                pred = out["pred_semantic_summary"].detach().cpu().numpy()[0]
                prompt_idx = int(out["prompt_logits"].detach().cpu().argmax(dim=-1).item())
                pred_onset = float(out["pred_onset_sec"].detach().cpu().reshape(-1)[0])
                pred_duration = float(out["pred_duration_sec"].detach().cpu().reshape(-1)[0])
                pred_lag = float(out["pred_lag_sec"].detach().cpu().reshape(-1)[0])
                subject = str(item["subject"])
                label = str(item["label"])
                stage = str(item["stage"])
                trial = int(item["trial_index"])
                sample_key = safe_name(f"{split_name}_{subject}_{label}_{stage}_t{trial:03d}")
                reference_path = resolve_audio_path(root, targets.semantic.audio_path(subject, trial))
                if not reference_path.exists():
                    continue
                reference_audio = load_wav_fixed(reference_path, sample_rate, n_samples, normalize="rms", target_rms=0.08)
                reference_file = f"reference/reference_{sample_key}.wav"
                save_wav(out_dir / reference_file, reference_audio, sample_rate)
                retrieved = retrieve_nearest(pred, train_bank, exclude_key=f"{subject}:{trial}")
                base = {
                    "sample_key": sample_key,
                    "subject": subject,
                    "label": label,
                    "stage": stage,
                    "trial_index": trial,
                    "split": split_name,
                    "retrieved_subject": retrieved["subject"],
                    "retrieved_label": retrieved["label"],
                    "retrieved_trial_index": retrieved["trial_index"],
                    "retrieval_cosine": retrieved["cosine"],
                    "pred_prompt": label_from_idx(eval_ds.label_vocab, prompt_idx),
                    "pred_onset_sec": pred_onset,
                    "pred_duration_sec": pred_duration,
                    "pred_lag_sec": pred_lag,
                    "best_lag_sec": "",
                    "generated_status": "codec_token_generation_with_predicted_lag",
                }
                append_manifest(manifest_rows, trace_rows, base, wav_type="original", file=reference_file, rms_value=rms(reference_audio), diagnostic=False)
                row_payload: dict[str, Any] = {**base, "reference_wav": str(out_dir / reference_file)}
                if args.include_retrieval:
                    recon_source = Path(str(retrieved["audio_path"]))
                    if recon_source.exists():
                        recon_audio = load_wav_fixed(recon_source, sample_rate, n_samples, normalize="rms", target_rms=0.08)
                        recon_file = f"retrieval_diagnostic/retrieval_{sample_key}.wav"
                        save_wav(out_dir / recon_file, recon_audio, sample_rate)
                        append_manifest(manifest_rows, trace_rows, base, wav_type="retrieval_diagnostic", file=recon_file, rms_value=rms(recon_audio), diagnostic=True)
                        row_payload["retrieval_diagnostic_wav"] = str(out_dir / recon_file)
                if args.include_generated_codec:
                    if codec_backend is None or targets.codec is None:
                        raise RuntimeError("Generated codec wavs require codec cache and local EnCodec backend")
                    pred_norm = out["pred_codec_seq"].detach().cpu().numpy()[0].astype(np.float32)
                    pred_raw = pred_norm * targets.codec.target_std.reshape(1, -1) + targets.codec.target_mean.reshape(1, -1)
                    latent_path = out_dir / "generated_codec_latents" / f"generated_codec_{sample_key}.npz"
                    np.savez_compressed(latent_path, pred_codec_norm=pred_norm, pred_codec_raw=pred_raw, pred_codec_token_ids=out["pred_codec_token_ids"].detach().cpu().numpy()[0])
                    audio = codec_backend.decode(pred_raw.astype(np.float32), decoder_scales=targets.codec.default_decoder_scales)
                    codec_sr = int(codec_backend.sample_rate)
                    audio = normalize_rms(pad_or_crop_audio(np.asarray(audio, dtype=np.float32), int(round(codec_sr * duration_sec))), target_rms=0.08, max_gain=8.0)
                    generated_file = f"generated_codec/generated_codec_{sample_key}.wav"
                    save_wav(out_dir / generated_file, audio.astype(np.float32), codec_sr)
                    append_manifest(manifest_rows, trace_rows, base, wav_type="generated_codec", file=generated_file, rms_value=rms(audio), diagnostic=False)
                    shifted = shift_audio(audio, pred_lag, sample_rate=codec_sr)
                    shifted_file = f"generated_codec_pred_lag/generated_codec_pred_lag_{sample_key}.wav"
                    save_wav(out_dir / shifted_file, shifted.astype(np.float32), codec_sr)
                    append_manifest(manifest_rows, trace_rows, base, wav_type="generated_codec_pred_lag", file=shifted_file, rms_value=rms(shifted), diagnostic=False)
                    row_payload["generated_codec_wav"] = str(out_dir / generated_file)
                    row_payload["generated_codec_pred_lag_wav"] = str(out_dir / shifted_file)
                    row_payload["generated_codec_latent"] = str(latent_path)
                rows.append(row_payload)
            if total_seen >= limit:
                break
    if not manifest_rows:
        raise RuntimeError("v12 synthesis wrote no wavs")
    manifest_path = out_dir / "listening_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    write_trace(out_dir / "generation_trace.csv", trace_rows)
    grouped_summary = organize_wavs(wav_dir=out_dir, manifest=manifest_path, mode="symlink")
    metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
    summary = {
        "event": "karaone_v12_synthesis",
        "checkpoint": str(resolve_bundle_path(args.checkpoint, BUNDLE_DIR)),
        "out_dir": str(out_dir),
        "split": args.split,
        "stages": list(stages),
        "n_pairs": len(rows),
        "include_retrieval": bool(args.include_retrieval),
        "include_generated_codec": bool(args.include_generated_codec),
        "waveform_status": "generated_codec_token_attempt_with_predicted_lag",
        "claim": "generated_codec_pred_lag wavs use EEG-predicted lag/onset diagnostics; retrieval_diagnostic wavs remain diagnostic only",
        "alignment_gate_pass": bool(metrics.get("subject_val_v12_alignment_gate_pass", False) and metrics.get("subject_test_v12_alignment_gate_pass", False)),
        "predicted_lag_generation_gate_pass": bool(metrics.get("subject_val_v12_predicted_lag_generation_gate_pass", False) and metrics.get("subject_test_v12_predicted_lag_generation_gate_pass", False)),
        "checkpoint_missing_keys": missing,
        "checkpoint_unexpected_keys": unexpected,
        "grouped_wavs": grouped_summary,
        "rows": rows,
    }
    write_json(out_dir / "v12_synthesis_summary.json", summary)
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# KaraOne v12 Wavs",
                "",
                "- `reference/`: ground-truth trial audio for evaluation.",
                "- `retrieval_diagnostic/`: train-audio retrieval baseline; diagnostic only.",
                "- `generated_codec/`: EEG-conditioned codec-token generation.",
                "- `generated_codec_pred_lag/`: generated codec shifted by model-predicted lag.",
                "- `generation_trace.csv`: per-wav provenance and timing fields.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def append_manifest(manifest_rows: list[dict[str, Any]], trace_rows: list[dict[str, Any]], base: dict[str, Any], *, wav_type: str, file: str, rms_value: float, diagnostic: bool) -> None:
    manifest_rows.append({**base, "wav_type": wav_type, "file": file, "rms": rms_value})
    trace_rows.append({**base, "wav_type": wav_type, "file": file, "rms": rms_value, "diagnostic_only": bool(diagnostic)})


def write_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_audio_path(root: Path, audio_path: str | Path) -> Path:
    path = Path(audio_path)
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()
