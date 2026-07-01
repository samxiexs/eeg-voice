"""Synthesize FEIS v3 reference, retrieval_diagnostic, and generated_codec wavs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.feis_v3.data import FEISV3AudioTokenBank, FEISV3ClusterBank, FEISV3Dataset
from src.feis_v3.model import FEISV3ModelConfig, FEISV3TokenGenerator
from src.feis_v3.synth import (
    decode_generated_codec,
    load_reference_audio,
    retrieve_diagnostic_audio,
    write_audio_comparison_figure,
    write_grouped_triplet,
    write_waveform_contact_sheet,
)
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, resolve_feis_root, save_wav


def parse_args():
    p = argparse.ArgumentParser(description="Synthesize FEIS v3 generated codec wavs.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "feis_v3_tokenized_generation.yaml"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="subject_test")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _move_batch(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    token_bank = FEISV3AudioTokenBank(ckpt["token_cache"])
    cluster_bank = FEISV3ClusterBank(ckpt["cluster_cache"])
    data_cfg = cfg.get("data", {})
    audio_cfg = cfg.get("audio", {})
    feis_root = resolve_feis_root(resolve_bundle_path(data_cfg.get("root", "../data/feis"), BUNDLE_DIR))
    ds = FEISV3Dataset(
        data_root=resolve_bundle_path(data_cfg.get("root", "../data/feis"), BUNDLE_DIR),
        token_bank=token_bank,
        split=args.split,
        stage=str(ckpt["stage"]),
        cluster_bank=cluster_bank,
        eeg_len=int(data_cfg.get("eeg_len", 1280)),
        include_anomalous=bool(data_cfg.get("include_anomalous", False)),
        subject_val=str(data_cfg.get("subject_val", "20")),
        subject_test=str(data_cfg.get("subject_test", "21")),
        negative_stage=str(data_cfg.get("negative_stage", "resting")),
        allow_negative_train=bool(ckpt.get("allow_negative_train", False)),
    )
    model = FEISV3TokenGenerator(FEISV3ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    out_dir = ensure_dir(args.out_dir or Path(ckpt["run_dir"]))
    for sub in [
        "reference",
        "retrieval_diagnostic",
        "generated_codec",
        "grouped_wavs",
        "waveform_compare",
        "waveform_compare/generated_codec",
        "waveform_compare/retrieval_diagnostic",
    ]:
        ensure_dir(out_dir / "wavs" / sub)
    limit = len(ds) if int(args.limit) <= 0 else min(int(args.limit), len(ds))
    rows = []
    with torch.no_grad():
        for idx in range(limit):
            item = ds[idx]
            batch = _move_batch(
                {
                    "eeg": item["eeg"].unsqueeze(0),
                    "stage_idx": item["stage_idx"].unsqueeze(0),
                    "eeg_valid_len": item["eeg_valid_len"].unsqueeze(0),
                    "channel_cluster_id": item["channel_cluster_id"].unsqueeze(0),
                },
                device,
            )
            out = model(**batch)
            sem_hist = out["semantic_logits"].softmax(dim=-1).mean(dim=1).squeeze(0).detach().cpu().numpy()
            retrieval_idx = retrieve_diagnostic_audio(token_bank, sem_hist, label=str(item["label"]))
            generated_tokens = out["codec_logits"].argmax(dim=-1).squeeze(0).detach().cpu().numpy()
            generated = decode_generated_codec(token_bank, generated_tokens)
            reference = load_reference_audio(
                feis_root,
                str(item["audio_path"]),
                sample_rate=token_bank.sample_rate,
                duration_sec=float(audio_cfg.get("duration_sec", 1.0)),
            )
            retrieval = load_reference_audio(
                feis_root,
                token_bank.audio_paths[retrieval_idx],
                sample_rate=token_bank.sample_rate,
                duration_sec=float(audio_cfg.get("duration_sec", 1.0)),
            )
            tag = f"{item['sample_key']}_{item['label']}_{item['stage']}"
            reference_path = out_dir / "wavs" / "reference" / f"{tag}_01_original_reference.wav"
            retrieval_path = out_dir / "wavs" / "retrieval_diagnostic" / f"{tag}_02_retrieval_diagnostic.wav"
            generated_path = out_dir / "wavs" / "generated_codec" / f"{tag}_03_generated_codec.wav"
            save_wav(reference_path, reference, token_bank.sample_rate)
            save_wav(retrieval_path, retrieval, token_bank.sample_rate)
            save_wav(generated_path, generated, token_bank.sample_rate)
            retrieval_fig = write_audio_comparison_figure(
                out_dir / "wavs" / "waveform_compare" / "retrieval_diagnostic" / f"{tag}_original_vs_retrieval_diagnostic.png",
                reference,
                retrieval,
                token_bank.sample_rate,
                f"{tag}: original vs retrieval_diagnostic",
            )
            generated_fig = write_audio_comparison_figure(
                out_dir / "wavs" / "waveform_compare" / "generated_codec" / f"{tag}_original_vs_generated_codec.png",
                reference,
                generated,
                token_bank.sample_rate,
                f"{tag}: original vs generated_codec",
            )
            grouped = write_grouped_triplet(
                out_dir,
                str(item["subject_id"]),
                str(item["label"]),
                int(item["trial_index"].item()),
                reference,
                retrieval,
                generated,
                token_bank.sample_rate,
            )
            rows.append(
                {
                    "sample_key": str(item["sample_key"]),
                    "subject_id": str(item["subject_id"]),
                    "label": str(item["label"]),
                    "stage": str(item["stage"]),
                    "trial_index": int(item["trial_index"].item()),
                    "repetition_index": int(item["repetition_index"].item()),
                    "audio_path": str(item["audio_path"]),
                    "audio_sha1": str(item["audio_sha1"]),
                    "wav_type": "triplet",
                    "claim_status": ckpt.get("claim_status", "diagnostic generated codec attempt"),
                    "original_reference": str(reference_path),
                    "retrieval_diagnostic": str(retrieval_path),
                    "generated_codec": str(generated_path),
                    "retrieval_diagnostic_figure": retrieval_fig,
                    "generated_codec_figure": generated_fig,
                    "grouped_original_reference": grouped["original_reference"],
                    "grouped_retrieval_diagnostic": grouped["retrieval_diagnostic"],
                    "grouped_generated_codec": grouped["generated_codec"],
                }
            )
    manifest_path = out_dir / "wavs" / "listening_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "sample_key",
            "subject_id",
            "label",
            "stage",
            "trial_index",
            "repetition_index",
            "audio_path",
            "audio_sha1",
            "wav_type",
            "claim_status",
            "original_reference",
            "retrieval_diagnostic",
            "generated_codec",
            "retrieval_diagnostic_figure",
            "generated_codec_figure",
            "grouped_original_reference",
            "grouped_retrieval_diagnostic",
            "grouped_generated_codec",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_waveform_contact_sheet(out_dir / "wavs" / "waveform_compare" / "original_vs_retrieval_diagnostic_contact_sheet.html", rows, generated=False)
    write_waveform_contact_sheet(out_dir / "wavs" / "waveform_compare" / "original_vs_generated_codec_contact_sheet.html", rows, generated=True)
    print(f"[done] wrote {len(rows)} FEIS v3 triplets to {out_dir / 'wavs'}")


if __name__ == "__main__":
    main()
