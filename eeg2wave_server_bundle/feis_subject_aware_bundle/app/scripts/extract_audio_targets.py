from __future__ import annotations

import argparse
import sys
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.audio_features import AudioFeatureConfig, extract_template_audio_features
from src.utils import load_simple_yaml, resolve_bundle_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract cached FEIS subject-aware audio targets.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "alignment.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--backend", default=None, help="auto | hubert_local | spectral_fallback_v1")
    parser.add_argument("--target-kind", default=None, help="hubert_pooled | hubert_sequence | encodec_latent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_simple_yaml(args.config)
    data_root = resolve_bundle_path(args.data_root or config["data"]["root"], BUNDLE_DIR)
    output_path = resolve_bundle_path(
        args.output_path or config["targets"]["cache_path"],
        BUNDLE_DIR,
    )
    audio_cfg = config["audio"]
    target_cfg = config["targets"]
    feature_config = AudioFeatureConfig(
        sample_rate=int(audio_cfg["sample_rate"]),
        duration_sec=float(audio_cfg["duration_sec"]),
        normalize=str(audio_cfg.get("normalize", "rms")),
        target_rms=float(audio_cfg.get("target_rms", 0.08)),
        max_gain=float(audio_cfg.get("max_gain", 10.0)),
        backend=str(args.backend or target_cfg.get("backend", "auto")),
        target_kind=str(args.target_kind or target_cfg.get("target_kind", "hubert_pooled")),
        ssl_model_name_or_path=str(
            resolve_bundle_path(
                target_cfg.get("ssl_model_name_or_path", target_cfg.get("hubert_model_name_or_path", "facebook/hubert-base-ls960")),
                BUNDLE_DIR,
            )
        ),
        codec_model_name_or_path=str(
            resolve_bundle_path(
                target_cfg.get("codec_model_name_or_path", "facebook/encodec_24khz"),
                BUNDLE_DIR,
            )
        ),
        local_files_only=bool(target_cfg.get("local_files_only", True)),
        spectral_bins=int(target_cfg.get("spectral_bins", 48)),
        sequence_target_steps=int(target_cfg.get("sequence_target_steps", 16)),
        codec_bandwidth=float(target_cfg.get("codec_bandwidth", 6.0)),
    )
    metadata = extract_template_audio_features(
        feis_root=data_root,
        output_path=output_path,
        config=feature_config,
    )
    print(f"Saved target cache to {output_path}")
    print(f"Feature backend: {metadata['feature_backend']}")
    print(
        f"Templates: {metadata['num_templates']} | target_kind={metadata['target_kind']} | "
        f"target_shape={tuple(metadata['target_shape'])}"
    )


if __name__ == "__main__":
    main()
