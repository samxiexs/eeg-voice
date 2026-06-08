from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import ensure_dir, load_simple_yaml, write_json


CODE_FILES = [
    "README.md",
    "RUN_GUIDE.md",
    "requirements.txt",
    "configs/alignment.yaml",
    "configs/alignment_ssl_local.yaml",
    "configs/waveform_protocol.yaml",
    "configs/config.yaml",
    "scripts/extract_audio_targets.py",
    "scripts/train_alignment.py",
    "scripts/eval_alignment.py",
    "scripts/train_waveform_protocol.py",
    "scripts/eval_waveform_protocol.py",
    "scripts/audit_pipeline.py",
    "src/__init__.py",
    "src/utils.py",
    "src/model.py",
    "src/losses.py",
    "src/dataset.py",
    "src/audio_features.py",
    "src/alignment_model.py",
    "src/alignment_losses.py",
    "src/eval_utils.py",
    "src/subject_conditioned_waveform.py",
]

ARTIFACT_DIRS = [
    "outputs/audio_targets",
    "outputs_alignment",
    "outputs_waveform_protocol",
    "outputs-0603-v1",
    "outputs-0603-v2",
    "outputs-0603-v3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a single-folder FEIS upload bundle for server transfer.")
    parser.add_argument("--bundle-root", default="server_bundle/feis_subject_aware_bundle")
    parser.add_argument("--data-root", default="eeg2wave_demo_bundle/data/feis")
    parser.add_argument("--ssl-model-dir", default=None, help="Local HuBERT/Wav2Vec2 directory to copy into bundle/models/")
    parser.add_argument("--ssl-model-name", default="hubert-base-ls960", help="Destination subfolder name under bundle/models/")
    parser.add_argument("--artifacts", nargs="*", default=ARTIFACT_DIRS)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def validate_local_ssl_model(path: Path) -> dict[str, object]:
    required_any = [
        "model.safetensors",
        "pytorch_model.bin",
    ]
    required_soft = [
        "config.json",
        "preprocessor_config.json",
    ]
    missing_soft = [name for name in required_soft if not (path / name).exists()]
    has_weight = any((path / name).exists() for name in required_any)
    return {
        "path": str(path),
        "exists": path.exists(),
        "has_weight_file": has_weight,
        "missing_soft_files": missing_soft,
        "looks_valid": path.exists() and has_weight and (path / "config.json").exists(),
    }


def write_server_readme(bundle_root: Path, ssl_model_name: str, has_ssl_model: bool) -> None:
    lines = [
        "# FEIS Server Bundle",
        "",
        "This folder is intended to be copied to a server as a single unit.",
        "",
        "## Layout",
        "",
        "```text",
        f"{bundle_root.name}/",
        "  app/",
        "  data/feis/",
        "  models/",
        "  artifacts/",
        "  reports/",
        "```",
        "",
        "## Local SSL Model Path",
        "",
        "The alignment SSL config points to:",
        "",
        f"`app/configs/alignment_ssl_local.yaml -> targets.ssl_model_name_or_path = ../models/{ssl_model_name}`",
        "",
    ]
    if has_ssl_model:
        lines.extend(
            [
                "The SSL model directory has already been copied into the bundle.",
                "",
                "Use it like this:",
                "",
                "```bash",
                "cd app",
                "python scripts/extract_audio_targets.py --config configs/alignment_ssl_local.yaml",
                "python scripts/train_alignment.py --config configs/alignment_ssl_local.yaml --protocol G",
                "```",
            ]
        )
    else:
        lines.extend(
            [
                "No local HuBERT/Wav2Vec2 model was copied yet.",
                "",
                f"Put your local model snapshot under `models/{ssl_model_name}/`, then rerun target extraction:",
                "",
                "```bash",
                "cd app",
                "python scripts/extract_audio_targets.py --config configs/alignment_ssl_local.yaml",
                "```",
            ]
        )
    (bundle_root / "RUN_SERVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rewrite_alignment_ssl_config(path: Path, ssl_model_name: str) -> None:
    config = load_simple_yaml(path)
    config["targets"]["ssl_model_name_or_path"] = f"../models/{ssl_model_name}"
    config["targets"]["backend"] = "ssl_local"
    path.write_text(
        "\n".join(
            [
                "data:",
                f"  root: {config['data']['root']}",
                f"  stage: {config['data']['stage']}",
                f"  protocol: {config['data']['protocol']}",
                f"  subject_id: {config['data']['subject_id']}",
                f"  holdout_subject_id: {config['data']['holdout_subject_id']}",
                f"  include_anomalous: {str(config['data']['include_anomalous']).lower()}",
                f"  ablation_mode: {config['data']['ablation_mode']}",
                "",
                "audio:",
                f"  sample_rate: {config['audio']['sample_rate']}",
                f"  duration_sec: {config['audio']['duration_sec']}",
                f"  normalize: {config['audio']['normalize']}",
                f"  target_rms: {config['audio']['target_rms']}",
                f"  max_gain: {config['audio']['max_gain']}",
                "",
                "targets:",
                f"  cache_path: {config['targets']['cache_path']}",
                f"  backend: {config['targets']['backend']}",
                f"  ssl_model_name_or_path: {config['targets']['ssl_model_name_or_path']}",
                f"  local_files_only: {str(config['targets']['local_files_only']).lower()}",
                f"  spectral_bins: {config['targets']['spectral_bins']}",
                "",
                "model:",
                f"  n_channels_eeg: {config['model']['n_channels_eeg']}",
                f"  hidden_dim: {config['model']['hidden_dim']}",
                f"  latent_dim: {config['model']['latent_dim']}",
                f"  use_label_head: {str(config['model']['use_label_head']).lower()}",
                f"  use_subject_demo_head: {str(config['model']['use_subject_demo_head']).lower()}",
                f"  subject_embedding_dim: {config['model']['subject_embedding_dim']}",
                "",
                "train:",
                f"  batch_size: {config['train']['batch_size']}",
                f"  lr: {config['train']['lr']}",
                f"  weight_decay: {config['train']['weight_decay']}",
                f"  epochs: {config['train']['epochs']}",
                f"  grad_clip: {config['train']['grad_clip']}",
                f"  num_workers: {config['train']['num_workers']}",
                f"  seed: {config['train']['seed']}",
                f"  lambda_cosine: {config['train']['lambda_cosine']}",
                f"  lambda_mse: {config['train']['lambda_mse']}",
                f"  lambda_prosody: {config['train']['lambda_prosody']}",
                f"  lambda_cls: {config['train']['lambda_cls']}",
                "",
                "output:",
                f"  root: {config['output']['root']}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    bundle_root = Path(args.bundle_root).resolve()
    source_data_root = Path(args.data_root).resolve()
    if args.clean:
        reset_dir(bundle_root)
    else:
        ensure_dir(bundle_root)

    app_root = ensure_dir(bundle_root / "app")
    data_root = ensure_dir(bundle_root / "data")
    models_root = ensure_dir(bundle_root / "models")
    artifacts_root = ensure_dir(bundle_root / "artifacts")
    reports_root = ensure_dir(bundle_root / "reports")

    for relpath in CODE_FILES:
        src = BUNDLE_DIR / relpath
        dst = app_root / relpath
        copy_file(src, dst)

    copy_tree(source_data_root, data_root / "feis")

    copied_artifacts = []
    for relpath in args.artifacts:
        src = (BUNDLE_DIR / relpath).resolve()
        dst = artifacts_root / Path(relpath).name
        if src.exists():
            copy_tree(src, dst)
            copied_artifacts.append(str(dst.relative_to(bundle_root)))

    repo_reports = [
        Path("reports/feis_subject_aware_audit.json"),
        Path("outputs/pipeline_audit.json"),
    ]
    for report_rel in repo_reports:
        report_src = BUNDLE_DIR.parent / report_rel if not report_rel.is_absolute() else report_rel
        if report_src.exists():
            copy_file(report_src, reports_root / report_src.name)

    ssl_model_report = None
    if args.ssl_model_dir:
        ssl_src = Path(args.ssl_model_dir).resolve()
        ssl_dst = models_root / args.ssl_model_name
        copy_tree(ssl_src, ssl_dst)
        ssl_model_report = validate_local_ssl_model(ssl_dst)
    else:
        ssl_dst = models_root / args.ssl_model_name
        ensure_dir(ssl_dst)
        (ssl_dst / "README.txt").write_text(
            "Place your local HuBERT or Wav2Vec2 snapshot here before uploading or before running SSL target extraction.\n",
            encoding="utf-8",
        )
        ssl_model_report = validate_local_ssl_model(ssl_dst)

    rewrite_alignment_ssl_config(app_root / "configs" / "alignment_ssl_local.yaml", args.ssl_model_name)
    write_server_readme(bundle_root, args.ssl_model_name, bool(ssl_model_report["looks_valid"]))

    manifest = {
        "bundle_root": str(bundle_root),
        "data_root": str(data_root / "feis"),
        "ssl_model": ssl_model_report,
        "copied_artifacts": copied_artifacts,
        "app_config": str(app_root / "configs" / "alignment_ssl_local.yaml"),
    }
    write_json(bundle_root / "bundle_manifest.json", manifest)
    print(f"Prepared server bundle at {bundle_root}")
    print(f"SSL model target dir: {ssl_dst}")
    print(f"Copied artifacts: {len(copied_artifacts)}")


if __name__ == "__main__":
    main()
