#!/usr/bin/env bash
set -euo pipefail

# One-command runner for the audio-only adapted-HuBERT-token-to-wav decoder.
#
# Modes:
#   full  (default): train/resume -> P02 latent metrics -> P02 wavs/figures
#   final: train/resume, then export train subjects, P02 and authorised MM21
#
# Examples:
#   DEVICE=mps bash run_karaone_0711v1_hubert_roundtrip.sh full
#   ALLOW_FINAL_TEST=1 DEVICE=mps bash run_karaone_0711v1_hubert_roundtrip.sh final

cd "$(dirname "$0")"

mode="${1:-full}"
python_bin="${PYTHON:-python3}"
config="${CONFIG:-configs/karaone_0711v1.yaml}"
stage="${STAGE:-overt_like}"
seed="${SEED:-11}"
device="${DEVICE:-auto}"
epochs="${EPOCHS:-}"
limit="${LIMIT:-}"
output_root="../artifacts/outputs_karaone_0711v1"
cache="../artifacts/karaone_0711v1/karaone_0711v1_${stage}_adapted_audio_targets_s${seed}.npz"
run_dir="${output_root}/karaone_0711v1_${stage}_hubert_roundtrip_s${seed}"
last_checkpoint="${run_dir}/checkpoints/last.pt"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-karaone-hubert-roundtrip}"
export PYTHONUNBUFFERED=1

usage() {
  cat <<'EOF'
Usage:
  bash run_karaone_0711v1_hubert_roundtrip.sh [full|final]

Environment overrides:
  PYTHON=python3       Python executable (default: python3)
  DEVICE=mps           mps, cuda, cpu, or auto (default: auto)
  CONFIG=configs/...   0711v1 YAML config
  STAGE=overt_like     overt_like or thinking
  SEED=11              run seed
  EPOCHS=80            optional training epoch override
  LIMIT=5              optional wav/PNG export limit; diagnostic only

Modes:
  full   Train/resume, then evaluate and synthesize P02 only.
  final  Train/resume, then evaluate and synthesize subject_train, P02, and MM21.
         MM21 requires ALLOW_FINAL_TEST=1.
EOF
}

preflight() {
  [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
  [[ -d ../data/karaone ]] || { echo "Missing KaraOne data: ../data/karaone" >&2; exit 2; }
  [[ -f "$cache" ]] || {
  echo "Missing adapted-HuBERT-token/EnCodec cache: $cache" >&2
    echo "Run the 0711v1 audio_ssl/audio_cache stage first." >&2
    exit 2
  }
  [[ -d ../models/encodec_24khz ]] || { echo "Missing local EnCodec checkpoint: ../models/encodec_24khz" >&2; exit 2; }
  "$python_bin" -c 'import torch, transformers, yaml, scipy; print("[hubert_roundtrip] torch=" + torch.__version__ + "; device=" + ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))'
}

run_full() {
  local command=("$python_bin" scripts/train_karaone_0711v1_hubert_roundtrip.py --config "$config" --stage "$stage" --seed "$seed" --phase all)
  if [[ -f "$last_checkpoint" ]]; then
    echo "[hubert_roundtrip] resuming completed/partial run from: $last_checkpoint"
    command+=(--resume-training "$last_checkpoint")
  fi
  if [[ "$device" != "auto" ]]; then command+=(--device "$device"); fi
  if [[ -n "$epochs" ]]; then command+=(--epochs "$epochs"); fi
  if [[ -n "$limit" ]]; then command+=(--limit "$limit"); fi
  "${command[@]}"
}

run_export_phase() {
  local phase="$1"
  local split="$2"
  local allow_final_test="${3:-0}"
  local command=("$python_bin" scripts/train_karaone_0711v1_hubert_roundtrip.py --config "$config" --stage "$stage" --seed "$seed" --phase "$phase" --split "$split")
  if [[ "$allow_final_test" == "1" ]]; then command+=(--allow-final-test); fi
  if [[ "$device" != "auto" ]]; then command+=(--device "$device"); fi
  if [[ "$phase" == "synthesize" && -n "$limit" ]]; then command+=(--limit "$limit"); fi
  "${command[@]}"
}

case "$mode" in
  full)
    preflight
    run_full
    echo "[hubert_roundtrip] P02 round-trip run complete: $run_dir"
    ;;
  final)
    if [[ "${ALLOW_FINAL_TEST:-}" != "1" ]]; then
      echo "Refusing MM21 access. Re-run with ALLOW_FINAL_TEST=1 to explicitly authorise final test evaluation/export." >&2
      exit 2
    fi
    preflight
    run_full
    run_export_phase evaluate subject_train
    run_export_phase synthesize subject_train
    run_export_phase evaluate subject_test 1
    run_export_phase synthesize subject_test 1
    echo "[hubert_roundtrip] subject_train + P02 + explicitly authorised MM21 run complete: $run_dir"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
