#!/usr/bin/env bash
set -euo pipefail

# One-command 0715 pipeline. It is independent of all 0711 checkpoints/caches.

cd "$(dirname "$0")"

mode="${1:-full}"
python_bin="${PYTHON:-python}"
config="${CONFIG:-configs/karaone_0715.yaml}"
device="${DEVICE:-auto}"
seed="15"
limit="${LIMIT:-}"
audio_epochs="${AUDIO_EPOCHS:-}"
eeg_epochs="${EEG_EPOCHS:-}"
force_audio_retrain="${FORCE_AUDIO_RETRAIN:-0}"

cache="../artifacts/karaone_0715/karaone_0715_encodec_codes_s${seed}.npz"
output_root="../artifacts/outputs_karaone_0715"
audio_run="${output_root}/karaone_0715_audio_codec_s${seed}"
eeg_run="${output_root}/karaone_0715_eeg_align_s${seed}"

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-karaone-0715}"

usage() {
  cat <<'EOF'
Usage:
  bash run_karaone_0715.sh [probe|prepare|audio|audio_audit|eeg|full|final]

Modes:
  probe    Run leakage/signal baselines on train subjects + P02; MM21 stays locked.
  prepare  Build resumable exact EnCodec discrete-code cache.
  audio    Train/resume voice-code encoder + masked code decoder.
  audio_audit  Run P02 audio-only MaskGIT/code/wav gate before EEG training.
  eeg      Train/resume clearing-calibrated overt-EEG alignment model.
  full     probe -> prepare -> audio -> audio-only gate -> eeg -> P02 metrics/wavs/figures.
  final    full -> train-split export -> authorised MM21 metrics/wavs/figures.

Environment overrides:
  DEVICE=mps          mps, cuda, cpu, or auto
  PYTHON=python       Python executable
  AUDIO_EPOCHS=60    Optional audio-model epoch override
  EEG_EPOCHS=80      Optional EEG-model epoch override
  LIMIT=5            Optional per-split synthesis limit for diagnostics

Final-test controls:
  ALLOW_FINAL_TEST=1 is mandatory for final mode.
  If the P02 gate fails, MM21 remains locked. For explicitly exploratory output only,
  also set ALLOW_EXPLORATORY=1; such output is not reportable as a successful test.
EOF
}

preflight() {
  [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
  [[ -f ../data/karaone/trials.csv ]] || { echo "Missing KaraOne data: ../data/karaone" >&2; exit 2; }
  [[ -d ../models/encodec_24khz ]] || { echo "Missing local EnCodec model: ../models/encodec_24khz" >&2; exit 2; }
  "$python_bin" -c 'import torch, numpy, scipy, sklearn, transformers, yaml, tqdm; print("[0715] torch=" + torch.__version__ + "; runtime_device=" + ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))'
}

run_probe() {
  local command=("$python_bin" scripts/diagnose_karaone_0715_signal.py --config "$config")
  "${command[@]}"
}

run_prepare() {
  local command=("$python_bin" scripts/build_karaone_0715_audio_cache.py --config "$config")
  if [[ "$device" != "auto" ]]; then command+=(--device "$device"); fi
  "${command[@]}"
}

run_audio() {
  local command=("$python_bin" scripts/train_karaone_0715.py --phase audio --config "$config")
  if [[ "$device" != "auto" ]]; then command+=(--device "$device"); fi
  if [[ -n "$audio_epochs" ]]; then command+=(--audio-epochs "$audio_epochs"); fi
  if [[ -f "${audio_run}/checkpoints/last.pt" && "${force_audio_retrain}" != "1" ]]; then
    echo "[0715] resuming audio model: ${audio_run}/checkpoints/last.pt"
    command+=(--resume "${audio_run}/checkpoints/last.pt")
  elif [[ "${force_audio_retrain}" == "1" ]]; then
    echo "[0715] forcing a fresh supervised audio run (ignoring existing checkpoint)"
  fi
  "${command[@]}"
}

run_audio_audit() {
  local command=("$python_bin" scripts/audit_karaone_0715_audio_roundtrip.py --config "$config")
  if [[ "$device" != "auto" ]]; then command+=(--device "$device"); fi
  if [[ -n "$limit" ]]; then command+=(--export-limit "$limit"); fi
  "${command[@]}"
}

run_eeg() {
  local command=("$python_bin" scripts/train_karaone_0715.py --phase eeg --config "$config")
  if [[ "$device" != "auto" ]]; then command+=(--device "$device"); fi
  if [[ -n "$eeg_epochs" ]]; then command+=(--eeg-epochs "$eeg_epochs"); fi
  if [[ "${ALLOW_EXPLORATORY:-0}" == "1" ]]; then command+=(--allow-failed-gate); fi
  if [[ -f "${eeg_run}/checkpoints/last.pt" ]]; then
    echo "[0715] resuming EEG model: ${eeg_run}/checkpoints/last.pt"
    command+=(--resume "${eeg_run}/checkpoints/last.pt")
  fi
  "${command[@]}"
}

run_evaluate() {
  local split="$1"
  local allow_test="${2:-0}"
  local command=("$python_bin" scripts/train_karaone_0715.py --phase evaluate --split "$split" --config "$config")
  if [[ "$device" != "auto" ]]; then command+=(--device "$device"); fi
  if [[ "$allow_test" == "1" ]]; then command+=(--allow-final-test); fi
  if [[ "${ALLOW_EXPLORATORY:-0}" == "1" ]]; then command+=(--allow-failed-gate); fi
  "${command[@]}"
}

run_synthesis() {
  local split="$1"
  local allow_test="${2:-0}"
  local command=("$python_bin" scripts/synthesize_karaone_0715.py --split "$split" --config "$config")
  if [[ "$device" != "auto" ]]; then command+=(--device "$device"); fi
  if [[ -n "$limit" ]]; then command+=(--limit "$limit"); fi
  if [[ "$allow_test" == "1" ]]; then command+=(--allow-final-test); fi
  if [[ "${ALLOW_EXPLORATORY:-0}" == "1" ]]; then command+=(--allow-failed-gate); fi
  "${command[@]}"
}

run_full() {
  run_probe
  run_prepare
  run_audio
  run_audio_audit
  run_eeg
  run_evaluate subject_val
  run_synthesis subject_val
}

case "$mode" in
  probe)
    preflight
    run_probe
    ;;
  prepare)
    preflight
    run_prepare
    ;;
  audio)
    preflight
    [[ -f "$cache" ]] || { echo "Missing cache; run prepare first: $cache" >&2; exit 2; }
    run_audio
    ;;
  audio_audit)
    preflight
    [[ -f "$cache" ]] || { echo "Missing cache; run prepare first: $cache" >&2; exit 2; }
    [[ -f "${audio_run}/checkpoints/best.pt" ]] || { echo "Missing audio checkpoint; run audio first." >&2; exit 2; }
    run_audio_audit
    ;;
  eeg)
    preflight
    [[ -f "$cache" ]] || { echo "Missing cache; run prepare first: $cache" >&2; exit 2; }
    [[ -f "${audio_run}/checkpoints/best.pt" ]] || { echo "Missing audio checkpoint; run audio first." >&2; exit 2; }
    run_eeg
    ;;
  full)
    preflight
    run_full
    echo "[0715] development run complete; inspect P02 gate and outputs: $eeg_run"
    ;;
  final)
    if [[ "${ALLOW_FINAL_TEST:-0}" != "1" ]]; then
      echo "Refusing MM21 access. Set ALLOW_FINAL_TEST=1 for an explicitly authorised final run." >&2
      exit 2
    fi
    preflight
    run_full
    run_evaluate subject_train
    run_synthesis subject_train
    run_evaluate subject_test 1
    run_synthesis subject_test 1
    echo "[0715] all-split export complete: $eeg_run"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
