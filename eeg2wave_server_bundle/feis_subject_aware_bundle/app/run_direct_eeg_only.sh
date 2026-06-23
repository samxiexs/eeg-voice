#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${APP_DIR}"

TAG="${TAG:-$(date +%m%d_%H%M)}"
STAGES="${STAGES:-stimuli,thinking}"
EPOCHS="${EPOCHS:-30}"
SPLIT="${SPLIT:-test_holdout}"
SYNTH_LIMIT="${SYNTH_LIMIT:-32}"
QC_CELLS="${QC_CELLS:-24}"
SAVE_WAV="${SAVE_WAV:-12}"

echo "===== EEG-only FEIS run ====="
echo "APP_DIR=${APP_DIR}"
echo "TAG=${TAG}"
echo "STAGES=${STAGES}"
echo "EPOCHS=${EPOCHS}"
echo "SPLIT=${SPLIT}"

DEVICE_ARGS=()
if [[ -n "${DEVICE:-}" ]]; then
  DEVICE_ARGS=(--device "${DEVICE}")
fi

python scripts/direct_train.py \
  --config configs/direct_eeg2speech.yaml \
  --stages "${STAGES}" \
  --run-suffix "${TAG}" \
  --epochs "${EPOCHS}" \
  "${DEVICE_ARGS[@]}"

RUN_DIR="../artifacts/outputs_direct/direct_${STAGES//,/_}_${TAG}"
CKPT="${RUN_DIR}/checkpoints/best.pt"

python scripts/direct_recon_eval.py \
  --config configs/direct_eeg2speech.yaml \
  --checkpoint "${CKPT}" \
  --split "${SPLIT}" \
  --qc-cells "${QC_CELLS}" \
  --save-wav "${SAVE_WAV}" \
  "${DEVICE_ARGS[@]}"

python scripts/direct_synthesize.py \
  --config configs/direct_eeg2speech.yaml \
  --checkpoint "${CKPT}" \
  --split "${SPLIT}" \
  --limit "${SYNTH_LIMIT}" \
  "${DEVICE_ARGS[@]}"

echo "Done."
