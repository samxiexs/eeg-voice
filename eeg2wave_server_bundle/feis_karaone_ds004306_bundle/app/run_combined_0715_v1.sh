#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
CONFIG="${CONFIG:-${BUNDLE_DIR}/app/configs/combined_0715_v1.yaml}"
CACHE="${CACHE:-${BUNDLE_DIR}/artifacts/combined_0715_v1/cache/combined_0715_encodec_codes.npz}"
KARAONE_BUNDLE="${KARAONE_BUNDLE:-$(cd "${BUNDLE_DIR}/../karaone_overt_recon_bundle" && pwd)}"
KARAONE_PYTHON_BIN="${KARAONE_PYTHON_BIN:-${PYTHON_BIN}}"
KARAONE_AUDIO_EPOCHS="${KARAONE_AUDIO_EPOCHS:-60}"
KARAONE_AUDIO_CHECKPOINT="${KARAONE_AUDIO_CHECKPOINT:-${KARAONE_BUNDLE}/artifacts/outputs_karaone_0715/karaone_0715_audio_codec_s15/checkpoints/best.pt}"

ensure_audio_initialization() {
  if [[ -f "${KARAONE_AUDIO_CHECKPOINT}" ]]; then
    echo "[combined] supervised KaraOne audio initialization: ${KARAONE_AUDIO_CHECKPOINT}"
    return
  fi
  echo "[combined] supervised KaraOne audio checkpoint not found; building it first"
  echo "[combined] KaraOne bundle: ${KARAONE_BUNDLE}"
  (cd "${KARAONE_BUNDLE}/app" && \
    PYTHON="${KARAONE_PYTHON_BIN}" AUDIO_EPOCHS="${KARAONE_AUDIO_EPOCHS}" bash ./run_karaone_0715.sh prepare)
  (cd "${KARAONE_BUNDLE}/app" && \
    PYTHON="${KARAONE_PYTHON_BIN}" AUDIO_EPOCHS="${KARAONE_AUDIO_EPOCHS}" bash ./run_karaone_0715.sh audio)
  [[ -f "${KARAONE_AUDIO_CHECKPOINT}" ]] || {
    echo "KaraOne supervised audio training did not produce ${KARAONE_AUDIO_CHECKPOINT}" >&2
    exit 2
  }
}

case "${1:-}" in
  probe)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/diagnose_combined_0715_signal.py" --config "${CONFIG}" "${@:2}"
    ;;
  cache)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/build_combined_0715_audio_cache.py" --config "${CONFIG}" --output "${CACHE}" "${@:2}"
    ;;
  audit-audio)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/audit_combined_0715_audio_roundtrip.py" --config "${CONFIG}" --cache "${CACHE}" "${@:2}"
    ;;
  train-audio)
    ensure_audio_initialization
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/train_combined_0715.py" --phase audio --config "${CONFIG}" --cache "${CACHE}" --audio-init-checkpoint "${KARAONE_AUDIO_CHECKPOINT}" "${@:2}"
    ;;
  train-eeg)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/train_combined_0715.py" --phase eeg --config "${CONFIG}" --cache "${CACHE}" "${@:2}"
    ;;
  validate)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/train_combined_0715.py" --phase evaluate --split validation --config "${CONFIG}" --cache "${CACHE}" "${@:2}"
    ;;
  test)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/train_combined_0715.py" --phase evaluate --split test --config "${CONFIG}" --cache "${CACHE}" "${@:2}"
    ;;
  *)
    echo "usage: $0 {probe|cache|audit-audio|train-audio|train-eeg|validate|test} [options]" >&2
    exit 2
    ;;
esac
