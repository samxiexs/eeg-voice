#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
CONFIG="${CONFIG:-${BUNDLE_DIR}/app/configs/combined_0715_v1.yaml}"
CACHE="${CACHE:-${BUNDLE_DIR}/artifacts/combined_0715_v1/cache/combined_0715_encodec_codes.npz}"

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
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/train_combined_0715.py" --phase audio --config "${CONFIG}" --cache "${CACHE}" "${@:2}"
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
