#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_ROOT="/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/data/processed/thinking_waveform_pairs/feis"
TARGET_ROOT="${BUNDLE_DIR}/data/feis"

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "Missing source FEIS directory: ${SOURCE_ROOT}" >&2
  exit 1
fi

if [[ -e "${TARGET_ROOT}" ]]; then
  echo "Target already exists: ${TARGET_ROOT}" >&2
  echo "Remove it manually if you want to copy again." >&2
  exit 1
fi

mkdir -p "${BUNDLE_DIR}/data"
cp -R "${SOURCE_ROOT}" "${TARGET_ROOT}"
mkdir -p "${BUNDLE_DIR}/outputs/checkpoints" "${BUNDLE_DIR}/outputs/recon_wavs" "${BUNDLE_DIR}/outputs/metrics"

echo "Bundle data prepared at: ${TARGET_ROOT}"
echo "Bundle is now ready for server upload."
