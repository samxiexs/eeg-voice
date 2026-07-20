#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"

"${PYTHON_BIN}" "${BUNDLE_DIR}/scripts/preprocess_combined_eeg.py" \
  --data-root "${BUNDLE_DIR}/data" \
  --output-root "${BUNDLE_DIR}/eeg_output" \
  --ds-modalities auditory "$@"
