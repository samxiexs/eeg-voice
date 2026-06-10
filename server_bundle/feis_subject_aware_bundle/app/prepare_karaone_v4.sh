#!/usr/bin/env bash
set -euo pipefail

# Prepare KaraOne from raw subject archives for v4 multi-dataset training.
#
# Run from the bundle app directory:
#   bash prepare_karaone_v4.sh
#
# The script calls the repo-level preprocessor with --prefer-karaone-archives.
# Each subject archive is unpacked into a temporary directory, only the files
# needed for EEG/audio alignment are extracted, and the temp directory is
# removed by the preprocessor after that subject is written to the final bundle.

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${APP_DIR}/../../.." && pwd)"

RAW_KARAONE_ROOT="${RAW_KARAONE_ROOT:-${REPO_ROOT}/data/KaraOne}"
BUNDLE_DATA_ROOT="${BUNDLE_DATA_ROOT:-${APP_DIR}/../data}"
PROCESSED_KARAONE_ROOT="${PROCESSED_KARAONE_ROOT:-${BUNDLE_DATA_ROOT}/karaone}"
CODEC_MODEL="${CODEC_MODEL:-${APP_DIR}/../models/encodec_24khz}"
TARGET_OUT="${TARGET_OUT:-${APP_DIR}/../artifacts/audio_targets/karaone_trial_encodec_latents.npz}"
TEMP_ROOT="${TEMP_ROOT:-/tmp}"
RUN_TARGETS="${RUN_TARGETS:-1}"
EXTRACT_STEPS="${EXTRACT_STEPS:-150}"
DURATION_SEC="${DURATION_SEC:-2.0}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TEMP_ROOT}/numba-cache}"

echo "===== KaraOne v4 Preparation ====="
echo "APP_DIR=${APP_DIR}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "RAW_KARAONE_ROOT=${RAW_KARAONE_ROOT}"
echo "PROCESSED_KARAONE_ROOT=${PROCESSED_KARAONE_ROOT}"
echo "CODEC_MODEL=${CODEC_MODEL}"
echo "TARGET_OUT=${TARGET_OUT}"
echo "TEMP_ROOT=${TEMP_ROOT}"
echo "RUN_TARGETS=${RUN_TARGETS}"
if [[ -n "${KARAONE_SUBJECTS:-}" ]]; then
  echo "KARAONE_SUBJECTS=${KARAONE_SUBJECTS}"
else
  echo "KARAONE_SUBJECTS=all archives found under RAW_KARAONE_ROOT"
fi

if [[ ! -d "${RAW_KARAONE_ROOT}" ]]; then
  echo "Missing RAW_KARAONE_ROOT: ${RAW_KARAONE_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/scripts/preprocess_thinking_waveform_pairs.py" ]]; then
  echo "Missing repo preprocessor: ${REPO_ROOT}/scripts/preprocess_thinking_waveform_pairs.py" >&2
  exit 1
fi

mkdir -p "${BUNDLE_DATA_ROOT}" "$(dirname "${TARGET_OUT}")"

echo
echo "===== Dependency preflight ====="
python - <<'PY'
import importlib
import sys

missing = []
for module in ["pandas", "mne", "soundfile"]:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append((module, f"{type(exc).__name__}: {exc}"))

if missing:
    print("Missing required packages for KaraOne archive preprocessing:")
    for module, error in missing:
        print(f"  - {module}: {error}")
    print()
    print("Install them in the active conda env, preferably with one of:")
    print("  conda install -c conda-forge pandas mne-base pysoundfile")
    print('  python -m pip install -i https://pypi.org/simple "pandas>=2.2,<3.0" "mne>=1.8,<2.0" "soundfile>=0.12,<1.0"')
    sys.exit(1)

print("pandas/mne/soundfile OK")
PY

echo
echo "===== Step 1/2: archive -> processed KaraOne bundle ====="
if [[ -n "${KARAONE_SUBJECTS:-}" ]]; then
  # Space-separated, for example:
  #   KARAONE_SUBJECTS="MM05 MM08" bash prepare_karaone_v4.sh
  # shellcheck disable=SC2206
  SUBJECT_LIST=(${KARAONE_SUBJECTS})
  python "${REPO_ROOT}/scripts/preprocess_thinking_waveform_pairs.py" \
    --datasets karaone \
    --karaone-root "${RAW_KARAONE_ROOT}" \
    --output-root "${BUNDLE_DATA_ROOT}" \
    --temp-root "${TEMP_ROOT}" \
    --prefer-karaone-archives \
    --karaone-subjects "${SUBJECT_LIST[@]}"
else
  python "${REPO_ROOT}/scripts/preprocess_thinking_waveform_pairs.py" \
    --datasets karaone \
    --karaone-root "${RAW_KARAONE_ROOT}" \
    --output-root "${BUNDLE_DATA_ROOT}" \
    --temp-root "${TEMP_ROOT}" \
    --prefer-karaone-archives
fi

echo
echo "Processed KaraOne files:"
python - <<PY
from pathlib import Path
root = Path("${PROCESSED_KARAONE_ROOT}")
print("root", root)
print("segments.csv", (root / "segments.csv").exists())
print("trials.csv", (root / "trials.csv").exists())
print("subjects", len(list((root / "subjects").glob("*.npz"))) if (root / "subjects").exists() else 0)
print("wavs", len(list((root / "audio").rglob("*.wav"))) if (root / "audio").exists() else 0)
PY

if [[ "${RUN_TARGETS}" != "1" ]]; then
  echo
  echo "Skipping EnCodec target extraction because RUN_TARGETS=${RUN_TARGETS}."
  exit 0
fi

if [[ ! -d "${CODEC_MODEL}" ]]; then
  echo "Missing CODEC_MODEL: ${CODEC_MODEL}" >&2
  exit 1
fi

echo
echo "===== Step 2/2: processed wavs -> EnCodec latent cache ====="
cd "${APP_DIR}"
if [[ -n "${TARGET_LIMIT:-}" ]]; then
  python scripts/v3_extract_karaone_targets.py \
    --karaone-root "${PROCESSED_KARAONE_ROOT}" \
    --codec-model "${CODEC_MODEL}" \
    --out "${TARGET_OUT}" \
    --duration-sec "${DURATION_SEC}" \
    --extract-steps "${EXTRACT_STEPS}" \
    --limit "${TARGET_LIMIT}"
else
  python scripts/v3_extract_karaone_targets.py \
    --karaone-root "${PROCESSED_KARAONE_ROOT}" \
    --codec-model "${CODEC_MODEL}" \
    --out "${TARGET_OUT}" \
    --duration-sec "${DURATION_SEC}" \
    --extract-steps "${EXTRACT_STEPS}"
fi

echo
echo "Done. Final artifacts:"
echo "  processed data: ${PROCESSED_KARAONE_ROOT}"
echo "  latent cache:   ${TARGET_OUT}"
