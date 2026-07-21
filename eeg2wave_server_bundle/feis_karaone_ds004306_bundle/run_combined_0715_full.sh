#!/usr/bin/env bash
set -euo pipefail

# One-command EEG-to-reconstruction run for the combined FEIS/KaraOne/ds004306
# pipeline.  Existing audio cache/checkpoint artifacts are reused by default.
# The individual Python programs retain their tqdm progress bars; this wrapper
# adds a stage-level progress bar and stops immediately on the first failure.

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
export PYTHON_BIN

# Default: reuse cache/audio -> train EEG (40 epochs) -> validate -> synthesize
# all validation samples for all three datasets -> create comparison plots.
#
# Optional controls:
#   RUN_AUDIO=1       explicitly rerun combined audio fine-tuning
#   REBUILD_CACHE=1   explicitly rebuild EnCodec cache v2
#   RUN_PRECHECKS=1   rerun preprocessing QC, signal probe, and audio audit
#   REBUILD_KARAONE=1 rebuild KaraOne normalized EEG first (can be long)
#   RUN_SYNTHESIS=0   stop after EEG validation
#   SYNTHESIS_LIMIT=12 synthesize only 12 validation trials per dataset (-1=all)
#   COMBINED_DEVICE=mps|cuda|cpu pass an explicit model/codec device
#   EEG_EPOCHS=40     number of combined EEG epochs
#   KARAONE_AUDIO_CHECKPOINT=/path/to/best.pt select the supervised audio init
#   KARAONE_AUDIO_EPOCHS=60 epochs for automatic KaraOne audio pretraining
#   RETRAIN_KARAONE_AUDIO=1 force fresh KaraOne audio pretraining (with RUN_AUDIO=1)
#   PLOT_COMPARISONS=0 skip waveform pair plots after synthesis
RUN_NAME="${RUN_NAME:-0721v1}"
REBUILD_KARAONE="${REBUILD_KARAONE:-0}"
REBUILD_CACHE="${REBUILD_CACHE:-0}"
RUN_PRECHECKS="${RUN_PRECHECKS:-0}"
RUN_AUDIO="${RUN_AUDIO:-0}"
RUN_SYNTHESIS="${RUN_SYNTHESIS:-1}"
SYNTHESIS_LIMIT="${SYNTHESIS_LIMIT:--1}"
ALLOW_FAILED_GATE="${ALLOW_FAILED_GATE:-1}"
EEG_EPOCHS="${EEG_EPOCHS:-40}"
PLOT_COMPARISONS="${PLOT_COMPARISONS:-1}"
PLOT_LIMIT="${PLOT_LIMIT:-${SYNTHESIS_LIMIT}}"
export ALLOW_FAILED_GATE

SOURCE_ARTIFACT_ROOT="${BUNDLE_DIR}/artifacts/combined_0715_v1"
ARTIFACT_ROOT="${BUNDLE_DIR}/artifacts/${RUN_NAME}"
CACHE_PATH="${CACHE:-${SOURCE_ARTIFACT_ROOT}/cache/combined_0715_encodec_codes.npz}"
if [[ "${RUN_AUDIO}" == "1" ]]; then
  AUDIO_CHECKPOINT="${COMBINED_AUDIO_CHECKPOINT:-${ARTIFACT_ROOT}/audio/checkpoints/best.pt}"
else
  AUDIO_CHECKPOINT="${COMBINED_AUDIO_CHECKPOINT:-${SOURCE_ARTIFACT_ROOT}/audio/checkpoints/best.pt}"
fi
EEG_CHECKPOINT="${ARTIFACT_ROOT}/eeg/checkpoints/best.pt"
SYNTHESIS_OUTPUT="${SYNTHESIS_OUTPUT:-${ARTIFACT_ROOT}/samples}"

if [[ "${PYTHON_BIN}" == */* ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python executable not found or not executable: ${PYTHON_BIN}" >&2
    exit 2
  fi
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found on PATH: ${PYTHON_BIN}" >&2
  exit 2
fi

cd "${BUNDLE_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ARTIFACT_ROOT}/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

TOTAL_STEPS=2
if [[ "${REBUILD_KARAONE}" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
if [[ "${RUN_PRECHECKS}" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 3))
fi
if [[ "${REBUILD_CACHE}" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
if [[ "${RUN_AUDIO}" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
if [[ "${RUN_SYNTHESIS}" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 4))
  if [[ "${PLOT_COMPARISONS}" == "1" ]]; then
    TOTAL_STEPS=$((TOTAL_STEPS + 1))
  fi
fi

CURRENT_STEP=0
CURRENT_LABEL="starting"

draw_progress() {
  local completed="$1"
  local label="$2"
  local state="$3"
  local width=32
  local filled=0
  local empty=32
  if (( TOTAL_STEPS > 0 )); then
    filled=$((completed * width / TOTAL_STEPS))
    empty=$((width - filled))
  fi
  local bar=""
  if (( filled > 0 )); then
    bar="$(printf '%*s' "${filled}" '' | tr ' ' '#')"
  fi
  if (( empty > 0 )); then
    bar+="$(printf '%*s' "${empty}" '' | tr ' ' '.')"
  fi
  printf '\r[combined-0715] [%s] %d/%d %-34s %s' \
    "${bar}" "${completed}" "${TOTAL_STEPS}" "${label}" "${state}"
  if (( completed >= TOTAL_STEPS )) && [[ "${state}" == "done" ]]; then
    printf '\n'
  fi
}

on_error() {
  local status="$?"
  printf '\n'
  draw_progress "${CURRENT_STEP}" "${CURRENT_LABEL}" "FAILED (exit ${status})"
  printf '\nPipeline stopped. Inspect the last command output above.\n' >&2
  exit "${status}"
}
trap on_error ERR

run_step() {
  local label="$1"
  shift
  CURRENT_LABEL="${label}"
  draw_progress "${CURRENT_STEP}" "${label}" "running"
  printf '\n\n===== %s =====\n' "${label}"
  "$@"
  CURRENT_STEP=$((CURRENT_STEP + 1))
  draw_progress "${CURRENT_STEP}" "${label}" "done"
  printf '\n'
}

run_optional_device_step() {
  local label="$1"
  shift
  if [[ -n "${COMBINED_DEVICE:-}" ]]; then
    run_step "${label}" "$@" --device "${COMBINED_DEVICE}"
  else
    run_step "${label}" "$@"
  fi
}

require_nonempty_file() {
  local path="$1"
  local description="$2"
  if [[ ! -s "${path}" ]]; then
    echo "ERROR: missing ${description}: ${path}" >&2
    return 2
  fi
  echo "[combined-0715] reuse ${description}: ${path}"
}

printf '[combined-0715] run=%s; output=%s\n' "${RUN_NAME}" "${ARTIFACT_ROOT}"
printf '[combined-0715] mode: cache=%s; audio=%s; EEG epochs=%s; synthesis=%s\n' \
  "$([[ "${REBUILD_CACHE}" == "1" ]] && printf rebuild || printf reuse)" \
  "$([[ "${RUN_AUDIO}" == "1" ]] && printf retrain || printf reuse)" \
  "${EEG_EPOCHS}" \
  "$([[ "${RUN_SYNTHESIS}" == "1" ]] && printf validation-all-datasets || printf disabled)"

if [[ "${REBUILD_KARAONE}" == "1" ]]; then
  run_step "KaraOne valid-length preprocessing" \
    bash "${BUNDLE_DIR}/run_preprocess.sh" --datasets karaone --overwrite
fi

if [[ "${RUN_PRECHECKS}" == "1" ]]; then
  run_step "Preprocessing QC / verify-only" \
    bash "${BUNDLE_DIR}/run_preprocess.sh" --verify-only

  run_step "Leakage-safe EEG signal probe" \
    bash "${BUNDLE_DIR}/app/run_combined_0715_v1.sh" probe
fi

if [[ "${REBUILD_CACHE}" == "1" ]]; then
  run_step "Build EnCodec cache v2" \
    bash "${BUNDLE_DIR}/app/run_combined_0715_v1.sh" cache --rebuild
fi

require_nonempty_file "${CACHE_PATH}" "EnCodec cache v2"

if [[ "${RUN_PRECHECKS}" == "1" ]]; then
  run_optional_device_step "Validation-only EnCodec round-trip audit" \
    bash "${BUNDLE_DIR}/app/run_combined_0715_v1.sh" audit-audio
fi

if [[ "${RUN_AUDIO}" == "1" ]]; then
  run_optional_device_step "Supervised KaraOne init + combined audio fine-tuning" \
    bash "${BUNDLE_DIR}/app/run_combined_0715_v1.sh" train-audio \
    --output-root "${ARTIFACT_ROOT}"
fi

require_nonempty_file "${AUDIO_CHECKPOINT}" "combined audio checkpoint"

run_optional_device_step "Train combined EEG model" \
  bash "${BUNDLE_DIR}/app/run_combined_0715_v1.sh" train-eeg \
  --epochs "${EEG_EPOCHS}" \
  --audio-checkpoint "${AUDIO_CHECKPOINT}" \
  --output-root "${ARTIFACT_ROOT}"

require_nonempty_file "${EEG_CHECKPOINT}" "combined EEG checkpoint"

run_optional_device_step "Run validation evaluation" \
  bash "${BUNDLE_DIR}/app/run_combined_0715_v1.sh" validate \
  --audio-checkpoint "${AUDIO_CHECKPOINT}" \
  --eeg-checkpoint "${EEG_CHECKPOINT}" \
  --output-root "${ARTIFACT_ROOT}"

if [[ "${RUN_SYNTHESIS}" == "1" ]]; then
  SYNTHESIS_GATE_ARGS=()
  if [[ "${ALLOW_FAILED_GATE}" == "1" ]]; then
    SYNTHESIS_GATE_ARGS+=(--allow-failed-gate)
  fi

  for DATASET in feis karaone ds004306; do
    run_optional_device_step "Validation synthesis: ${DATASET}" \
      "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/synthesize_combined_0715.py" \
      --cache "${CACHE_PATH}" \
      --audio-checkpoint "${AUDIO_CHECKPOINT}" \
      --eeg-checkpoint "${EEG_CHECKPOINT}" \
      --dataset "${DATASET}" \
      --split validation \
      --validation-gate "${ARTIFACT_ROOT}/eeg/metrics/validation_gate.json" \
      --limit "${SYNTHESIS_LIMIT}" \
      --output "${SYNTHESIS_OUTPUT}" \
      "${SYNTHESIS_GATE_ARGS[@]}"
  done

  run_step "Audit structural + EEG-specific reconstruction" \
    bash "${BUNDLE_DIR}/app/run_combined_0715_v1.sh" audit-reconstruction \
    --synthesis-root "${SYNTHESIS_OUTPUT}" \
    --output-root "${ARTIFACT_ROOT}"

  if [[ "${PLOT_COMPARISONS}" == "1" ]]; then
    run_step "Generate reference/reconstruction pair plots" \
      "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/plot_combined_0715_pairs.py" \
      --synthesis-root "${SYNTHESIS_OUTPUT}" \
      --output "${SYNTHESIS_OUTPUT}" \
      --limit "${PLOT_LIMIT}"
  fi
fi

draw_progress "${CURRENT_STEP}" "pipeline" "done"
printf '\nCombined 0715 pipeline completed successfully.\n'
printf 'Locked test was not run; review validation metrics/gate before using --allow-final-test.\n'
