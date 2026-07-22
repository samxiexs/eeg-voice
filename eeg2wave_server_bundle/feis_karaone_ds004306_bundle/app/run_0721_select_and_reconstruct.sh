#!/usr/bin/env bash
set -euo pipefail

# Reuse the existing EnCodec cache and fine-tuned audio model. Decode KaraOne
# validation for each EEG candidate, choose the checkpoint with the strongest
# EEG-specific reconstruction against controls, then generate all validation
# WAVs and comparison plots with the selected checkpoint. Locked test is never
# accessed.

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
CONFIG="${CONFIG:-${BUNDLE_DIR}/app/configs/combined_0715_v1.yaml}"
RUN_ROOT="${RUN_ROOT:-${BUNDLE_DIR}/artifacts/0721v1}"
SOURCE_ROOT="${SOURCE_ROOT:-${BUNDLE_DIR}/artifacts/combined_0715_v1}"
CACHE_PATH="${CACHE:-${SOURCE_ROOT}/cache/combined_0715_encodec_codes.npz}"
AUDIO_CHECKPOINT="${AUDIO_CHECKPOINT:-${SOURCE_ROOT}/audio/checkpoints/best.pt}"
ORIGINAL_GATE="${ORIGINAL_GATE:-${RUN_ROOT}/eeg/metrics/validation_gate.json}"
SELECTION_ROOT="${SELECTION_ROOT:-${RUN_ROOT}/checkpoint_selection}"
SELECTED_CHECKPOINT="${SELECTED_CHECKPOINT:-${RUN_ROOT}/eeg/checkpoints/selected.pt}"
SELECTION_REPORT="${SELECTION_REPORT:-${RUN_ROOT}/eeg/metrics/checkpoint_selection.json}"
FINAL_OUTPUT="${FINAL_OUTPUT:-${RUN_ROOT}/selected_samples}"
SELECTION_LIMIT="${SELECTION_LIMIT:--1}"
SYNTHESIS_LIMIT="${SYNTHESIS_LIMIT:--1}"
PLOT_LIMIT="${PLOT_LIMIT:-${SYNTHESIS_LIMIT}}"
INCLUDE_PERIODIC_CANDIDATES="${INCLUDE_PERIODIC_CANDIDATES:-1}"

cd "${BUNDLE_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_ROOT}/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "ERROR: missing $2: $1" >&2
    exit 2
  fi
}

require_file "${CACHE_PATH}" "EnCodec cache"
require_file "${AUDIO_CHECKPOINT}" "fine-tuned audio checkpoint"
require_file "${RUN_ROOT}/eeg/checkpoints/best.pt" "EEG proxy-best checkpoint"
require_file "${RUN_ROOT}/eeg/checkpoints/last.pt" "EEG final checkpoint"

CANDIDATE_NAMES=("proxy_best" "last")
CANDIDATE_PATHS=(
  "${RUN_ROOT}/eeg/checkpoints/best.pt"
  "${RUN_ROOT}/eeg/checkpoints/last.pt"
)

if [[ "${INCLUDE_PERIODIC_CANDIDATES}" == "1" && -d "${RUN_ROOT}/eeg/checkpoints/candidates" ]]; then
  while IFS= read -r checkpoint; do
    [[ -n "${checkpoint}" ]] || continue
    stem="$(basename "${checkpoint}" .pt)"
    CANDIDATE_NAMES+=("${stem}")
    CANDIDATE_PATHS+=("${checkpoint}")
  done < <(find "${RUN_ROOT}/eeg/checkpoints/candidates" -type f -name 'epoch_*.pt' | sort)
fi

OPTIONAL_DEVICE=()
if [[ -n "${COMBINED_DEVICE:-}" ]]; then
  OPTIONAL_DEVICE=(--device "${COMBINED_DEVICE}")
fi

TOTAL_STEPS=$((${#CANDIDATE_PATHS[@]} + 7))
CURRENT_STEP=0
CURRENT_LABEL="starting"

draw_progress() {
  local completed="$1"
  local label="$2"
  local state="$3"
  local width=32
  local filled=$((completed * width / TOTAL_STEPS))
  local empty=$((width - filled))
  local bar=""
  if (( filled > 0 )); then bar="$(printf '%*s' "${filled}" '' | tr ' ' '#')"; fi
  if (( empty > 0 )); then bar+="$(printf '%*s' "${empty}" '' | tr ' ' '.')"; fi
  printf '\r[0721-select] [%s] %d/%d %-38s %s' \
    "${bar}" "${completed}" "${TOTAL_STEPS}" "${label}" "${state}"
}

on_error() {
  local status="$?"
  printf '\n'
  draw_progress "${CURRENT_STEP}" "${CURRENT_LABEL}" "FAILED (${status})"
  printf '\nSelection/reconstruction stopped. Inspect the error above.\n' >&2
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

echo "[0721-select] audio is reused, not fine-tuned: ${AUDIO_CHECKPOINT}"
echo "[0721-select] candidate count: ${#CANDIDATE_PATHS[@]}"
echo "[0721-select] selection split: KaraOne validation only; locked test disabled"

SELECTOR_ARGS=()
for index in "${!CANDIDATE_PATHS[@]}"; do
  name="${CANDIDATE_NAMES[$index]}"
  checkpoint="${CANDIDATE_PATHS[$index]}"
  candidate_output="${SELECTION_ROOT}/${name}/samples"
  manifest="${candidate_output}/karaone/validation/synthesis_manifest.json"
  run_step "Decode candidate ${name}" \
    "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/synthesize_combined_0715.py" \
    --config "${CONFIG}" \
    --cache "${CACHE_PATH}" \
    --audio-checkpoint "${AUDIO_CHECKPOINT}" \
    --eeg-checkpoint "${checkpoint}" \
    --dataset karaone \
    --split validation \
    --validation-gate "${ORIGINAL_GATE}" \
    --allow-failed-gate \
    --limit "${SELECTION_LIMIT}" \
    --output "${candidate_output}" \
    "${OPTIONAL_DEVICE[@]}"
  SELECTOR_ARGS+=(--candidate "${name}" "${checkpoint}" "${manifest}")
done

run_step "Select decoded-validation checkpoint" \
  "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/select_combined_0721_checkpoint.py" \
  "${SELECTOR_ARGS[@]}" \
  --output-checkpoint "${SELECTED_CHECKPOINT}" \
  --output-report "${SELECTION_REPORT}"

run_step "Validate selected checkpoint" \
  "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/train_combined_0715.py" \
  --phase evaluate \
  --split validation \
  --config "${CONFIG}" \
  --cache "${CACHE_PATH}" \
  --audio-checkpoint "${AUDIO_CHECKPOINT}" \
  --eeg-checkpoint "${SELECTED_CHECKPOINT}" \
  --output-root "${RUN_ROOT}" \
  "${OPTIONAL_DEVICE[@]}"

for dataset in feis karaone ds004306; do
  run_step "Generate selected audio: ${dataset}" \
    "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/synthesize_combined_0715.py" \
    --config "${CONFIG}" \
    --cache "${CACHE_PATH}" \
    --audio-checkpoint "${AUDIO_CHECKPOINT}" \
    --eeg-checkpoint "${SELECTED_CHECKPOINT}" \
    --dataset "${dataset}" \
    --split validation \
    --validation-gate "${RUN_ROOT}/eeg/metrics/validation_gate.json" \
    --allow-failed-gate \
    --limit "${SYNTHESIS_LIMIT}" \
    --output "${FINAL_OUTPUT}" \
    "${OPTIONAL_DEVICE[@]}"
done

run_step "Audit selected reconstruction" \
  "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/audit_combined_0715_reconstruction.py" \
  --synthesis-root "${FINAL_OUTPUT}" \
  --output-root "${RUN_ROOT}"

run_step "Plot selected reconstruction pairs" \
  "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/plot_combined_0715_pairs.py" \
  --synthesis-root "${FINAL_OUTPUT}" \
  --output "${FINAL_OUTPUT}" \
  --limit "${PLOT_LIMIT}"

draw_progress "${CURRENT_STEP}" "selection + reconstruction" "done"
printf '\n\nSelected checkpoint: %s\n' "${SELECTED_CHECKPOINT}"
printf 'Selection report:    %s\n' "${SELECTION_REPORT}"
printf 'Generated audio:     %s/<dataset>/validation/\n' "${FINAL_OUTPUT}"
printf 'Locked test was not accessed. A failed selection/gate remains exploratory.\n'
