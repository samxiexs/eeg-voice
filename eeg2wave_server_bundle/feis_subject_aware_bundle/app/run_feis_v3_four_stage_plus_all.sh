#!/usr/bin/env bash
set -euo pipefail

# Run FEIS v3 in the modes requested for the non-articulator stages:
#   1) fit + simulate each stage independently:
#      stimuli, thinking, speaking, resting
#   2) fit + simulate a combined all_non_articulators mode:
#      stimuli + thinking + speaking + resting
#
# The combined mode remains label-aware: FEISV3Dataset keeps `label_idx` per row,
# token targets remain subject-label audio variants, and the batch sampler groups
# by subject-label so labels are not merged into an undifferentiated pool.
#
# Fitting/simulation can be separated:
#   FIT=1 SIMULATE=0 ./run_feis_v3_four_stage_plus_all.sh full 50 mytag
#   FIT=0 SIMULATE=1 ./run_feis_v3_four_stage_plus_all.sh full 50 mytag

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BUNDLE_DIR"

MODE="${1:-smoke}"
EPOCHS="${2:-1}"
SUITE_TAG="${3:-$(date +%Y%m%d_%H%M%S)}"

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-configs/feis_v3_tokenized_generation.yaml}"
ALIGNER="${ALIGNER:-hybrid}"
DEVICE="${DEVICE:-cpu}"
SYNTH_SPLIT="${SYNTH_SPLIT:-subject_test}"
SYNTH_LIMIT="${SYNTH_LIMIT:-3}"
MAX_STEPS="${MAX_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
FIT="${FIT:-1}"
SIMULATE="${SIMULATE:-1}"

if [ "$MODE" = "smoke" ]; then
  MAX_STEPS="${MAX_STEPS:-2}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  EVAL_LIMIT="${EVAL_LIMIT:-16}"
  SYNTH_LIMIT="${SYNTH_LIMIT:-3}"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-$BUNDLE_DIR/../artifacts/matplotlib_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/feis-cache}"
mkdir -p "$MPLCONFIGDIR" "$BUNDLE_DIR/../artifacts/outputs_feis/logs"

SUMMARY_CSV="$BUNDLE_DIR/../artifacts/outputs_feis/feis_v3_four_stage_plus_all_${SUITE_TAG}.csv"
printf "mode,stage,allow_negative_train,run_dir,checkpoint,summary_report,wav_manifest\n" > "$SUMMARY_CSV"

echo "[suite] mode=$MODE epochs=$EPOCHS tag=$SUITE_TAG aligner=$ALIGNER device=$DEVICE"
echo "[suite] FIT=$FIT SIMULATE=$SIMULATE summary=$SUMMARY_CSV"

run_dir_for() {
  local stage="$1"
  local tag="$2"
  printf "%s/../artifacts/outputs_feis/feis_v3_tokenized_generation_%s_%s_%s" "$BUNDLE_DIR" "$stage" "$ALIGNER" "$tag"
}

fit_one() {
  local stage="$1"
  local tag="$2"
  local allow_negative_train="$3"
  local run_dir
  run_dir="$(run_dir_for "$stage" "$tag")"

  echo
  echo "===== FIT stage=$stage tag=$tag allow_negative_train=$allow_negative_train ====="
  echo "+ $PYTHON scripts/build_feis_v3_tokens.py --config $CONFIG --stage $stage"
  "$PYTHON" scripts/build_feis_v3_tokens.py --config "$CONFIG" --stage "$stage"

  CLUSTER_CMD=("$PYTHON" scripts/build_feis_v3_clusters.py --config "$CONFIG" --stage "$stage")
  if [ "$allow_negative_train" = "1" ]; then
    CLUSTER_CMD+=(--allow-negative-train)
  fi
  echo "+ ${CLUSTER_CMD[*]}"
  "${CLUSTER_CMD[@]}"

  TRAIN_CMD=(
    "$PYTHON" scripts/train_feis_v3.py
    --config "$CONFIG"
    --stage "$stage"
    --epochs "$EPOCHS"
    --run-suffix "$tag"
    --aligner "$ALIGNER"
    --device "$DEVICE"
  )
  if [ -n "$MAX_STEPS" ]; then
    TRAIN_CMD+=(--max-steps "$MAX_STEPS")
  fi
  if [ -n "$BATCH_SIZE" ]; then
    TRAIN_CMD+=(--batch-size "$BATCH_SIZE")
  fi
  if [ -n "$EVAL_LIMIT" ]; then
    TRAIN_CMD+=(--eval-limit "$EVAL_LIMIT")
  fi
  if [ "$allow_negative_train" = "1" ]; then
    TRAIN_CMD+=(--allow-negative-train)
  fi
  if [ -n "${INIT_FROM:-}" ]; then
    TRAIN_CMD+=(--init-from "$INIT_FROM")
  fi
  echo "+ ${TRAIN_CMD[*]}"
  "${TRAIN_CMD[@]}"
}

simulate_one() {
  local mode_name="$1"
  local stage="$2"
  local tag="$3"
  local allow_negative_train="$4"
  local run_dir ckpt report manifest
  run_dir="$(run_dir_for "$stage" "$tag")"
  ckpt="$run_dir/checkpoints/best.pt"
  report="$run_dir/reports/feis_v3_run_summary.md"
  manifest="$run_dir/wavs/listening_manifest.csv"

  echo
  echo "===== SIMULATE stage=$stage tag=$tag ====="
  if [ ! -f "$ckpt" ]; then
    echo "Missing checkpoint for simulation: $ckpt" >&2
    exit 1
  fi

  echo "+ $PYTHON scripts/synthesize_feis_v3.py --checkpoint $ckpt"
  "$PYTHON" scripts/synthesize_feis_v3.py \
    --config "$CONFIG" \
    --checkpoint "$ckpt" \
    --split "$SYNTH_SPLIT" \
    --out-dir "$run_dir" \
    --limit "$SYNTH_LIMIT" \
    --device "$DEVICE"

  echo "+ $PYTHON scripts/organize_feis_v3_wavs.py --run-dir $run_dir"
  "$PYTHON" scripts/organize_feis_v3_wavs.py --run-dir "$run_dir"

  echo "+ $PYTHON scripts/summarize_feis_v3_run.py --run-dir $run_dir"
  "$PYTHON" scripts/summarize_feis_v3_run.py --run-dir "$run_dir"

  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "$mode_name" "$stage" "$allow_negative_train" "$run_dir" "$ckpt" "$report" "$manifest" >> "$SUMMARY_CSV"
}

run_one() {
  local mode_name="$1"
  local stage="$2"
  local tag="$3"
  local allow_negative_train="$4"

  if [ "$FIT" = "1" ]; then
    fit_one "$stage" "$tag" "$allow_negative_train"
  fi
  if [ "$SIMULATE" = "1" ]; then
    simulate_one "$mode_name" "$stage" "$tag" "$allow_negative_train"
  fi
}

# Exclude only the shortest stage: articulators.
run_one "single_stage" "stimuli" "${SUITE_TAG}_stimuli" "0"
run_one "single_stage" "thinking" "${SUITE_TAG}_thinking" "0"
run_one "single_stage" "speaking" "${SUITE_TAG}_speaking" "0"

# Resting is explicitly allowed here because the user requested it as a training
# mode. Treat the result as diagnostic/control, not speech-generation evidence.
run_one "single_stage_control_train" "resting" "${SUITE_TAG}_resting" "1"

echo
echo "===== COMBINED LABEL-AWARE MODE ====="
echo "[combined] stage=all_non_articulators expands to stimuli thinking speaking resting."
echo "[combined] Labels remain separate through label_idx, subject-label audio keys, and the repeat-aware sampler."
run_one "combined_label_aware" "all_non_articulators" "${SUITE_TAG}_all_non_articulators" "1"

echo
echo "[done] FEIS v3 four-stage plus all suite finished."
echo "summary_csv: $SUMMARY_CSV"
