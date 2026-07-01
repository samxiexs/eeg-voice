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
#
# KaraOne-style two-stage fitting is the default:
#   ALIGNMENT FIT -> CODEC GENERATION FIT -> SIMULATE
# Set TWO_STAGE=0 to use the older joint fit path.

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
TWO_STAGE="${TWO_STAGE:-1}"
ALIGN_EPOCHS="${ALIGN_EPOCHS:-$EPOCHS}"
CODEC_EPOCHS="${CODEC_EPOCHS:-$EPOCHS}"

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
printf "mode,stage,allow_negative_train,two_stage,alignment_run_dir,alignment_checkpoint,codec_run_dir,codec_checkpoint,summary_report,wav_manifest\n" > "$SUMMARY_CSV"

echo "[suite] mode=$MODE epochs=$EPOCHS tag=$SUITE_TAG aligner=$ALIGNER device=$DEVICE"
echo "[suite] FIT=$FIT SIMULATE=$SIMULATE TWO_STAGE=$TWO_STAGE ALIGN_EPOCHS=$ALIGN_EPOCHS CODEC_EPOCHS=$CODEC_EPOCHS summary=$SUMMARY_CSV"

DEVICE="$("$PYTHON" - "$DEVICE" <<'PY'
import sys
import torch

requested = sys.argv[1]
if requested == "auto":
    if torch.cuda.is_available():
        print("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("mps")
    else:
        print("cpu")
elif requested == "cuda" and not torch.cuda.is_available():
    raise SystemExit(
        "Requested DEVICE=cuda, but this PyTorch is not CUDA-enabled. "
        f"torch={torch.__version__} cuda_built={torch.backends.cuda.is_built()} "
        f"cuda_available={torch.cuda.is_available()}. Use DEVICE=cpu here, "
        "or run on a machine/env with a CUDA-enabled PyTorch build."
    )
elif requested == "mps" and not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
    raise SystemExit(
        "Requested DEVICE=mps, but MPS is not available in this PyTorch/runtime. Use DEVICE=cpu."
    )
else:
    print(requested)
PY
)"
if [ "$DEVICE" = "mps" ]; then
  export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
fi
echo "[suite] resolved_device=$DEVICE"

run_dir_for() {
  local stage="$1"
  local tag="$2"
  printf "%s/../artifacts/outputs_feis/feis_v3_tokenized_generation_%s_%s_%s" "$BUNDLE_DIR" "$stage" "$ALIGNER" "$tag"
}

build_caches() {
  local stage="$1"
  local allow_negative_train="$2"

  echo
  echo "===== CACHE stage=$stage allow_negative_train=$allow_negative_train ====="
  echo "+ $PYTHON scripts/build_feis_v3_tokens.py --config $CONFIG --stage $stage"
  "$PYTHON" scripts/build_feis_v3_tokens.py --config "$CONFIG" --stage "$stage"

  CLUSTER_CMD=("$PYTHON" scripts/build_feis_v3_clusters.py --config "$CONFIG" --stage "$stage")
  if [ "$allow_negative_train" = "1" ]; then
    CLUSTER_CMD+=(--allow-negative-train)
  fi
  echo "+ ${CLUSTER_CMD[*]}"
  "${CLUSTER_CMD[@]}"
}

train_phase_one() {
  local stage="$1"
  local tag="$2"
  local allow_negative_train="$3"
  local phase="$4"
  local epochs="$5"
  local init_from="${6:-}"

  echo
  if [ "$phase" = "alignment" ]; then
    echo "===== ALIGNMENT FIT stage=$stage tag=$tag ====="
  elif [ "$phase" = "codec" ]; then
    echo "===== CODEC GENERATION FIT stage=$stage tag=$tag ====="
  else
    echo "===== JOINT FIT stage=$stage tag=$tag ====="
  fi

  TRAIN_CMD=(
    "$PYTHON" scripts/train_feis_v3.py
    --config "$CONFIG"
    --stage "$stage"
    --epochs "$epochs"
    --run-suffix "$tag"
    --aligner "$ALIGNER"
    --phase "$phase"
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
  if [ -n "$init_from" ]; then
    TRAIN_CMD+=(--init-from "$init_from")
  elif [ -n "${INIT_FROM:-}" ]; then
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
  local run_dir ckpt report manifest alignment_run_dir alignment_ckpt
  run_dir="$(run_dir_for "$stage" "$tag")"
  ckpt="$run_dir/checkpoints/best.pt"
  alignment_run_dir=""
  alignment_ckpt=""
  if [ "$TWO_STAGE" = "1" ]; then
    alignment_run_dir="$(run_dir_for "$stage" "${tag/_codec/_alignment}")"
    alignment_ckpt="$alignment_run_dir/checkpoints/best.pt"
  fi
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

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$mode_name" "$stage" "$allow_negative_train" "$TWO_STAGE" "$alignment_run_dir" "$alignment_ckpt" "$run_dir" "$ckpt" "$report" "$manifest" >> "$SUMMARY_CSV"
}

run_one() {
  local mode_name="$1"
  local stage="$2"
  local tag="$3"
  local allow_negative_train="$4"
  local sim_tag="$tag"

  if [ "$FIT" = "1" ]; then
    build_caches "$stage" "$allow_negative_train"
    if [ "$TWO_STAGE" = "1" ]; then
      local align_tag="${tag}_alignment"
      local codec_tag="${tag}_codec"
      local align_run_dir align_ckpt
      align_run_dir="$(run_dir_for "$stage" "$align_tag")"
      align_ckpt="$align_run_dir/checkpoints/best.pt"
      train_phase_one "$stage" "$align_tag" "$allow_negative_train" "alignment" "$ALIGN_EPOCHS"
      train_phase_one "$stage" "$codec_tag" "$allow_negative_train" "codec" "$CODEC_EPOCHS" "$align_ckpt"
      sim_tag="$codec_tag"
    else
      train_phase_one "$stage" "$tag" "$allow_negative_train" "joint" "$EPOCHS"
    fi
  elif [ "$TWO_STAGE" = "1" ]; then
    sim_tag="${tag}_codec"
  fi
  if [ "$SIMULATE" = "1" ]; then
    simulate_one "$mode_name" "$stage" "$sim_tag" "$allow_negative_train"
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
