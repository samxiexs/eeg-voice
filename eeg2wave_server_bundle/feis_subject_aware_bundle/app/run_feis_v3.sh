#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BUNDLE_DIR"

MODE="${1:-smoke}"
STAGE="${2:-stimuli}"
EPOCHS="${3:-1}"
TAG="${4:-$(date +%Y%m%d_%H%M%S)}"

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-configs/feis_v3_tokenized_generation.yaml}"
ALIGNER="${ALIGNER:-hybrid}"
TRAIN_PHASE="${TRAIN_PHASE:-joint}"
DEVICE="${DEVICE:-cpu}"
SYNTH_SPLIT="${SYNTH_SPLIT:-subject_test}"
SYNTH_LIMIT="${SYNTH_LIMIT:-3}"
MAX_STEPS="${MAX_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
ALLOW_NEGATIVE_TRAIN="${ALLOW_NEGATIVE_TRAIN:-0}"

if [ "$MODE" = "smoke" ]; then
  MAX_STEPS="${MAX_STEPS:-2}"
  SYNTH_LIMIT="${SYNTH_LIMIT:-3}"
  BATCH_SIZE="${BATCH_SIZE:-16}"
  EVAL_LIMIT="${EVAL_LIMIT:-32}"
fi

RUN_DIR="$BUNDLE_DIR/../artifacts/outputs_feis/feis_v3_tokenized_generation_${STAGE}_${ALIGNER}_${TAG}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$BUNDLE_DIR/../artifacts/matplotlib_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/feis-cache}"
if [ "$DEVICE" = "mps" ] || [ "$DEVICE" = "auto" ]; then
  export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
fi
mkdir -p "$MPLCONFIGDIR" "$BUNDLE_DIR/../artifacts/outputs_feis/logs"

echo "[config] mode=$MODE stage=$STAGE aligner=$ALIGNER train_phase=$TRAIN_PHASE epochs=$EPOCHS tag=$TAG device=$DEVICE"
echo "[run_dir] $RUN_DIR"

echo "+ $PYTHON scripts/build_feis_v3_tokens.py --config $CONFIG --stage $STAGE"
"$PYTHON" scripts/build_feis_v3_tokens.py --config "$CONFIG" --stage "$STAGE"

CLUSTER_CMD=("$PYTHON" scripts/build_feis_v3_clusters.py --config "$CONFIG" --stage "$STAGE")
if [ "$ALLOW_NEGATIVE_TRAIN" = "1" ]; then
  CLUSTER_CMD+=(--allow-negative-train)
fi
echo "+ ${CLUSTER_CMD[*]}"
"${CLUSTER_CMD[@]}"

TRAIN_CMD=(
  "$PYTHON" scripts/train_feis_v3.py
  --config "$CONFIG"
  --stage "$STAGE"
  --epochs "$EPOCHS"
  --run-suffix "$TAG"
  --aligner "$ALIGNER"
  --phase "$TRAIN_PHASE"
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
if [ "$ALLOW_NEGATIVE_TRAIN" = "1" ]; then
  TRAIN_CMD+=(--allow-negative-train)
fi
if [ -n "${INIT_FROM:-}" ]; then
  TRAIN_CMD+=(--init-from "$INIT_FROM")
fi

echo "+ ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

echo "+ $PYTHON scripts/synthesize_feis_v3.py --checkpoint $RUN_DIR/checkpoints/best.pt"
"$PYTHON" scripts/synthesize_feis_v3.py \
  --config "$CONFIG" \
  --checkpoint "$RUN_DIR/checkpoints/best.pt" \
  --split "$SYNTH_SPLIT" \
  --out-dir "$RUN_DIR" \
  --limit "$SYNTH_LIMIT" \
  --device "$DEVICE"

echo "+ $PYTHON scripts/organize_feis_v3_wavs.py --run-dir $RUN_DIR"
"$PYTHON" scripts/organize_feis_v3_wavs.py --run-dir "$RUN_DIR"

echo "+ $PYTHON scripts/summarize_feis_v3_run.py --run-dir $RUN_DIR"
"$PYTHON" scripts/summarize_feis_v3_run.py --run-dir "$RUN_DIR"

echo "[done] $RUN_DIR"
