#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BUNDLE_DIR"

CONDA_ROOT="${CONDA_ROOT:-/opt/anaconda3}"
CONDA_ENV="${CONDA_ENV:-eegvoice}"
CONFIG="${CONFIG:-configs/feis_mel_align.yaml}"
STAGES="${STAGES:-stimuli thinking speaking resting}"
EPOCHS="${EPOCHS:-40}"
RUN_SUFFIX="${RUN_SUFFIX:-mel_align_v1}"
DEVICE="${DEVICE:-cpu}"
SPLIT="${SPLIT:-test_holdout}"
SYNTH_LIMIT="${SYNTH_LIMIT:-24}"
MAX_WAVEFORM_FIGS="${MAX_WAVEFORM_FIGS:-24}"
MAX_STEPS="${MAX_STEPS:-}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$BUNDLE_DIR/../artifacts/matplotlib_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/feis-cache}"
mkdir -p "$MPLCONFIGDIR"

if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
  echo "Missing conda init script: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
  exit 1
fi

conda activate "$CONDA_ENV"

python - <<'PY'
import sys
import torch
import scipy
print(f"[env] python={sys.executable}")
print(f"[env] torch={torch.__version__}")
print(f"[env] scipy={scipy.__version__}")
PY

echo "[config] stages=$STAGES"
echo "[config] epochs=$EPOCHS run_suffix=$RUN_SUFFIX device=$DEVICE split=$SPLIT"

echo "+ python scripts/build_feis_mel_targets.py --config $CONFIG"
python scripts/build_feis_mel_targets.py --config "$CONFIG"

for STAGE in $STAGES; do
  RUN_DIR="$BUNDLE_DIR/../artifacts/outputs_mel/feis_mel_${STAGE}_${RUN_SUFFIX}"
  WAV_DIR="$RUN_DIR/final_wavs_${SPLIT}"

  echo
  echo "===== Stage: $STAGE ====="
  TRAIN_CMD=(
    python scripts/feis_mel_train.py
    --config "$CONFIG"
    --stage "$STAGE"
    --epochs "$EPOCHS"
    --run-suffix "$RUN_SUFFIX"
    --device "$DEVICE"
  )
  if [ -n "$MAX_STEPS" ]; then
    TRAIN_CMD+=(--max-steps "$MAX_STEPS")
  fi
  echo "+ ${TRAIN_CMD[*]}"
  "${TRAIN_CMD[@]}"

  echo "+ python scripts/feis_mel_eval.py --checkpoint $RUN_DIR/checkpoints/best.pt"
  python scripts/feis_mel_eval.py \
    --config "$CONFIG" \
    --checkpoint "$RUN_DIR/checkpoints/best.pt" \
    --split "$SPLIT" \
    --device "$DEVICE"

  echo "+ python scripts/feis_mel_synthesize.py --checkpoint $RUN_DIR/checkpoints/best.pt"
  python scripts/feis_mel_synthesize.py \
    --config "$CONFIG" \
    --checkpoint "$RUN_DIR/checkpoints/best.pt" \
    --split "$SPLIT" \
    --out-dir "$WAV_DIR" \
    --limit "$SYNTH_LIMIT" \
    --diverse-labels \
    --device "$DEVICE"

  echo "+ python scripts/direct_make_run_figures.py --wav-dir $WAV_DIR"
  python scripts/direct_make_run_figures.py \
    --wav-dir "$WAV_DIR" \
    --max-waveforms "$MAX_WAVEFORM_FIGS"

  echo "[stage done] $STAGE"
  echo "  run_dir: $RUN_DIR"
  echo "  training_figures: $RUN_DIR/figures"
  echo "  wavs: $WAV_DIR"
  echo "  waveform_figures: $WAV_DIR/waveform_compare"
done

echo
echo "[done] all requested stages finished."
