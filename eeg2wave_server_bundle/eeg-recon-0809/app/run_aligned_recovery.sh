#!/usr/bin/env bash
# Recovery v3 driver.  Environment knobs:
#   ALIGNED_CONFIG    aligned config (default configs/aligned_speech_local_v1.yaml; v2 data: configs/aligned_speech_local_v2.yaml)
#   ALIGNED_RUN_ROOT  output folder under outputs/ (default aligned_recovery_v3)
#   ALIGNED_SEED      seed for m0/full (default 322); ALIGNED_SEEDS  space-separated seeds for sweep
#   ALIGNED_MIX       probability of same-sentence EEG averaging during training (default 0)
#   ALIGNED_DEVICE    auto|cpu|mps|cuda
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv-aligned-local/bin/python}"
export ALIGNED_TARGET_CACHE_NAME=targets_adapted.h5
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
SEED="${ALIGNED_SEED:-322}"
DEVICE="${ALIGNED_DEVICE:-auto}"
CONFIG="${ALIGNED_CONFIG:-$PROJECT_ROOT/configs/aligned_speech_local_v1.yaml}"
MIX="${ALIGNED_MIX:-0}"
RUN_ROOT="${ALIGNED_RUN_ROOT:-aligned_recovery_v3}"
case "$SEED" in ''|*[!0-9]*) echo 'ALIGNED_SEED must be an integer' >&2; exit 2;; esac
BASE="$PROJECT_ROOT/outputs/$RUN_ROOT"
DECODER="${ALIGNED_DECODER:-$PROJECT_ROOT/outputs/$(basename "${CONFIG%.yaml}")/adapt/best_checkpoint.pt}"
HIFIGAN="${ALIGNED_HIFIGAN:-$PROJECT_ROOT/outputs/aligned_speech_local_v1/hifigan/best}"
run() {
  echo "Recovery v3: config=$(basename "$CONFIG") seed=$SEED device=$DEVICE mix=$MIX root=$RUN_ROOT"
  "$PYTHON_BIN" app/aligned_recovery.py --config "$CONFIG" --decoder "$DECODER" --seed "$SEED" --device "$DEVICE" "$@"
}
case "${1:-m0}" in
  linear|m0|pilot|full|evaluate|sweep)
    mkdir -p logs
    bash "$0" "_$1" 2>&1 | tee -a "logs/${RUN_ROOT}_${1}_seed${SEED}.log"
    ;;
  _linear)
    # G1 gate: linear envelope tracking on validation contents; no network involved.
    "$PYTHON_BIN" app/linear_envelope_check.py --config "$CONFIG" --output "$BASE/linear_envelope_check"
    ;;
  _m0)
    # Closed-loop sanity on 50 train-fold trials: can the masked objective be fit at all?
    run --mode m0 --updates 600 --eval-every 100 --batch-size 8 --warmup 50 --output "$BASE/m0_seed$SEED"
    "$PYTHON_BIN" - "$BASE/m0_seed$SEED/best_passed.pt" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'app')
from aligned_recovery import load_checkpoint
p = Path(sys.argv[1])
if not p.exists() or not load_checkpoint(p)['evaluation']['m0_passed']:
    sys.exit('Recovery M0 has not passed; inspect metrics before full training.')
print('Recovery M0 objective gate passed.')
PY
    ;;
  _pilot)
    # Full train fold; a small validation cohort is diagnostic, never test selection.
    run --mode pilot --updates 600 --eval-every 200 --output "$BASE/pilot_seed$SEED"
    ;;
  _full)
    # Fresh initialization (an M0 memorization checkpoint is a gate, not a starting point).
    # Acoustic-dominant weights + positional code: the recipe that passed every
    # validation control on 2026-09-15 (outputs/aligned_recovery_v3/full_seed322_positional).
    run --mode full --updates 4000 --eval-every 200 \
      --sequence-weight 0 --delta-weight 0 --contrastive-weight 0.5 --mix-same-content "$MIX" \
      --m0-checkpoint "$BASE/m0_seed$SEED/best_passed.pt" \
      --output "$BASE/full_seed${SEED}_positional"
    ;;
  _evaluate)
    # Formal validation report (bootstrap CIs) and waveform export for the passing checkpoint.
    "$PYTHON_BIN" app/evaluate_aligned_recovery.py --config "$CONFIG" --device "$DEVICE" --hifigan "$HIFIGAN" \
      --checkpoint "$BASE/full_seed${SEED}_positional/best_passed.pt" --role validation \
      --output "$BASE/eval_validation_seed$SEED" --export-wavs --tail predicted
    ;;
  _sweep)
    # Replication over seeds, then a seed-level summary with t-intervals.
    dirs=()
    for seed in ${ALIGNED_SEEDS:-322 323 324}; do
      ALIGNED_SEED="$seed" bash "$0" _m0
      ALIGNED_SEED="$seed" bash "$0" _full
      dirs+=("$BASE/full_seed${seed}_positional")
    done
    "$PYTHON_BIN" app/aggregate_recovery_runs.py "${dirs[@]}" --output "$BASE/sweep_summary.json"
    ;;
  *) echo 'usage: bash app/run_aligned_recovery.sh linear|m0|pilot|full|evaluate|sweep' >&2; exit 2;;
esac
