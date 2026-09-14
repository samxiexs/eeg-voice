#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv-aligned-local/bin/python}"
export ALIGNED_TARGET_CACHE_NAME=targets_adapted.h5
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
SEED="${ALIGNED_SEED:-${ALIGNED_SEEDS:-322}}"
DEVICE="${ALIGNED_DEVICE:-auto}"
case "$SEED" in ''|*[!0-9]*) echo 'ALIGNED_SEED must be an integer' >&2; exit 2;; esac
BASE="$PROJECT_ROOT/outputs/aligned_recovery_v2"
run() {
  echo "Recovery v2: seed=$SEED, device=$DEVICE, physical batch=8"
  "$PYTHON_BIN" app/aligned_recovery.py --seed "$SEED" --device "$DEVICE" "$@"
}
case "${1:-m0}" in
  m0|pilot|full)
    mkdir -p logs
    bash "$0" "_$1" 2>&1 | tee -a "logs/aligned_recovery_${1}_seed${SEED}.log"
    ;;
  _m0)
    run --mode m0 --updates 600 --eval-every 100 --output "$BASE/m0_seed$SEED"
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
    # M0 weights contain only train-fold observations. A passing v2 gate is required.
    run --mode full --updates 10000 --eval-every 250 \
      --initialize "$BASE/m0_seed$SEED/best_passed.pt" \
      --m0-checkpoint "$BASE/m0_seed$SEED/best_passed.pt" \
      --output "$BASE/full_seed$SEED"
    ;;
  *) echo 'usage: bash app/run_aligned_recovery.sh m0|pilot|full' >&2; exit 2;;
esac
