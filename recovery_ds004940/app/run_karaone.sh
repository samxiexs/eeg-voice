#!/usr/bin/env bash
# KaraOne imagined-speech route.  Stages:
#   prepare    audit + preprocess the 14 participants into artifacts/karaone/shards (~16 s each)
#   baselines  Riemannian/LDA prompt decoding per stage with block-aware CV, permutation p-values, artifact controls
#   transfer   DS004940-pretrained encoder embeddings on KaraOne + heard->imagined cross-stage test
#   all        the three above
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv-aligned-local/bin/python}"
export ALIGNED_TARGET_CACHE_NAME=targets_adapted.h5
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
PERMUTATIONS="${KARAONE_PERMUTATIONS:-200}"
CHECKPOINT="${KARAONE_ENCODER:-$PROJECT_ROOT/outputs/aligned_recovery_v3/full_seed322_positional/best_passed.pt}"
mkdir -p logs
case "${1:-all}" in
  prepare|baselines|transfer|all)
    bash "$0" "_$1" 2>&1 | grep -v -i "warning" | tee -a "logs/karaone_$1.log"
    ;;
  _prepare)   "$PYTHON_BIN" scripts/prepare_karaone.py ;;
  _baselines) "$PYTHON_BIN" app/karaone.py baselines --permutations "$PERMUTATIONS" ;;
  _transfer)  "$PYTHON_BIN" app/karaone.py transfer --permutations "$PERMUTATIONS" --checkpoint "$CHECKPOINT" ;;
  _all)       bash "$0" _prepare; bash "$0" _baselines; bash "$0" _transfer ;;
  *) echo 'usage: bash app/run_karaone.sh prepare|baselines|transfer|all' >&2; exit 2 ;;
esac
