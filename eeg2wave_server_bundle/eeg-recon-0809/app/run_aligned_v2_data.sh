#!/usr/bin/env bash
# v2 data route: both DS004940 tasks (Active + Passive), 22 participants
# (bad-channel limit 0.20), content roles pinned to v1.  Reuses the v1
# train-fold-adapted HuBERT teacher and HiFi-GAN (leak-free because the train
# contents are identical); only the target cache and the acoustic decoder are
# rebuilt against the v2 manifest.  Stages, in order:
#   prepare  audit + materialize EEG shards (slowest: reads every BDF twice)
#   cache    HuBERT targets for the v2 manifest with the v1 adapted teacher
#   audio    train-fold acoustic decoder for the v2 manifest
#   all      the three above
# then:  ALIGNED_CONFIG=configs/aligned_speech_local_v2.yaml ALIGNED_RUN_ROOT=aligned_recovery_v3_data_v2 \
#        bash app/run_aligned_recovery.sh linear|m0|full|evaluate|sweep
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv-aligned-local/bin/python}"
CONFIG="$PROJECT_ROOT/configs/aligned_speech_local_v2.yaml"
V1_OUTPUT="$PROJECT_ROOT/outputs/aligned_speech_local_v1"
V2_OUTPUT="$PROJECT_ROOT/outputs/aligned_speech_local_v2"
DEVICE="${ALIGNED_DEVICE:-auto}"
export ALIGNED_TARGET_CACHE_NAME=targets_adapted.h5
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR="$PROJECT_ROOT/artifacts/local_matplotlib"
mkdir -p "$MPLCONFIGDIR" logs
run() { "$PYTHON_BIN" app/aligned_speech.py --config "$CONFIG" "$@"; }
case "${1:-all}" in
  prepare|cache|audio|all)
    bash "$0" "_$1" 2>&1 | tee -a "logs/aligned_v2_data_$1.log"
    ;;
  _prepare)
    "$PYTHON_BIN" app/aligned_prepare_v2.py --config "$CONFIG" --materialize
    ;;
  _cache)
    test -f "$V1_OUTPUT/hubert/best/config.json" || { echo "missing v1 adapted HuBERT: $V1_OUTPUT/hubert/best" >&2; exit 1; }
    run cache --hubert "$V1_OUTPUT/hubert/best" --device "$DEVICE"
    ;;
  _audio)
    run train-audio --stage adapt --device "$DEVICE" --batch-size "${ALIGNED_BATCH:-1}" --output "$V2_OUTPUT/adapt"
    ;;
  _all)
    bash "$0" _prepare
    bash "$0" _cache
    bash "$0" _audio
    echo "v2 data ready. Next: ALIGNED_CONFIG=$CONFIG ALIGNED_RUN_ROOT=aligned_recovery_v3_data_v2 bash app/run_aligned_recovery.sh linear && ... m0 && ... full"
    ;;
  *) echo 'usage: bash app/run_aligned_v2_data.sh prepare|cache|audio|all' >&2; exit 2;;
esac
