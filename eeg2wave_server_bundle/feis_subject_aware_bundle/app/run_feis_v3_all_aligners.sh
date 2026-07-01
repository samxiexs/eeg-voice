#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BUNDLE_DIR"

STAGES="${STAGES:-stimuli speaking thinking}"
ALIGNERS="${ALIGNERS:-mlp clip ctc ot perceiver hybrid}"
ALIGN_EPOCHS="${ALIGN_EPOCHS:-50}"
MODE="${MODE:-full}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

for STAGE in $STAGES; do
  for ALIGNER_NAME in $ALIGNERS; do
    echo
    echo "===== FEIS v3 stage=$STAGE aligner=$ALIGNER_NAME ====="
    ALIGNER="$ALIGNER_NAME" ./run_feis_v3.sh "$MODE" "$STAGE" "$ALIGN_EPOCHS" "${RUN_TAG}_${ALIGNER_NAME}"
  done
done
