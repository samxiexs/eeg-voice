#!/usr/bin/env bash
# Subject-free EEG -> speech generation, one heavy process at a time (16 GB machine), throttled for quiet.
# No participant index anywhere: not a model input, and no participant-keyed augmentation
# (background mixing and covariance recolouring are off); participant ids only split the data
# into training people and held-out people.
#
#   bash scripts/run_mfcc_queue.sh tonight     # cache -> folds -> diffusion (~8 h at THROTTLE=1.0)
#   bash scripts/run_mfcc_queue.sh cache       # EEG / audio cache of the diffusion decoder (~4 min)
#   bash scripts/run_mfcc_queue.sh folds       # two encoders, each on one half of the training sentences
#                                              # (CLIP to HuBERT L9 + weighted MFCC-80 + envelope/onset, DS004940 + Broderick)
#   bash scripts/run_mfcc_queue.sh diffusion   # MFCC-80 diffusion (std-weighted) conditioned by the fold encoders,
#                                              # export 240 validation trials with every control, compare
# Knobs: THROTTLE (1.0 = 50 % duty cycle)  SEED (322)  FOLD_UPDATES (2000)  DIFFUSION_UPDATES (8000)
#        PERSON_DIM (32): data-derived person vector from the person's own calibration EEG (participant ids only
#        label which windows share a person in the training loss); 0 = the plain subject-free chain (2026-10-02 baseline)
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export ALIGNED_TARGET_CACHE_NAME=targets_adapted.h5 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=.venv-aligned-local/bin/python
GR=app/generative_recovery.py
OUT=outputs/generative_recovery
UNIVERSAL=outputs/universal
THROTTLE="${THROTTLE:-1.0}"
SEED="${SEED:-322}"
FOLD_UPDATES="${FOLD_UPDATES:-2000}"
DIFFUSION_UPDATES="${DIFFUSION_UPDATES:-8000}"
PERSON_DIM="${PERSON_DIM:-32}"
TAG=$([ "$PERSON_DIM" -gt 0 ] && echo "_person" || echo "")
RUN="subject_free_mfcc$TAG"
FOLDS=("$UNIVERSAL/fold0${TAG}_seed$SEED/training_state.pt" "$UNIVERSAL/fold1${TAG}_seed$SEED/training_state.pt")
PERSON=()
if [ "$PERSON_DIM" -gt 0 ]; then PERSON=(--person-dim "$PERSON_DIM" --person-windows 8 --person-weight 0.1); fi
ENCODER=(--initialize-trunk outputs/broderick2018/trunk/best.pt --heldout-subjects 4
         --shift-max 10 --channel-gain 0.15 --mix-same-content 0.2 --mix-partners 3 --mix-start 0.8 --mix-anneal-epochs 12
         --spatial-geometry 0.5 --spatial-subset 0.5 --spatial-keep 0.25 1 --spatial-reference 0.3 --spatial-conduction 0.3
         --spectral-space mfcc --continuous-root artifacts/speech_continuous/broderick2018 --continuous-weight 0.5
         --continuous-batch 16 --continuous-mix 0.3)

busy() { pgrep -f "universal_train.py|generative_recovery.py (train|export|cache)|marion_transfer.py" >/dev/null; }

finished() {  # run dir, budget: its metrics reach the budget
  [ -f "$1/metrics.json" ] && $PY -c "import json,sys; h=json.load(open('$1/metrics.json'))['history']; sys.exit(0 if h and h[-1]['update'] >= $2 else 1)"
}

stage_cache() {
  [ -f "$OUT/cache/complete.json" ] || $PY $GR cache
}

stage_folds() {  # final weights after FOLD_UPDATES (training_state.pt), never selected on validation
  for half in 0 1; do
    local out="$UNIVERSAL/fold${half}${TAG}_seed$SEED"
    finished "$out" "$FOLD_UPDATES" && continue
    $PY app/universal_train.py --seed "$SEED" --throttle "$THROTTLE" --output "$out" "${ENCODER[@]}" ${PERSON[@]+"${PERSON[@]}"} \
      --content-half "$half" --crossfit-seed 322 --updates "$FOLD_UPDATES" --eval-every "$FOLD_UPDATES" 2>&1 | tee -a "logs/fold${half}${TAG}_seed$SEED.log"
  done
}

stage_diffusion() {
  local flags=(--crossfit --universal-encoders "${FOLDS[@]}")
  if [ ! -f "$OUT/$RUN/last.pt" ]; then
    $PY $GR train --run "$RUN" "${flags[@]}" --updates "$DIFFUSION_UPDATES" --eval-every 500 --throttle "$THROTTLE" \
      --condition-noise 0.1 --condition-dropout 0.05 2>&1 | tee -a "logs/$RUN.log"
  fi
  if [ ! -f "$OUT/$RUN/export_validation/index.json" ]; then
    $PY $GR export "${flags[@]}" --checkpoint "$OUT/$RUN/best.pt" --export-output "$OUT/$RUN/export_validation" \
      --limit 240 --guidance 2 --steps 50 2>&1 | tee -a "logs/$RUN.log"
  fi
  $PY $GR compare --export-output "$OUT/$RUN/export_validation" 2>&1 | tee -a "logs/$RUN.log"
}

main() {
  if busy; then echo "another training/export process is running; refusing to overlap" >&2; exit 1; fi
  mkdir -p logs
  case "${1:-}" in
    cache) stage_cache ;;
    folds) stage_folds ;;
    diffusion) stage_diffusion ;;
    tonight) stage_cache; stage_folds; stage_diffusion ;;
    *) echo "usage: $0 tonight|cache|folds|diffusion" >&2; exit 2 ;;
  esac
}
# Everything above is only definitions, and this line is read whole before it runs,
# so editing this file while a queue is running cannot change what the running queue does.
main "$@"; exit $?
