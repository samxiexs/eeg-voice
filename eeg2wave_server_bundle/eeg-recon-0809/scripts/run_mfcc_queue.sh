#!/usr/bin/env bash
# MFCC-80 route, one heavy process at a time (16 GB machine).  Finished steps are skipped.
#
# Stage A — the diffusion decoder in standardised MFCC-80 (app/mfcc.py; exactly invertible, so the
# audio ceiling is the log-mel one) on the existing cross-fitted v3 conditioning, and the mel decoder
# re-exported on the same 240 validation trials so both are scored with the MFCC metrics:
#   bash scripts/run_mfcc_queue.sh a
#   bash scripts/run_mfcc_queue.sh a2          # the same, v-loss weighted by coefficient standard deviation (--coefficient-weights std)
#
# Stage B — the subject-free encoder with the MFCC-80 loss and Broderick continuous speech, its two
# cross-fitting halves, the diffusion decoder on top of them, and the Marion listen->imagine probe:
#   bash scripts/run_mfcc_queue.sh data        # Broderick shards + HuBERT/mel/acoustic targets
#   bash scripts/run_mfcc_queue.sh encoder     # full encoder (selection on unseen DS004940 participants)
#   bash scripts/run_mfcc_queue.sh folds       # --content-half 0 / 1, final weights, no selection
#   bash scripts/run_mfcc_queue.sh diffusion   # MFCC-80 diffusion on the subject-free cross-fitted features
#   bash scripts/run_mfcc_queue.sh marion      # listen->imagine gate on encoder features (new and old encoder)
#   bash scripts/run_mfcc_queue.sh b           # all of stage B in order
# Knobs: THROTTLE (0.5)  SEED (322)  ENCODER_UPDATES (3000)  FOLD_UPDATES (2000)
#        DIFFUSION_FLAGS (extra generative_recovery.py train flags, e.g. "--coefficient-weights low")
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export ALIGNED_TARGET_CACHE_NAME=targets_adapted.h5 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=.venv-aligned-local/bin/python
GR=app/generative_recovery.py
OUT=outputs/generative_recovery
EXPORT="--crossfit --limit 240 --guidance 2 --steps 50"
THROTTLE="${THROTTLE:-0.5}"
SEED="${SEED:-322}"
ENCODER_UPDATES="${ENCODER_UPDATES:-3000}"
FOLD_UPDATES="${FOLD_UPDATES:-2000}"
DIFFUSION_FLAGS="${DIFFUSION_FLAGS:-}"
CONTINUOUS=artifacts/speech_continuous/broderick2018
UNIVERSAL=outputs/universal
TRUNK=outputs/broderick2018/trunk/best.pt
# The subject-free speech recipe (trial + spatial augmentation, Broderick trunk; outputs/universal/speech_seed322) ...
RECIPE=(--initialize-trunk "$TRUNK" --heldout-subjects 4
        --shift-max 10 --channel-gain 0.15 --background-mix 0.5 --background-alpha 0.2 0.8
        --mix-same-content 0.2 --mix-partners 3 --mix-start 0.8 --mix-anneal-epochs 12
        --spatial-geometry 0.5 --spatial-subset 0.5 --spatial-keep 0.25 1 --spatial-reference 0.3
        --spatial-conduction 0.3 --spatial-recolour 0.3)
# ... plus the MFCC-80 spectral target and the continuous-speech domain.
MFCC_CONTINUOUS=(--spectral-space mfcc --continuous-root "$CONTINUOUS" --continuous-weight 0.5 --continuous-batch 16
                 --continuous-mix 0.3)

busy() { pgrep -f "universal_train.py|aligned_recovery.py|generative_recovery.py (train|crossfit|export)|envelope_decoder.py" >/dev/null; }

stage_a() {
  [ -f "$OUT/cache/complete.json" ] || $PY $GR cache
  if [ ! -f "$OUT/crossfit_mfcc/last.pt" ]; then
    $PY $GR train --space mfcc --run crossfit_mfcc --crossfit --updates 8000 --eval-every 500 \
      --condition-noise 0.1 --condition-dropout 0.05
  fi
  if [ ! -f "$OUT/crossfit_mfcc/export_validation/index.json" ]; then
    $PY $GR export $EXPORT --checkpoint "$OUT/crossfit_mfcc/best.pt" --export-output "$OUT/crossfit_mfcc/export_validation"
  fi
  $PY $GR compare --export-output "$OUT/crossfit_mfcc/export_validation"
  # The mel decoder (reported 2026-09-19) on the same trials and noise seeds; its old comparison is kept.
  mel="$OUT/crossfit/export_validation"
  [ -f "$mel/comparison_2026-09-19.json" ] || cp "$mel/comparison.json" "$mel/comparison_2026-09-19.json"
  if [ ! -d "$mel/waveforms" ]; then
    $PY $GR export $EXPORT --checkpoint "$OUT/crossfit/best.pt" --export-output "$mel"
  fi
  $PY $GR compare --export-output "$mel"
}

stage_a2() {
  local run=crossfit_mfcc_std
  if [ ! -f "$OUT/$run/last.pt" ]; then
    $PY $GR train --space mfcc --coefficient-weights std --run "$run" --crossfit --updates 8000 --eval-every 500 \
      --condition-noise 0.1 --condition-dropout 0.05
  fi
  if [ ! -f "$OUT/$run/export_validation/index.json" ]; then
    $PY $GR export $EXPORT --checkpoint "$OUT/$run/best.pt" --export-output "$OUT/$run/export_validation"
  fi
  $PY $GR compare --export-output "$OUT/$run/export_validation"
}

stage_data() {
  [ -f "$CONTINUOUS/manifest.csv" ] || $PY scripts/prepare_broderick_windows.py --output "$CONTINUOUS"
  [ -f "$CONTINUOUS/targets_hubert.h5" ] || $PY scripts/cache_broderick_targets.py --output "$CONTINUOUS/targets_hubert.h5"
}

universal() {  # output-dir, flags...
  local out="$1"; shift
  $PY app/universal_train.py --seed "$SEED" --throttle "$THROTTLE" --output "$out" "${RECIPE[@]}" "${MFCC_CONTINUOUS[@]}" "$@" \
    2>&1 | tee -a "logs/$(basename "$out").log"
}

stage_encoder() {
  local out="$UNIVERSAL/mfcc_broderick_seed$SEED"
  if [ ! -f "$out/training_state.pt" ] || ! $PY -c "import torch,sys; sys.exit(0 if torch.load('$out/training_state.pt', weights_only=False)['complete'] else 1)"; then
    universal "$out" --updates "$ENCODER_UPDATES" --eval-every 250
  fi
}

stage_folds() {
  for half in 0 1; do
    local out="$UNIVERSAL/mfcc_broderick_half${half}_seed$SEED"
    if [ ! -f "$out/training_state.pt" ] || ! $PY -c "import torch,sys; sys.exit(0 if torch.load('$out/training_state.pt', weights_only=False)['complete'] else 1)"; then
      universal "$out" --content-half "$half" --crossfit-seed 322 --updates "$FOLD_UPDATES" --eval-every "$FOLD_UPDATES"
    fi
  done
}

stage_diffusion() {
  local run=universal_mfcc
  local folds=("$UNIVERSAL/mfcc_broderick_half0_seed$SEED/training_state.pt" "$UNIVERSAL/mfcc_broderick_half1_seed$SEED/training_state.pt")
  local flags=(--conditioner universal --universal-encoders "${folds[@]}")
  [ -f "$OUT/cache/complete.json" ] || $PY $GR cache
  if [ ! -f "$OUT/$run/last.pt" ]; then
    # shellcheck disable=SC2086  # DIFFUSION_FLAGS is a flag list
    $PY $GR train --space mfcc --run "$run" --crossfit "${flags[@]}" --updates 8000 --eval-every 500 \
      --condition-noise 0.1 --condition-dropout 0.05 $DIFFUSION_FLAGS
  fi
  if [ ! -f "$OUT/$run/export_validation/index.json" ]; then
    $PY $GR export $EXPORT "${flags[@]}" --checkpoint "$OUT/$run/best.pt" --export-output "$OUT/$run/export_validation"
  fi
  $PY $GR compare --export-output "$OUT/$run/export_validation"
}

stage_marion() {
  local new="$UNIVERSAL/mfcc_broderick_seed$SEED/best_metric.pt" old="$UNIVERSAL/speech_seed322/best_metric.pt"
  [ -f outputs/marion2021/imagery_gate_encoder_mfcc_broderick.json ] || \
    $PY app/marion_imagery_gate.py --features encoder --encoder "$new" --output outputs/marion2021/imagery_gate_encoder_mfcc_broderick.json
  [ -f outputs/marion2021/imagery_gate_encoder_speech.json ] || \
    $PY app/marion_imagery_gate.py --features encoder --encoder "$old" --output outputs/marion2021/imagery_gate_encoder_speech.json
}

main() {
  if busy; then echo "another training/export process is running; refusing to overlap" >&2; exit 1; fi
  mkdir -p logs
  case "${1:-}" in
    a) stage_a ;;
    a2) stage_a2 ;;
    data) stage_data ;;
    encoder) stage_encoder ;;
    folds) stage_folds ;;
    diffusion) stage_diffusion ;;
    marion) stage_marion ;;
    b) stage_data; stage_encoder; stage_folds; stage_diffusion; stage_marion ;;
    *) echo "usage: $0 a|a2|data|encoder|folds|diffusion|marion|b" >&2; exit 2 ;;
  esac
}
# Everything above is only definitions, and this line is read whole before it runs,
# so editing this file while a queue is running cannot change what the running queue does.
main "$@"; exit $?
