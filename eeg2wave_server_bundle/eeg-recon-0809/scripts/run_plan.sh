#!/usr/bin/env bash
# The plan end to end, one heavy process at a time (16 GB machine).  Finished steps are skipped.
#
#   bash scripts/run_plan.sh gates      # linear floor + stimulus-locking gate (minutes)
#   bash scripts/run_plan.sh listen     # stage 1: listening pretraining (~1-2 h)
#   bash scripts/run_plan.sh imagery    # stage 2: 4 variants x 5 subject folds (~7 h)
#   bash scripts/run_plan.sh report     # pooled tables
#   bash scripts/run_plan.sh reconstruct  # imagined speech -> speech: targets, decoders, 5 within-person folds, report
#   bash scripts/run_plan.sh all
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv-aligned-local/bin/python}
FOLDS=${FOLDS:-"0 1 2 3 4"}
IMAGERY_OUT=${IMAGERY_OUT:-outputs/imagery}          # e.g. IMAGERY_DATASETS=chisco IMAGERY_OUT=outputs/imagery_chisco
IMAGERY_DATASETS=${IMAGERY_DATASETS:-}               # empty: every dataset of configs/plan.yaml that has a store
RECON_DATASETS=${RECON_DATASETS:-"thinking_out_loud bci2020 karaone"}

gates() {
  [ -f outputs/baseline.json ] || $PY scripts/baseline.py
  [ -f outputs/isc_marion2021.json ] || $PY scripts/isc.py marion2021
  [ -f outputs/isc_ds004940.json ] || $PY scripts/isc.py ds004940
}

listen() {
  [ -f outputs/listen/evaluation.json ] || $PY scripts/train.py listen --out outputs/listen 2>&1 | tee logs/listen.log
}

imagery() {
  run() {            # one variant of one fold; a failure is logged and the queue moves on
    local name=$1 f=$2; shift 2
    [ -f "$IMAGERY_OUT/${name}_f$f/evaluation.json" ] && return 0
    $PY scripts/train.py imagery --fold "$f" --out "$IMAGERY_OUT/${name}_f$f" \
      ${IMAGERY_DATASETS:+--datasets $IMAGERY_DATASETS} "$@" 2>&1 \
      | tee "logs/$(basename "$IMAGERY_OUT")_${name}_f$f.log" || echo "FAILED imagery ${name} fold $f"
  }
  for f in $FOLDS; do
    run scratch "$f"                                       # all modalities, random init
    if [ -f outputs/listen/model.pt ]; then                # H1: listening pretraining helps imagery
      run pretrained "$f" --init outputs/listen/model.pt
    fi
    run imagined_only "$f" --only-target                   # H2: spoken / heard trials help imagery
    run person "$f" --set person_dim=32                    # P1: person vector from the person's own unlabelled EEG
  done
}

report() {
  $PY scripts/report.py "$IMAGERY_OUT" --reference scratch --json "$IMAGERY_OUT/summary.json"
}

reconstruct() {    # targets need macOS `say` (copy artifacts/audio to run elsewhere); decoders, then folds, report
  for d in $RECON_DATASETS; do
    [ -f "artifacts/audio/$d.npz" ] || $PY scripts/reconstruct.py targets --datasets "$d" 2>&1 | tee -a logs/reconstruct_targets.log
    [ -f "outputs/reconstruct/decoders/$d.pt" ] || $PY scripts/reconstruct.py decoder --datasets "$d" 2>&1 \
      | tee "logs/decoder_$d.log" || echo "FAILED decoder $d"
  done
  for f in $FOLDS; do                 # fold by fold, so every dataset has a first result early
    for d in $RECON_DATASETS; do
      [ -f "outputs/reconstruct/f$f/$d/summary.json" ] && continue
      [ -f "outputs/reconstruct/decoders/$d.pt" ] || continue
      $PY scripts/reconstruct.py run --datasets "$d" --fold "$f" 2>&1 | tee "logs/reconstruct_${d}_f$f.log" \
        || echo "FAILED reconstruct $d fold $f"
    done
    $PY scripts/reconstruct.py report || echo "FAILED report"
  done
}

case "${1:-all}" in
  gates) gates ;;
  listen) listen ;;
  imagery) imagery ;;
  report) report ;;
  reconstruct) reconstruct ;;
  all) gates; listen; imagery; report ;;
  *) echo "usage: $0 gates|listen|imagery|report|reconstruct|all" >&2; exit 2 ;;
esac
