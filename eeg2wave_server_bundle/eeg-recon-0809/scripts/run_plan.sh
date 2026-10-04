#!/usr/bin/env bash
# The plan end to end, one heavy process at a time (16 GB machine).  Finished steps are skipped.
#
#   bash scripts/run_plan.sh gates      # linear floor + stimulus-locking gate (minutes)
#   bash scripts/run_plan.sh listen     # stage 1: listening pretraining (~1-2 h)
#   bash scripts/run_plan.sh imagery    # stage 2: 4 variants x 5 subject folds (~7 h)
#   bash scripts/run_plan.sh report     # pooled tables
#   bash scripts/run_plan.sh all
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv-aligned-local/bin/python}
FOLDS=${FOLDS:-"0 1 2 3 4"}

gates() {
  [ -f outputs/baseline.json ] || $PY scripts/baseline.py
  [ -f outputs/isc_marion2021.json ] || $PY scripts/isc.py marion2021
  [ -f outputs/isc_ds004940.json ] || $PY scripts/isc.py ds004940
}

listen() {
  [ -f outputs/listen/evaluation.json ] || $PY scripts/train.py listen --out outputs/listen 2>&1 | tee logs/listen.log
}

imagery() {
  for f in $FOLDS; do
    run() { local name=$1; shift; [ -f "outputs/imagery/${name}_f$f/evaluation.json" ] ||
            $PY scripts/train.py imagery --fold "$f" --out "outputs/imagery/${name}_f$f" "$@" 2>&1 | tee "logs/imagery_${name}_f$f.log"; }
    run scratch                                         # all modalities, random init
    run pretrained --init outputs/listen/model.pt      # H1: listening pretraining helps imagery
    run imagined_only --only-target                     # H2: spoken / heard trials help imagery
    run person --set person_dim=32                      # P1: person vector from the person's own unlabelled EEG
  done
}

report() {
  $PY scripts/report.py outputs/imagery --reference scratch --json outputs/imagery/summary.json
}

case "${1:-all}" in
  gates) gates ;;
  listen) listen ;;
  imagery) imagery ;;
  report) report ;;
  all) gates; listen; imagery; report ;;
  *) echo "usage: $0 gates|listen|imagery|report|all" >&2; exit 2 ;;
esac
