#!/usr/bin/env bash
# The plan end to end, one heavy process at a time (16 GB machine).  Finished steps are skipped.
#
#   bash scripts/run_plan.sh reconstruct  # items (BCI2020, Thinking Out Loud): targets, decoders, 5 folds, report
#   bash scripts/run_plan.sh sentences    # Chisco sentences, linear encoders: targets, 5 folds of held-out runs, report
#   bash scripts/run_plan.sh deep         # Chisco sentences, the deep encoder (outputs/sentences_deep)
#   bash scripts/run_plan.sh control      # positive control: the reading epochs (sentence on screen), fold 0
#   bash scripts/run_plan.sh round        # all of it fold by fold (sentences linear, deep, items); outputs/index.html
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv-aligned-local/bin/python}
FOLDS=${FOLDS:-"0 1 2 3 4"}
RECON_DATASETS=${RECON_DATASETS:-"thinking_out_loud bci2020"}

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

sentences() {      # $1: linear | deep.  Targets need macOS `say`; fold by fold, report after each fold
  local encoder=$1 out=outputs/sentences
  [ "$encoder" = deep ] && out=outputs/sentences_deep
  [ -f artifacts/audio/chisco_sentences.h5 ] || $PY scripts/sentences.py targets 2>&1 | tee -a logs/sentences_targets.log
  for f in $FOLDS; do
    [ -f "$out/f$f/chisco/summary.json" ] && continue
    $PY scripts/sentences.py run --fold "$f" --encoder "$encoder" 2>&1 | tee "logs/sentences_${encoder}_f$f.log" \
      || echo "FAILED sentences $encoder fold $f"
    $PY scripts/sentences.py report --encoder "$encoder" || echo "FAILED sentences report"
  done
}

control() {        # the same linear pipeline on the reading epochs: how far the pipeline reaches when content is there
  [ -f outputs/sentences_read/f0/chisco/summary.json ] && return 0
  $PY scripts/sentences.py run --fold 0 --modalities read --out outputs/sentences_read 2>&1 | tee logs/sentences_read_f0.log
  $PY scripts/sentences.py report --out outputs/sentences_read
}

round() {          # one full round, fold by fold, so every pipeline has a first result early
  [ -f artifacts/audio/chisco_sentences.h5 ] || $PY scripts/sentences.py targets 2>&1 | tee -a logs/sentences_targets.log
  for f in $FOLDS; do
    FOLDS=$f sentences linear
    FOLDS=$f sentences deep
    FOLDS=$f reconstruct
    $PY scripts/overview.py || echo "FAILED overview"
  done
}

case "${1:-}" in
  reconstruct) reconstruct ;;
  sentences) sentences linear ;;
  deep) sentences deep ;;
  control) control ;;
  round) round ;;
  *) echo "usage: $0 reconstruct|sentences|deep|control|round" >&2; exit 2 ;;
esac
