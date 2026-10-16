#!/usr/bin/env bash
# VAE + diffusion route on DS004940 (app/vaediff*.py).  Every stage is started by hand; none starts another.
#
#   bash scripts/run_vaediff.sh ingest     download -> preprocess -> verify -> delete raw, one subject at a time
#   bash scripts/run_vaediff.sh test       unit tests + synthetic end-to-end smoke run (about 15 s)
#   bash scripts/run_vaediff.sh audio      stage 1: audio posterior + mel diffusion decoder (no EEG), ~40 min
#   bash scripts/run_vaediff.sh eeg        stage 2: EEG posterior, weights tuned every 50 updates, ~60 min
#   bash scripts/run_vaediff.sh evaluate   test sentences, seen and unseen people (needs CHECKPOINT or the run's best.pt)
#   bash scripts/run_vaediff.sh status     last tuning rounds of the stage-2 run
#
# Environment: RUN (stage-2 run name, default main), AUDIO_RUN (default audio), EXTRA (more flags for the stage).
# Logs go to logs/vaediff_<stage>_<run>.log; follow them with `tail -f`.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-../.venv-aligned-local/bin/python}"
export PYTHONPATH=app:app/src PYTHONWARNINGS=ignore PYTORCH_ENABLE_MPS_FALLBACK=1
RUN="${RUN:-main}"; AUDIO_RUN="${AUDIO_RUN:-audio}"; EXTRA="${EXTRA:-}"
mkdir -p logs

busy() {
  if pgrep -f "app/vaediff.py (audio|eeg)" >/dev/null; then
    echo "another vaediff training process is running; one heavy job at a time on this Mac" >&2; exit 1
  fi
}

case "${1:-}" in
  ingest)
    $PY app/vaediff_data.py all 2>&1 | tee -a logs/vaediff_ingest.log ;;
  test)
    $PY -m unittest tests.test_vaediff -v ;;
  audio)
    busy
    caffeinate -i $PY app/vaediff.py audio --run "$AUDIO_RUN" --minutes 40 --updates 6000 --eval-every 500 $EXTRA \
      2>&1 | tee -a "logs/vaediff_audio_${AUDIO_RUN}.log" ;;
  eeg)
    busy
    caffeinate -i $PY app/vaediff.py eeg --run "$RUN" --stage1 "outputs/vaediff/${AUDIO_RUN}/audio_best.pt" \
      --minutes 60 --updates 10000 --eval-every 500 --tune-every 50 $EXTRA 2>&1 | tee -a "logs/vaediff_eeg_${RUN}.log" ;;
  evaluate)
    CHECKPOINT="${CHECKPOINT:-outputs/vaediff/${RUN}/best.pt}"
    $PY app/vaediff.py evaluate --checkpoint "$CHECKPOINT" $EXTRA 2>&1 | tee -a "logs/vaediff_evaluate_${RUN}.log" ;;
  status)
    $PY app/vaediff.py status --run "$RUN" ;;
  *)
    sed -n '2,13p' "$0"; exit 2 ;;
esac
