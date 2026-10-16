#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$BUNDLE_DIR/data/ds004940}"
MODE="${MODE:-active}"
SUBJECTS="${SUBJECTS:-001 002 003 004}"

[[ -d "$DATA_DIR" ]] || { echo "ERROR: missing $DATA_DIR; run download first." >&2; exit 1; }
[[ -f "$DATA_DIR/dataset_description.json" ]] || { echo "ERROR: missing dataset_description.json" >&2; exit 1; }
[[ -d "$DATA_DIR/stimuli" ]] || { echo "ERROR: missing stimuli directory" >&2; exit 1; }

case "$MODE" in
  active) TASKS=(N400Active) ;;
  active_passive) TASKS=(N400Active N400Passive) ;;
  *) echo "ERROR: MODE must be active or active_passive (got: $MODE)" >&2; exit 2 ;;
esac

if [[ "$SUBJECTS" == "all" ]]; then
  SUBJECT_LIST=()
  for n in $(seq 1 22); do SUBJECT_LIST+=("$(printf '%03d' "$n")"); done
else
  SUBJECT_LIST=()
  for raw_subject in $SUBJECTS; do
    [[ "$raw_subject" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid subject: $raw_subject" >&2; exit 2; }
    SUBJECT_LIST+=("$(printf '%03d' "$((10#$raw_subject))")")
  done
fi

stimuli_count=$(find "$DATA_DIR/stimuli" -type f -iname '*.wav' | wc -l | tr -d ' ')
echo "mode=$MODE"
echo "subjects=${SUBJECT_LIST[*]}"
echo "stimulus_wav_files=$stimuli_count"
(( stimuli_count > 0 )) || { echo "ERROR: no stimulus WAV files found" >&2; exit 1; }

for subject in "${SUBJECT_LIST[@]}"; do
  eeg_dir="$DATA_DIR/sub-$subject/eeg"
  for task in "${TASKS[@]}"; do
    eeg_count=$(find "$eeg_dir" -maxdepth 1 -type f -iname "*${task}*_eeg.bdf" 2>/dev/null | wc -l | tr -d ' ')
    event_count=$(find "$eeg_dir" -maxdepth 1 -type f -iname "*${task}*events.tsv" 2>/dev/null | wc -l | tr -d ' ')
    echo "sub-$subject task=$task raw_eeg_files=$eeg_count event_files=$event_count"
    (( eeg_count > 0 )) || { echo "ERROR: missing BDF for sub-$subject $task" >&2; exit 1; }
    (( event_count > 0 )) || { echo "ERROR: missing events.tsv for sub-$subject $task" >&2; exit 1; }
  done
done

echo "OK: DS004940 requested raw EEG, events and original audio are present."
