#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$BUNDLE_DIR/data/ds004940}"
S3_URI="s3://openneuro.org/ds004940"

# Default: Active task + four subjects for a small end-to-end pilot.
MODE="${MODE:-active}"
SUBJECTS="${SUBJECTS:-001 002 003 004}"
INCLUDE_CODE="${INCLUDE_CODE:-0}"

command -v aws >/dev/null 2>&1 || {
  echo "ERROR: aws CLI is required. Install/configure it, then rerun." >&2
  exit 127
}

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

mkdir -p "$DATA_DIR"
echo "[ds004940] destination: $DATA_DIR"
echo "[ds004940] mode: $MODE; subjects: ${SUBJECT_LIST[*]}"
echo "[ds004940] downloading shared metadata and original stimulus WAVs"

ROOT_ARGS=(
  --exclude '*'
  --include 'README*'
  --include 'CHANGES*'
  --include 'dataset_description.json'
  --include 'participants.*'
  --include 'task-*.json'
  --include '.bidsignore'
  --include 'stimuli/**'
)
if [[ "$INCLUDE_CODE" == "1" ]]; then ROOT_ARGS+=(--include 'code/**'); fi
aws s3 sync --no-sign-request "$S3_URI" "$DATA_DIR" "${ROOT_ARGS[@]}" --only-show-errors

for subject in "${SUBJECT_LIST[@]}"; do
  echo "[ds004940] downloading sub-$subject: ${TASKS[*]}"
  SUBJECT_ARGS=(--exclude '*')
  for task in "${TASKS[@]}"; do
    # Includes the task's BDF, events.tsv and BIDS sidecars.
    SUBJECT_ARGS+=(--include "sub-${subject}/eeg/*${task}*")
  done
  aws s3 sync --no-sign-request "$S3_URI" "$DATA_DIR" "${SUBJECT_ARGS[@]}" --only-show-errors
done

mkdir -p "$BUNDLE_DIR/reports"
date -u +%Y-%m-%dT%H:%M:%SZ > "$BUNDLE_DIR/reports/download_completed_utc.txt"
echo "[ds004940] download complete"
