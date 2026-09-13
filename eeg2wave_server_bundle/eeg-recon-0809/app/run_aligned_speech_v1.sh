#!/usr/bin/env bash
# Run stages separately; full EEG runs require actual reviewed audio and M0 evidence.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ALIGNED_CONFIG="${ALIGNED_CONFIG:-$PROJECT_ROOT/configs/aligned_speech_v1.yaml}"
ALIGNED_OUTPUT="${ALIGNED_OUTPUT:-$PROJECT_ROOT/outputs/aligned_speech_v1}"
ALIGNED_CORPUS="${ALIGNED_CORPUS:-$PROJECT_ROOT/artifacts/aligned_speech_v1/librispeech}"
ALIGNED_DEVICE="${ALIGNED_DEVICE:-cuda}"
ALIGNED_BATCH="${ALIGNED_BATCH:-4}"
EEG_GATE_ARGS=(--audio-review "$ALIGNED_OUTPUT/audio_review.json" --m0-report "$ALIGNED_OUTPUT/m0_report.json")
if [[ "${ALIGNED_OBJECTIVE_ONLY:-0}" == "1" ]]; then
  EEG_GATE_ARGS=(--objective-only --m0-report "$ALIGNED_OUTPUT/m0_report.json")
fi
stage="${1:-readiness}"
run() { "$PYTHON_BIN" "$SCRIPT_DIR/aligned_speech.py" --config "$ALIGNED_CONFIG" "$@"; }
case "$stage" in
  readiness) run readiness ;;
  report) run report --output-root "$ALIGNED_OUTPUT" ;;
  probe) run probe --device "$ALIGNED_DEVICE" --batch-size "$ALIGNED_BATCH" --output "$ALIGNED_OUTPUT/resource_probe.json" ;;
  prepare) run prepare --materialize ;;
  cache) run cache --device "$ALIGNED_DEVICE" ;;
  corpus)
    : "${LIBRISPEECH_ROOT:?set LIBRISPEECH_ROOT to the directory containing train-clean-100 and dev-clean}"
    run corpus-manifest --corpus-root "$LIBRISPEECH_ROOT" --output "$ALIGNED_CORPUS/manifest.csv"
    run cache --device "$ALIGNED_DEVICE" --manifest "$ALIGNED_CORPUS/manifest.csv" --output "$ALIGNED_CORPUS/targets.h5"
    ;;
  audio)
    run train-audio --stage pretrain --device "$ALIGNED_DEVICE" --batch-size "$ALIGNED_BATCH" \
      --manifest "$ALIGNED_CORPUS/manifest.csv" --cache "$ALIGNED_CORPUS/targets.h5" \
      --output "$ALIGNED_OUTPUT/pretrain"
    run train-audio --stage adapt --device "$ALIGNED_DEVICE" --batch-size "$ALIGNED_BATCH" \
      --initialize "$ALIGNED_OUTPUT/pretrain/best_checkpoint.pt" --output "$ALIGNED_OUTPUT/adapt"
    run train-audio --stage mfcc --device "$ALIGNED_DEVICE" --batch-size "$ALIGNED_BATCH" --output "$ALIGNED_OUTPUT/mfcc"
    run export --kind audio --role validation --device "$ALIGNED_DEVICE" \
      --checkpoint "$ALIGNED_OUTPUT/adapt/best_checkpoint.pt" --mfcc-checkpoint "$ALIGNED_OUTPUT/mfcc/best_checkpoint.pt" \
      --output "$ALIGNED_OUTPUT/audio_listening"
    ;;
  m0)
    run train-eeg --stage align --m0 --lag-ms 0 --max-steps 5000 --device "$ALIGNED_DEVICE" --batch-size "$ALIGNED_BATCH" \
      --initialize "$ALIGNED_OUTPUT/adapt/best_checkpoint.pt" --output "$ALIGNED_OUTPUT/m0_align"
    run train-eeg --stage finetune --m0 --lag-ms 0 --max-steps 5000 --device "$ALIGNED_DEVICE" --batch-size "$ALIGNED_BATCH" \
      --initialize "$ALIGNED_OUTPUT/m0_align/best_checkpoint.pt" --output "$ALIGNED_OUTPUT/m0_finetune"
    m0_checkpoint="$ALIGNED_OUTPUT/m0_finetune/best_checkpoint.pt"
    if [[ ! -f "$m0_checkpoint" ]]; then m0_checkpoint="$ALIGNED_OUTPUT/m0_finetune/last_checkpoint.pt"; fi
    run evaluate --m0 --role train --device "$ALIGNED_DEVICE" \
      --checkpoint "$m0_checkpoint" --output "$ALIGNED_OUTPUT/m0_report.json"
    ;;
  full)
    for seed in 31 47 73; do
      candidates=()
      for lag in 0 100 200 300 400; do
        trial_root="$ALIGNED_OUTPUT/seed-$seed/lag-$lag"
        run train-eeg --stage align --seed "$seed" --lag-ms "$lag" --device "$ALIGNED_DEVICE" --batch-size "$ALIGNED_BATCH" \
          "${EEG_GATE_ARGS[@]}" \
          --initialize "$ALIGNED_OUTPUT/adapt/best_checkpoint.pt" --output "$trial_root/align"
        run train-eeg --stage finetune --seed "$seed" --lag-ms "$lag" --device "$ALIGNED_DEVICE" --batch-size "$ALIGNED_BATCH" \
          "${EEG_GATE_ARGS[@]}" \
          --initialize "$trial_root/align/best_checkpoint.pt" --output "$trial_root/finetune"
        candidate="$trial_root/finetune/best_checkpoint.pt"
        if [[ ! -f "$candidate" ]]; then candidate="$trial_root/finetune/last_checkpoint.pt"; fi
        candidates+=("$candidate")
      done
      selection="$ALIGNED_OUTPUT/seed-$seed/selection.json"
      run select-lag --device "$ALIGNED_DEVICE" --checkpoints "${candidates[@]}" --output "$selection"
      selected_checkpoint="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "$selection")"
      run evaluate --role test --device "$ALIGNED_DEVICE" --checkpoint "$selected_checkpoint" --selection "$selection" \
        --output "$ALIGNED_OUTPUT/seed-$seed/test_evaluation.json"
      run export --kind eeg --role test --device "$ALIGNED_DEVICE" --checkpoint "$selected_checkpoint" --selection "$selection" \
        --output "$ALIGNED_OUTPUT/seed-$seed/eeg_listening"
    done
    ;;
  *) echo "stage must be readiness|report|probe|prepare|cache|corpus|audio|m0|full" >&2; exit 2 ;;
esac
