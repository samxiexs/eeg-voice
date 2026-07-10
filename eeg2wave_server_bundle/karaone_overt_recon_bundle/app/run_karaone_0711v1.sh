#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

phase="${1:-audit}"
seed="${SEED:-11}"
stage="${STAGE:-overt_like}"
config="${CONFIG:-configs/karaone_0711v1.yaml}"
run=(python scripts/train_karaone_0711v1.py --config "$config" --stage "$stage" --seed "$seed")
name="karaone_0711v1_${stage}"
out="../artifacts/outputs_karaone_0711v1"

case "$phase" in
  audit|audio_ssl|eeg_ssl)
    "${run[@]}" --phase "$phase"
    ;;
  align_global)
    : "${RESUME:?Set RESUME to the EEG-SSL checkpoint.}"
    "${run[@]}" --phase align_global --resume "$RESUME"
    ;;
  align_token)
    : "${RESUME:?Set RESUME to the global-alignment checkpoint.}"
    : "${GATE:?Set GATE to the passed global validation_gate.json.}"
    "${run[@]}" --phase align_token --resume "$RESUME" --gate "$GATE"
    ;;
  flow)
    : "${RESUME:?Set RESUME to the token-alignment checkpoint.}"
    : "${GATE:?Set GATE to the passed token validation_gate.json.}"
    "${run[@]}" --phase flow --resume "$RESUME" --gate "$GATE"
    ;;
  evaluate)
    : "${RESUME:?Set RESUME to the locked alignment checkpoint.}"
    "${run[@]}" --phase evaluate --resume "$RESUME" --allow-final-test
    ;;
  all)
    "${run[@]}" --phase audit
    "${run[@]}" --phase audio_ssl
    "${run[@]}" --phase eeg_ssl
    eeg_checkpoint="$out/${name}_eeg_ssl_s${seed}/checkpoints/best.pt"
    "${run[@]}" --phase align_global --resume "$eeg_checkpoint"
    global_gate="$out/${name}_align_global_s${seed}/metrics/validation_gate.json"
    python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$global_gate" || { echo "0711v1 stopped: global semantic gate did not pass."; exit 0; }
    global_checkpoint="$out/${name}_align_global_s${seed}/checkpoints/best.pt"
    "${run[@]}" --phase align_token --resume "$global_checkpoint" --gate "$global_gate"
    token_gate="$out/${name}_align_token_s${seed}/metrics/validation_gate.json"
    python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$token_gate" || { echo "0711v1 stopped: token semantic gate did not pass."; exit 0; }
    token_checkpoint="$out/${name}_align_token_s${seed}/checkpoints/best.pt"
    "${run[@]}" --phase flow --resume "$token_checkpoint" --gate "$token_gate"
    ;;
  *)
    echo "Unknown phase: $phase" >&2
    exit 2
    ;;
esac
