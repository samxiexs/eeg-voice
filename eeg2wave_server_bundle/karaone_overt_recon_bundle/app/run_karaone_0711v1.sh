#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

phase="${1:-audit}"
seed="${SEED:-11}"
stage="${STAGE:-overt_like}"
config="${CONFIG:-configs/karaone_0711v1.yaml}"
python_bin="${PYTHON:-python}"
run=("$python_bin" scripts/train_karaone_0711v1.py --config "$config" --stage "$stage" --seed "$seed")
name="karaone_0711v1_${stage}"
out="../artifacts/outputs_karaone_0711v1"

passed_gate() {
  "$python_bin" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$1"
}

preflight() {
  [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
  [[ -d ../data/karaone ]] || { echo "Missing KaraOne data: ../data/karaone" >&2; exit 2; }
  [[ -d ../../feis_subject_aware_bundle/models/hubert-base-ls960 ]] || { echo "Missing local HuBERT checkpoint." >&2; exit 2; }
  [[ -d ../models/encodec_24khz ]] || { echo "Missing local EnCodec checkpoint." >&2; exit 2; }
  "$python_bin" -c 'import torch, transformers, yaml, scipy; print(f"[0711v1] torch={torch.__version__}; device=" + ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))'
}

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
  all|full)
    preflight
    "${run[@]}" --phase audit
    "${run[@]}" --phase audio_ssl
    "${run[@]}" --phase eeg_ssl
    eeg_checkpoint="$out/${name}_eeg_ssl_s${seed}/checkpoints/best.pt"
    "${run[@]}" --phase align_global --resume "$eeg_checkpoint"
    global_gate="$out/${name}_align_global_s${seed}/metrics/validation_gate.json"
    passed_gate "$global_gate" || { echo "0711v1 stopped safely: global semantic gate did not pass; MM21 and flow were not accessed."; exit 0; }
    global_checkpoint="$out/${name}_align_global_s${seed}/checkpoints/best.pt"
    "${run[@]}" --phase align_token --resume "$global_checkpoint" --gate "$global_gate"
    token_gate="$out/${name}_align_token_s${seed}/metrics/validation_gate.json"
    passed_gate "$token_gate" || { echo "0711v1 stopped safely: token semantic gate did not pass; MM21 and flow were not accessed."; exit 0; }
    token_checkpoint="$out/${name}_align_token_s${seed}/checkpoints/best.pt"
    "${run[@]}" --phase flow --resume "$token_checkpoint" --gate "$token_gate"
    flow_checkpoint="$out/${name}_flow_s${seed}/checkpoints/last.pt"
    # This is the one authorised MM21 access in a successful full run.
    "${run[@]}" --phase evaluate --resume "$token_checkpoint" --allow-final-test
    "$python_bin" scripts/synthesize_karaone_0711v1.py \
      --config "$config" --stage "$stage" --seed "$seed" \
      --encoder "$token_checkpoint" --flow "$flow_checkpoint" --gate "$token_gate" \
      --split subject_test --allow-final-test
    echo "0711v1 full run complete: $out/${name}_flow_s${seed}"
    ;;
  *)
    echo "Unknown phase: $phase" >&2
    exit 2
    ;;
esac
