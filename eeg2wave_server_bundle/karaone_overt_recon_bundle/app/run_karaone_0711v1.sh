#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

phase="${1:-audit}"
seed="${SEED:-11}"
stage="${STAGE:-overt_like}"
config="${CONFIG:-configs/karaone_0711v1.yaml}"
python_bin="${PYTHON:-python}"
if [[ -n "${ALLOW_EXPLORATORY+x}" ]]; then
  allow_exploratory="$ALLOW_EXPLORATORY"
else
  allow_exploratory="$("$python_bin" -c 'import sys,yaml; print(str(yaml.safe_load(open(sys.argv[1])).get("run", {}).get("allow_exploratory_without_gate", False)).lower())' "$config")"
fi
run=("$python_bin" scripts/train_karaone_0711v1.py --config "$config" --stage "$stage" --seed "$seed")
name="karaone_0711v1_${stage}"
out="../artifacts/outputs_karaone_0711v1"

passed_gate() {
  "$python_bin" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$1"
}

is_true() {
  [[ "$1" == "true" || "$1" == "TRUE" || "$1" == "1" ]]
}

phase_done() {
  # Usage: phase_done <latest_metrics.json> <checkpoint> <expected_epochs>
  "$python_bin" - "$1" "$2" "$3" <<'PY'
import json
import sys
from pathlib import Path

metrics = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
expected = int(sys.argv[3])
if not metrics.is_file() or not checkpoint.is_file():
    raise SystemExit(1)
try:
    epoch = int(json.loads(metrics.read_text())["epoch"])
except (ValueError, KeyError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if epoch >= expected else 1)
PY
}

resume_training_arg() {
  local checkpoint="$1"
  if [[ -f "$checkpoint" ]]; then
    printf '%s\n' "--resume-training" "$checkpoint"
  fi
}

epochs_for() {
  "$python_bin" - "$config" "$1" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
key = sys.argv[2]
print(int(cfg[key]["epochs"]))
PY
}

exploratory_arg=()
if is_true "$allow_exploratory"; then
  exploratory_arg=(--allow-gate-bypass)
fi

preflight() {
  [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
  [[ -d ../data/karaone ]] || { echo "Missing KaraOne data: ../data/karaone" >&2; exit 2; }
  [[ -d ../../feis_subject_aware_bundle/models/hubert-base-ls960 ]] || { echo "Missing local HuBERT checkpoint." >&2; exit 2; }
  [[ -d ../models/encodec_24khz ]] || { echo "Missing local EnCodec checkpoint." >&2; exit 2; }
  "$python_bin" -c 'import torch, transformers, yaml, scipy; print(f"[0711v1] torch={torch.__version__}; device=" + ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))'
}

case "$phase" in
  audit|audio_ssl|audio_cache|eeg_ssl)
    "${run[@]}" --phase "$phase"
    ;;
  align_global)
    : "${RESUME:?Set RESUME to the EEG-SSL checkpoint.}"
    "${run[@]}" --phase align_global --resume "$RESUME"
    ;;
  align_token)
    : "${RESUME:?Set RESUME to the global-alignment checkpoint.}"
    : "${GATE:?Set GATE to the passed global validation_gate.json.}"
    "${run[@]}" --phase align_token --resume "$RESUME" --gate "$GATE" "${exploratory_arg[@]}"
    ;;
  flow)
    : "${RESUME:?Set RESUME to the token-alignment checkpoint.}"
    : "${GATE:?Set GATE to the passed token validation_gate.json.}"
    "${run[@]}" --phase flow --resume "$RESUME" --gate "$GATE" "${exploratory_arg[@]}"
    ;;
  evaluate)
    : "${RESUME:?Set RESUME to the locked alignment checkpoint.}"
    "${run[@]}" --phase evaluate --resume "$RESUME" --allow-final-test
    ;;
  all|full)
    preflight
    "${run[@]}" --phase audit
    audio_checkpoint="$out/${name}_audio_ssl_s${seed}/checkpoints/best.pt"
    audio_cache="../artifacts/karaone_0711v1/karaone_0711v1_${stage}_adapted_audio_targets_s${seed}.npz"
    if [[ -f "$audio_cache" ]]; then
      echo "[0711v1] reusing completed adapted audio cache: $audio_cache"
    elif [[ -f "$audio_checkpoint" ]]; then
      echo "[0711v1] audio_ssl checkpoint found; rebuilding only the interrupted target cache."
      "${run[@]}" --phase audio_cache --resume "$audio_checkpoint"
    else
      "${run[@]}" --phase audio_ssl
    fi
    eeg_checkpoint="$out/${name}_eeg_ssl_s${seed}/checkpoints/best.pt"
    eeg_last="$out/${name}_eeg_ssl_s${seed}/checkpoints/last.pt"
    if phase_done "$out/${name}_eeg_ssl_s${seed}/metrics/latest_metrics.json" "$eeg_checkpoint" "$(epochs_for eeg_ssl)"; then
      echo "[0711v1] reusing completed eeg_ssl checkpoint: $eeg_checkpoint"
    else
      eeg_resume=( $(resume_training_arg "$eeg_last") )
      "${run[@]}" --phase eeg_ssl "${eeg_resume[@]}"
    fi
    global_last="$out/${name}_align_global_s${seed}/checkpoints/last.pt"
    global_checkpoint="$out/${name}_align_global_s${seed}/checkpoints/best.pt"
    if phase_done "$out/${name}_align_global_s${seed}/metrics/latest_metrics.json" "$global_checkpoint" "$(epochs_for alignment)"; then
      echo "[0711v1] reusing completed align_global checkpoint: $global_checkpoint"
    else
      global_resume=( $(resume_training_arg "$global_last") )
      "${run[@]}" --phase align_global --resume "$eeg_checkpoint" "${global_resume[@]}"
    fi
    global_gate="$out/${name}_align_global_s${seed}/metrics/validation_gate.json"
    if ! passed_gate "$global_gate"; then
      if ! is_true "$allow_exploratory"; then
        echo "0711v1 stopped safely: global semantic gate did not pass; MM21 and flow were not accessed."
        exit 0
      fi
      echo "0711v1 exploratory mode: bypassing failed global gate; all-split output will be diagnostic-only."
    fi
    token_last="$out/${name}_align_token_s${seed}/checkpoints/last.pt"
    token_checkpoint="$out/${name}_align_token_s${seed}/checkpoints/best.pt"
    if phase_done "$out/${name}_align_token_s${seed}/metrics/latest_metrics.json" "$token_checkpoint" "$(epochs_for alignment)"; then
      echo "[0711v1] reusing completed align_token checkpoint: $token_checkpoint"
    else
      token_resume=( $(resume_training_arg "$token_last") )
      "${run[@]}" --phase align_token --resume "$global_checkpoint" --gate "$global_gate" "${exploratory_arg[@]}" "${token_resume[@]}"
    fi
    token_gate="$out/${name}_align_token_s${seed}/metrics/validation_gate.json"
    token_passed=true
    if ! passed_gate "$token_gate"; then
      token_passed=false
      if ! is_true "$allow_exploratory"; then
        echo "0711v1 stopped safely: token semantic gate did not pass; MM21 and flow were not accessed."
        exit 0
      fi
      echo "0711v1 exploratory mode: bypassing failed token gate; all-split output will be diagnostic-only."
    fi
    flow_checkpoint="$out/${name}_flow_s${seed}/checkpoints/last.pt"
    if phase_done "$out/${name}_flow_s${seed}/metrics/latest_metrics.json" "$flow_checkpoint" "$(epochs_for flow)"; then
      echo "[0711v1] reusing completed flow checkpoint: $flow_checkpoint"
    else
      flow_resume=( $(resume_training_arg "$flow_checkpoint") )
      "${run[@]}" --phase flow --resume "$token_checkpoint" --gate "$token_gate" "${exploratory_arg[@]}" "${flow_resume[@]}"
    fi
    if [[ "$token_passed" != true ]] || ! passed_gate "$global_gate"; then
      "$python_bin" scripts/synthesize_karaone_0711v1.py \
        --config "$config" --stage "$stage" --seed "$seed" \
        --encoder "$token_checkpoint" --flow "$flow_checkpoint" --gate "$token_gate" \
        --split all --allow-all-splits-diagnostic "${exploratory_arg[@]}"
      echo "0711v1 exploratory flow complete: all wavs/figures are diagnostic-only and not reportable as a decoding result."
      exit 0
    fi
    # This is the one authorised MM21 access in a successful full run.
    "${run[@]}" --phase evaluate --resume "$token_checkpoint" --allow-final-test
    "$python_bin" scripts/synthesize_karaone_0711v1.py \
      --config "$config" --stage "$stage" --seed "$seed" \
      --encoder "$token_checkpoint" --flow "$flow_checkpoint" --gate "$token_gate" \
      --split all --allow-all-splits-diagnostic
    echo "0711v1 full run complete: $out/${name}_flow_s${seed}"
    ;;
  *)
    echo "Unknown phase: $phase" >&2
    exit 2
    ;;
esac
