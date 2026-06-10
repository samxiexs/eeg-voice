#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SUBJECT="${SUBJECT:-01}"
HOLDOUT="${HOLDOUT:-21}"
RUN_ENCODEC="${RUN_ENCODEC:-1}"
RUN_BASELINES="${RUN_BASELINES:-0}"
RUN_WAVEFORM="${RUN_WAVEFORM:-${RUN_BASELINES}}"
RUN_POOLED="${RUN_POOLED:-${RUN_BASELINES}}"
REFRESH_TARGETS="${REFRESH_TARGETS:-0}"

ENCODEC_MODEL_DIR="../models/encodec_24khz"

step() {
  printf "\n===== %s =====\n" "$1"
}

run_python() {
  echo "+ $*"
  "$@"
}

step "Config"
echo "Working directory: ${SCRIPT_DIR}"
echo "SUBJECT=${SUBJECT}"
echo "HOLDOUT=${HOLDOUT}"
echo "RUN_WAVEFORM=${RUN_WAVEFORM}"
echo "RUN_POOLED=${RUN_POOLED}"
echo "RUN_ENCODEC=${RUN_ENCODEC}"
echo "REFRESH_TARGETS=${REFRESH_TARGETS}"

step "Extract target caches"
extract_if_needed() {
  local config="$1"
  local cache="$2"
  if [[ -f "${cache}" && "${REFRESH_TARGETS}" != "1" ]]; then
    echo "+ using cached targets ${cache}"
  else
    run_python python scripts/extract_audio_targets.py --config "${config}"
  fi
}

if [[ "${RUN_POOLED}" == "1" ]]; then
  extract_if_needed configs/alignment_ssl_local.yaml ../artifacts/audio_targets/feis_subject_templates_ssl.npz
else
  echo "+ skipping pooled HuBERT cache refresh (RUN_POOLED=0)"
fi
extract_if_needed configs/alignment_hubert_seq_local.yaml ../artifacts/audio_targets/feis_subject_templates_hubert_seq_t16.npz
if [[ "${RUN_ENCODEC}" == "1" ]]; then
  if [[ ! -d "${ENCODEC_MODEL_DIR}" ]]; then
    echo "ERROR: Missing local EnCodec model directory: ${ENCODEC_MODEL_DIR}" >&2
    echo "Download it first, for example:" >&2
    echo "  huggingface-cli download facebook/encodec_24khz --local-dir ${ENCODEC_MODEL_DIR}" >&2
    exit 1
  fi
  extract_if_needed configs/alignment_encodec_local.yaml ../artifacts/audio_targets/feis_subject_templates_encodec_latents.npz
fi

step "Analyze target spaces"
if [[ "${RUN_POOLED}" == "1" ]]; then
  run_python python scripts/analyze_alignment_space.py --config configs/alignment_ssl_local.yaml
fi
run_python python scripts/analyze_alignment_space.py --config configs/alignment_hubert_seq_local.yaml
if [[ "${RUN_ENCODEC}" == "1" ]]; then
  run_python python scripts/analyze_alignment_space.py --config configs/alignment_encodec_local.yaml
fi

if [[ "${RUN_WAVEFORM}" == "1" ]]; then
  step "A. Raw waveform baseline / Protocol G"
  run_python python scripts/train_waveform_protocol.py --config configs/waveform_protocol.yaml --protocol G
  run_python python scripts/eval_waveform_protocol.py --config configs/waveform_protocol.yaml --protocol G

  step "A. Raw waveform baseline / Protocol S"
  run_python python scripts/train_waveform_protocol.py --config configs/waveform_protocol.yaml --protocol S --subject "${SUBJECT}"
  run_python python scripts/eval_waveform_protocol.py --config configs/waveform_protocol.yaml --protocol S --subject "${SUBJECT}"

  step "A. Raw waveform baseline / Protocol U"
  run_python python scripts/train_waveform_protocol.py --config configs/waveform_protocol.yaml --protocol U --holdout-subject "${HOLDOUT}"
  run_python python scripts/eval_waveform_protocol.py --config configs/waveform_protocol.yaml --protocol U --holdout-subject "${HOLDOUT}"
fi

if [[ "${RUN_POOLED}" == "1" ]]; then
  step "Legacy pooled HuBERT / Protocol G"
  run_python python scripts/train_alignment.py --config configs/alignment_ssl_local.yaml --protocol G
  run_python python scripts/eval_alignment.py --config configs/alignment_ssl_local.yaml --protocol G --split test

  step "Legacy pooled HuBERT / Protocol S"
  run_python python scripts/train_alignment.py --config configs/alignment_ssl_local.yaml --protocol S --subject "${SUBJECT}"
  run_python python scripts/eval_alignment.py --config configs/alignment_ssl_local.yaml --protocol S --subject "${SUBJECT}" --split test

  step "Legacy pooled HuBERT / Protocol U"
  run_python python scripts/train_alignment.py --config configs/alignment_ssl_local.yaml --protocol U --holdout-subject "${HOLDOUT}"
  run_python python scripts/eval_alignment.py --config configs/alignment_ssl_local.yaml --protocol U --holdout-subject "${HOLDOUT}" --split test
fi

step "B. Sequence HuBERT / Protocol G"
run_python python scripts/train_alignment.py --config configs/alignment_hubert_seq_local.yaml --protocol G
run_python python scripts/eval_alignment.py --config configs/alignment_hubert_seq_local.yaml --protocol G --split test
run_python python scripts/audit_recon_audio.py --eval-json ../artifacts/outputs_alignment/g_thinking_none_hubert_seq_t16/metrics/test_evaluation.json --output-dir ../artifacts/outputs_alignment/g_thinking_none_hubert_seq_t16/metrics

step "B. Sequence HuBERT / Protocol S"
run_python python scripts/train_alignment.py --config configs/alignment_hubert_seq_local.yaml --protocol S --subject "${SUBJECT}"
run_python python scripts/eval_alignment.py --config configs/alignment_hubert_seq_local.yaml --protocol S --subject "${SUBJECT}" --split test
run_python python scripts/audit_recon_audio.py --eval-json "../artifacts/outputs_alignment/s_thinking_none_hubert_seq_t16_subject_${SUBJECT}/metrics/test_evaluation.json" --output-dir "../artifacts/outputs_alignment/s_thinking_none_hubert_seq_t16_subject_${SUBJECT}/metrics"

step "B. Sequence HuBERT / Protocol U"
run_python python scripts/train_alignment.py --config configs/alignment_hubert_seq_local.yaml --protocol U --holdout-subject "${HOLDOUT}"
run_python python scripts/eval_alignment.py --config configs/alignment_hubert_seq_local.yaml --protocol U --holdout-subject "${HOLDOUT}" --split test
run_python python scripts/audit_recon_audio.py --eval-json "../artifacts/outputs_alignment/u_thinking_none_hubert_seq_t16_holdout_${HOLDOUT}/metrics/test_evaluation.json" --output-dir "../artifacts/outputs_alignment/u_thinking_none_hubert_seq_t16_holdout_${HOLDOUT}/metrics"

if [[ "${RUN_ENCODEC}" == "1" ]]; then
  step "C. EnCodec latent / Protocol G"
  run_python python scripts/train_alignment.py --config configs/alignment_encodec_local.yaml --protocol G
  run_python python scripts/eval_alignment.py --config configs/alignment_encodec_local.yaml --protocol G --split test
  run_python python scripts/audit_recon_audio.py --eval-json ../artifacts/outputs_alignment/g_thinking_none_encodec_latent/metrics/test_evaluation.json --output-dir ../artifacts/outputs_alignment/g_thinking_none_encodec_latent/metrics

  step "C. EnCodec latent / Protocol S"
  run_python python scripts/train_alignment.py --config configs/alignment_encodec_local.yaml --protocol S --subject "${SUBJECT}"
  run_python python scripts/eval_alignment.py --config configs/alignment_encodec_local.yaml --protocol S --subject "${SUBJECT}" --split test
  run_python python scripts/audit_recon_audio.py --eval-json "../artifacts/outputs_alignment/s_thinking_none_encodec_latent_subject_${SUBJECT}/metrics/test_evaluation.json" --output-dir "../artifacts/outputs_alignment/s_thinking_none_encodec_latent_subject_${SUBJECT}/metrics"

  step "C. EnCodec latent / Protocol U"
  run_python python scripts/train_alignment.py --config configs/alignment_encodec_local.yaml --protocol U --holdout-subject "${HOLDOUT}"
  run_python python scripts/eval_alignment.py --config configs/alignment_encodec_local.yaml --protocol U --holdout-subject "${HOLDOUT}" --split test
  run_python python scripts/audit_recon_audio.py --eval-json "../artifacts/outputs_alignment/u_thinking_none_encodec_latent_holdout_${HOLDOUT}/metrics/test_evaluation.json" --output-dir "../artifacts/outputs_alignment/u_thinking_none_encodec_latent_holdout_${HOLDOUT}/metrics"
fi

append_if_exists() {
  local arr_name="$1"
  local flag="$2"
  local path="$3"
  if [[ -f "${path}" ]]; then
    eval "${arr_name}+=(\"\${flag}\" \"\${path}\")"
  else
    echo "+ optional artifact missing, report will omit: ${path}"
  fi
}

step "Combined report / Protocol G"
REPORT_G=(
  python scripts/report_phase2.py
  --config configs/alignment_hubert_seq_local.yaml
  --protocol G
  --split test
  --sequence-eval ../artifacts/outputs_alignment/g_thinking_none_hubert_seq_t16/metrics/test_evaluation.json
  --space-summary ../artifacts/outputs_alignment/template_space/feis_subject_templates_hubert_seq_t16/space_summary.json
  --output-path ../artifacts/outputs_alignment/g_thinking_none_hubert_seq_t16/metrics/test_phase_report_compare_all.md
)
append_if_exists REPORT_G --alignment-eval ../artifacts/outputs_alignment/g_thinking_none/metrics/test_evaluation.json
append_if_exists REPORT_G --codec-eval ../artifacts/outputs_alignment/g_thinking_none_encodec_latent/metrics/test_evaluation.json
append_if_exists REPORT_G --waveform-eval ../artifacts/outputs_waveform_protocol/g_thinking_none/metrics/test_metrics.json
run_python "${REPORT_G[@]}"

step "Combined report / Protocol S"
REPORT_S=(
  python scripts/report_phase2.py
  --config configs/alignment_hubert_seq_local.yaml
  --protocol S
  --subject "${SUBJECT}"
  --split test
  --sequence-eval "../artifacts/outputs_alignment/s_thinking_none_hubert_seq_t16_subject_${SUBJECT}/metrics/test_evaluation.json"
  --space-summary ../artifacts/outputs_alignment/template_space/feis_subject_templates_hubert_seq_t16/space_summary.json
  --output-path "../artifacts/outputs_alignment/s_thinking_none_hubert_seq_t16_subject_${SUBJECT}/metrics/test_phase_report_compare_all.md"
)
append_if_exists REPORT_S --alignment-eval "../artifacts/outputs_alignment/s_thinking_none_subject_${SUBJECT}/metrics/test_evaluation.json"
append_if_exists REPORT_S --codec-eval "../artifacts/outputs_alignment/s_thinking_none_encodec_latent_subject_${SUBJECT}/metrics/test_evaluation.json"
append_if_exists REPORT_S --waveform-eval "../artifacts/outputs_waveform_protocol/s_thinking_none_subject_${SUBJECT}/metrics/test_metrics.json"
run_python "${REPORT_S[@]}"

step "Combined report / Protocol U"
REPORT_U=(
  python scripts/report_phase2.py
  --config configs/alignment_hubert_seq_local.yaml
  --protocol U
  --holdout-subject "${HOLDOUT}"
  --split test
  --sequence-eval "../artifacts/outputs_alignment/u_thinking_none_hubert_seq_t16_holdout_${HOLDOUT}/metrics/test_evaluation.json"
  --space-summary ../artifacts/outputs_alignment/template_space/feis_subject_templates_hubert_seq_t16/space_summary.json
  --output-path "../artifacts/outputs_alignment/u_thinking_none_hubert_seq_t16_holdout_${HOLDOUT}/metrics/test_phase_report_compare_all.md"
)
append_if_exists REPORT_U --alignment-eval "../artifacts/outputs_alignment/u_thinking_none_holdout_${HOLDOUT}/metrics/test_evaluation.json"
append_if_exists REPORT_U --codec-eval "../artifacts/outputs_alignment/u_thinking_none_encodec_latent_holdout_${HOLDOUT}/metrics/test_evaluation.json"
append_if_exists REPORT_U --waveform-eval "../artifacts/outputs_waveform_protocol/u_thinking_none_holdout_${HOLDOUT}/metrics/test_metrics.json"
run_python "${REPORT_U[@]}"

step "Done"
echo "All requested FEIS Phase 3 runs finished."
