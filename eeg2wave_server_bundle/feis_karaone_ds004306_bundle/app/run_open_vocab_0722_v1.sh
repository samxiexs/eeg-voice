#!/usr/bin/env bash
# The optional argument lists below are intentionally empty in the default
# G1/project-only run. Bash 3.2 (the macOS system shell) treats an empty array
# expansion as an unbound variable under `set -u`, so retain fail-fast command
# handling without nounset for this CLI forwarding wrapper.
set -eo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
PROJECT_ONLY="${PROJECT_ONLY:-0}"
if [[ -z "${CONFIG:-}" ]]; then
  if [[ "${PROJECT_ONLY}" == "1" ]]; then
    CONFIG="${BUNDLE_DIR}/app/configs/open_vocab_0722_project_hubert_v1.yaml"
  else
    CONFIG="${BUNDLE_DIR}/app/configs/open_vocab_0722_v1.yaml"
  fi
fi
DEVICE_ARGS=()
[[ -n "${DEVICE:-}" ]] && DEVICE_ARGS=(--device "${DEVICE}")
GENERALIZATION="${GENERALIZATION:-g1}"
HOLDOUT_ARGS=()
[[ -n "${HOLDOUT_LABEL:-}" ]] && HOLDOUT_ARGS=(--holdout-label "${HOLDOUT_LABEL}")
PROJECT_ONLY_ARGS=()
[[ "${PROJECT_ONLY}" == "1" ]] && PROJECT_ONLY_ARGS=(--project-audio-only)
SHARED_INIT_ARGS=()
[[ -n "${SHARED_AUDIO_INIT:-}" ]] && SHARED_INIT_ARGS=(--shared-init-checkpoint "${SHARED_AUDIO_INIT}")
COMPUTE_XLSR="${COMPUTE_XLSR:-1}"
XLSR_ARGS=()
[[ "${COMPUTE_XLSR}" == "1" ]] && XLSR_ARGS=(--compute-xlsr)
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ "${PROJECT_ONLY}" == "1" ]]; then
    OUTPUT_ROOT="${BUNDLE_DIR}/artifacts/open_vocab_0722_project_hubert_v1"
  else
    OUTPUT_ROOT="${BUNDLE_DIR}/artifacts/open_vocab_0722_v1"
  fi
fi

bar() {
  local current="$1" total="$2" label="$3" width=32 filled empty
  filled=$((current * width / total)); empty=$((width - filled))
  printf '[openvoice-0722] ['
  printf '%*s' "${filled}" '' | tr ' ' '#'
  printf '%*s' "${empty}" '' | tr ' ' '.'
  printf '] %d/%d %s\n' "${current}" "${total}" "${label}"
}

run_synthesis() {
  local dataset="$1" split="${2:-validation}"
  "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/synthesize_open_vocab_0722.py" \
    --config "${CONFIG}" --dataset "${dataset}" --split "${split}" \
    --generalization "${GENERALIZATION}" "${HOLDOUT_ARGS[@]}" "${XLSR_ARGS[@]}" "${PROJECT_ONLY_ARGS[@]}" "${DEVICE_ARGS[@]}"
}

case "${1:-}" in
  prepare)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/prepare_open_vocab_0722.py" --config "${CONFIG}" "${@:2}"
    ;;
  public-manifest)
    : "${LIBRITTS_ROOT:?Set LIBRITTS_ROOT to the extracted LibriTTS clean root}"
    : "${AISHELL_ROOT:?Set AISHELL_ROOT to the extracted AISHELL-1 wav root}"
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/build_open_vocab_public_manifest.py" --config "${CONFIG}" --english-root "${LIBRITTS_ROOT}" --chinese-root "${AISHELL_ROOT}" "${@:2}"
    ;;
  public-cache)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/build_open_vocab_public_cache.py" --config "${CONFIG}" "${DEVICE_ARGS[@]}" "${@:2}"
    ;;
  teachers)
    args=(--config "${CONFIG}" "${DEVICE_ARGS[@]}")
    [[ "${PROJECT_ONLY}" == "1" ]] && args+=(--project-only)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/build_open_vocab_teacher_cache.py" "${args[@]}" "${@:2}"
    ;;
  train-audio|pretrain-eeg|train-eeg|validate|test)
    phase="${1}"
    args=(--config "${CONFIG}" --generalization "${GENERALIZATION}" "${HOLDOUT_ARGS[@]}" "${DEVICE_ARGS[@]}")
    [[ "${PROJECT_ONLY}" == "1" ]] && args+=(--project-audio-only)
    case "${phase}" in
      train-audio) args+=(--phase audio "${SHARED_INIT_ARGS[@]}") ;;
      pretrain-eeg) args+=(--phase eeg-pretrain) ;;
      train-eeg) args+=(--phase eeg) ;;
      validate) args+=(--phase evaluate --split validation) ;;
      test) args+=(--phase evaluate --split test --allow-final-test) ;;
    esac
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/train_open_vocab_0722.py" "${args[@]}" "${@:2}"
    ;;
  select-eeg)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/select_open_vocab_0722_checkpoint.py" --config "${CONFIG}" --generalization "${GENERALIZATION}" "${HOLDOUT_ARGS[@]}" "${PROJECT_ONLY_ARGS[@]}" "${DEVICE_ARGS[@]}" "${@:2}"
    ;;
  synthesize)
    dataset="${2:?usage: $0 synthesize {karaone|feis|ds004306} [validation|test] [options]}"
    split="${3:-validation}"
    shift $(( $# >= 3 ? 3 : 2 ))
    if [[ "${split}" == "test" ]]; then
      exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/synthesize_open_vocab_0722.py" --config "${CONFIG}" --dataset "${dataset}" --split test --allow-final-test --generalization "${GENERALIZATION}" "${HOLDOUT_ARGS[@]}" "${XLSR_ARGS[@]}" "${PROJECT_ONLY_ARGS[@]}" "${DEVICE_ARGS[@]}" "$@"
    fi
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/synthesize_open_vocab_0722.py" --config "${CONFIG}" --dataset "${dataset}" --split validation --generalization "${GENERALIZATION}" "${HOLDOUT_ARGS[@]}" "${XLSR_ARGS[@]}" "${PROJECT_ONLY_ARGS[@]}" "${DEVICE_ARGS[@]}" "$@"
    ;;
  audit-model)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/audit_open_vocab_0722_model.py" --config "${CONFIG}" "${PROJECT_ONLY_ARGS[@]}" "${DEVICE_ARGS[@]}" "${@:2}"
    ;;
  gate)
    manifest="${SYNTHESIS_MANIFEST:-${OUTPUT_ROOT}/synthesis/${GENERALIZATION}/${HOLDOUT_LABEL:-all}/karaone/validation/synthesis_manifest.json}"
    args=(--config "${CONFIG}" --synthesis-manifest "${manifest}" --generalization "${GENERALIZATION}")
    args+=("${PROJECT_ONLY_ARGS[@]}")
    [[ -n "${DENSE_BASELINE_REPORT:-}" ]] && args+=(--dense-baseline-report "${DENSE_BASELINE_REPORT}")
    [[ -n "${SEED_SUMMARY:-}" ]] && args+=(--seed-summary "${SEED_SUMMARY}")
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/gate_open_vocab_0722.py" "${args[@]}" "${@:2}"
    ;;
  plot)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/plot_open_vocab_0722_pairs.py" --manifest "${2:?synthesis_manifest.json is required}" "${@:3}"
    ;;
  track-b)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/preprocess_open_vocab_0722_track_b.py" --config "${CONFIG}" "${@:2}"
    ;;
  ablation-config)
    stage="${2:?stage is required}"
    output="${3:-${OUTPUT_ROOT}/ablation_configs/${stage}.yaml}"
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/make_open_vocab_0722_ablation.py" --config "${CONFIG}" --stage "${stage}" --output "${output}"
    ;;
  seed-config)
    seed="${2:?seed must be 15, 31 or 47}"
    output="${3:-${OUTPUT_ROOT}/seed_configs/seed${seed}.yaml}"
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/make_open_vocab_0722_seed_config.py" --config "${CONFIG}" --seed "${seed}" --output "${output}"
    ;;
  aggregate-seeds)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/aggregate_open_vocab_0722_seeds.py" "${@:2}"
    ;;
  compare-dense-moe)
    exec "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/compare_open_vocab_0722_dense_moe.py" "${@:2}"
    ;;
  all)
    total=13
    bar 0 "${total}" "prepare + leakage audit"
    "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/prepare_open_vocab_0722.py" --config "${CONFIG}"
    bar 1 "${total}" "public speech manifest"
    if [[ "${PROJECT_ONLY}" != "1" && ! -f "${OUTPUT_ROOT}/manifests/public_audio_manifest.csv" ]]; then
      : "${LIBRITTS_ROOT:?Set LIBRITTS_ROOT}"
      : "${AISHELL_ROOT:?Set AISHELL_ROOT}"
      "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/build_open_vocab_public_manifest.py" --config "${CONFIG}" --english-root "${LIBRITTS_ROOT}" --chinese-root "${AISHELL_ROOT}"
    fi
    bar 2 "${total}" "frozen EnCodec public cache"
    if [[ "${PROJECT_ONLY}" != "1" ]]; then "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/build_open_vocab_public_cache.py" --config "${CONFIG}" "${DEVICE_ARGS[@]}"; fi
    bar 3 "${total}" "frozen XLS-R/XLM-R teacher cache"
    teacher_args=(--config "${CONFIG}" "${DEVICE_ARGS[@]}"); [[ "${PROJECT_ONLY}" == "1" ]] && teacher_args+=(--project-only)
    "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/build_open_vocab_teacher_cache.py" "${teacher_args[@]}"
    bar 4 "${total}" "label-free audio prior"
    "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/train_open_vocab_0722.py" --config "${CONFIG}" --phase audio "${PROJECT_ONLY_ARGS[@]}" "${SHARED_INIT_ARGS[@]}" "${DEVICE_ARGS[@]}"
    bar 5 "${total}" "EEG self-supervised pretraining"
    "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/train_open_vocab_0722.py" --config "${CONFIG}" --phase eeg-pretrain --generalization "${GENERALIZATION}" "${HOLDOUT_ARGS[@]}" "${PROJECT_ONLY_ARGS[@]}" "${DEVICE_ARGS[@]}"
    bar 6 "${total}" "paired EEG-audio training"
    "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/train_open_vocab_0722.py" --config "${CONFIG}" --phase eeg --generalization "${GENERALIZATION}" "${HOLDOUT_ARGS[@]}" "${PROJECT_ONLY_ARGS[@]}" "${DEVICE_ARGS[@]}"
    bar 7 "${total}" "decoded validation checkpoint selection"
    "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/select_open_vocab_0722_checkpoint.py" --config "${CONFIG}" --generalization "${GENERALIZATION}" "${HOLDOUT_ARGS[@]}" "${PROJECT_ONLY_ARGS[@]}" "${DEVICE_ARGS[@]}"
    bar 8 "${total}" "ten-mode validation synthesis"
    for dataset in karaone feis ds004306; do run_synthesis "${dataset}" validation; done
    bar 9 "${total}" "montage/router/model audit"
    "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/audit_open_vocab_0722_model.py" --config "${CONFIG}" "${PROJECT_ONLY_ARGS[@]}" "${DEVICE_ARGS[@]}"
    bar 10 "${total}" "formal validation gate"
    manifest="${OUTPUT_ROOT}/synthesis/${GENERALIZATION}/${HOLDOUT_LABEL:-all}/karaone/validation/synthesis_manifest.json"
    gate_args=(--config "${CONFIG}" --synthesis-manifest "${manifest}" --generalization "${GENERALIZATION}")
    gate_args+=("${PROJECT_ONLY_ARGS[@]}")
    [[ -n "${DENSE_BASELINE_REPORT:-}" ]] && gate_args+=(--dense-baseline-report "${DENSE_BASELINE_REPORT}")
    [[ -n "${SEED_SUMMARY:-}" ]] && gate_args+=(--seed-summary "${SEED_SUMMARY}")
    "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/gate_open_vocab_0722.py" "${gate_args[@]}"
    bar 11 "${total}" "comparison plots"
    for dataset in karaone feis ds004306; do
      "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/plot_open_vocab_0722_pairs.py" --manifest "${OUTPUT_ROOT}/synthesis/${GENERALIZATION}/${HOLDOUT_LABEL:-all}/${dataset}/validation/synthesis_manifest.json"
    done
    bar 12 "${total}" "embedding evaluation"
    "${PYTHON_BIN}" "${BUNDLE_DIR}/app/scripts/train_open_vocab_0722.py" --config "${CONFIG}" --phase evaluate --split validation --generalization "${GENERALIZATION}" "${HOLDOUT_ARGS[@]}" "${PROJECT_ONLY_ARGS[@]}" "${DEVICE_ARGS[@]}"
    bar 13 "${total}" "complete; locked test was not accessed"
    ;;
  *)
    echo "usage: $0 {prepare|public-manifest|public-cache|teachers|train-audio|pretrain-eeg|train-eeg|select-eeg|validate|test|synthesize|audit-model|gate|plot|track-b|ablation-config|seed-config|aggregate-seeds|compare-dense-moe|all} [options]" >&2
    exit 2
    ;;
esac
