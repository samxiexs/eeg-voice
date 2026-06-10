#!/usr/bin/env bash
set -euo pipefail

# Full FEIS-only v3 run:
#   FEIS EnCodec targets -> speaking teacher -> thinking main -> eval -> recon metrics -> wav synthesis
#
# Default:
#   bash run_feis_v3_full.sh
#
# Optional:
#   PROTOCOL=S SUBJECT=01 bash run_feis_v3_full.sh
#   PROTOCOL=U HOLDOUT=21 bash run_feis_v3_full.sh
#   SYNTH_LIMIT=64 RECON_LIMIT=128 bash run_feis_v3_full.sh

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${APP_DIR}"

TAG="${TAG:-$(date +%m%d_%H%M)}"
PROTOCOL="${PROTOCOL:-G}"
SUBJECT="${SUBJECT:-01}"
HOLDOUT="${HOLDOUT:-21}"
SYNTH_LIMIT="${SYNTH_LIMIT:-32}"
RECON_LIMIT="${RECON_LIMIT:-64}"

protocol_args() {
  case "${PROTOCOL}" in
    G)
      ;;
    S)
      printf '%s\n' --subject "${SUBJECT}"
      ;;
    U)
      printf '%s\n' --holdout-subject "${HOLDOUT}"
      ;;
    *)
      echo "Unsupported PROTOCOL=${PROTOCOL}; use G, S, or U." >&2
      exit 1
      ;;
  esac
}

device_args() {
  if [[ -n "${DEVICE:-}" ]]; then
    printf '%s\n' --device "${DEVICE}"
  fi
}

run_prefix() {
  local stage="$1"
  local suffix="$2"
  local lower
  lower="$(printf '%s' "${PROTOCOL}" | tr '[:upper:]' '[:lower:]')"
  printf '%s_%s_%s' "${lower}" "${stage}" "${suffix}"
}

CONTEXT_SUFFIX=""
if [[ "${PROTOCOL}" == "S" ]]; then
  CONTEXT_SUFFIX="_subject_${SUBJECT}"
elif [[ "${PROTOCOL}" == "U" ]]; then
  CONTEXT_SUFFIX="_holdout_${HOLDOUT}"
fi

SPEAKING_SUFFIX="speaking_teacher_${TAG}${CONTEXT_SUFFIX}"
THINKING_SUFFIX="thinking_main_${TAG}${CONTEXT_SUFFIX}"
SPEAKING_RUN="$(run_prefix speaking "${SPEAKING_SUFFIX}")"
THINKING_RUN="$(run_prefix thinking "${THINKING_SUFFIX}")"
SPEAKING_CKPT="../artifacts/outputs_v3/${SPEAKING_RUN}/checkpoints/best.pt"
THINKING_CKPT="../artifacts/outputs_v3/${THINKING_RUN}/checkpoints/best.pt"

echo "===== FEIS v3 full run ====="
echo "APP_DIR=${APP_DIR}"
echo "TAG=${TAG}"
echo "PROTOCOL=${PROTOCOL}"
if [[ "${PROTOCOL}" == "S" ]]; then
  echo "SUBJECT=${SUBJECT}"
elif [[ "${PROTOCOL}" == "U" ]]; then
  echo "HOLDOUT=${HOLDOUT}"
fi
echo "SPEAKING_RUN=${SPEAKING_RUN}"
echo "THINKING_RUN=${THINKING_RUN}"
echo "SYNTH_LIMIT=${SYNTH_LIMIT}"
echo "RECON_LIMIT=${RECON_LIMIT}"

echo
echo "===== Step 1/6: extract FEIS EnCodec targets ====="
python scripts/extract_audio_targets.py \
  --config configs/alignment_encodec_local.yaml \
  --backend encodec_latent

echo
echo "===== Step 2/6: train speaking teacher ====="
python scripts/v3_train.py \
  --config configs/v3_encodec.yaml \
  --protocol "${PROTOCOL}" \
  --stage speaking \
  --run-suffix "${SPEAKING_SUFFIX}" \
  $(protocol_args) \
  $(device_args)

echo
echo "===== Step 3/6: train thinking main ====="
python scripts/v3_train.py \
  --config configs/v3_encodec.yaml \
  --protocol "${PROTOCOL}" \
  --stage thinking \
  --run-suffix "${THINKING_SUFFIX}" \
  --init-from "${SPEAKING_CKPT}" \
  --distill-teacher "${SPEAKING_CKPT}" \
  --teacher-stage speaking \
  $(protocol_args) \
  $(device_args)

echo
echo "===== Step 4/6: eval retrieval/classification ====="
python scripts/v3_eval.py \
  --config configs/v3_encodec.yaml \
  --checkpoint "${THINKING_CKPT}" \
  --protocol "${PROTOCOL}" \
  --stage thinking \
  --split test \
  --out "../artifacts/outputs_v3/${THINKING_RUN}/metrics/test_eval.json" \
  $(protocol_args) \
  $(device_args)

echo
echo "===== Step 5/6: eval reconstruction metrics ====="
python scripts/v3_recon_eval.py \
  --config configs/v3_encodec.yaml \
  --checkpoint "${THINKING_CKPT}" \
  --protocol "${PROTOCOL}" \
  --stage thinking \
  --split test \
  --limit "${RECON_LIMIT}" \
  --out "../artifacts/outputs_v3/${THINKING_RUN}/metrics/recon_eval.json" \
  $(protocol_args) \
  $(device_args)

echo
echo "===== Step 6/6: synthesize wavs ====="
python scripts/v3_synthesize.py \
  --config configs/v3_encodec.yaml \
  --checkpoint "${THINKING_CKPT}" \
  --protocol "${PROTOCOL}" \
  --stage thinking \
  --out-dir "../artifacts/outputs_v3/${THINKING_RUN}/recon_wavs" \
  --limit "${SYNTH_LIMIT}" \
  $(protocol_args) \
  $(device_args)

echo
echo "Done."
echo "Metrics:"
echo "  ../artifacts/outputs_v3/${THINKING_RUN}/metrics/test_eval.json"
echo "  ../artifacts/outputs_v3/${THINKING_RUN}/metrics/recon_eval.json"
echo "Wavs:"
echo "  ../artifacts/outputs_v3/${THINKING_RUN}/recon_wavs"
