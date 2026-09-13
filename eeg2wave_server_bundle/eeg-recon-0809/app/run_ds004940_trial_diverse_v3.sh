#!/usr/bin/env bash
# Resumable DS004940 v3 trial-diversity pilot.  It never mutates v2 outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export DATA_CONFIG="${DATA_CONFIG:-$PROJECT_ROOT/configs/training_data_v4_ds004940_fixed.yaml}"
export PILOT_CONFIG="${PILOT_CONFIG:-$PROJECT_ROOT/configs/ds004940_trial_diverse_v3.yaml}"
source "$SCRIPT_DIR/lib/joint_pilot_common.sh"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-ds004940_trial_diverse_v3}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$RUN_ROOT/$EXPERIMENT_NAME}"
V3_FROM="${V3_FROM:-all}" # all | m0 | m1 | renderer | export | resume
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
M0_STEPS="${M0_STEPS:-5000}"
M0_BATCH_SIZE="${M0_BATCH_SIZE:-2}"
M1_EPOCHS="${M1_EPOCHS:-50}"
M1_BATCH_SIZE="${M1_BATCH_SIZE:-4}"
M1_CONTENTS_PER_BATCH="${M1_CONTENTS_PER_BATCH:-2}"
M1_SUBJECTS_PER_CONTENT="${M1_SUBJECTS_PER_CONTENT:-2}"
THERMAL_MODE="${THERMAL_MODE:-1}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-90}"
NATIVE_RENDERER_STEPS="${NATIVE_RENDERER_STEPS:-2000}"
DIFFUSION_DENOISE="${DIFFUSION_DENOISE:-0}"
DIFFUSION_TRAIN_STEPS="${DIFFUSION_TRAIN_STEPS:-2000}"
EXPORT_SAMPLES_PER_TRIAL="${EXPORT_SAMPLES_PER_TRIAL:-1}"

case "$V3_FROM" in all|m0|m1|renderer|export|resume) ;; *) echo "V3_FROM must be all, m0, m1, renderer, export, or resume" >&2; exit 2 ;; esac
case "$DIFFUSION_DENOISE" in 0|1) ;; *) echo "DIFFUSION_DENOISE must be 0 or 1" >&2; exit 2 ;; esac
case "$THERMAL_MODE" in 0|1) ;; *) echo "THERMAL_MODE must be 0 or 1" >&2; exit 2 ;; esac

if [[ "$THERMAL_MODE" == "1" ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.75}"
  export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.40}"
fi

start_joint_log "$EXPERIMENT_NAME"
require_joint_runtime
require_local_hubert
cd "$PROJECT_ROOT"
echo "WARNING: v3 is exploratory until pairing/listening gates are independently completed."
echo "experiment_root=$EXPERIMENT_ROOT resume_from=$V3_FROM thermal_mode=$THERMAL_MODE"

ARTIFACT="artifacts/training_data/v4_ds004940_fixed"
M0_MANIFEST="$ARTIFACT/manifests/manifest_explore_m0_ds004940_trial_diverse_v3.csv"
M0_TARGETS="$ARTIFACT/speech_targets/speech_targets_explore_m0_ds004940_trial_diverse_v3.h5"
M0_NORMALIZER="$ARTIFACT/normalizers/explore_m0_ds004940_trial_diverse_v3_joint_ood_fold-0.json"
M1_MANIFEST="$ARTIFACT/manifests/manifest_explore_stage2_ds004940_trial_diverse_v3.csv"
M1_SPLIT="$ARTIFACT/splits/stage2_ds004940_trial_diverse_v3_fold-0.csv"
M1_TARGETS="$ARTIFACT/speech_targets/speech_targets_explore_stage2_ds004940_trial_diverse_v3.h5"
M1_NORMALIZER="$ARTIFACT/normalizers/explore_stage2_ds004940_trial_diverse_v3_fold-0.json"

if [[ "$V3_FROM" == "all" || "$V3_FROM" == "m0" || "$V3_FROM" == "resume" ]]; then
  joint_run "$PYTHON_BIN" scripts/prepare_m0_artifacts.py --data-config "$DATA_CONFIG" --pilot-config "$PILOT_CONFIG" \
    --hubert-local-path "$HUBERT_LOCAL_PATH" --artifact-set explore_m0_ds004940_trial_diverse_v3
  joint_run "$PYTHON_BIN" app/audit_fixed_window.py --manifest "$M0_MANIFEST" \
    --output "$EXPERIMENT_ROOT/d0_fixed_window_m0.json"
  joint_run "$PYTHON_BIN" app/train_joint.py --config "$PILOT_CONFIG" --mode ds004940 --stage overfit --seed 31 --explore \
    --max-steps "$M0_STEPS" --batch-size "$M0_BATCH_SIZE" --checkpoint-every "$CHECKPOINT_EVERY" --output-root "$EXPERIMENT_ROOT"
  joint_run "$PYTHON_BIN" app/evaluate_joint.py --checkpoint "$EXPERIMENT_ROOT/overfit/ds004940/seed-31/checkpoint.pt" \
    --dataset ds004940 --role train
fi

M0_EVALUATION="$EXPERIMENT_ROOT/overfit/ds004940/seed-31/evaluation_ds004940_train.json"
require_v3_m0_gate() {
  joint_run "$PYTHON_BIN" - "$M0_EVALUATION" <<'PY'
import json, sys
from pathlib import Path
path=Path(sys.argv[1])
if not path.exists(): raise SystemExit(f"M1 blocked: missing M0 evaluation {path}")
result=json.loads(path.read_text())
if not result.get("gate",{}).get("passed",False):
    raise SystemExit("M1 blocked: v3 M0 gate failed: "+json.dumps(result.get("gate",{}).get("checks",{}),sort_keys=True))
print("v3_m0_gate=pass")
PY
}

if [[ "$V3_FROM" == "all" || "$V3_FROM" == "m1" || "$V3_FROM" == "resume" ]]; then
  require_v3_m0_gate
  joint_run "$PYTHON_BIN" scripts/prepare_stage2_split.py --data-config "$DATA_CONFIG" --pilot-config "$PILOT_CONFIG" \
    --explore --materialize --hubert-local-path "$HUBERT_LOCAL_PATH"
  for seed in $(pilot_seeds); do
    joint_run "$PYTHON_BIN" app/train_joint.py --config "$PILOT_CONFIG" --mode ds004940 --stage generalization --seed "$seed" --explore \
      --max-epochs "$M1_EPOCHS" --batch-size "$M1_BATCH_SIZE" --contents-per-batch "$M1_CONTENTS_PER_BATCH" \
      --subjects-per-content "$M1_SUBJECTS_PER_CONTENT" --checkpoint-every "$CHECKPOINT_EVERY" --output-root "$EXPERIMENT_ROOT"
    checkpoint="$EXPERIMENT_ROOT/generalization/ds004940/seed-$seed/checkpoint.pt"
    joint_run "$PYTHON_BIN" app/evaluate_joint.py --checkpoint "$checkpoint" --dataset ds004940 --role validation
    joint_run "$PYTHON_BIN" app/evaluate_joint.py --checkpoint "$checkpoint" --dataset ds004940 --role test
    [[ "$COOLDOWN_SECONDS" -gt 0 ]] && sleep "$COOLDOWN_SECONDS"
  done
fi

if [[ "$V3_FROM" == "all" || "$V3_FROM" == "renderer" || "$V3_FROM" == "export" || "$V3_FROM" == "resume" ]]; then
  require_v3_m0_gate
  for required in "$M1_MANIFEST" "$M1_SPLIT" "$M1_TARGETS" "$M1_NORMALIZER"; do
    [[ -f "$required" ]] || { echo "renderer/export continuation requires artifact: $required" >&2; exit 2; }
  done
  if [[ "$V3_FROM" != "export" ]]; then
    joint_run "$PYTHON_BIN" app/train_native_audio_renderer.py --config "$PILOT_CONFIG" --manifest "$M1_MANIFEST" \
      --split "$M1_SPLIT" --targets "$M1_TARGETS" --normalizer "$M1_NORMALIZER" \
      --output "$EXPERIMENT_ROOT/native_audio_renderer" --max-steps "$NATIVE_RENDERER_STEPS" --checkpoint-every "$CHECKPOINT_EVERY"
    if [[ "$DIFFUSION_DENOISE" == "1" ]]; then
      checkpoint="$EXPERIMENT_ROOT/generalization/ds004940/seed-31/checkpoint.pt"
      joint_run "$PYTHON_BIN" app/train_native_mel_diffusion.py --config "$PILOT_CONFIG" --manifest "$M1_MANIFEST" \
        --split "$M1_SPLIT" --targets "$M1_TARGETS" --normalizer "$M1_NORMALIZER" \
        --renderer "$EXPERIMENT_ROOT/native_audio_renderer/checkpoint.pt" --eeg-checkpoint "$checkpoint" \
        --output "$EXPERIMENT_ROOT/native_mel_diffusion" --max-steps "$DIFFUSION_TRAIN_STEPS" --checkpoint-every "$CHECKPOINT_EVERY"
    fi
  fi
  [[ -f "$EXPERIMENT_ROOT/native_audio_renderer/checkpoint.pt" ]] || {
    echo "export requires a renderer checkpoint: $EXPERIMENT_ROOT/native_audio_renderer/checkpoint.pt" >&2
    exit 2
  }
  if [[ "$DIFFUSION_DENOISE" == "1" ]]; then
    [[ -f "$EXPERIMENT_ROOT/native_mel_diffusion/checkpoint.pt" ]] || {
      echo "random qualitative export requires a diffusion checkpoint: $EXPERIMENT_ROOT/native_mel_diffusion/checkpoint.pt" >&2
      exit 2
    }
  fi
  for seed in $(pilot_seeds); do
    checkpoint="$EXPERIMENT_ROOT/generalization/ds004940/seed-$seed/checkpoint.pt"
    [[ -f "$checkpoint" ]] || { echo "export requires completed checkpoint: $checkpoint" >&2; exit 2; }
    export_args=(--checkpoint "$checkpoint" --renderer "$EXPERIMENT_ROOT/native_audio_renderer/checkpoint.pt" --role test --max-pairs 32 \
      --output "$EXPERIMENT_ROOT/native_audio_pairs/seed-$seed/test" --sampling-mode deterministic)
    if [[ "$DIFFUSION_DENOISE" == "1" ]]; then
      export_args+=(--diffusion-mode on --diffusion-checkpoint "$EXPERIMENT_ROOT/native_mel_diffusion/checkpoint.pt" \
        --sampling-mode random --samples-per-trial "$EXPORT_SAMPLES_PER_TRIAL")
    else
      export_args+=(--diffusion-mode off)
    fi
    joint_run "$PYTHON_BIN" app/export_conditioned_audio_pairs.py "${export_args[@]}"
  done
fi

echo "v3 pipeline complete. All outputs remain exploratory."
