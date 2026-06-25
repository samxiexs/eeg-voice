#!/usr/bin/env bash
set -euo pipefail
#
# Trains each FEIS stage independently (stimuli/thinking/speaking/resting) and writes a
# four-stage summary that now includes the HONEST controls per stage
# (pred_over_labelprior_pcc_gain, pred_over_zeroeeg_pcc_gain, content_over_chance,
# eeg_informative). Read the table across stages: if `resting` (no speech) is ~as good as
# `speaking`, and gains ~0 / eeg_informative=false, the mel matches the label prior, not EEG.
#
# NeuroTalk overt->imagined transfer (improvement lever, ROI #5): pretrain on the strongest
# stage then fine-tune a weaker one via --init-from, e.g.
#   python scripts/feis_mel_train.py --stage speaking --run-suffix overt
#   python scripts/feis_mel_train.py --stage thinking --run-suffix from_overt \
#       --init-from ../artifacts/outputs_mel/feis_mel_speaking_overt/checkpoints/best.pt
#
# Generative decoder head: DECODER={regression|diffusion|flow}. flow = conditional flow
# matching (NeuroSonic/Voicebox), deterministic ODE sampling, fewer steps, no mean-collapse:
#   DECODER=flow bash run_full_four_stage_feis_mel.sh

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BUNDLE_DIR"

CONDA_ROOT="${CONDA_ROOT:-/opt/anaconda3}"
CONDA_ENV="${CONDA_ENV:-eegvoice}"
CONFIG="${CONFIG:-configs/feis_mel_align.yaml}"
STAGES="${STAGES:-stimuli thinking speaking resting}"
EPOCHS="${EPOCHS:-40}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_SUFFIX="${RUN_SUFFIX:-mel_align_${RUN_TAG}}"
DEVICE="${DEVICE:-cpu}"
SPLIT="${SPLIT:-test_holdout}"
SYNTH_LIMIT="${SYNTH_LIMIT:-24}"
MAX_WAVEFORM_FIGS="${MAX_WAVEFORM_FIGS:-24}"
MAX_STEPS="${MAX_STEPS:-}"
DECODER="${DECODER:-regression}"   # regression | diffusion | flow (conditional flow matching, NeuroSonic/Voicebox)
OUTPUT_ROOT="$BUNDLE_DIR/../artifacts/outputs_mel"
SUMMARY_CSV="$OUTPUT_ROOT/four_stage_summary_${RUN_TAG}.csv"
SUMMARY_JSON="$OUTPUT_ROOT/four_stage_summary_${RUN_TAG}.json"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$BUNDLE_DIR/../artifacts/matplotlib_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/feis-cache}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_ROOT"

if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
  echo "Missing conda init script: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
  exit 1
fi

conda activate "$CONDA_ENV"

python - <<'PY'
import sys
import torch
import scipy
print(f"[env] python={sys.executable}")
print(f"[env] torch={torch.__version__}")
print(f"[env] scipy={scipy.__version__}")
PY

echo "[config] stages=$STAGES decoder=$DECODER"
echo "[config] epochs=$EPOCHS run_tag=$RUN_TAG run_suffix=$RUN_SUFFIX device=$DEVICE split=$SPLIT"
echo "[summary] $SUMMARY_CSV"

SUMMARY_CSV="$SUMMARY_CSV" SUMMARY_JSON="$SUMMARY_JSON" python - <<'PY'
import csv
import json
import os
from pathlib import Path

csv_path = Path(os.environ["SUMMARY_CSV"])
json_path = Path(os.environ["SUMMARY_JSON"])
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        "stage", "target_kind", "decoder_kind", "channel_moe", "diffusion_enabled", "gan_enabled",
        "content_top1", "retrieval_top1", "mel_PCC", "DTW_MCD", "pred_to_label_bank_dtw",
        "mean_mel_baseline_dtw", "pred_beats_mean",
        "pred_over_labelprior_pcc_gain", "pred_over_zeroeeg_pcc_gain", "content_over_chance", "eeg_informative",
        "best_checkpoint", "run_dir", "wav_dir",
        "figures_dir", "waveform_compare_dir",
    ])
json_path.write_text(json.dumps([], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

echo "+ python scripts/build_feis_mel_targets.py --config $CONFIG"
python scripts/build_feis_mel_targets.py --config "$CONFIG"

for STAGE in $STAGES; do
  RUN_DIR="$BUNDLE_DIR/../artifacts/outputs_mel/feis_mel_${STAGE}_${RUN_SUFFIX}"
  WAV_DIR="$RUN_DIR/final_wavs_${SPLIT}"

  echo
  echo "===== Stage: $STAGE ====="
  TRAIN_CMD=(
    python scripts/feis_mel_train.py
    --config "$CONFIG"
    --stage "$STAGE"
    --epochs "$EPOCHS"
    --run-suffix "$RUN_SUFFIX"
    --device "$DEVICE"
    --decoder "$DECODER"
  )
  if [ -n "$MAX_STEPS" ]; then
    TRAIN_CMD+=(--max-steps "$MAX_STEPS")
  fi
  echo "+ ${TRAIN_CMD[*]}"
  "${TRAIN_CMD[@]}"

  echo "+ python scripts/feis_mel_eval.py --checkpoint $RUN_DIR/checkpoints/best.pt"
  python scripts/feis_mel_eval.py \
    --config "$CONFIG" \
    --checkpoint "$RUN_DIR/checkpoints/best.pt" \
    --split "$SPLIT" \
    --device "$DEVICE"

  echo "+ python scripts/feis_mel_synthesize.py --checkpoint $RUN_DIR/checkpoints/best.pt"
  python scripts/feis_mel_synthesize.py \
    --config "$CONFIG" \
    --checkpoint "$RUN_DIR/checkpoints/best.pt" \
    --split "$SPLIT" \
    --out-dir "$WAV_DIR" \
    --limit "$SYNTH_LIMIT" \
    --diverse-labels \
    --device "$DEVICE"

  echo "+ python scripts/direct_make_run_figures.py --wav-dir $WAV_DIR"
  python scripts/direct_make_run_figures.py \
    --wav-dir "$WAV_DIR" \
    --max-waveforms "$MAX_WAVEFORM_FIGS"

  SUMMARY_CSV="$SUMMARY_CSV" SUMMARY_JSON="$SUMMARY_JSON" STAGE="$STAGE" RUN_DIR="$RUN_DIR" WAV_DIR="$WAV_DIR" python - <<'PY'
import csv
import json
import os
from pathlib import Path

import torch

stage = os.environ["STAGE"]
run_dir = Path(os.environ["RUN_DIR"])
wav_dir = Path(os.environ["WAV_DIR"])
csv_path = Path(os.environ["SUMMARY_CSV"])
json_path = Path(os.environ["SUMMARY_JSON"])
ckpt_path = run_dir / "checkpoints" / "best.pt"
metrics_path = run_dir / "metrics" / "test_metrics.json"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
holdout = metrics.get("test_holdout", {})
row = {
    "stage": stage,
    "target_kind": ckpt.get("target_kind", ""),
    "decoder_kind": ckpt.get("decoder_kind", "regression"),
    "channel_moe": bool(ckpt.get("channel_moe", False)),
    "diffusion_enabled": bool(ckpt.get("diffusion_enabled", False)),
    "gan_enabled": bool(ckpt.get("gan_enabled", False)),
    "content_top1": holdout.get("content_top1", ""),
    "retrieval_top1": holdout.get("retrieval_top1", ""),
    "mel_PCC": holdout.get("mel_PCC", ""),
    "DTW_MCD": holdout.get("DTW_MCD", ""),
    "pred_to_label_bank_dtw": holdout.get("pred_to_label_bank_dtw", ""),
    "mean_mel_baseline_dtw": holdout.get("mean_mel_baseline_dtw", ""),
    "pred_beats_mean": holdout.get("pred_beats_mean", ""),
    # Honest controls (the cross-stage view: resting should be the worst / not informative).
    "pred_over_labelprior_pcc_gain": holdout.get("pred_over_labelprior_pcc_gain", ""),
    "pred_over_zeroeeg_pcc_gain": holdout.get("pred_over_zeroeeg_pcc_gain", ""),
    "content_over_chance": holdout.get("content_over_chance", ""),
    "eeg_informative": holdout.get("eeg_informative", ""),
    "best_checkpoint": str(ckpt_path),
    "run_dir": str(run_dir),
    "wav_dir": str(wav_dir),
    "figures_dir": str(run_dir / "figures"),
    "waveform_compare_dir": str(wav_dir / "waveform_compare"),
}
fields = [
    "stage", "target_kind", "decoder_kind", "channel_moe", "diffusion_enabled", "gan_enabled",
    "content_top1", "retrieval_top1", "mel_PCC", "DTW_MCD", "pred_to_label_bank_dtw",
    "mean_mel_baseline_dtw", "pred_beats_mean",
    "pred_over_labelprior_pcc_gain", "pred_over_zeroeeg_pcc_gain", "content_over_chance", "eeg_informative",
    "best_checkpoint", "run_dir", "wav_dir",
    "figures_dir", "waveform_compare_dir",
]
with csv_path.open("a", encoding="utf-8", newline="") as handle:
    csv.DictWriter(handle, fieldnames=fields).writerow(row)
payload = json.loads(json_path.read_text(encoding="utf-8"))
payload.append(row)
json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

  echo "[stage done] $STAGE"
  echo "  run_dir: $RUN_DIR"
  echo "  training_figures: $RUN_DIR/figures"
  echo "  wavs: $WAV_DIR"
  echo "  waveform_figures: $WAV_DIR/waveform_compare"
done

echo
echo "[done] all requested stages finished."
echo "  summary_csv: $SUMMARY_CSV"
echo "  summary_json: $SUMMARY_JSON"
