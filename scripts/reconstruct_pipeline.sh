#!/usr/bin/env bash
# Full reconstruction pipeline (about 7 h on an M2 Pro), one job at a time:
#   1-2  audio-side decoders, WavLM route and shared-embedding route (no EEG), each with a test report
#   3-4  EEG bridge on each route (decoder and EEG tower frozen), each with a three-scale test evaluation
#   5    zero-shot: the embedding-route decoder on the EEG embedding without EEG training
# Progress: tail -f logs/reconstruct/<run>/pipeline.log   Stop: kill the PID in artifacts/reconstruct/<run>/pipeline.pid
set -u
cd "$(dirname "$0")/.."
RUN=${1:-v3}; AUDIO_MIN=${AUDIO_MIN:-70}; EEG_MIN=${EEG_MIN:-120}
PY=.venv-aligned-local/bin/python; LOG=logs/reconstruct/$RUN; mkdir -p "$LOG" "artifacts/reconstruct/$RUN"
echo $$ > "artifacts/reconstruct/$RUN/pipeline.pid"
step() { local name=$1; shift; echo "$(date '+%F %T') start $name" >> "$LOG/pipeline.log"
  if "$PY" scripts/reconstruct.py "$@" > "$LOG/$name.log" 2>&1; then echo "$(date '+%F %T') done  $name" >> "$LOG/pipeline.log"; return 0
  else echo "$(date '+%F %T') FAILED $name (see $LOG/$name.log)" >> "$LOG/pipeline.log"; return 1; fi; }
step audio-wavlm audio --run "$RUN/audio-wavlm" --route wavlm --minutes "$AUDIO_MIN" --validate-minutes 5
step audio-embedding audio --run "$RUN/audio-embedding" --route embedding --minutes "$AUDIO_MIN" --validate-minutes 5
if [ -f "artifacts/reconstruct/$RUN/audio-wavlm/audio_decoder.pt" ]; then
  step eeg-wavlm train --run "$RUN/eeg-wavlm" --audio "$RUN/audio-wavlm" --minutes "$EEG_MIN" --validate-minutes 10 && \
  step eval-eeg-wavlm evaluate --run "$RUN/eeg-wavlm"
fi
if [ -f "artifacts/reconstruct/$RUN/audio-embedding/audio_decoder.pt" ]; then
  step eeg-embedding train --run "$RUN/eeg-embedding" --audio "$RUN/audio-embedding" --minutes "$EEG_MIN" --validate-minutes 10 && \
  step eval-eeg-embedding evaluate --run "$RUN/eeg-embedding"
  step zeroshot zeroshot --run "$RUN/zeroshot" --audio "$RUN/audio-embedding" && step eval-zeroshot evaluate --run "$RUN/zeroshot"
fi
echo "$(date '+%F %T') pipeline finished" >> "$LOG/pipeline.log"
