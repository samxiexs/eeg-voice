#!/usr/bin/env bash
# What is running, how far it got, and whether anything stopped or failed.
# Live view: while true; do clear; bash scripts/status.sh; sleep 30; done
cd "$(dirname "$0")/.."
mtime() { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1"; }      # macOS / Linux
date
echo "--- running"
jobs=$(ps -eo pid,etime,%cpu,command | grep -E "scripts/(train|prepare|download|baseline|isc|reconstruct)\.py|run_plan\.sh" | grep -v grep | cut -c1-150)
if [ -n "$jobs" ]; then echo "$jobs"; else echo "!! nothing is running"; fi
echo "--- training (latest logs)"
for log in $(ls -t logs/listen.log logs/imagery_*.log 2>/dev/null | head -2); do
  step=$(grep '"step"' "$log" | tail -1 | sed -E 's/.*"step": ([0-9]+), "seconds": ([0-9]+).*/step \1, \2 s/')
  age=$(( $(date +%s) - $(mtime "$log") ))
  echo "$log: ${step:-starting} (updated ${age}s ago)"
done
for log in $(ls -t logs/decoder*.log logs/reconstruct*.log 2>/dev/null | head -2); do
  age=$(( $(date +%s) - $(mtime "$log") ))
  echo "$log: $(grep -v Warn "$log" | tail -1 | cut -c1-140) (updated ${age}s ago)"
done
echo "--- reconstruction runs finished: $(ls outputs/reconstruct/f*/*/summary.json 2>/dev/null | wc -l | tr -d ' ')"
echo "--- imagery runs finished: $(ls outputs/imagery/*/evaluation.json 2>/dev/null | wc -l | tr -d ' ') / 20"
for parts in artifacts/store/*.parts; do                                    # a per-person conversion in progress
  [ -d "$parts" ] && echo "--- $(basename "$parts" .parts) parts converted: $(ls "$parts" | tr '\n' ' ')"
done
for log in $(find logs -name '*.log' -mmin -120 2>/dev/null); do        # logs written in the last 2 h
  error=$(grep -E "Error|FAILED|non-finite" "$log" | grep -v Warning | tail -1 | cut -c1-120)
  [ -n "$error" ] && echo "!! $log: $error"
done
exit 0
