#!/bin/bash
# Replicate study (Ash-approved 2026-08-12): 10 fresh worlds per wavelength,
# whisper ON, global-scope khop sweep each. Seeds 101-110.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
for L in L1 L2 L3; do
  for seed in 101 102 103 104 105 106 107 108 109 110; do
    cell="SYNTH_HR_${L}_B1_S${seed}"
    log="logs/${cell}_khop_global.log"
    echo "=== $(date +%H:%M:%S) $cell ==="
    uv run python code/manipulation_checks.py "$cell" >> logs/replicate_checks.log 2>&1 || { echo "CHECKS FAILED: $cell"; exit 1; }
    uv run python code/run_experiment.py --dataset "$cell" --family khop --scope global > "$log" 2>&1 || { echo "STAGE FAILED: $cell — stopping per on-errors rule"; exit 1; }
    tail -2 "$log"
  done
done
echo "=== ALL REPLICATES DONE $(date +%H:%M:%S) ==="
