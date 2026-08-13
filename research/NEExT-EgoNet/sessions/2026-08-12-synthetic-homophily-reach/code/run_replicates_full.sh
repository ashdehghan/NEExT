#!/bin/bash
# Full replicate battery (Ash-approved 2026-08-13): the three series the first
# replicate pass did not cover, 10 worlds each per wavelength.
#   B1 local (30 jobs) + B0 global (30) + B0 local (30) = 90 jobs.
# B0 replicate worlds generate on demand; each gets a manipulation check.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
for L in L1 L2 L3; do
  for seed in 101 102 103 104 105 106 107 108 109 110; do
    b1="SYNTH_HR_${L}_B1_S${seed}"
    b0="SYNTH_HR_${L}_B0_S${seed}"
    for job in "$b1 local" "$b0 global" "$b0 local"; do
      cell=${job% *}; scope=${job#* }
      log="logs/${cell}_khop_${scope}.log"
      if grep -q "] done" "$log" 2>/dev/null; then echo "skip $cell $scope (done)"; continue; fi
      if [ "$cell" = "$b0" ] && [ "$scope" = "global" ]; then
        uv run python code/manipulation_checks.py "$cell" >> logs/replicate_checks.log 2>&1 || { echo "CHECKS FAILED: $cell"; exit 1; }
      fi
      echo "=== $(date +%H:%M:%S) $cell $scope ==="
      uv run python code/run_experiment.py --dataset "$cell" --family khop --scope "$scope" > "$log" 2>&1 || { echo "STAGE FAILED: $cell $scope — stopping per on-errors rule"; exit 1; }
      tail -2 "$log"
    done
  done
done
echo "=== FULL BATTERY DONE $(date +%H:%M:%S) ==="
