#!/bin/bash
# Phase 3 Part B (Ash-approved 2026-08-13): beta-sweep regime morphing at
# lambda=0.20, seeds 101-105 shared across beta; plus chore D: nodefeat on
# all batch-1 replicate worlds.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
run_stage() {
  local cell=$1 fam=$2 scope=$3
  local log="logs/${cell}_${fam}_${scope}.log"
  if grep -q "] done" "$log" 2>/dev/null; then echo "skip $cell $fam $scope"; return 0; fi
  echo "=== $(date +%H:%M:%S) $cell $fam $scope ==="
  uv run python code/run_experiment.py --dataset "$cell" --family "$fam" --scope "$scope" > "$log" 2>&1 \
    || { echo "STAGE FAILED: $cell $fam $scope — stopping per on-errors rule"; exit 1; }
  tail -2 "$log"
}
# New beta cells: checks + full battery
for B in B2 B3 B4; do
  for seed in 101 102 103 104 105; do
    cell="SYNTH_HR_L2_${B}_S${seed}"
    uv run python code/manipulation_checks.py "$cell" >> logs/beta_sweep_checks.log 2>&1 || { echo "CHECKS FAILED: $cell"; exit 1; }
    run_stage "$cell" khop global
    run_stage "$cell" khop local
    run_stage "$cell" nodefeat local
  done
done
# nodefeat for the sweep's B0/B1 endpoints (khop already cached from batch 1)
for B in B0 B1; do
  for seed in 101 102 103 104 105; do
    run_stage "SYNTH_HR_L2_${B}_S${seed}" nodefeat local
  done
done
# Chore D: nodefeat on the remaining batch-1 replicate worlds
for L in L1 L3; do
  for B in B0 B1; do
    for seed in 101 102 103 104 105 106 107 108 109 110; do
      run_stage "SYNTH_HR_${L}_${B}_S${seed}" nodefeat local
    done
  done
done
for B in B0 B1; do
  for seed in 106 107 108 109 110; do
    run_stage "SYNTH_HR_L2_${B}_S${seed}" nodefeat local
  done
done
echo "=== BETA SWEEP + CHORE DONE $(date +%H:%M:%S) ==="
