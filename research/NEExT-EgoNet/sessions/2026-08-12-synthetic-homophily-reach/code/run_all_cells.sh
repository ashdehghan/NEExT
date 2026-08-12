#!/bin/bash
# Full grid: 6 cells x (khop local, khop global, nodefeat). Sequential; logs per stage.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
for cell in SYNTH_HR_L1_B1 SYNTH_HR_L2_B1 SYNTH_HR_L3_B1 SYNTH_HR_L1_B0 SYNTH_HR_L2_B0 SYNTH_HR_L3_B0; do
  for stage in "khop local" "nodefeat local" "khop global"; do
    fam=${stage% *}; scope=${stage#* }
    log="logs/${cell}_${fam}_${scope}.log"
    echo "=== $(date +%H:%M:%S) $cell $fam $scope ==="
    uv run python code/run_experiment.py --dataset "$cell" --family "$fam" --scope "$scope" > "$log" 2>&1
    rc=$?
    tail -3 "$log"
    if [ $rc -ne 0 ]; then echo "STAGE FAILED ($rc): $cell $fam $scope — stopping per on-errors rule"; exit $rc; fi
  done
done
echo "=== ALL CELLS DONE $(date +%H:%M:%S) ==="
