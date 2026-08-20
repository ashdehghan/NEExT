#!/bin/bash
# k=5 extension battery (Ash-approved): new khop cells only, one job at a time.
set -u
cd "$(dirname "$0")"
mkdir -p ext_logs
run() {
  local sess=$1 ks=$2 scope=$3
  local tag="$(basename $sess)_k${ks//,/}_${scope}"
  echo "=== $(date +%H:%M:%S) $tag ==="
  (cd ../../.. && uv run --no-sync python research/NEExT-EgoNet/sessions/$sess/code/run_experiment.py --family khop --scope $scope --k-hops $ks) \
    > "ext_logs/${tag}.log" 2>&1 || { echo "STAGE FAILED: $tag — stopping per on-errors rule"; exit 1; }
  tail -3 "ext_logs/${tag}.log"
}
run 2026-08-06-egonet-signal-airports-eu 4,5 local
run 2026-08-06-egonet-signal-airports-eu 4,5 global
run 2026-08-06-egonet-signal-roman-empire 5 local
run 2026-08-06-egonet-signal-roman-empire 5 global
run 2026-08-06-egonet-signal-wiki-chameleon 4,5 global
run 2026-08-06-egonet-signal-wiki-chameleon 4,5 local
echo "=== EXTENSION BATTERY DONE $(date +%H:%M:%S) ==="
