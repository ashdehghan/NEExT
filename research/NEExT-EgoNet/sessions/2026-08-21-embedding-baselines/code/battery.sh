#!/bin/bash
# Embedding-baselines battery (Ash-approved stages only). One job at a time,
# stop on the first error (on-errors rule), every stage logged under logs/.
# Run from anywhere: paths resolve from this file.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SESSION="$(dirname "$HERE")"
REPO="$(cd "$SESSION/../../../.." && pwd)"
SEP="$REPO/research/NEExT-EgoNet/sessions/2026-08-13-leakage-control/code/run_separation.py"
RUN="$HERE/run_embeddings.py"
mkdir -p "$SESSION/logs"
export PYTHONHASHSEED=0

stage() {
  local tag=$1; shift
  echo "=== $(date +%H:%M:%S) $tag ==="
  (cd "$REPO" && uv run --no-sync python "$@") > "$SESSION/logs/${tag}.log" 2>&1 \
    || { echo "STAGE FAILED: $tag — stopping per on-errors rule (see logs/${tag}.log)"; exit 1; }
  grep -v -i "warning\|deprecat" "$SESSION/logs/${tag}.log" | tail -3
}

STAGES="${1:-all}"
want() { [ "$STAGES" = "all" ] || [[ ",$STAGES," == *",$1,"* ]]; }

# 1. small graphs: seed + dimension sensitivity
want airports  && stage airports_seeds   "$RUN" --dataset airports  --methods all --seeds 42,43,44
want airports  && stage airports_d128    "$RUN" --dataset airports  --methods all --dims 128
want chameleon && stage chameleon_seeds  "$RUN" --dataset chameleon --methods all --seeds 42,43,44
want chameleon && stage chameleon_d128   "$RUN" --dataset chameleon --methods all --dims 128
# 2. large graphs: seed 42, D=16
want roman     && stage roman            "$RUN" --dataset roman     --methods all
want amazon    && stage amazon           "$RUN" --dataset amazon    --methods all
# 3. region holdout on the cached representations
want separation && stage sep_roman       "$SEP" --target roman_emb
want separation && stage sep_amazon      "$SEP" --target amazon_emb
want separation && stage sep_chameleon   "$SEP" --target chameleon_emb
# 4. synthetic calibration anchors (if budget allows)
want synth     && stage synth_b0         "$RUN" --dataset synth_b0  --methods all
want synth     && stage synth_b1         "$RUN" --dataset synth_b1  --methods all
want synth     && stage sep_synth        "$SEP" --target synth_emb
# 5. aggregation, stats, tables, figures
want analysis  && stage collect          "$HERE/collect_results.py"
want analysis  && stage stats            "$HERE/stats_tests.py"
want analysis  && stage tables           "$HERE/make_tables.py"
want analysis  && stage fig_overview     "$HERE/plot_baselines_overview.py"
want analysis  && stage fig_separation   "$HERE/plot_baselines_separation.py"
echo "=== BATTERY DONE $(date +%H:%M:%S) ==="
