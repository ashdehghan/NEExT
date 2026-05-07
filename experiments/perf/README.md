# Custom Feature Performance Experiments

This sandbox is for diagnosing NEExT custom node feature performance before
deciding on framework API changes.

The harness generates deterministic Reddit-like synthetic graph families and
runs custom feature functions through the public `NEExT.compute_node_features()`
path. It records run-level timings, per-graph feature timings, result shapes,
fingerprints, warning counts, and coarse memory indicators.

## Quick Start

From the repo root:

```bash
python3 experiments/perf/custom_feature_benchmark.py --preset smoke
```

More useful local run:

```bash
python3 experiments/perf/custom_feature_benchmark.py \
  --preset mixed \
  --features cheap,triangles,ego_graphlet \
  --n-jobs 1,2,-1 \
  --backends loky,threading \
  --include-combined
```

Results are written to `experiments/perf/results/`, which is ignored by git.

## Workloads

- `cheap`: cheap degree-derived feature; mostly measures NEExT, DataFrame,
  scheduling, and serialization overhead.
- `triangles`: medium NetworkX triangle/clustering feature.
- `ego_graphlet`: expensive graphlet-style proxy using per-node radius-2 ego
  graph counts.

Presets:

- `smoke`: very small sanity check.
- `quick`: small mixed-size run.
- `mixed`: many small graphs plus a few larger sparse graphs.
- `stress`: larger local stress run.

## Backends

- `loky`: NEExT's default process backend. This remains the notebook-safe
  choice for notebook-defined custom functions, but it can spend significant
  time serializing large graphs and feature functions.
- `threading`: NEExT's thread backend for cases where process serialization
  dominates runtime. It avoids graph pickling between workers, but Python
  GIL-bound custom code may not speed up.

## Output Files

- `dataset_summary.json`: graph sizes, densities, sample sizes, and optional
  pickle-size estimates.
- `run_summary.csv`: one row per benchmark scenario.
- `feature_events.csv`: one row per custom feature call per graph.

Use matching fingerprints and result shapes to confirm equivalent outputs across
parallel settings before comparing timings.
