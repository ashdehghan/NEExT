# Session: egonet-poc

| | |
|---|---|
| **Started** | 2026-07-31 |
| **Last active** | 2026-07-31 |
| **Status** | active |

## Goal

Proof of concept: does classifying a node by the **embedding of its k-hop egonet** work?
Decompose a single labeled graph into one egonet per node (center node's label = egonet's
graph label), embed egonets with NEExT's structural-feature + Wasserstein pipeline, and
classify — versus honest baselines that isolate what the egonet embedding adds.

## Approach

- Dataset: **AIRPORTS_USA** (HF anomalypoint/NEExT; 1190 nodes, 13.6k edges,
  `activity_quartile`, 4 classes, structurally-driven labels). Structural-only (no native
  node features enter the egonets).
- Sweep k ∈ {1, 2}; all nodes as centers (`sample_fraction=1.0`, seed 13).
- All 11 structural node features (`feature_vector_length=3`) → `approx_wasserstein`
  embeddings (dim 16, seed 42) → XGBoost, 10× stratified 70/30 hold-out (seed 42+i).
- Baselines under the identical split protocol: **majority class** and
  **center-node structural features** (same 11 features on the full source graph, no
  egonet decomposition).

## Key findings

See [notes/results.md](notes/results.md).

## Next steps

- Filled in as the session progresses.

## Layout

- `notes/` — running lab-notebook notes + committed results summary (tracked)
- `code/` — `datasets.py` (cached HF loader), `run_experiment.py` (tracked)
- `data/` — cached HF downloads (gitignored)
- `logs/` — run logs (gitignored)
- `outputs/` — results.csv, feature importances, config.json (gitignored)
- `figures/` — egonet size distributions etc. (gitignored)
