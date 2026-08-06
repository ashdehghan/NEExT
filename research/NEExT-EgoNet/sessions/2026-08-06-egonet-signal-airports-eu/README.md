# Session: Egonet signal test — Air Traffic Europe (Experiment 1, new research line)

| | |
|---|---|
| **Started** | 2026-08-06 |
| **Last active** | 2026-08-06 |
| **Status** | active |

## Goal

First experiment of the reset research line: is there predictive power in the
egonet induced around a node — enough to classify the node from the egonet
alone? Yes/no signal detection under ONE fixed configuration, no sweeps.

## Approach

- **Dataset:** `AIRPORTS_EUROPE` (399 nodes, 5,993 edges, 4 activity-quartile
  classes → 25% uniform floor). Every node is a center.
- **Pipeline:** 2-hop egonet per node (label = center's class) → 4 structural
  features per member (`degree_centrality`, `clustering_coefficient`,
  `page_rank`, `closeness_centrality`), `feature_vector_length=1`, computed
  inside the egonet → approx-Wasserstein embedding (dim 4) → XGBoost under
  10× stratified 70/30 shared splits (`lib.nodeclass`).
- **Floors:** `majority_floor` and the new `permutation_floor` (same pipeline,
  training labels shuffled — the honest random model; added to
  `lib/nodeclass/evaluate.py` this session, unit-tested).
- Figures follow `lib/containment/plotstyle` (academic style, legend outside
  the axes — standing convention).

## Key findings

- **Signal: yes.** Accuracy 0.463 ± 0.043 vs 0.249 (permuted) / 0.250
  (majority); macro-F1 0.462; OVR-AUC 0.753. Boxes don't overlap the floors
  across 10 splits — `figures/exp1_signal_box.{pdf,png}`.
- Permutation floor lands exactly at chance → no leakage in the harness.
- **Saturation observed, signal survives:** median 2-hop egonet spans 278/399
  nodes (70% of the graph; p90 = 369). Heavily overlapping near-global bags
  still separate activity quartiles.
- Runtime: 14 s end-to-end.

## Next steps

- k=1 comparison (smaller, less saturated bags) — is k=2's coverage helping or
  hurting?
- More features / longer feature vectors; other datasets (AIRPORTS_USA next).
- Compare against a non-egonet baseline (center-node full-graph features) to
  see whether the *neighborhood distribution* is doing work beyond the node
  itself.

## Layout

- `notes/` — running lab-notebook notes (tracked)
- `code/` — scripts and notebooks (tracked)
- `data/` — datasets (gitignored)
- `logs/` — run logs (gitignored)
- `outputs/` — embeddings, models, checkpoints (gitignored)
- `figures/` — generated plots (gitignored)
