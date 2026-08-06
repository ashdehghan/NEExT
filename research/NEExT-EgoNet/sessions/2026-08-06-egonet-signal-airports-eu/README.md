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

- **Signal: yes, and smaller bags win.** Accuracy vs the permutation floor
  (0.249 ± 0.061): k=1 **0.536 ± 0.024**, k=2 0.463 ± 0.043, k=3
  0.434 ± 0.043 — every k clears the floor decisively, and accuracy falls
  monotonically with k (`figures/exp1_signal_box.{pdf,png}`).
- Coverage explains the ordering: median egonet = 16 nodes (4% of graph) at
  k=1, 278 (70%) at k=2, 394 (99%) at k=3. As bags saturate toward the whole
  graph the embeddings blur together — the signal is local.
- Permutation floor lands exactly at chance → no leakage in the harness
  (`permutation_floor` added to `lib/nodeclass` this session, unit-tested).
- Per Ash: figure reports accuracy only, permutation floor as the baseline;
  majority floor + macro-F1/AUC stay recorded in `outputs/` (majority's
  macro-F1 of 0.10 is an artifact of single-class prediction, misleading on
  a near-balanced 4-class task).
- Runtime: 22 s end-to-end for the full k-sweep.

## Next steps

- Center-node-features baseline (full-graph features of the center only) to
  isolate what the neighborhood distribution adds beyond the node itself.
- More features / longer feature vectors; other datasets (AIRPORTS_USA next).
- k=1 looks like the default construction for follow-ups on dense graphs.

## Layout

- `notes/` — running lab-notebook notes (tracked)
- `code/` — scripts and notebooks (tracked)
- `data/` — datasets (gitignored)
- `logs/` — run logs (gitignored)
- `outputs/` — embeddings, models, checkpoints (gitignored)
- `figures/` — generated plots (gitignored)
