# Session: Egonet embeddings as node-level features (9-dataset benchmark)

| | |
|---|---|
| **Started** | 2026-08-06 |
| **Last active** | 2026-08-06 |
| **Status** | parked (2026-08-06 — benchmark never ran to completion; research line reset, kept for the record) |

## Goal

Revisiting the original PoC question, properly powered: is the NEExT Wasserstein
embedding of a node's egonet a good *per-node* representation? Against what, and
*when*? The goal is a label-type taxonomy: which label mechanisms favor
graph-level (egonet) embeddings vs classic node embeddings vs plain structural
features — extending the resolution-matching thesis to the node-classification
setting.

## Approach

- **Datasets (9)**, chosen by hypothesized label mechanism (see `code/config.py`):
  structural-role (AIRPORTS_USA, ROMAN_EMPIRE), local-property (MINESWEEPER,
  TOLOKERS), community/homophily (CORA, POLBLOGS, EMAIL_EU_CORE), outlier
  (BOOKS, REDDIT_PYGOD). ~3000 stratified sampled centers per dataset; every
  method evaluated on the same sampled node set under shared splits
  (10× stratified 70/30, XGBoost).
- **Egonet side**: 5 constructions — hop k=1, hop k=2 (dense-graph guarded),
  random-walk bags at 3 parameter settings — each represented as
  approx-Wasserstein dim 16 AND pooled member features (mean/max/p90/min/p10,
  visit-weighted mean on walk bags). Size-only kept as a diagnostic floor.
- **Baselines**: karateclub 1.3.3 (installed `--no-deps`; every method
  smoke-verified) — DeepWalk, Node2Vec (p=q=1), Node2Vec-BFS (p=.25, q=4),
  Walklets, GraphWave (structural), Role2Vec (structural); plus NEExT
  structural features on the full graph (the PoC winner), degree+core floor,
  majority floor. struc2vec proper has no reputable maintained implementation —
  GraphWave/Role2Vec stand in for the structural-embedding family.
- **Metrics**: accuracy + macro-F1 + macro-OVR-AUC; binary datasets add
  ROC-AUC / PR-AUC (headline for the imbalanced outlier tasks).
- **Shared lib**: `lib/nodeclass/` (new, unit-tested) + reused
  `lib/containment` splits/runio/representations/plotstyle.

Reproduce:

```
uv run python code/run_benchmark.py --smoke     # 5-min end-to-end check
uv run python code/run_benchmark.py             # full matrix (resumable)
uv run python code/plot_results.py              # figures from outputs/ only
```

Every cell persists its full representation matrix (parquet), per-split
predictions with class probabilities, per-stage timings, and config — any
post-hoc analysis replots from `outputs/` without recomputation.

## Key findings

(pending — see notes/results.md)

## Next steps

(pending)

## Layout

- `notes/` — running lab-notebook notes (tracked)
- `code/` — scripts and notebooks (tracked)
- `data/` — datasets (gitignored)
- `logs/` — run logs (gitignored)
- `outputs/` — per-cell run dirs, representation parquets, node tables (gitignored)
- `figures/` — generated plots (gitignored)
