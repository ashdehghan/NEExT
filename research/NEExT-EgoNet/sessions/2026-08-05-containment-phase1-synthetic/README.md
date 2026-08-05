# Session: containment-phase1-synthetic

| | |
|---|---|
| **Started** | 2026-08-05 |
| **Last active** | 2026-08-05 |
| **Status** | active |

## Goal

Phase 1 of the containment-search program: map the synthetic **detectability
surface** — when can a bag-level representation tell "this k-hop egonet
contains a planted anomaly" from "it doesn't", as a function of prevalence,
radius, base topology, and anomaly structure?

Reframing under test: egonet = MIL bag, label = OR over members
(`y_contains`), vs the PoC's center-label question. Thesis: distributional
embeddings (Wasserstein) are coarse-resolution instruments and should win
coarse-resolution questions. Full design rationale: manuscript NEExT-EgoNet
(Manuscript app `p_134274d1`), framework + experimental-design sections.

## Approach

- Grid: {er, ba} × {hub, clique, tail} × π ∈ {.005, .01, .02, .05, .10} × k ∈ {1, 2} = 60 cells
- n = 3000 nodes, 800 sampled centers per cell; igraph backend, n_jobs=-1
- Representations per cell: wasserstein (dim 16), pooled_all (mean+max+p90),
  pooled_max, size_only (confound baseline), node_oracle (full-graph quality
  reference)
- Protocol: XGBoost, 10× stratified 30% hold-out, split seeds shared across
  representations; ROC-AUC / PR-AUC primary; degenerate cells recorded with
  label balance (saturation data, not failures)
- Shared code: `../../lib/containment/` (unit-tested); runner
  `code/run_sweep.py` (resumable, `--smoke` mode); `code/plot_results.py`
  regenerates all figures from saved CSVs only

**Gate to phase 2:** some region of the surface where wasserstein or pooled
beats size_only by a clear margin. No region ⇒ the containment signal is bag
size all the way down ⇒ negative result, program stops.

## Key findings

(pending sweep completion — see `logs/sweep.log`, `results.csv`,
`outputs/<run_id>/`)

## Next steps

- Fold results into manuscript `sections/results_phase1.tex` + figures
- Gate decision → phase 2 (BOOKS, REDDIT_PYGOD, MINESWEEPER contrast)

## Layout

- `notes/` — running lab-notebook notes (tracked)
- `code/` — scripts and notebooks (tracked)
- `data/` — datasets (gitignored)
- `logs/` — run logs (gitignored)
- `outputs/` — per-cell run artifacts (gitignored)
- `figures/` — generated plots (gitignored)
