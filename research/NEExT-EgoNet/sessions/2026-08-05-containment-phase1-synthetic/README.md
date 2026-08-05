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

Sweep complete: 60/60 cells (6 saturation-degenerate, all hub π≥0.02 — as the
1−(1−π)^s model predicts). **Gate to phase 2: PASS.**

- 23/54 non-degenerate cells: best fair rep beats size_only by ≥ +0.05 AUC.
  Peaks: BA/tail π=.02 k=1 → 0.87 vs 0.51 (+0.35); BA/tail π=.005 k=1 →
  0.81 vs 0.54 (+0.27); BA/hub π=.02 k=2 → 0.89 vs 0.66 (+0.24).
- Anomaly difficulty ordering (mean margin): tail +0.10 > hub +0.05 >
  clique +0.03. Tails = point anomalies → containment-friendliest. Hubs
  saturate early (contained in every neighbor's bag; pos_rate far above the
  uniform curve). Cliques invisible at k=1 (don't fit the bag), emerge at
  k=2 (BA π=.05: 0.67→0.91).
- **Dilution measured:** BA/tail π=.01: 0.89 (k=1, med bag 12) → 0.74 (k=2,
  med bag 109). Radius rule splits by signature: match k to the anomaly's
  scale, then check π·s for balance.
- **Pooling verdict (honest):** pooled_all wins 21 cells, pooled_max 19,
  wasserstein 14. Pooled moments suffice for extremum signatures; wasserstein
  earns its keep in hard low-signal regimes (ER/tail k=2: leads at every π).
  Phase 2 contract: wass + pooled co-primary.
- node_oracle ≥0.98 nearly everywhere — planted anomalies trivial at
  full-graph cost; margins measure what survives bag-level compression.

Artifacts: `results.csv` (all cells × reps × splits), `outputs/<run_id>/`
(config/metrics/bag_predictions/bag_table), `figures/` via
`code/plot_results.py` (CSV-only regeneration). Manuscript updated:
`sections/results_phase1.tex` + 4 figures, compiles clean.

## Next steps

- Phase 2 (next session): BOOKS → REDDIT_PYGOD → MINESWEEPER, same matrix,
  reuse lib/containment unchanged; add dataset loader to lib (copy PoC
  `datasets.py` pattern).
- Consider per-anomaly-type feature ablation informed by pooling verdict.

## Layout

- `notes/` — running lab-notebook notes (tracked)
- `code/` — scripts and notebooks (tracked)
- `data/` — datasets (gitignored)
- `logs/` — run logs (gitignored)
- `outputs/` — per-cell run artifacts (gitignored)
- `figures/` — generated plots (gitignored)
