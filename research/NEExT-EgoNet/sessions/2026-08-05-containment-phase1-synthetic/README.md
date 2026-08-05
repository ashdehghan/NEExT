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

Sweep complete: 60/60 cells; 10 degenerate under the uniform <10-minority-bags
rule (6 single-class + 4 reclassified in the 2026-08-05 audit — the guard
entered mid-sweep; 9 of 10 are hub π≥0.02, as 1−(1−π)^s predicts).
**Gate to phase 2: PASS.**

- 20/50 valid cells: best fair rep beats size_only by ≥ +0.05 AUC (corrected
  in audit; two former headline cells rested on 1–3 minority test bags/split).
  Peak: BA/tail π=.02 k=1 → 0.87 vs 0.51 (+0.35); BA/tail π=.01 → +0.17 at
  both radii.
- Anomaly difficulty ordering (mean margin, valid cells): tail +0.08 >
  hub +0.04 > clique +0.03. Tails = point anomalies → containment-friendliest. Hubs
  saturate early (contained in every neighbor's bag; pos_rate far above the
  uniform curve). Cliques invisible at k=1 (don't fit the bag), emerge at
  k=2 (BA π=.05: 0.67→0.91).
- **Dilution measured:** BA/tail π=.01: 0.89 (k=1, med bag 12) → 0.74 (k=2,
  med bag 109). Radius rule splits by signature: match k to the anomaly's
  scale, then check π·s for balance.
- **Pooling verdict (honest):** pooled_all wins 21 valid cells, pooled_max 17,
  wasserstein 12. Pooled moments suffice for extremum signatures; wasserstein
  earns its keep in hard low-signal regimes (ER/tail k=2: leads at every π).
  Phase 2 contract: wass + pooled co-primary.
- node_oracle (leak-free, audit-fixed): median 0.93, range 0.54–1.00 —
  readily but not trivially detectable given label-scarce centers; at k=1
  low π the oracle sits BELOW the fair bag methods (they see ~10× more
  positive examples). Margins measure what survives bag-level compression.

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
