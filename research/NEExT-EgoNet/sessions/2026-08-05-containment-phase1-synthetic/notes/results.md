# Phase-1 results log

Full analysis lives in README.md (Key findings) and the manuscript
(`sections/results_phase1.tex`). This note records analysis details that
belong to the lab notebook rather than the paper.

## Margin table (best fair rep − size_only, mean ROC-AUC over 10 splits)

23/54 non-degenerate cells ≥ +0.05. Top cells:

| cell | wass | pooled_all | pooled_max | size_only | margin |
|---|---|---|---|---|---|
| ba/tail π=.02 k=1 | .827 | .865 | .807 | .514 | +.351 |
| ba/tail π=.005 k=1 | .807 | .786 | .790 | .537 | +.270 |
| er/tail π=.005 k=1 | .622 | .569 | .652 | .383 | +.269 |
| ba/hub π=.02 k=2 | .848 | .848 | .893 | .655 | +.238 |
| ba/tail π=.01 k=1 | .890 | .887 | .883 | .716 | +.174 |
| ba/clique π=.05 k=2 | .863 | .906 | .898 | .740 | +.165 |

Win counts among fair reps: pooled_all 21, pooled_max 19, wasserstein 14.
Mean margin by anomaly: tail +.081/.094 (wass/pooled), hub +.009/.049,
clique +.008/.027.

## Oddities & caveats worth remembering

- `er/tail π=.005 k=1` has size_only at **0.383** — below chance. Size is
  anti-correlated there (tail bags are *small*), and XGBoost on 2 features
  with 6 positive bags in test is high-variance. Treat sub-0.5 baselines as
  noise, not signal.
- Six degenerate cells are ALL hub-type at π≥.02: a hub is a member of every
  neighbor's bag, so hub containment saturates far above the uniform curve.
  For hub-like targets the informative regime is π ≤ .01 or k=0-ish radii.
- `node_oracle` = 0.5 in two cells (er/hub π≥.05 k=1): saturated cells where
  <2 negative *centers* exist for training — oracle degenerates with the
  labels; recorded as-is.
- Runtime: full 60-cell sweep ≈ 55 min wall-clock total across the two
  launches (k=1 cells 40-65s; k=2 45-340s, ER/hub π=.1 k=2 the worst at
  338s). n=3000, 800 centers, n_jobs=-1, igraph.
- Crash mid-sweep (fixed): stratified split needs ≥2 minority members;
  evaluate.py now marks minority<10 as degenerate (MIN_MINORITY_BAGS).

## Ideas parked for later phases

- Phase-2 feature ablation: which of the 11 features carry tail vs clique
  signal (feature importances are in each run's model — not persisted; add
  if needed).
- Wasserstein wins concentrate in low-signal regimes — check whether that
  holds on real data before claiming it in the trim-down.
- Consider k=0.5-style "center + sampled neighbors" bags for hub-type
  targets to dodge early saturation.

## Audit remediation (2026-08-05, session 3)

**Oracle leakage fixed.** The original node_oracle trained on ALL bag centers
(incl. test bags). Replaced by `evaluate_node_oracle` (lib/containment/
evaluate.py): per-split training on train-bag centers only, AND test bags
scored on non-training-center members only — the regression test showed that
without the exclusion, memorized member labels still leak through bag overlap
(0.909 AUC on random labels; now ~chance). All 60 cells' oracle rows
regenerated (`code/rerun_oracle.py`; rebuilt bags verified identical to
stored; fair rows asserted byte-untouched).

**Clean oracle:** median 0.933, range 0.539–0.999 over valid cells (was
"≥0.98 almost everywhere" under the leak). At k=1 low π the label-starved
oracle (~4–8 positive centers) sits BELOW the fair bag-level methods — they
learn from ~10× more positive bags. The old 0.5/0.736 saturated-cell values
are gone; those cells now report degenerate_oracle_training or are excluded.

**Uniform degeneracy rule applied retroactively.** MIN_MINORITY_BAGS=10
entered mid-sweep; 4 early cells with minority<10 carried "ok" rows:
ba_tail_p0.005_k1 (9 minority — was the +.270 headline cell),
ba_hub_p0.02_k2 (5 — was +.238), er_hub_p0.1_k1 (all-0.5 row),
er_tail_p0.005_k1 (the sub-chance size_only cell). All four reclassified
degenerate. **Corrected headline: 20 of 50 valid cells ≥ +0.05; peak +0.351
(ba_tail_p0.02_k1) unchanged. Ordering (tail .084 > hub .039 > clique .029)
and pooling verdict (pooled_all 21 / pooled_max 17 / wass 12) unchanged.**
Figures regenerated with the rule applied (plot_results.load_results).

Manuscript updated accordingly (abstract, results_phase1 incl. new tab:phase1
rows, protocols appendix, limitations: +transductive preprocessing, +split
dependence). Phase-2 to-do reaffirmed: add min/p10 to pooled stats.
