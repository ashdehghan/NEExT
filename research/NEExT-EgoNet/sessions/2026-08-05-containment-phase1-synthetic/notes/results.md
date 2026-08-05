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
