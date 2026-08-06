# Session: walk-bags

| | |
|---|---|
| **Started** | 2026-08-06 |
| **Last active** | 2026-08-06 |
| **Status** | done |

## Goal

Does random-walk bag construction (feature/random-walk-egonets, PR #14) buy
what the theory promised: a smooth score field WITHOUT flattening the anomaly
spikes (C1, landscape harness), and better containment margins over size-only
(C2, phase-1 subset)? Runs on a LOCAL integration branch (dev + feature);
research tree lands on dev, core stays in the PR.

## Approach

C1 head-to-head: 3 landscape networks x 5 bag variants {hop_k1, hop_k2,
walk(0.15), walk_far(0.05), walk_unweighted}; every node embedded in one
space; metrics = smoothness ratio (edge/random |dnovelty|), spike percentile
at d=0, d=1 percentile (halo), hub-pull rho(novelty, degree).
C2: 12 phase-1 cells (er/ba x tail/clique x pi {.01,.02,.05}) with walk bags
under the audited protocol; representations wasserstein(weighted),
pooled_all(+min/p10, weighted mean), pooled_max, size_only; margins compared
against the FROZEN phase-1 hop results (not rerun).

## Key findings

**C1 (novelty-field properties):** default walk bags (min_visits=1) LOSE to
hop_k2 — the one-visit fringe balloons bags 5x, slows features 7x, and its
structural features swamp the bag distribution (weights help but not enough).
Size-matched walk bags (min_visits=3, median 57-67 nodes) FLIP it: rings
spike .80 vs .76, infiltrators smoother (.75 vs .81) at equal spike (.997),
hub confound ~0 everywhere vs hop_k2's -0.2..-0.57, cost comparable.
The smooth+spiky+degree-neutral combination — delivered, but only with the
noise floor. -> PR #14 default amended to min_visits=3 (commit 0a1792f).

**C2 (containment margins, 12 phase-1 cells vs frozen hop results):**
roughly tie on 6 cells, walk wins 2 (ba_clique .02 +.025, ba_tail .05 +.065),
hop wins 3 big ones (ba_tail .02 -.29, er_tail .01 -.13, ba_clique .05 -.11).
Mechanism: walk membership is itself degree-biased — walks rarely step down
into degree-1 dangling tails (1/deg chance) so tails are under-contained,
while dense structures are over-visited (cliques mildly helped). Bag
construction should MATCH the anomaly's relationship to the walk measure;
the two constructions are complementary, not ordered.

Data: outputs/headtohead.csv, outputs/containment_comparison.csv,
outputs/<run>/ per-run artifacts. Core feature: PR #14
(feature/random-walk-egonets, incl. the amended default).

## Next steps

- Ash reviews PR #14 (core + workbench + amended default).
- Landscape/walk direction decision still open (novelty-ranking cascade vs
  big-footprint gradients) — now informed: tuned walk bags give the smooth
  degree-neutral field the walk idea wanted.
- Complementarity idea: hop bags for point anomalies, walk bags for dense
  structures — a mixed-construction ensemble is the obvious next experiment.

## Layout

- `notes/` — running lab-notebook notes (tracked)
- `code/` — scripts and notebooks (tracked)
- `data/` — datasets (gitignored)
- `logs/` — run logs (gitignored)
- `outputs/` — embeddings, models, checkpoints (gitignored)
- `figures/` — generated plots (gitignored)
