# Session: Synthetic experiment 1 — planting the homophily-reach mechanism

| | |
|---|---|
| **Started** | 2026-08-12 |
| **Last active** | 2026-08-12 |
| **Status** | active |

## Goal

Causally validate the homophily-reach mechanism inferred from AMAZON_RATINGS: plant a
smooth label field with a dialable correlation length in a synthetic graph, run the
identical instrument, and test the dial-response prediction that the k-profile's plateau
tracks the measured homophily reach.

## Approach

Generator `lib/nodeclass/rgg_homophily_graph`: RGG on the unit torus (n=6000, deg≈8 —
polynomial neighborhood growth, so k=6 bags stay ≤2.4% of the graph and saturation never
confounds the plateau). Labels = 5-quantile bins of a smooth Fourier field with wavelength
λ (homophily dial) + 10% flip noise; faint coupling β scales connection radii by
(1 + β·field) so structure carries a weak label trace (β=0 = negative control). Grid:
λ ∈ {0.10, 0.20, 0.35} × β ∈ {0, 0.03} — 6 graphs, k=1..6 both scopes + node baseline +
permutation floor, 2000 centers, 10 shared splits, macro-F1. Predictions pre-registered in
`notes/predictions.md` BEFORE any classifier run; manipulation checks
(`code/manipulation_checks.py`) gate the runs.

## Key findings

- **P1** (checks): passed after one honest recalibration — β 0.06→0.03 (marginal degree
  η² was too bright), and reach re-defined as the distance-homophily curve's first zero
  crossing (the single-frequency field oscillates). Measured reach 2.7 / 5.1 / 8.2 hops.
- **P4, the money result — CONFIRMED**: last k with a significant paired gain (p<.05)
  tracks measured reach in every series; global β=0.03: k*=2/3/6 vs reach 2.7/5.1/8.2.
  Figure: `figures/synth_money_kstar_vs_reach.pdf`, profiles in
  `figures/synth_profiles_grid.pdf`.
- **P2 REFUTED (headline surprise)**: β=0 controls are far above the floor and also
  plateau at reach — **locality leakage**: overlapping bags act as a position hash and
  positional centralities decode the smooth field with zero planted structural signal.
  The B1−B0 gap (+.016/.031/.046 at plateau, global) is the true planted-coupling effect.
- **P3 confirmed**: ascending profiles, global ≥ local, node baseline just above floor.
- Interpretation: homophily reach causally sets where polling stops paying; on real
  homophilous graphs part of the *level* likely enters via transductive locality rather
  than structural signatures.

## Next steps

- Discuss with Ash: manuscript synthetic section + how to present the locality-leakage
  finding (it reframes part of the Amazon story).
- Possible follow-up: quantify the leak on real data (e.g., a position-scrambled control
  or non-overlapping-bag ablation); remaining synthetic mechanisms (node-encoded,
  relational/motif, density morphing).

## Layout

- `notes/` — predictions (pre-registered) + lab notebook (tracked)
- `code/` — generator wiring, runner, checks, figures (tracked)
- `data/` — datasets (gitignored)
- `logs/` — run logs (gitignored)
- `outputs/` — embeddings, models, checkpoints (gitignored)
- `figures/` — generated plots (gitignored)
