# Pre-registered predictions — synthetic homophily-reach experiment
Written 2026-08-12, BEFORE any pipeline run on these graphs.

Grid: RGG torus n=6000, mean deg ~8; wavelength L1=0.10 / L2=0.20 / L3=0.35;
coupling B0=0.0 / B1=0.06; labels = 5 field quantiles + 10% flip noise.

P1 (manipulation checks): marginal eta2 of every feature <= ~0.02 at B1 and
   ~0 at B0; distance-homophily excess decays with a length that increases
   L1 < L2 < L3; degree independent of label at B0.
P2 (negative control): at B0, EVERY method (node baseline, all bags, both
   scopes) is statistically at the permutation floor (paired t, p > 0.05).
P3 (B1 signature): ascending macro-F1 radius profile in both scopes;
   global >= local at every k; node baseline slightly above floor.
P4 (MONEY): the profile's plateau position k* tracks the measured homophily
   reach: k*(L1) < k*(L2) < k*(L3), with gains beyond k* ~ 0 while bags
   remain far below graph coverage (<10%).

## Addendum after manipulation checks (still BEFORE any classifier run)
- beta recalibrated 0.06 -> 0.03 to meet the registered faintness target
  (marginal deg eta2 <= ~0.02); B1 cells regenerated.
- The single-frequency Fourier field OSCILLATES: homophily excess crosses
  zero and goes negative at ~half-wavelength (clearest on L1: negative
  d3-6, rebound d8). Reach is therefore measured as the FIRST ZERO
  CROSSING: L1 ~2.5, L2 ~5.5, L3 ~7.5 hops (monotone in wavelength, as
  required). Refined expectation for P4: on L1 the profile may DIP after
  its early plateau (anti-correlated ring pollutes wide bags) — a sharper
  version of the polling prediction, noted here before running.

## Phase 3 pre-registration: beta-sweep regime morphing (2026-08-13, BEFORE runs)

Design: lambda fixed at 0.20 (L2); beta in {0, .03, .10, .20, .40}; seeds
101-105 SHARED across beta (same points+field per seed -> paired across
beta); khop both scopes + nodefeat + floor per world. Plus chore: nodefeat
on all 60 batch-1 replicate worlds.

- P-B1 (checks): marginal degree eta2 rises steeply with beta (~beta^2:
  ~.02 / ~.15 / ~.4 at .03/.10/.20+, saturating under the radius clip);
  homophily h_adj and measured reach UNDISTURBED across beta.
- P-B2: node baseline rises with beta from just-above-floor toward a
  strong ceiling (Airports-like readout of the node's own connectivity).
- P-B3 (money): bag advantage = best bag minus node baseline is positive
  in the faint regime and shrinks toward zero/negative as beta grows —
  the Amazon regime morphs into the Airports regime along one dial.
- P-B4: the global k-profile flattens earlier as beta grows (per-node
  signal loud enough that breadth stops mattering sooner).
