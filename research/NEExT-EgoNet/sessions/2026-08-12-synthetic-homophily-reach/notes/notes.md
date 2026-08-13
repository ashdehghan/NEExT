# Notes

<!-- Running lab notebook: dated entries, newest at the bottom. -->

## YYYY-MM-DD

- 

## 2026-08-12 — manipulation checks (P1) + recalibration

First check round FAILED P1 on two counts (recorded honestly):
- beta=0.06 too bright: marginal deg eta2 .033-.072 vs registered <=~.02 target.
  Recalibrated beta -> 0.03 (eta2 ~ beta^2); B1 cells regenerated.
- Single-frequency Fourier field oscillates: L1 excess negative d3-6, spurious
  rebound at d8 -> naive "excess>.02" reach unreliable. Reach redefined as
  FIRST ZERO CROSSING (interpolated). Registered in predictions.md addendum
  BEFORE any classifier run, incl. refined L1-dip expectation.

Second round PASSES P1 (outputs/manipulation_checks.csv):
  cell    deg_eta2  h_adj  reach0x
  L1_B0   .004      .296   2.83     L1_B1  .007  .298  2.68
  L2_B0   .001      .488   5.44     L2_B1  .023  .487  5.10
  L3_B0   .001      .584   8.14     L3_B1  .020  .585  8.22
- faint: B1 marginals .007/.023/.020 (L2 marginally over .02 — accepted, "~")
- dark controls: B0 <= .004
- reach monotone in lambda, well separated; B0 twins match B1 (coupling does
  not distort homophily). h_adj rises with lambda as expected.

Probes: k=6 bag median 141 nodes (2.4% of graph) -> no saturation confound;
local k=6 feature pass ~2.4 min per 2000 bags. Grid launched:
code/run_all_cells.sh — 6 cells x (khop local, nodefeat, khop global).

Env note: matplotlib (advanced extra) got pruned by a uv auto-sync; board
plots via Plotly, print figures after the grid with --extra advanced sync.

## 2026-08-12 — grid results (all 18 stages, exit 0)

Verdicts on pre-registered predictions:
- P2 REFUTED (the headline): beta=0 controls NOT at floor in either scope.
  L1_B0 local .22->.28, L2_B0 local .21->.33, L3_B0 global .32->.48 (floors
  ~.20). With zero planted coupling, structure cannot carry the label ->
  the signal is LOCALITY LEAKAGE, two channels: (a) overlapping k-hop bags
  make nearby nodes' Wasserstein embeddings nearly identical (position
  hash), (b) positional full-graph centralities (closeness/eigen/pagerank)
  decode the smooth field directly (node_struct .24-.29 at beta=0 despite
  deg eta2 <= .004). Leak grows with wavelength and ALSO plateaus at reach
  (it polls the same field).
- P3 CONFIRMED: B1 ascending profiles both scopes, global >= local
  everywhere, node_struct just above floor (.25/.32/.34).
- P4 (money) CONFIRMED: last-significant-paired-gain k (p<.05, 10 shared
  splits): global B1 2/3/6, global B0 4/5/6, local B0 3/4/5 vs measured
  reach 2.7/5.1/8.2 — monotone in every series. Local B1 5/5/6: L1 k=5
  gain matches the registered oscillation-rebound caveat (positive band
  d7-8 recaptured by wide bags).
- Planted-coupling effect = B1-B0 gap at plateau: +.016/.031/.046 (global),
  grows with lambda (polling aggregates the faint radius trace).

Interpretation for the paper: homophily-reach as plateau-setter is
causally validated (dial-response). Attribution nuance: on homophilous
graphs part of the LEVEL of egonet-embedding performance can enter via
transductive locality (bag identity), not structural signatures — the
beta=0 control isolates that channel; real-data ascending profiles
(Amazon) likely mix both. Manuscript untouched pending discussion.

Artifacts: results.csv (18 stages), outputs/manipulation_checks.csv,
outputs/distance_homophily_curves.csv, figures/synth_profiles_grid.pdf,
figures/synth_money_kstar_vs_reach.pdf, figures/synth_distance_homophily_v2.pdf.

## 2026-08-12 — 10-replicate study (Ash-approved) + ring-by-ring mechanism test

Motivation: Ash challenged "plateau at reach" (visual mismatch) and the
single-draw noise in L1. Two responses:

1. Ring-by-ring prediction test: from the measured agreement curve ALONE,
   predict the k-profile shape. Two aggregation models:
   - naive averaging  S(k)=sum(n_d e_d)/sqrt(sum n_d)  -> predicts DECLINE
   - adaptive weighting S(k)=sqrt(sum n_d e_d^2)       -> predicts saturation
   (code/analysis_polling_prediction.py, single-seed first pass)

2. Replicates: 10 fresh worlds per wavelength (seeds 101-110, whisper ON,
   global sweep + per-replicate checks; code/run_replicates.sh; 30 jobs,
   ~1.5h sequential on the 4-core box). datasets.py now takes _S<seed> ids;
   manipulation_checks.py takes cell ids via argv and merges CSVs.

Replicate check stats (mean±std across 10): reach 2.9±0.2 / 5.9±0.9 /
8.6±0.5 (dial robust across draws); deg eta2 .013/.018/.027.

RESULTS (code/analysis_replicates.py, figures/polling_prediction_replicated):
- Adaptive-polling prediction matches replicate-mean profiles: r = 0.92 /
  0.97 / 0.99. Naive averaging refuted: r = -0.86 / -0.29 / +0.91 (its
  predicted decline never happens). The classifier acts as an adaptive
  poller: the profile SHAPE is predictable from the agreement curve alone.
- L1 dip VERDICT: noise. Across 10 worlds, lift(k4)-lift(k3) = -0.001
  (std .010, p=.70). Gone in the replicate mean; bands smooth.
- Claim sharpened (post-Ash-challenge): not "plateau sits at reach" but
  (a) plateau position moves monotonically with the reach dial, and
  (b) the full profile shape is predicted by adaptive polling of the
  measured agreement curve.

Board figures now board-sized w/ readable fonts + true aspect (plot_board.py).

## 2026-08-13 — full battery + manuscript section

Full replicate battery (Ash-approved): 90 more jobs (B1 local, B0 both
scopes x 10 seeds x 3 lambda), all green — every series in the grid now
10-world. B0 replicate checks: eta2 <= .003 (dark), homophily/reach
indistinguishable from B1 twins. Replicate-median plateau k*:
global B1 3.5/4/6, local B1 4/5.5/6, global B0 4.5/4.5/6, local B0 3/5/6
vs reach 2.9/5.9/8.6 — non-decreasing in all four series.

Manuscript p_4a49e253: new Section "Synthetic Mechanism Validation"
(sections/synthetic.tex, input after analysis) — series framing, generator
+ construction figure, manipulation-check table (replicate mean±std),
setup w/ pre-registered predictions, results (dial response), ring-by-ring
prediction figure, locality-leakage full treatment, established/limitations.
Figures: synth_construction / synth_profiles (all series 10-world, lift
units) / synth_prediction. Compiles 0 errors (15 pp). Multi-agent review
workflow launched (3 lenses + adversarial verify); fixes to follow.

## 2026-08-13 — beta-sweep results (P-B1..B4 verdicts)

All 5x5 sweep worlds + nodefeat endpoints complete (chore done too:
node_struct now exists for all 60 batch-1 replicate worlds).

- P-B1 CONFIRMED, quantitatively: deg eta2 = .001/.014/.150/.379/.575 at
  beta 0/.03/.10/.20/.40 (registered guess ~beta^2: .02/.15/.4 — dead on
  through beta=.2; saturates at .4 under the radius clip). Homophily/reach
  undisturbed through beta=.2; at beta=.4 h_adj dips to .499 and reach
  measurement noisier (7.6+-2.4) — strong coupling perturbs the graph
  itself; noted as a dial limit.
- P-B2 CONFIRMED: node-baseline lift .098/.106/.212/.290/.365 (5 worlds
  each). Note beta=0 node lift ~.10 = pure positional leakage baseline.
- P-B3 CONFIRMED in direction, crossing NOT reached: bag advantage
  .136/.137/.122/.075/.036 — monotone shrink from beta=.03, still positive
  at beta=.4 (generator clip caps beta at .5, so full inversion to the
  Airports sign was out of reach). The two curves cross (node lift
  overtakes bag advantage) between beta=.03 and .10.
- P-B4: to assess in the fuller analysis (profile flattening vs beta).
Figure: figures/morphing_box (house style, boxes over 5 worlds).
