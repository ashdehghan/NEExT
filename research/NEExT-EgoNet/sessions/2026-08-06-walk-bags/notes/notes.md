# Notes

<!-- Running lab notebook: dated entries, newest at the bottom. -->

## YYYY-MM-DD

- 

## C1 head-to-head findings (2026-08-06)

Full table: outputs/headtohead.csv. Metrics: smoothness_ratio = median edge
|dnov| / median random-pair |dnov| (lower = smoother; anomaly-incident edges
excluded), spike_pctile = background percentile of anomaly-median novelty,
hub_pull_rho = spearman(novelty, degree).

1. DEFAULT walk bags (min_visits=1) underperform hop_k2: bigger (median
   266-478 vs 68-83), slower (106-276s vs 10-30s per network), spikes WORSE
   (rings .56 vs .76; infiltrators .77 vs .99), smoothness no better. The
   1-visit fringe dominates the induced subgraph; visit weights shrink its
   mass in the embedding but its FEATURES still swamp the bag distribution
   (weights help: weighted beats unweighted on every spike — but not enough).
2. SIZE-MATCHED walk bags (min_visits=3, median 57-67) FLIP the verdict:
   - rings: spike .801 vs hop_k2 .758 (walk wins the hard case)
   - infiltrators: smoother (.747 vs .807) AND equal spike (.997 vs .991)
     -> the smooth+spiky combination neither hop radius gave us
   - hub confound ~0 everywhere (-0.02/-0.03 vs hop_k2 -0.20/-0.57)
   - cost comparable to hop_k2 (22-25s)
3. ACTION: core default min_visits=1 is a bad default (5x bags, 7x cost,
   worse quality). Amend PR: default min_visits=3 (documented as a noise
   floor relative to n_walks*walk_length visit events).
4. walk_far (restart .05): bigger bags, no consistent gain over .15.
