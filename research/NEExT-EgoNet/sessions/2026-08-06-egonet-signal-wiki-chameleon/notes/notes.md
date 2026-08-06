# Notes

<!-- Running lab notebook: dated entries, newest at the bottom. -->

## YYYY-MM-DD

- 
# Session created 2026-08-06 (Experiment 3): Chameleon full pattern, Ash-approved
# option: 1000 stratified centers after timing probe showed 2.5 h at full n.

- Five chained stages, ~55 min total (local khop 42 min of it; global khop
  35 s thanks to cached 29 s feature pass — small graph).
- Results (floor .197/.224, majority .230): node_struct .635. Local khop:
  k1 .555, k2 .629, k3 .550 — NON-MONOTONE, peaks at k2 (new profile!).
  Global khop: .616/.660/.590 — also peaks at k2, and GLOBAL K2 .660 BEATS
  the node baseline (+.024) — first bag rep to top node features on a dense
  graph. Walks local: tight .454 < default .467 < wide .478 (locality
  ordering REVERSED, another first); walks global: .551/.538/.505 (normal
  ordering), all below khop.
- Prediction on record only half held: node features strong (airports-like)
  but traffic is partly NEIGHBORHOOD-encoded — the 2-hop region's
  composition of true global features carries signal the page's own profile
  lacks. Chameleon is not an airports clone; it's a third regime (dense +
  partially neighborhood-encoded label).

- Figure consolidation (Ash chose Layout A, merged axis, after seeing both
  protos on the board): promoted to lib/nodeclass/figures.py
  (dataset_overview_figure); all three sessions now render one overview
  figure via thin code/plot_results_overview.py wrappers. Manuscript: 4
  figure floats -> 3 (one per dataset), Chameleon subsection added (table
  + overview figure + 4 findings incl. global k2 > node baseline; triptych
  framing closes the section). Compiles clean, pdf_rev 7.
