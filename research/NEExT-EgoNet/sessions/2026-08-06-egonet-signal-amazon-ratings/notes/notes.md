# Notes

<!-- Running lab notebook: dated entries, newest at the bottom. -->

## YYYY-MM-DD

- 
# Session created 2026-08-06 (Experiment 4): AMAZON_RATINGS, Roman-twin leg.
# Pre-flight gates: k1/k2/k3 medians 6/28/57 (0.23% max — no saturation);
# 50-bag probe put local k3 at 3.5 min — no trims needed. 3000 centers.

- Five stages complete (~57 min total; feature pass 45.8 min -> cached
  parquet). Floor .303/.304 (majority share .368 — accuracy compressed;
  macro-F1 is the honest lens). node_struct .384.
- khop local .319/.371/.389; khop global .347/.390/.400 — ASCENDING profile
  in both scopes (first ever), global >= local at every k, global k3 .400
  edges past the node baseline (+.016). Walks: local .331/.336/.354,
  global .343/.343/.355 — wide best (reversed locality, matches ascending
  profile).
- Prediction on record mostly WRONG: expected Roman-like (local k1 wins,
  dilution decay). Got: wider-is-better, global>=local, bag>node only at
  k3 global and modestly. Rating class looks weakly structural overall
  (field spans .30-.40); whatever signal exists is diffuse and global.
- Prelim reading: co-purchase rating signal is neither in the node profile
  nor the immediate fragment — it accumulates slowly with neighborhood
  coverage. Fourth regime? Discuss with Ash before manuscript.

- Mechanism-analysis session (2026-08-07, with Ash on the terminal):
  measured eta2 feature map (11x4; airports=degree family .6; roman=two
  blocks flow .3 + shape .27; chameleon all marginals <=.046 vs
  multivariate .635 -> INTERACTION CODE; amazon dark), adjusted homophily
  (+.206/-.045/+.035/+.150), chameleon walk ESS check (ESS/size ~.15,
  10-14% mass on top-20 hubs vs 0.9% census), amazon distance-homophily
  curve (peaks d=2 at .385 vs .271 baseline, above baseline through d=8 ->
  polling frame predicts continued ascent at k4/5; run in flight).
  Manuscript: analysis section added (sections/analysis.tex) + orange eta2
  heatmap figure + ESS table; Amazon results paragraph now forward-refs it.
  lsme==basic_expansion==degree at vlen 1 (footnoted).

- k4/k5 extension landed (local 29 min re-run incl k1-3 jitter re-eval;
  global 56 s on cache). Bag medians k4=111, k5=214 (0.87% of graph).
  Global: .297 -> .302 (k4) -> .300 (k5) — PLATEAU at ~.30, no downturn,
  far below coverage: consistent with distance-homophily reach thinning
  (curve prediction CONFIRMED). Local still climbing shallowly to .299.
  Note: local k1-3 re-run values shifted within noise (multithreaded
  XGBoost nondeterminism, e.g. k1 .211->.203); manuscript table matches
  current artifacts. Manuscript: exp4 table k1-5, figure regenerated,
  Results + analysis verdict sentences updated. pdf_rev 11.
