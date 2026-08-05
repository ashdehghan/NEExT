# Notes

<!-- Running lab notebook: dated entries, newest at the bottom. -->

## YYYY-MM-DD

- 

## Smoothness mechanism test (2026-08-05, follow-up brainstorm with Ash)

Q: why is the field salt-and-pepper — shouldn't adjacent nodes' bags embed nearby?
Test: per-edge Jaccard(bag_u, bag_v) vs |Δnovelty| (anomaly-incident edges excluded),
bags reconstructed from persisted edges.csv (sizes verified vs bag_table).
Code: code/analyze_smoothness.py → outputs/smoothness_stats.csv,
figures/smoothness_mechanism.{pdf,png}.

Findings:
- Median Jaccard at k=1 ≈ 0.12–0.14 (!): adjacent bags share ~1/8 of their mass.
  The "side-by-side ⇒ same subgraph" premise simply fails in sparse graphs.
- Overlap→smoothness confirmed in direction everywhere: ρ ≈ −0.19..−0.22 (k=1),
  −0.06..−0.10 (k=2), all p<1e-5. More overlap → smaller jump.
- The k=1→k=2 smoothness flip is a REGIME shift: k=2 edge overlap distribution
  centered at J≈0.36 (LFR; ER k=2 only 0.15 — community structure creates overlap).
- Within-regime ρ is modest ⇒ second mechanism: small-sample variance. 9-member
  bags sample a spatially uncorrelated, heavy-tailed feature field; single-member
  churn (esp. betweenness outliers under z-scoring) dominates residual jumps.

Conclusion: overlap sets the smoothness regime; sampling variance sets roughness
within it. Fix ranking: (1) distance-weighted bag membership (attacks both causes);
(2) post-hoc score diffusion (cheap, treats symptoms). Ash to pick direction.
