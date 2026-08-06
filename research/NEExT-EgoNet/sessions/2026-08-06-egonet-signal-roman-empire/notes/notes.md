# Notes

<!-- Running lab notebook: dated entries, newest at the bottom. -->

## YYYY-MM-DD

- 
# (session created 2026-08-06; first pass per Ash: local k1/k2 + floor only)

- ROMAN_EMPIRE first pass (79 s): 3000 stratified centers of 22,662 nodes,
  17 classes survive min_count=10 filter (majority share .14). Bags tiny as
  predicted: k1 median 4, k2 median 7 (0.03% of graph) — no saturation.
  Results (local, fall-vl1): k1 .308+-.008, k2 .266+-.012 vs permuted floor
  .100+-.010 (majority .140). Signal 3x chance BUT inversion prediction
  WRONG so far: k1 > k2 even with 4-node bags. Macro-F1 low (.16) — heavy
  class imbalance, small classes missed. AUC .74/.72. Scratchpad: airports
  plots cleared, exp2 figure up (Ash-directed). Next candidates: node_struct
  ceiling, global scope, k=3, walk bags.

- k=3/k=4 extension (Ash; pre-flight reach checked first: k3 median 13, k4
  median 19, max 111 — no blowup risk, ran full k1-4 in 128 s). Results:
  k1 .306, k2 .262, k3 .180, k4 .137 (~majority .140) vs floor .100.
  Clean monotone decay, NO inversion anywhere. Key point: no saturation
  confound here (k4 = 0.08% of graph) — the decay is pure DILUTION: wider
  bags mix in structurally irrelevant context. Same law as airports at the
  opposite density extreme: signal lives in the immediate neighborhood.

- Global-scope run (runner restructured: ONE full-graph feature pass reused
  across k via project_source_features). Pass took 1343 s (~22 min — over my
  1-5 min estimate; culprit feature TBD, suspect lsme/load at 22.6k nodes);
  rest 74 s. Results: global k1 .228, k2 .149, k3 .125, k4 .118 vs local
  .306/.262/.180/.137, floor .112. COMPLETE REVERSAL of airports: global
  loses everywhere here. Interpretation: sparse graph -> global per-node
  features nearly degenerate (most words structurally identical at full-graph
  scale), so bag distributions of global values carry little; the in-bag
  fragment SHAPE is what discriminates. Scope interacts with graph character:
  dense+node-encoded labels -> global robust; sparse+heterophilous -> local
  wins. Paired figure exp2_roman_scopes; board updated (Ash-directed).
