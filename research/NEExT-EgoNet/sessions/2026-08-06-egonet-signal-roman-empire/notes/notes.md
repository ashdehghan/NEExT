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
