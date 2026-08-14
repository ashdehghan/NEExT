# Pre-registered predictions — Phase 3 Part A: separated-split leakage control
Registered 2026-08-13, BEFORE any separated-split evaluation runs.

Method: region-holdout splits (lib/nodeclass/separated_splits.py) — test
centers grown as 3 contiguous BFS regions (~30% of centers), train centers
discarded within buffer B hops of any test center; B in {0,1,2,4,6,8,10},
10 region-seed repeats; permutation floor under identical splits. All
representations are the CACHED artifacts from the source sessions.

- P-A1 (method gate): synthetic L2_B0 (zero coupling) global-scope bags
  COLLAPSE toward the permutation floor as B grows past the bag-overlap
  scale (B >= 2k). If B0 does not collapse, the splitter is broken — stop.
- P-A2: synthetic L2_B1 retains a residual lift at large B comparable to
  the B1-B0 gap under shared splits (~0.02-0.03 macro-F1) — the planted
  structural share survives separation.
- P-A3 (the real-data measurement, direction open):
  * Amazon: the ascending global profile shrinks by some Delta under
    separation; Delta is the quantity of interest. If Delta ~ the whole
    lift, the ascending profile was mostly locality; if small, structural.
  * Chameleon: node_struct largely survives (own-feature interaction code,
    not neighbor identity); global k2's +0.023 margin over node_struct is
    the open question.
- P-A4 (accounting): larger B shrinks the train pool; results reported
  with retained-center counts; skipped splits (missing train class)
  reported, never silently dropped.

## P-A5 pre-registration: Roman Empire separation (2026-08-14, BEFORE the run)

The local-fragment mechanism (analysis: a word's role is written in the
SHAPE of its 4-7 word fragment, not in where the fragment sits) predicts:
- Roman's LOCAL k1-2 lift largely SURVIVES separation — fragment shape is
  a repeating grammatical pattern, not a positional signature; the same
  configurations recur in every region of the word graph.
- The GLOBAL scope stays collapsed/near-floor as under shared splits.
- Sparse graph (deg 2.9, 22.6k nodes) -> regional separation is feasible
  (unlike Chameleon); expect usable train pools out to B >= 4; 18
  imbalanced classes may cost some skipped splits at large B (guards
  count them).
