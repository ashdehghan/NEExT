# Notes

<!-- Running lab notebook: dated entries, newest at the bottom. -->

## YYYY-MM-DD

- 
## 2026-08-13 — synthetic anchor results (P-A1 gate)

GATE PASSES, more strongly than registered:
- P-A1: B0 (zero coupling) sits AT the permutation floor at EVERY buffer,
  including B=0 — merely making test centers contiguous regions (no buffer)
  removes the entire leakage signal (~.28 lift under shared splits -> ~0).
  Leakage requires interleaved train/test neighbors; regional holdout kills
  it. Splitter validated causally.
- P-A2: B1 retains a large lift under separation (~.10 at k4/B0, decaying
  to ~.06 at B10 as the train pool shrinks 1400->568) — LARGER than the
  registered ~.02-.03 (the shared-splits B1-B0 gap). Reading: leakage and
  structural signal are REDUNDANT under shared splits (leakage masks the
  structural channel's marginal value); separation reveals the structural
  share directly. Pre-registered numeric guess wrong, direction right —
  recorded honestly.
- 0 skipped splits; train pools reported per B.

Next: amazon + chameleon separation evals queued AFTER the beta sweep
(avoid CPU contention that inflated this run 3x).

## 2026-08-13 — real-data separation results (P-A3 verdicts)

AMAZON (train pool at B=0 is 2,100 = exactly the shared-splits training
size, so no small-sample confound at the headline comparison):
- Global k4 lift: +.026 at B0 (was +.103 shared) — ~75% of the lift was
  transductive locality leakage. node_struct: ~+.01 (was +.088) — ~85-90%
  leakage. Local k4: +.020 (was ~.093).
- The residual ~+.02 structural share (global k3/k4) is stable across
  buffers out to B=10 — consistent with a real but small structural
  signal, and strikingly close to the synthetic B1-B0 gap at the plateau.
- Reading: Amazon's LEVEL was mostly position-matching; the SHAPE story
  (plateau at reach) was validated causally on synthetics and still stands.

CHAMELEON (dense small-world graph — separation barely exists: train pool
700/298/104/7 at B=0/1/2/4; results beyond B=1 are near-degenerate, 574 of
980 split-rows skipped by the guards):
- At B=0: global k2 +.203 (was +.445), node_struct +.188 (was +.442) —
  regionalization alone halves BOTH. The interaction-code signal is partly
  region-specific (or regional class skew); bag-over-node margin at B0
  (+.015) is within noise — the shared-splits +.023 (p=.002) is neither
  preserved nor refuted at 10 repeats.
- Dense-graph limit recorded honestly: the control cannot fully separate
  a graph whose diameter is ~bag scale.

Fix log (one root cause each, no blind retries): node_struct parquet never
persisted -> rebuilt rep from cached source features; degenerate tiny
train sets crashed XGB -> skip guards in evaluate + floor; all-skipped
block crashed concat -> empty guard; one path slip on relaunch.

## 2026-08-14 — Roman separation results (P-A5 verdict)

P-A5 CONFIRMED, cleanly: local k1 lift +.090-.094 under separation vs
+.098 shared (~95% survives), local k2 +.103-.109 vs +.106 (~100%),
global k1 +.060-.073 vs +.076, node_struct +.075-.081 vs +.085 — ALL
essentially flat across B=0..10. Sparse graph separates for free (train
pool 2100 -> 2084 at B=10; 0 skipped splits). The relational-fragment
code is position-free, exactly as the mechanism account predicts: the
same grammatical configurations recur in every region, so a classifier
trained on one region reads fragments in another without loss.
Three-way contrast complete: Roman survives ~fully / Amazon collapses to
~+.02 / Chameleon halves at B=0 and cannot separate further.
