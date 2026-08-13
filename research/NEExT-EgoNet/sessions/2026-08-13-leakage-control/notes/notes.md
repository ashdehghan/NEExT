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
