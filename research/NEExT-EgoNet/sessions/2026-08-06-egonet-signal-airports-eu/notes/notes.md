# Notes

<!-- Running lab notebook: dated entries, newest at the bottom. -->

## 2026-08-06

- Session created as Experiment 1 of the reset research line (Ash + agent
  design on the terminal). Design locked before running: ONE config, no
  sweeps; box plot with approaches on the x-axis vs floors.
- Added `permutation_floor` to `lib/nodeclass/evaluate.py` (train-label
  shuffle under shared splits) + unit test (`test_permutation_floor_severs_signal`).
  Rationale: majority floor doesn't inherit pipeline artifacts; the permuted
  floor does, so it doubles as a leakage tripwire.
- Run (14 s): acc .463±.043 / F1 .462 / AUC .753 vs floors at .25 & chance.
  Permuted floor mean .249 → harness clean.
- `khop_reach` k=2: median 278/399 nodes (70%), p90 369, max 390. Saturation
  is real and the signal survives it — worth a k=1 follow-up to see whether
  smaller bags sharpen or blur the separation.
- Figure `figures/exp1_signal_box.{pdf,png}` (plotstyle, legend above axes,
  panels (a) accuracy / (b) macro-F1). Pushed PNG to the terminal scratchpad.

- Ash follow-up: drop majority from the figure (keep in ledger), accuracy
  only, add k=1 and k=3. Re-run (22 s): k=1 .536±.024 > k=2 .463 > k=3 .434,
  all >> floor .249. Coverage medians 4% / 70% / 99% — monotone degradation
  with saturation; the signal is local. Figure regenerated (single panel).

- All-features run (feature_list=["all"], 11 features, vlen 1, Wass dim 11,
  tag fall-vl1): k1 .556±.039 (was .536 with 4 feats), k2 .467 (~unchanged),
  k3 .435 (~unchanged), permuted .246. Runtime 2m40s (k1 12s / k2 55s /
  k3 90s — betweenness+load+lsme dominate on big bags). Verdict: +0.02 at
  k=1 (within 1 std), nothing at k>=2 — feature richness doesn't rescue
  saturated bags; ranking unchanged. Figure regenerated; scratchpad rebuilt
  with fresh items (in-place image patches don't refresh assets).

- feature_scope shipped in core lib (45b3353); reviewed diff post-hoc: row
  order matches local mode, indicator-merge guard, no new lint. Session runner
  gained --scope; global runs tagged fall-vl1-glob.
- Global-scope k-sweep (24 s vs 2m40s local): k1 .524, k2 .491, k3 .444 vs
  local .556/.467/.435. DIFFERENT PROFILES: local peaks at k1 and decays with
  saturation; global flatter, wins at k2, tighter boxes (std ~halves at k>=2)
  — features fixed, only bag composition varies. Best cell: local k1.
  Paired figure exp1_signal_box (+ _scopes copy); scratchpad updated (Ash ok).

- Walk-bag runs (Ash-approved tuned trio: tight l5/r.30/mv3, default l10/r.15/
  mv3, wide l20/r.05/mv2; n_walks=100, weighted). Bag medians 30/100/231 —
  exactly the k1(16)-k2(278) gap. Local: tight .519, default .525, wide .472.
  Global: tight .544, default .532, wide .537 — FLAT across sizes; wide
  global beats wide local by +.065 (visit weights keep mass near center as
  bags sprawl). Best cells overall: local k1 .556 > global walk_tight .544.
  Runtimes 74s local / 19s global. Figure exp1_walk_box; scratchpad = two
  plots only (khop + walk), per Ash.

- Node-features baseline (Ash-approved): node's own 11 full-graph features
  straight to XGBoost (no bags/embedding), same splits. Result .592+-.039 —
  BEATS every egonet cell (best .556). Mirrors old PoC on AIRPORTS_USA.
  Added as --family nodefeat; both figures now carry the baseline box
  (dark grey) next to Random. Scratchpad updated (Ash ok); manuscript NOT
  touched. Open question sharpened: where does the neighborhood add signal
  beyond the node itself?
