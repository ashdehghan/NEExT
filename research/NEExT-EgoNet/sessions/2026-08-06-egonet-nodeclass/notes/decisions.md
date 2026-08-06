# Decisions log — egonet-nodeclass session

## karateclub install (2026-08-06, Phase 0 gate)

- Plain `uv pip install karateclub==1.3.3` would downgrade networkx 3.5→2.6.3,
  numpy 1.26→1.22, pandas 2.3→1.3 (verified via `--dry-run`) — rejected.
- Installed `karateclub==1.3.3 --no-deps`, then `gensim 4.4.0, pygsp,
  python-louvain, decorator` (resolver-clean additions; smart-open/wrapt came
  along). Core stack untouched (verified: NEExT 0.3.10 imports, nx 3.5,
  np 1.26.4, pd 2.3.1).
- Smoke on `karate_club_graph` (shape, finite, same-seed determinism ×2):
  - DeepWalk / Node2Vec / Walklets / Role2Vec: OK, deterministic (workers=1).
  - GraphWave: FAILED with pygsp 0.6.1 (`Heat.__init__() ... 'tau'` — API
    rename). Pinned **pygsp 0.5.1** (the version karateclub was built
    against): OK, deterministic. pygsp 0.5.1 is 2017-era but is only used by
    GraphWave, and emits only SyntaxWarnings under py3.12.
- **Verdict: all 5 methods usable, none dropped.** karateclub is NOT added to
  pyproject (research-env-only dependency); recorded here + in the runner's
  per-cell config (`karateclub_version`).
- struc2vec proper: no reputable maintained package (reference impl is
  Python-2-era; shenweichen/GraphEmbedding unmaintained). GraphWave + Role2Vec
  are the structural-embedding representatives. Decided with Ash.

## Guard policy (fixed BEFORE the sweep, applied uniformly)

- Rare-class filter: a class needs ≥10 sampled centers, applied once per
  dataset before sampling and re-applied to the sample — never mid-sweep.
- hop_k2 dense guard: skip (status="guarded") when median k=2 neighborhood
  size over sampled centers > max(2000, 0.10·n). Reach stats persisted either
  way (`<ds>__hop_k2_reach.json`).

## Dimension fairness

- dimensions=16 wherever the method has a true bottleneck knob (DeepWalk,
  Node2Vec ×2, Walklets 4×4, Role2Vec) — matches Wasserstein-16.
- GraphWave's width is 2·sample_number characteristic-function samples, not a
  learned bottleneck; truncating to 8 samples would cripple the method more
  than fairness demands. It runs at its natural 200 and `n_features` is
  reported everywhere. PCA-16 sensitivity cell only if its verdict becomes
  load-bearing.
- Egonet pooled spans ~165 columns (5 stats × 33 feature cols) — comparison
  is method-level under a fixed classifier and shared splits, not
  dimension-matched everywhere. Stated in the notes/discussion.

## walk_tight amendment (before any walk_tight cell entered the final matrix)

- Original walk_tight (wl=5, rp=.30, **min_visits=5**) produced a single-node
  egonet on EMAIL_EU_CORE (high-degree center: 100 short walks spread too thin
  for any neighbor to reach 5 visits) → ZeroDivisionError inside NEExT's
  feature layer. On TOLOKERS (avg deg 88) this would be epidemic, not rare.
- Amended to **min_visits=3** — same floor as walk_default, so the tight/default
  contrast isolates the walk-shape effect (short, high-restart walks). Applied
  uniformly: the crashed EMAIL construction had persisted nothing, and the
  three AIRPORTS walk_tight cells from the earlier run were deleted and rerun
  under the new params. No mixed-parameter cells exist in the matrix.
- Runner hardened at the same time: a construction/baseline failure now writes
  status="error" cells (traceback in config.json) instead of killing the run.

## Determinism

- karateclub: workers=1 + seed + np.random.seed → bit-exact in smoke;
  recorded as "best-effort" (gensim guarantees only single-threaded).
- Egonet side: NEExT seeded RNG convention (self-contained, no global state).
