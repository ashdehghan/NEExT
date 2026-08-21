# Notes — embedding-baselines session

## 2026-08-21

- Plan approved by Ash (families proximity + structural; libs where sound, from-scratch
  otherwise; shared splits + region holdout). Lib layer built: `lib/nodeclass/{graph_utils,
  walks, emb_struc2vec, emb_role2vec, emb_graphwave, embed_registry}.py`, 28 tests, suite
  77 green. pecanpy found non-reproducible (parallel numba walker) → own walker.
- Runner smoke on AIRPORTS: center gate OK (node table == signal session's bag_table.csv),
  all six methods embed in < 30 s.
- **AIRPORTS evaluation (seed 42, D=16, 10 shared splits, macro-F1):** deepwalk .338,
  node2vec_bfs .351, node2vec_dfs .368 (floors .23–.25); struc2vec .584, role2vec .463,
  graphwave .587 — vs node features .590 and best bag (local k1) .554. P4 holds: structural
  embeddings reach the node-features ceiling and beat every bag; proximity barely clears the
  floor.
- Timing probe launched (chameleon all; roman deepwalk/role2vec/graphwave/struc2vec).
