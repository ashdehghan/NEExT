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
- **Battery complete (04:58 UTC), exit 0.** Measured: Roman struc2vec 23 min (all skip-gram,
  5 epochs hs+neg), DeepWalk 2.5 min, GraphWave 85 s / 1.9 GB peak; Amazon struc2vec 24 min,
  GraphWave 138 s / 2.0 GB. Whole battery ≈ 2.5 h.
- Shared-split results: see README key findings / outputs/stats_shared_splits.csv. Region
  holdout: outputs/stats_region_holdout.csv. Figures: figures/baselines_{overview,separation}.
- Manuscript: sections/baselines.tex written + main.tex \input after structural_share; 16 bib
  entries added; compiles 0/0 (rev 23). Three-agent review pass launched (stats vs CSV,
  technical vs code, clarity/overclaims).
- Predictions scorecard: P4 ✓ (Airports), P3 ✓ (Roman), P2 ✓ (Amazon; proximity lift decays
  .045→.033→.008), P5 ✓ (anchors), P1 ✗ on Chameleon (bags beat all embeddings), ✓ on Amazon.
- Review pass applied (3 agents): stats check = 0 table discrepancies, 7 prose fixes (Amazon
  GraphWave +.055 is position — dies at B=0; DeepWalk Amazon B=4 significantly negative; bag
  Chameleon lift .44 over its paired global floor; D=128 structural ≤ +.016; "ordering" claim
  narrowed to family level); technical = 11 precision fixes (k* rule wording, Role2Vec reduced
  features, DeepWalk first-order, objectives sentence, floor-row scope footnote, |V| notation,
  GraphWave native width); clarity = P1–P5 scored explicitly, closing softened, GraphWave named
  on β=0 anchor. plot_baselines_separation: bag reference lift now over the same-scope floor.
  Manuscript rev 24 compiles 0/0.
