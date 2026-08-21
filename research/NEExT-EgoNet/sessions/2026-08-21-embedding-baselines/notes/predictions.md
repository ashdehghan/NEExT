# Pre-registered predictions — embedding baselines (written 2026-08-21, before any evaluation ran)

Families: **proximity** (DeepWalk, node2vec BFS p=.25/q=4, node2vec DFS p=4/q=.25) encode
*position* in the graph; **structural** (struc2vec, Role2Vec, GraphWave) encode role and are
position-free by construction. Egonet bags are position-free except through the
locality-leakage channel measured in the paper's structural-share section.

Same centers, same 10 stratified 70/30 splits (seed 42), same XGBoost, per-method
permutation floor. Macro-F1 headline.

- **P1 (shared splits).** On the homophilous / position-rich graphs (Amazon Ratings,
  Chameleon) the best proximity embedding scores at or above the best egonet bag; on
  heterophilous Roman Empire proximity embeddings are weak (near the node baseline or below).
- **P2 (region holdout, Amazon + Chameleon).** The proximity lift over the paired floor
  collapses toward zero already at B = 0–2 (position matching dies when test regions are
  contiguous), while structural embeddings retain a lift comparable to their shared-split lift.
- **P3 (region holdout, Roman).** Structural embeddings survive at every buffer (like local
  k=1/k=2 bags); proximity embeddings do not.
- **P4 (Airports — struc2vec's own benchmark).** struc2vec and GraphWave score high (close to
  the node-features baseline, above every bag), proximity embeddings score low: activity
  quartile is a role label, not a community label.
- **P5 (synthetic anchors, if run).** On the β=0 world proximity embeddings score far above the
  floor under shared splits and die at B=0; structural embeddings sit at the floor on β=0 and
  carry a lift on β=0.03.

Secondary expectations: Role2Vec's ceiling is the label purity of its type partition (few
hundred types) — expect it below struc2vec everywhere; GraphWave at the reference's auto scales
is very smooth (s ≈ 5–25 on 24k nodes) and may underperform on the sparse graphs.
