# Decisions log — embedding-baselines session

## Sourcing (2026-08-21, with Ash)
- Families: proximity + structural only (no spectral, no GNN). Libraries where sound,
  from-scratch (with paper references + tests) where not. Evaluate under shared splits AND
  region holdout.
- gensim 4.4.0 installed (skip-gram for every walk method); pecanpy 2.0.9 installed but its
  parallel numba walker is NOT reproducible run-to-run even at workers=1 → DeepWalk/node2vec
  walks come from `lib/nodeclass/walks.py` (numba rejection sampling, exact node2vec transition
  probabilities, tested) and pecanpy is an optional `route="pecanpy"` cross-check.
- karateclub NOT installed (GPLv3; pure-Python walker 20–40 min per config at 24k nodes; its
  GraphWave deviates from the paper; its Role2Vec is a WL/Doc2Vec variant). Wired as optional
  `walklets` / `netmf` / `*_kc` methods — only if the battery budget allows.
- struc2vec, Role2Vec (original), GraphWave implemented from the papers in
  `lib/nodeclass/emb_*.py`; validation: DTW vs brute force, mirrored karate (f_k(u, mirror) = 0
  exactly; embedding mirror cosine ≥ .9), barbell roles, Chebyshev vs dense eigh ≤ 1e-6, subset
  columns exact, determinism.

## Policies
- dim = 16 for every learned bottleneck (≥ bags' D = 11); GraphWave reports its native width
  (100 = 2 × 25 t-points × 2 scales); sensitivity dim = 128 on airports + chameleon.
- seed 42 primary; 43/44 on airports + chameleon (seed sensitivity).
- Word2Vec objective per method = its reference: DeepWalk hs; node2vec neg-5; struc2vec hs=1
  (+ gensim-era default neg=5, as the reference code); Role2Vec neg-5. Epochs: struc2vec 5
  (reference default), others 1. 10 walks × 80, window 10.
- struc2vec: OPT1 (compressed sequences, exact DTW — not FastDTW), OPT2 (2⌊log2 n⌋+1 candidates,
  seeded within-degree order), OPT3 (k* = largest k ≤ 6 whose mean ring is still populated),
  q_stay = 0.3 (reference code).
- Role2Vec features: degree, triangles, open wedges, core number (3-node graphlet orbits + k-core;
  the paper's default is 4-node motif counts via PGD), vertical log binning α = .5.
- GraphWave: reference auto scales (l1 ≈ 1/n), t ∈ linspace(0,100,25), Chebyshev order
  max(30, 2·s_max), computed only for the evaluation centers (exact per column).
- Every embedding is fit unsupervised on the full graph (transductive, like bags and node
  features); region holdout is the leakage control for all of them.
- Center gate: node_table must equal the signal session's node_struct bag_table.csv byte-for-byte
  before anything runs (`<DS>__center_gate.json`).
