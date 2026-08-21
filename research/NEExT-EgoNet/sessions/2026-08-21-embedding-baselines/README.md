# Session: External node-embedding baselines (proximity vs structural) — 2026-08-21

| | |
|---|---|
| **Started** | 2026-08-21 |
| **Last active** | 2026-08-21 |
| **Status** | active (battery complete 2026-08-21; manuscript section written) |

## Goal

Place the egonet bags among classic node embeddings on the paper's four datasets, under the
SAME centers / 10 shared splits / XGBoost / floors — and use the region-holdout instrument to
measure each method's structural share. Proximity embeddings (DeepWalk, node2vec) encode
position; structural embeddings (struc2vec, Role2Vec, GraphWave) are position-free. The
baselines therefore test the paper's mechanism account (position vs structure), not just rank
methods. Predictions registered in `notes/predictions.md` before any evaluation.

## Approach

- Methods: `lib/nodeclass/embed_registry.py` — DeepWalk / node2vec (own numba walks + gensim),
  struc2vec / Role2Vec / GraphWave implemented from the papers (`lib/nodeclass/emb_*.py`,
  tests in `lib/tests/`). Optional karateclub cross-checks (walklets, netmf, *_kc).
- Protocol: `code/run_embeddings.py` rebuilds each dataset's canonical center table and asserts
  it equals the signal session's `node_struct` bag_table.csv (center gate), embeds on the full
  graph (dim 16, seed 42), evaluates under the shared splits with a per-method permutation floor,
  persists `representation.parquet` so `run_separation.py` can run region holdout on it.
- Policies + pins: `notes/decisions.md`.

Reproduce (from the repo root, always `--no-sync`):

```
uv run --no-sync python research/NEExT-EgoNet/sessions/2026-08-21-embedding-baselines/code/run_embeddings.py --dataset airports --methods all
bash research/NEExT-EgoNet/sessions/2026-08-21-embedding-baselines/code/battery.sh
```

## Key findings

Shared splits (macro-F1, seed 42, D=16): structural embeddings win where the label is a role
(Airports: struc2vec .584 / GraphWave .587 ≈ node features .590, > best bag .554, p≈.02;
Roman: GraphWave .148 > node .137, ≈ local k2 bag .158, proximity ON the floor .048–.058);
proximity wins where the label is homophilous (Amazon: node2vec-DFS .299 ≈ best bag .302,
structural at/below floor); on Chameleon every embedding is below node features .635 and the
global k2 bag .658 (GraphWave .601 best) — P1 did not hold there. Region holdout: Roman
structural lifts survive in full (GraphWave +.098/.101/.101 at B=0/2/4, Role2Vec +.050,
struc2vec +.025) while proximity ≤ +.013; Amazon node2vec-DFS lift .108 → .045/.033/.008
(mostly position); Chameleon everyone halves at B=0; synthetic β=0: proximity +.35 shared →
≤ +.07 at B=0, 0 at B=2, structural at floor; β=.03: GraphWave +.074 → +.033 at every buffer.
Seed range ≤ .024 (one .072), D=128 adds ≤ .056 to proximity, 0 to structural; no ordering
change. Manuscript: sections/baselines.tex (rev 23, compiles 0/0).

## Next steps

- Review-pass findings applied? (see notes/notes.md) · background/related-work paragraph
- Optional: karateclub cross-checks (walklets/netmf/role2vec_kc/graphwave_kc), RolX/ReFeX

## Layout

- `notes/` — predictions.md (pre-registered), decisions.md, notes.md (lab log)
- `code/` — config.py, datasets.py, run_embeddings.py, battery.sh, collect/stats/table/plot scripts
- `data/ logs/ outputs/ figures/` — gitignored artifacts (cached embeddings, run dirs, results.csv)
