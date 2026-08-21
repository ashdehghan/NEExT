# Session: External node-embedding baselines (proximity vs structural) — 2026-08-21

| | |
|---|---|
| **Started** | 2026-08-21 |
| **Last active** | 2026-08-21 |
| **Status** | active |

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

(pending)

## Next steps

(pending — see notes/notes.md)

## Layout

- `notes/` — predictions.md (pre-registered), decisions.md, notes.md (lab log)
- `code/` — config.py, datasets.py, run_embeddings.py, battery.sh, collect/stats/table/plot scripts
- `data/ logs/ outputs/ figures/` — gitignored artifacts (cached embeddings, run dirs, results.csv)
