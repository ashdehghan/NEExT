# Results — egonet embeddings as node-level features

(populated as the benchmark completes; every number regenerates from
`outputs/` — see README for the artifact contract)

## Q1 — Do egonet embeddings beat classic node embeddings anywhere?

(pending)

## Q2 — Label-type taxonomy: which label mechanisms favor which resolution?

(pending — the resolution-matching verdict; see fig_taxonomy_verdict)

## Q3 — Hop vs walk bags; walk-parameter sensitivity

(pending)

## Q4 — Wasserstein vs pooled on real data (phase-1 co-primary check)

(pending)

## Q5 — Cost accounting + AIRPORTS re-baseline vs the PoC

(pending — PoC pre-0.3.9 numbers were k=1 20 min / k=2 6.7 h on 1186 nodes)

## Q6 — Guards, exclusions, honest limitations

- hop_k2 dense guard triggers: (pending)
- karateclub exclusions: none (all 5 methods usable; GraphWave needed pygsp 0.5.1)
- GraphWave runs at its natural 200 features (not dim-matched; see decisions.md)
- Transductive caveat: node2vec-family embeddings are trained on the full graph
  (unlabeled test nodes included) — standard for this benchmark style, stated.

## Oddities log

(pending)
