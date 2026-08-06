# NEExT-EgoNet — Research Index

Research home for the EgoNet paper work. Companion manuscript: **NEExT-EgoNet**
(Manuscript app, project id `p_134274d1`).

**Research motivation:** use NEExT as a mechanism for building subgraph embeddings for
nodes — decompose a single labeled graph into k-hop egonets around each node (the center
node's label becomes the egonet's graph label), embed the egonets, and classify. Node
classification reframed as graph classification.

**Data:** single-graph node-classification datasets from the
[anomalypoint/NEExT](https://huggingface.co/datasets/anomalypoint/NEExT) Hugging Face
collection (also the backend of the Workbench Dataset Library).

## How this folder works

- Work happens in **sessions** — one folder per line of investigation, under `sessions/`,
  named `YYYY-MM-DD-topic-slug`.
- Start a new session with `./new-session.sh <topic-slug>`; it copies `_template/` and
  reminds you to add a row to the table below.
- Inside a session: `notes/` (lab notebook), `code/` (scripts + notebooks — tracked),
  `data/`, `logs/`, `outputs/`, `figures/` (all four gitignored; regenerable artifacts).
- Keep the session's `README.md` metadata current (status, findings, next steps) — it is
  the source of truth for that session; this table is the overview.

## Sessions

| Session | Status | Goal | Key findings |
|---|---|---|---|
| [2026-08-06-egonet-signal-airports-eu](sessions/2026-08-06-egonet-signal-airports-eu/) | active | **Experiment 1 (new research line):** signal detection — 2-hop egonet, 4 features (vlen 1), Wass dim 4, XGBoost vs majority + permutation floors on AIRPORTS_EUROPE | Signal YES: acc .463±.043 vs .25 floors, AUC .75; permutation floor exactly at chance (no leakage); median k=2 egonet spans 70% of graph yet classes separate; 14 s runtime |
| [2026-08-06-egonet-nodeclass](sessions/2026-08-06-egonet-nodeclass/) | parked | Egonet embeddings as node-level features: 5 constructions (hop k1/k2 + 3 walk variants) × {wass16, pooled} vs karateclub baselines (DeepWalk/Node2Vec/Walklets/GraphWave/Role2Vec) + structural features across 9 datasets picked by label mechanism | Benchmark never completed — research line parked 2026-08-06 (direction reset); lib/nodeclass + tests committed for the record |
| [2026-08-06-walk-bags](sessions/2026-08-06-walk-bags/) | done | Random-walk egonets (PR #14): C1 field-property head-to-head (5 bag variants x 3 networks) + C2 containment margins vs frozen phase-1 | min_visits=1 default was bad (fringe swamps bags) -> amended to 3; tuned walk bags = smooth+spiky+degree-neutral (rings spike .80 vs .76); containment: complementary not dominant (walks under-contain degree-1 tails, help cliques) |
| [2026-08-05-score-landscape](sessions/2026-08-05-score-landscape/) | done | Sanity-map the embedding-distance landscape (Ash's guided-walk idea): 3 realistic networks (LFR fraud rings, infiltrators, calibration tails), novelty+affinity fields, K and k sweeps | Bed of nails, not rolling hills: novelty spikes AT anomalous centers (infiltrators 99th pct at k=2) but NO gradient/halo (dilution); k=1 rough+informative, k=2 smooth+saturated; unsupervised novelty beats tiny-K affinity. Walk-as-gradient unsupported; novelty-ranking cascade survives |
| [2026-08-05-containment-phase1-synthetic](sessions/2026-08-05-containment-phase1-synthetic/) | done | Containment reframing (egonet=MIL bag, label=contains target?): synthetic detectability surface over {er,ba}×{hub,clique,tail}×π×k vs size-only confound | GATE PASS (audit-corrected): 20/50 valid cells beat size-only by ≥.05 AUC (peak +.35, BA/tail k=1 low π); oracle leak found+fixed, clean oracle median .93. Tail > hub > clique difficulty; dilution + saturation both measured; pooled features strong co-primary, wasserstein wins hard regimes |
| [2026-07-31-egonet-poc](sessions/2026-07-31-egonet-poc/) | active | Prove k-hop egonet embeddings carry node-class signal (AIRPORTS_USA, k ∈ {1,2}, vs baselines) | Mechanism works (0.56–0.63 acc vs 0.25 floor) but center-node full-graph features win (0.68); k=2 slow (6.7 h). Next: local-label datasets + combined signals |

<!-- Statuses: active | done | parked. Newest first. -->
