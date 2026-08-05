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
| [2026-08-05-containment-phase1-synthetic](sessions/2026-08-05-containment-phase1-synthetic/) | done | Containment reframing (egonet=MIL bag, label=contains target?): synthetic detectability surface over {er,ba}×{hub,clique,tail}×π×k vs size-only confound | GATE PASS: 23/54 cells beat size-only by ≥.05 AUC (peak +.35, BA/tail k=1 low π). Tail > hub > clique difficulty; dilution + saturation both measured; pooled features strong co-primary, wasserstein wins hard regimes |
| [2026-07-31-egonet-poc](sessions/2026-07-31-egonet-poc/) | active | Prove k-hop egonet embeddings carry node-class signal (AIRPORTS_USA, k ∈ {1,2}, vs baselines) | Mechanism works (0.56–0.63 acc vs 0.25 floor) but center-node full-graph features win (0.68); k=2 slow (6.7 h). Next: local-label datasets + combined signals |

<!-- Statuses: active | done | parked. Newest first. -->
