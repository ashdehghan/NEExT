# Results — egonet-poc (2026-07-31)

Dataset: AIRPORTS_USA (HF anomalypoint/NEExT), label `activity_quartile` (4 balanced
classes, ~297 each). 1190 nodes / 13 599 edges; 1186 nodes after largest-component
filtering (4 dropped) → 1186 egonets per k. Config: `outputs/config.json` (seeds:
egonet 13, embedding/ML 42; 10× stratified 70/30 hold-out, XGBoost).

## Headline table

| Method | k | Accuracy | Macro F1 | Runtime |
|---|---|---|---|---|
| majority class (floor) | — | 0.250 ± 0.000 | — | — |
| egonet embedding (approx_wasserstein, dim 16) | 1 | 0.556 ± 0.011 | 0.542 ± 0.008 | 20 min |
| egonet embedding (approx_wasserstein, dim 16) | 2 | 0.626 ± 0.027 | 0.624 ± 0.026 | **6.7 h** |
| **center-node structural features (baseline)** | — | **0.681 ± 0.027** | **0.677 ± 0.028** | 71 s |

## Egonet size distributions

| k | median nodes | p90 nodes | max nodes | median edges |
|---|---|---|---|---|
| 1 | 7 | 69.5 | 239 | 16 |
| 2 | 256 | 635.5 | 999 | 5 908 |

At k=2 the median egonet already contains ~22% of the whole graph — the airport network
has a small diameter, so 2-hop neighborhoods around hubs swallow most of the graph.

## Findings

1. **The mechanism works end-to-end.** Single graph → per-node egonets (center label
   inherited, label stripped from features) → structural features → Wasserstein
   embedding → classifier, all through NEExT's existing API, no library changes. Egonet
   embeddings carry real signal: 2.2–2.5× the majority floor.
2. **But the cheap baseline wins (here).** The center node's own structural feature
   vector on the full graph (11 features × 3 hops, 71 s) beats both egonet configs by
   5–12 accuracy points. On this dataset the label is a global structural role
   (activity quartile ≈ hub-ness), which node-level features computed on the *full*
   graph capture directly — while the egonet pipeline only sees each node's local
   subgraph, losing that global context.
3. **k=2 is prohibitively slow as-is:** 6.7 h vs 20 min at k=1, driven by big dense
   egonets × expensive features (betweenness, load, local efficiency) plus the known
   quadratic attribute-copy in `_build_egonet`. Anything beyond k=1 on small-diameter
   graphs needs cheaper feature sets, egonet-size caps, or the library hot-spot fixed.

## Interpretation / hypotheses for next session

- The comparison isn't apples-to-apples in information terms: the baseline's features
  are computed on the full graph (global information), the egonet pipeline's on the
  subgraphs only (local). The interesting scientific question is now: **where does
  local-subgraph information beat or complement global node features?**
  - Combine them: egonet embedding + center-node features as joint input (NEExT's
    `EmbeddingBuilder` has strategies for exactly this).
  - Datasets where labels depend on local neighborhood *pattern* rather than global
    role: anomaly/fraud sets (`is_outlier`), MINESWEEPER (`is_mine` is purely local),
    heterophilous sets (ROMAN_EMPIRE).
  - Sweep embedding dimension (16 may bottleneck: 33 feature cols available); try
    feature ablation (drop the 3 expensive features → big speedup, how much accuracy?).

## Threats to validity

- Single dataset, single embedding algorithm, one dim; no hyperparameter tuning on
  either side. Feature normalization is fit on the full collection (train/test scaler
  leak inherent to the current library pipeline, affects both arms equally).
- Repeated hold-out std understates variance vs. true resampling.
