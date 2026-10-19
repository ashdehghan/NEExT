# Ring-by-ring vs pooled ego-nets (EgoNEExT)

**Status:** ready to run. Findings: none yet (fill in the section at the end).

## Question

EgoNEExT represents a node by its k-hop ego-net, summarised as one distribution of structural features (a pooled bag).
Pooling treats a neighbour one hop away the same as one three hops away. Does keeping the hop rings separate,
[centre | ring 1 | ring 2 | ...], carry more label signal than the pooled bag at the same radius?

The idea came from the EgoNEExT team meeting in October 2026.

## What the notebook compares

On the same centres, splits and classifier as the EgoNEExT paper:

| name | representation |
|---|---|
| `node` | the centre's own 11 structural features (the paper's node-features baseline) |
| `pooled_k` | the paper's k-hop bag, approximate Wasserstein embedding |
| `ringmean_k` | centre features, then the mean feature vector of each ring 1..k, plus ring sizes (= NEExT's `feature_vector_length = k + 1`) |
| `ringblock_k` | NEExT `compute_ring_embeddings`: each ring embedded separately, concatenated |
| `ringexact_k` | the same with exact Wasserstein per ring (keeps each ring's distribution shape); only for k ≤ 2 |

All features use the global scope (each member's full-graph values). The notebook states three expectations before
the run; the "Reading the results" cell says how to check each.

## How to run

From the NEExT repository root, on the `dev` branch:

```bash
git pull
uv sync --extra dev --extra experiments --extra advanced
uv run jupyter lab shared_experiments/egonext-ring-vs-pooled/ring_vs_pooled.ipynb
```

Run all cells. The configuration cell sets:

- `DATASETS`: Air Traffic (Europe) and Wikipedia Chameleon by default; Roman Empire and Amazon Ratings are commented
  out. Uncomment them for the full study.
- `MAX_K` (default 3), `EXACT_MAX_K` (default 2), `N_JOBS` (default 4 CPU workers).

Datasets download on first use from the `anomalypoint/NEExT` collection on Hugging Face into `data/`. Full-graph
features are cached in `outputs/`. Both folders are git-ignored.

**Runtime on a laptop (4 workers).** The two default datasets take about 15 minutes in all. Each additional dataset
needs one full-graph feature pass, then cached: about 25 minutes for Roman Empire and 50 minutes for Amazon Ratings.

## Protocol

- Structure and label only (native node attributes dropped); largest connected component.
- Classes with fewer than 10 sampled nodes are removed.
- Centres: every node for Air Traffic; a stratified sample with seed 13 otherwise (1,000 for Chameleon, 3,000 for
  Roman Empire and Amazon Ratings), as in the paper.
- 10 stratified 70/30 splits (seeds 42 to 51), shared by every representation, so comparisons are paired.
- XGBoost, 200 trees, depth 4, seed 42. Headline metric: macro-F1; accuracy is kept.
- Permutation floor: the same pipeline on shuffled labels (seed 7), for `node` and the widest `ringmean`.

The configuration cell prints the NEExT version and commit; `outputs/results.csv` carries them on every row.

## What to send back

1. `outputs/results.csv` and `outputs/ring_vs_pooled.pdf`.
2. The summary table and the paired-test table from the notebook.
3. A few lines under "Findings" below: which expectations held, and anything that looked wrong (for example, a
   permutation floor above chance).

## Caveats

- This is the shared-split comparison. On homophilous graphs part of a neighbourhood's score can be position rather
  than structure; the paper's region-holdout evaluation separates them and is a natural follow-up.
- The paper's runner is not bit-for-bit repeatable run to run (about 0.003 macro-F1 on Roman Empire); differences that
  small are noise.
- With many comparisons, p-values near 0.05 are weak evidence.

## Findings

(To be filled in after the run: date, commit, datasets, the tables, and the reading.)
