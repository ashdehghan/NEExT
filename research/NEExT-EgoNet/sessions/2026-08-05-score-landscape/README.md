# Session: score-landscape

| | |
|---|---|
| **Started** | 2026-08-05 |
| **Last active** | 2026-08-05 |
| **Status** | done |

## Goal

Sanity study for Ash's guided-search idea: treat embedding distance as a
surface over the graph. Before building any walker — compute the landscape and
LOOK at it. Is it a navigable surface? Where does it glow, how far, how noisy,
as a function of reference size K and radius k?

## Approach

3 networks (n≈1500, seeded, connected by construction; generators in
`lib/containment/landscape_synthetic.py`, unit-tested):
- **calibration_tails** — ER + degree-1 tails (deliberately blatant; bug detector)
- **fraud_rings** — LFR background + quasi-cliques at ρ∈{0.9,0.6,0.3} mixed in one graph
- **infiltrators** — LFR + degree-preserving random rewires (purely relational anomaly)

Per (network, k∈{1,2}): egonet of EVERY node → features → one Wasserstein
embedding (single fit = consistent space) + pooled variant; persisted
embeddings/bag/meta/edges/layout CSVs. Fields (post-hoc, K∈{5,10,25,50}):
**novelty** (mean dist to K reference bags) and **affinity** (dist-to-neg −
dist-to-pos refs). Figures: field heatmaps on force layout, K-sweep, slope vs
hop-distance-to-anomaly, edge-vs-random-pair smoothness.

## Key findings

**Headline: the surface is a bed of nails, not rolling hills. Centering ≫ containing.**

1. **Novelty spikes exactly at anomalous CENTERS, with no halo.** Background-
   percentile of the anomaly-median spike: tails 99.5th; infiltrators 89th
   (k=1) → **99.1st (k=2, boxes fully disjoint)**; rings ρ=0.9 91st, ρ=0.6/0.3
   ~54–71st (marginal). At d≥1 hops the field is FLAT everywhere — no gradient.
2. **Containing ≠ novel (dilution made vivid):** at k=2 a neighbor's bag
   literally contains the infiltrator yet reads baseline (~0.53); the
   infiltrator-centered bag reads 1.02. One weird member among ~80 moves
   nothing; a weird *composition* (bag centered on the anomaly) moves everything.
3. **The walkability tension:** k=1 fields are informative but rough (edge
   diffs ≈ random-pair diffs — salt-and-pepper); k=2 fields become smooth
   (bags overlap) but labels saturate (rings pos_rate 0.84) and dilution
   erases the halo. Smoothness and signal trade off through the same knob.
4. **Novelty (unsupervised) beats affinity (supervised) at realistic K:** a
   K=25 random sample at 2% bag-positive rate caught 0 positives (calibration);
   with 1–7 positive refs the affinity fields are noise/inverted. The
   unsupervised how-far-from-typical field is the reliable one — no labels
   needed for the spike signal.
5. **Relational anomalies are the embedding's best case:** infiltrators (degree
   preserved, only wiring shape wrong) are near-perfectly separated at k=2 —
   and pooled z-scored features track closely (0.987 vs 0.991 pct); pooled
   remains a genuine rival everywhere except ρ=0.9 rings (0.91 wass vs 0.71 pooled... per-ring
   n=10, noisy).
6. **Subtlety dial partly works:** ρ=0.9 rings glow, ρ≤0.6 marginal — but one
   ring per density (10 members) is too few to pin the washout point;
   non-monotonicity (ρ=0.3 > ρ=0.6) is within placement noise.

**Verdict for the walk idea:** no evidence of a followable first-order gradient
on these networks — greedy neighbor-stepping has nothing to climb between
spikes. What the landscape DOES support: **novelty ranking as an unsupervised
detector** — probe bags, rank by novelty, drill into the top — i.e. the
cascade framing survives; the navigation framing doesn't (as tested). Possible
rescues (untested): anomalies with larger structural footprints (whole
anomalous communities) might create real basins; k=1 spikes + k=2 confirmation
as a two-scale probe.

Figures: `figures/` (maps/ksweep/slope/smooth per run). Numbers: computed from
`outputs/<run>/` CSVs (see notes/results.md).

## Next steps

- Discuss verdict with Ash: pivot walk → novelty-ranking cascade? Or test the
  larger-footprint (anomalous community) hypothesis before giving up on gradients?
- If keeping: fold a "score landscape" subsection + 1–2 restyled figures into
  the manuscript.
- Ring subtlety dial needs ≥5 rings per density for a real washout curve.

## Layout

- `notes/` — running lab-notebook notes (tracked)
- `code/` — scripts and notebooks (tracked)
- `data/` — datasets (gitignored)
- `logs/` — run logs (gitignored)
- `outputs/` — embeddings, models, checkpoints (gitignored)
- `figures/` — generated plots (gitignored)
