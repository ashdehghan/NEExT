"""Benchmark graph-embedding quality: GNN (GCN/GraphSAGE/GIN) vs. Wasserstein.

Network-free: uses NEExT's synthetic graph presets to build labeled graph
collections, computes node features, then for each embedding algorithm computes
graph embeddings and trains a downstream classifier. Prints a comparison of
downstream accuracy / AUC so we can confirm the GNN embeddings carry signal and
sanity-check the GNN default hyperparameters.

Run:  python benchmark_embeddings.py
"""

from __future__ import annotations

import argparse
import sys

from NEExT.framework import NEExT

# (algorithm_label, embedding_algorithm, architecture)
EMBEDDINGS = [
    ("Approx Wasserstein", "approx_wasserstein", None),
    ("GNN (GCN)", "gnn", "GCN"),
    ("GNN (GraphSAGE)", "gnn", "GraphSAGE"),
    ("GNN (GIN)", "gnn", "GIN"),
]

# (preset, kwargs) — separable -> harder, to see where GNN holds up.
PRESETS = [
    ("er_vs_ba", {"n_per_class": 40, "n_nodes": 45}),
    ("community_gradient", {}),
]


def run_preset(preset: str, preset_kwargs: dict, dimension: int, sample_size: int) -> list[dict]:
    nxt = NEExT(log_level="ERROR")
    collection = nxt.generate_synthetic_graphs(preset=preset, seed=42, **preset_kwargs)
    features = nxt.compute_node_features(collection, feature_list=["page_rank", "degree_centrality"], feature_vector_length=3)
    # Wasserstein dimension cannot exceed the feature-column count; cap for a fair
    # comparison (the GNN could go higher, but we hold dimension constant).
    dim = min(dimension, len(features.feature_columns))

    rows = []
    for label, algorithm, architecture in EMBEDDINGS:
        kwargs = {"embedding_algorithm": algorithm, "embedding_dimension": dim, "random_state": 42}
        if architecture is not None:
            kwargs["architecture"] = architecture
        embeddings = nxt.compute_graph_embeddings(collection, features, **kwargs)
        results = nxt.train_ml_model(collection, embeddings, model_type="classifier", sample_size=sample_size)
        rows.append(
            {
                "label": label,
                "accuracy_mean": results.get("accuracy_mean"),
                "accuracy_std": results.get("accuracy_std"),
                "auc_mean": results.get("auc_mean"),
                "f1_mean": results.get("f1_score_mean"),
            }
        )
    return rows


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "  -  "


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=8, help="embedding dimension")
    parser.add_argument("--sample-size", type=int, default=5, help="train_ml_model resamples")
    args = parser.parse_args(argv)

    for preset, preset_kwargs in PRESETS:
        print(f"\n=== preset: {preset} {preset_kwargs} | dim={args.dimension} ===")
        rows = run_preset(preset, preset_kwargs, args.dimension, args.sample_size)
        print(f"{'algorithm':22} {'acc_mean':>9} {'acc_std':>9} {'auc_mean':>9} {'f1_mean':>9}")
        for r in rows:
            print(f"{r['label']:22} {_fmt(r['accuracy_mean']):>9} {_fmt(r['accuracy_std']):>9} " f"{_fmt(r['auc_mean']):>9} {_fmt(r['f1_mean']):>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
