"""EgoNet PoC: node classification via k-hop egonet embeddings.

Pipeline per k: single labeled graph -> one egonet per node (center label
becomes graph label) -> structural node features -> approx_wasserstein
graph embeddings -> XGBoost classifier (repeated stratified hold-out).

Baselines under the identical split protocol:
  - majority_class: predict the training majority class.
  - center_node_features: the center node's structural feature vector
    computed on the FULL source graph (no egonet decomposition) -> same
    classifier. Isolates what the egonet embedding adds.

Outputs: outputs/results.csv, outputs/feature_importance_*.csv,
figures/*.png (matplotlib optional). Summary printed at the end goes into
notes/results.md by hand.
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_single_graph_dataset
from sklearn.model_selection import train_test_split

from NEExT import NEExT
from NEExT.embeddings.embeddings import Embeddings
from NEExT.ml_models.ml_models import MLModels

CONFIG = {
    "dataset": "AIRPORTS_USA",
    "label_column": "activity_quartile",
    "k_hops": [1, 2],
    "feature_list": ["all"],
    "feature_vector_length": 3,
    "embedding_algorithm": "approx_wasserstein",
    "embedding_dimension": 16,
    "egonet_seed": 13,
    "ml_seed": 42,
    "sample_size": 10,  # repeated hold-out iterations
    "test_size": 0.3,
}

SESSION_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = SESSION_ROOT / "outputs"
FIGURES = SESSION_ROOT / "figures"


def classify(collection, embeddings) -> dict:
    """XGBoost repeated stratified hold-out via NEExT's MLModels."""
    model = MLModels(
        graph_collection=collection,
        embeddings=embeddings,
        model_type="classifier",
        model_name="xgboost",
        compute_feature_importance=True,
        sample_size=CONFIG["sample_size"],
        test_size=CONFIG["test_size"],
        random_state=CONFIG["ml_seed"],
        n_jobs=1,
    )
    return model.compute()


def majority_baseline(labels: np.ndarray) -> dict:
    """Majority-class accuracy under the same split protocol as MLModels."""
    accuracies = []
    for i in range(CONFIG["sample_size"]):
        train_y, test_y = train_test_split(
            labels,
            test_size=CONFIG["test_size"],
            random_state=CONFIG["ml_seed"] + i,
            shuffle=True,
            stratify=labels,
        )
        majority = Counter(train_y).most_common(1)[0][0]
        accuracies.append(float(np.mean(test_y == majority)))
    return {"accuracy_mean": float(np.mean(accuracies)), "accuracy_std": float(np.std(accuracies))}


def center_node_embeddings(nxt, source_gc, egonets) -> Embeddings:
    """Center node's full-graph structural features, keyed by egonet id."""
    src_features = nxt.compute_node_features(
        source_gc,
        feature_list=CONFIG["feature_list"],
        feature_vector_length=CONFIG["feature_vector_length"],
        show_progress=False,
    )
    fdf = src_features.features_df.set_index("node_id")
    rows = []
    for egonet_id, (_, center_node_id) in sorted(egonets.egonet_to_graph_node_mapping.items()):
        row = {"graph_id": egonet_id}
        row.update(fdf.loc[center_node_id, src_features.feature_columns].to_dict())
        rows.append(row)
    return Embeddings(pd.DataFrame(rows), "center_node_features", src_features.feature_columns)


def egonet_size_stats(egonets) -> dict:
    node_counts = [len(g.nodes) for g in egonets.graphs]
    edge_counts = [g.G.number_of_edges() for g in egonets.graphs]
    return {
        "n_egonets": len(node_counts),
        "nodes_median": float(np.median(node_counts)),
        "nodes_p90": float(np.percentile(node_counts, 90)),
        "nodes_max": int(np.max(node_counts)),
        "edges_median": float(np.median(edge_counts)),
        "edges_max": int(np.max(edge_counts)),
        "_node_counts": node_counts,
    }


def save_size_histogram(node_counts, k):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping figure")
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(node_counts, bins=50)
    ax.set_xlabel("egonet size (nodes)")
    ax.set_ylabel("count")
    ax.set_title(f"{CONFIG['dataset']} egonet size distribution, k={k}")
    fig.tight_layout()
    fig.savefig(FIGURES / f"egonet_sizes_k{k}.png", dpi=150)
    plt.close(fig)


def result_row(method, k, res, extra=None) -> dict:
    row = {
        "dataset": CONFIG["dataset"],
        "method": method,
        "k": k,
        "embedding_algorithm": CONFIG["embedding_algorithm"] if method == "egonet_embedding" else "",
        "embedding_dimension": CONFIG["embedding_dimension"] if method == "egonet_embedding" else "",
        "accuracy_mean": round(res["accuracy_mean"], 4),
        "accuracy_std": round(res["accuracy_std"], 4),
        "f1_macro_mean": round(res["f1_score_mean"], 4) if "f1_score_mean" in res else "",
        "f1_macro_std": round(res["f1_score_std"], 4) if "f1_score_std" in res else "",
    }
    row.update(extra or {})
    return row


def main():
    t_start = time.time()
    nxt = NEExT(log_level="WARNING")
    rows = []

    edges_df, nodes_df = load_single_graph_dataset(
        CONFIG["dataset"], label_column=CONFIG["label_column"], structural_only=True
    )
    n_source_nodes = len(nodes_df)
    print(f"Loaded {CONFIG['dataset']}: {n_source_nodes} nodes, {len(edges_df)} edges")
    print(f"Label distribution: {dict(Counter(nodes_df[CONFIG['label_column']]))}")

    source_gc = nxt.load_single_graph_from_dfs(
        edges_df=edges_df, nodes_df=nodes_df, graph_id=CONFIG["dataset"], filter_largest_component=True
    )
    n_kept = source_gc.get_total_node_count()
    print(f"After largest-component filter: {n_kept} nodes ({n_source_nodes - n_kept} dropped)")

    baseline_done = False
    for k in CONFIG["k_hops"]:
        print(f"\n=== k={k} ===")
        t_k = time.time()
        egonets = nxt.compute_k_hop_egonets(
            source_gc,
            k_hop=k,
            egonet_feature_target=CONFIG["label_column"],
            sample_fraction=1.0,
            random_seed=CONFIG["egonet_seed"],
        )
        stats = egonet_size_stats(egonets)
        node_counts = stats.pop("_node_counts")
        print(f"Egonets: {stats}")
        save_size_histogram(node_counts, k)

        features = nxt.compute_node_features(
            egonets,
            feature_list=CONFIG["feature_list"],
            feature_vector_length=CONFIG["feature_vector_length"],
            show_progress=False,
        )
        embeddings = nxt.compute_graph_embeddings(
            egonets,
            features,
            embedding_algorithm=CONFIG["embedding_algorithm"],
            embedding_dimension=CONFIG["embedding_dimension"],
            random_state=CONFIG["ml_seed"],
        )
        res = classify(egonets, embeddings)
        elapsed = time.time() - t_k
        print(
            f"egonet_embedding k={k}: acc {res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}, "
            f"f1 {res['f1_score_mean']:.4f}±{res['f1_score_std']:.4f} ({elapsed:.0f}s)"
        )
        rows.append(result_row("egonet_embedding", k, res, {**stats, "seconds": round(elapsed, 1)}))
        res["feature_importance"].to_csv(OUTPUTS / f"feature_importance_egonet_k{k}.csv")

        if not baseline_done:
            # Baselines are k-independent: same centers, same labels, same splits.
            labels = np.array([g.graph_label for g in egonets.graphs])
            maj = majority_baseline(labels)
            print(f"majority_class: acc {maj['accuracy_mean']:.4f}±{maj['accuracy_std']:.4f}")
            rows.append(result_row("majority_class", "", maj))

            t_b = time.time()
            center_emb = center_node_embeddings(nxt, source_gc, egonets)
            res_b = classify(egonets, center_emb)
            print(
                f"center_node_features: acc {res_b['accuracy_mean']:.4f}±{res_b['accuracy_std']:.4f}, "
                f"f1 {res_b['f1_score_mean']:.4f}±{res_b['f1_score_std']:.4f} ({time.time() - t_b:.0f}s)"
            )
            rows.append(result_row("center_node_features", "", res_b, {"seconds": round(time.time() - t_b, 1)}))
            res_b["feature_importance"].to_csv(OUTPUTS / "feature_importance_center_node.csv")
            baseline_done = True

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUTS / "results.csv", index=False)
    (OUTPUTS / "config.json").write_text(json.dumps(CONFIG, indent=2))
    print(f"\nTotal: {time.time() - t_start:.0f}s. Results:\n{results_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
