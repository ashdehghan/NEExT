"""Per-feature label association (eta squared) for the Analysis section.

Regenerates the numbers behind fig:eta2-map in the manuscript: for each of
the four real datasets, the between-class share of variance of every
full-graph structural feature (one-way ANOVA eta^2 = SS_between / SS_total),
computed on all labelled nodes of the largest connected component, exactly
the node population the signal sessions used. Features are the L=1 pool
cached by each signal session's global-scope pass
(outputs/<DATASET>__source_features__fall-vl1.parquet); when a cache is
missing the pass is recomputed with NEExT and cached the same way.

Writes outputs/feature_eta2.csv (rows = features, columns = datasets) and
renders figures/feature_eta2_map.{pdf,png}. At L=1 lsme_0 and
basic_expansion_0 are rescalings of degree, so the map shows the nine
distinct measurements.

Run from the repo root: uv run --no-sync python <this file>
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
SESSIONS_ROOT = SESSION_ROOT.parent
RESEARCH_ROOT = SESSIONS_ROOT.parent
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(SESSION_ROOT / "code"))

import numpy as np
import pandas as pd

from lib.containment import plotstyle as ps

# (session directory, catalog id, label column, display name)
DATASETS = [
    ("2026-08-06-egonet-signal-airports-eu", "AIRPORTS_EUROPE", "activity_quartile", "Air Traffic"),
    ("2026-08-06-egonet-signal-roman-empire", "ROMAN_EMPIRE", "syntactic_role", "Roman Empire"),
    ("2026-08-06-egonet-signal-wiki-chameleon", "WIKIPEDIA_ARTICLES_CHAMELEON", "traffic_quintile", "Chameleon"),
    ("2026-08-06-egonet-signal-amazon-ratings", "AMAZON_RATINGS", "rating_class", "Amazon Ratings"),
]
FEATURE_TAG = "fall-vl1"

# Row labels for the figure; lsme_0 and basic_expansion_0 are degree at L=1.
DISPLAY = {
    "eigenvector_centrality_0": "eigenvector",
    "page_rank_0": "PageRank",
    "degree_centrality_0": "degree",
    "closeness_centrality_0": "closeness",
    "load_centrality_0": "load",
    "betweenness_centrality_0": "betweenness",
    "clustering_coefficient_0": "clustering",
    "betastar_0": r"$\beta^*$",
    "local_efficiency_0": "local efficiency",
}


def load_graph(session_dir: Path, catalog_id: str, label_column: str):
    """Largest-component graph collection + node table, as the signal sessions build them."""
    from NEExT import NEExT

    from datasets import load_single_graph_dataset  # this session's loader (same in every twin)
    from lib.nodeclass.data import build_node_table

    edges_df, nodes_df = load_single_graph_dataset(catalog_id, label_column=label_column, structural_only=True)
    nxt = NEExT(log_level="WARNING")
    collection = nxt.load_single_graph_from_dfs(
        edges_df=edges_df, nodes_df=nodes_df, graph_type="igraph", filter_largest_component=True
    )
    node_table, _ = build_node_table(collection, label_column)
    return nxt, collection, node_table


def load_features(session_dir: Path, catalog_id: str, nxt, collection) -> pd.DataFrame:
    cache = session_dir / "outputs" / f"{catalog_id}__source_features__{FEATURE_TAG}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    feats = nxt.compute_node_features(
        collection, feature_list=["all"], feature_vector_length=1, show_progress=False, n_jobs=-1
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.features_df.to_parquet(cache, index=False)
    return feats.features_df


def eta_squared(x: np.ndarray, y: np.ndarray) -> float:
    """One-way ANOVA effect size: between-class share of the total variance."""
    grand = x.mean()
    ss_total = float(((x - grand) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    ss_between = 0.0
    for cls in np.unique(y):
        xc = x[y == cls]
        ss_between += len(xc) * (xc.mean() - grand) ** 2
    return float(ss_between / ss_total)


def compute_table() -> pd.DataFrame:
    columns = {}
    for session_name, catalog_id, label_column, display in DATASETS:
        session_dir = SESSIONS_ROOT / session_name
        nxt, collection, node_table = load_graph(session_dir, catalog_id, label_column)
        feats = load_features(session_dir, catalog_id, nxt, collection)
        merged = node_table[["node_id", "y"]].rename(columns={"y": "label"}).merge(feats, on="node_id", how="inner", validate="one_to_one")
        y = merged["label"].to_numpy()
        feature_cols = [c for c in feats.columns if c not in ("node_id", "graph_id")]
        columns[display] = {c: eta_squared(merged[c].to_numpy(dtype=float), y) for c in feature_cols}
        print(f"{display}: {len(merged)} labelled nodes, {len(np.unique(y))} classes")
    return pd.DataFrame(columns)


def render(table: pd.DataFrame, path_stem: Path):
    import matplotlib.pyplot as plt

    ps.use_style()
    shown = table.loc[list(DISPLAY)].copy()
    shown = shown.sort_values(shown.columns[0], ascending=False)
    values = shown.to_numpy()

    fig, ax = plt.subplots(figsize=(ps.FULL_W, 3.4))
    im = ax.imshow(values, cmap="Oranges", vmin=0.0, vmax=max(0.65, values.max()), aspect="auto")
    ax.set_xticks(range(shown.shape[1]))
    ax.set_xticklabels(list(shown.columns))
    ax.set_yticks(range(shown.shape[0]))
    ax.set_yticklabels([DISPLAY[f] for f in shown.index])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.4 else ps.INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(r"$\eta^2$(feature $\sim$ label)")
    cbar.outline.set_visible(False)
    fig.tight_layout()
    ps.save(fig, path_stem)


def main():
    table = compute_table()
    out_dir = SESSION_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.index.name = "feature"
    table.to_csv(out_dir / "feature_eta2.csv", float_format="%.4f")
    print(table.round(3).to_string())
    fig_dir = SESSION_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    render(table, fig_dir / "feature_eta2_map")
    print("saved feature_eta2_map")


if __name__ == "__main__":
    main()
