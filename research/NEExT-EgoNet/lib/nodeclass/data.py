"""Canonical node tables: labels, rare-class filtering, center sampling.

The sampled node table is the single source of truth for one dataset's
benchmark: every representation is evaluated on exactly these nodes, in this
row order, so `shared_splits` yields identical partitions for every method.

The rare-class filter is applied uniformly, before any method runs — the
phase-1 lesson: a degeneracy rule introduced mid-sweep silently skews
headlines. A class survives iff it has at least MIN_CLASS_COUNT sampled
nodes, which guarantees >=3 examples of every class in each 30% test split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MIN_CLASS_COUNT = 10


def build_node_table(source_collection, label_column: str) -> "tuple[pd.DataFrame, dict]":
    """One row per node: node_id, y (0..C-1 int), y_raw.

    Classes are int-encoded in sorted order of their raw values (stringified
    for mixed types) so the encoding is deterministic across runs.
    """
    graph = source_collection.graphs[0]
    rows = [(n, graph.node_attributes[n][label_column]) for n in graph.nodes]
    table = pd.DataFrame(rows, columns=["node_id", "y_raw"]).sort_values("node_id").reset_index(drop=True)

    classes = sorted(table["y_raw"].unique(), key=str)
    mapping = {raw: i for i, raw in enumerate(classes)}
    table["y"] = table["y_raw"].map(mapping).astype(int)

    report = {
        "label_column": label_column,
        "n_nodes": len(table),
        "n_classes": len(classes),
        "class_map": {str(raw): i for raw, i in mapping.items()},
        "class_counts": {str(raw): int((table["y_raw"] == raw).sum()) for raw in classes},
    }
    return table[["node_id", "y", "y_raw"]], report


def filter_rare_classes(node_table: pd.DataFrame, min_count: int = MIN_CLASS_COUNT) -> "tuple[pd.DataFrame, dict]":
    """Drop nodes whose class has fewer than min_count members.

    Surviving classes keep their y codes (no re-encoding — codes stay
    comparable across the filter), and the report says exactly what was
    dropped so the notes can state it.
    """
    counts = node_table["y"].value_counts()
    kept_classes = counts[counts >= min_count].index
    kept = node_table[node_table["y"].isin(kept_classes)].reset_index(drop=True)
    dropped = node_table[~node_table["y"].isin(kept_classes)]
    report = {
        "min_count": min_count,
        "n_dropped_nodes": len(dropped),
        "n_dropped_classes": int(node_table["y"].nunique() - len(kept_classes)),
        "dropped_class_counts": {str(raw): int(c) for raw, c in dropped["y_raw"].value_counts().items()},
    }
    return kept, report


def sample_centers(node_table: pd.DataFrame, n_centers: int = 3000, seed: int = 13) -> pd.DataFrame:
    """Stratified-proportional sample of n_centers rows (all rows when fewer).

    The sample is re-filtered for rare classes (proportional sampling can push
    a borderline class under MIN_CLASS_COUNT) and returned sorted by node_id —
    the canonical row order every shared split derives from.
    """
    if len(node_table) > n_centers:
        idx, _ = train_test_split(
            np.arange(len(node_table)),
            train_size=n_centers,
            random_state=seed,
            shuffle=True,
            stratify=node_table["y"].to_numpy(),
        )
        sampled = node_table.iloc[idx]
    else:
        sampled = node_table
    sampled, _ = filter_rare_classes(sampled)
    return sampled.sort_values("node_id").reset_index(drop=True)
