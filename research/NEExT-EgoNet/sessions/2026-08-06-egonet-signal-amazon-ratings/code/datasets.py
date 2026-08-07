"""Dataset loading helpers for EgoNet experiments.

Downloads single-graph catalog datasets (Hugging Face: anomalypoint/NEExT,
via the URLs registered in NEExT.workbench.dataset_library) into the
session's data/raw/ cache and returns dataframes ready for
NEExT.load_single_graph_from_dfs.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from NEExT.workbench.dataset_library import get_catalog_dataset

SESSION_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = SESSION_ROOT / "data" / "raw"


def load_single_graph_dataset(
    catalog_id: str,
    label_column: Optional[str] = None,
    structural_only: bool = False,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Return (edges_df, nodes_df) for a single-graph catalog dataset.

    Files are downloaded once into data/raw/<catalog_id>/ and reused on
    subsequent calls (the library itself has no download cache).

    Args:
        catalog_id: Catalog id, e.g. "AIRPORTS_USA" (case-insensitive).
        label_column: The node-attribute column holding class labels.
        structural_only: If True, drop every nodes.csv column except
            node_id and label_column. This keeps native node features out
            of the egonets (pure-structure experiments) and avoids the
            per-egonet attribute-copy cost for wide feature matrices.
    """
    catalog = get_catalog_dataset(catalog_id)
    if catalog is None:
        raise ValueError(f"Unknown catalog id: {catalog_id}")
    if catalog.source_graph_shape != "single_graph":
        raise ValueError(f"{catalog.id} is not a single-graph dataset")

    frames = {}
    for key in ("nodes", "edges"):
        cache = RAW_DIR / catalog.id / f"{key}.csv"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            pd.read_csv(catalog.files[key]).to_csv(cache, index=False)
        frames[key] = pd.read_csv(cache)

    nodes_df, edges_df = frames["nodes"], frames["edges"]
    if label_column is not None and label_column not in nodes_df.columns:
        raise ValueError(f"Label column '{label_column}' not in nodes.csv columns: {list(nodes_df.columns)}")
    if structural_only:
        if label_column is None:
            raise ValueError("structural_only=True requires label_column")
        nodes_df = nodes_df[["node_id", label_column]].copy()
    return edges_df, nodes_df
