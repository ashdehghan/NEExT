"""Dataset loader for the embedding-baselines session.

Real datasets: the catalog loader of the signal sessions (downloads nodes.csv /
edges.csv once into this session's data/raw/). Synthetic anchors (SYNTH_HR_*):
delegated to the synthetic session's own loader so the cached worlds are
reused byte-for-byte. Deliberately imports nothing named `config` at module
level (run_separation.load_dataset pops `datasets`/`config` from sys.modules).
"""

import importlib.util
from pathlib import Path
from typing import Optional

import pandas as pd

SESSION_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = SESSION_ROOT / "data" / "raw"
SYNTH_LOADER = SESSION_ROOT.parent / "2026-08-12-synthetic-homophily-reach" / "code" / "datasets.py"


def _synthetic_loader():
    spec = importlib.util.spec_from_file_location("synth_datasets", SYNTH_LOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_single_graph_dataset


def load_single_graph_dataset(
    catalog_id: str,
    label_column: Optional[str] = None,
    structural_only: bool = False,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Return (edges_df, nodes_df) ready for NEExT.load_single_graph_from_dfs."""
    if catalog_id.startswith("SYNTH_HR"):
        return _synthetic_loader()(catalog_id, label_column=label_column, structural_only=structural_only)

    from NEExT.workbench.dataset_library import get_catalog_dataset

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
