"""Synthetic dataset loader: generates and caches planted-mechanism graphs.

Dataset ids encode the grid cell: SYNTH_HR_<L>_<B> where L in {L1,L2,L3}
(wavelength short/medium/long) and B in {B0,B1} (coupling off/weak).
An optional fifth part S<seed> (e.g. SYNTH_HR_L2_B1_S101) selects a
replicate world: same dials, different random draw of points and paint.
Without it the original exploratory seed (13) is used. Generated once per
id into data/raw/<id>/{edges,nodes}.csv (seeded, deterministic) so the
downstream pipeline is identical to the real-data sessions.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import sys

SESSION_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = SESSION_ROOT / "data" / "raw"
sys.path.insert(0, str(SESSION_ROOT.parents[1]))

from lib.nodeclass.synthetic import LABEL_COLUMN, rgg_homophily_graph  # noqa: E402

N = 6000
MEAN_DEGREE = 8.0
WAVELENGTHS = {"L1": 0.10, "L2": 0.20, "L3": 0.35}
COUPLINGS = {"B0": 0.0, "B1": 0.03}
SEED = 13


def load_single_graph_dataset(
    catalog_id: str,
    label_column: Optional[str] = None,
    structural_only: bool = False,
):
    """Generate (or load cached) synthetic grid cell; catalog-compatible signature."""
    parts = catalog_id.split("_")
    if len(parts) not in (4, 5) or parts[0] != "SYNTH" or parts[1] != "HR":
        raise ValueError(f"Unknown synthetic id: {catalog_id}")
    wavelength = WAVELENGTHS[parts[2]]
    coupling = COUPLINGS[parts[3]]
    if len(parts) == 5:
        if not parts[4].startswith("S") or not parts[4][1:].isdigit():
            raise ValueError(f"Bad replicate part in {catalog_id}: expected S<seed>, got {parts[4]}")
        seed = int(parts[4][1:])
    else:
        seed = SEED

    cache = RAW_DIR / catalog_id
    if not (cache / "edges.csv").exists():
        cache.mkdir(parents=True, exist_ok=True)
        edges_df, nodes_df = rgg_homophily_graph(
            n=N, mean_degree=MEAN_DEGREE, wavelength=wavelength, coupling=coupling, seed=seed
        )
        edges_df.to_csv(cache / "edges.csv", index=False)
        nodes_df.to_csv(cache / "nodes.csv", index=False)
    edges_df = pd.read_csv(cache / "edges.csv")
    nodes_df = pd.read_csv(cache / "nodes.csv")
    if label_column is not None and label_column not in nodes_df.columns:
        raise ValueError(f"Label column '{label_column}' not in nodes columns: {list(nodes_df.columns)}")
    return edges_df, nodes_df
