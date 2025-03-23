import numpy as np
import pandas as pd
from typing import Dict


def semi_supervised_set(
    features_df: pd.DataFrame,
    col: str = "is_outlier",
    hide_frac: Dict[int, float] = {0: 0.1, 1: 0.1},
    seed: int = 42,
):
    """Fot a graph with known labels randomly hide a fraction of labels"""
    _features_df = features_df.copy()
    np.random.seed(seed)

    for _cls, frac in hide_frac.items():
        if frac == 1:
            continue
        mask = _features_df[col] == _cls
        drop_indices = np.random.choice(_features_df[mask].index, size=int(len(_features_df[mask]) * frac), replace=False)
        _features_df.loc[drop_indices, col] = -1

    return _features_df


def initialize_graph(
    graph_data: Dict[str, str],
    frac: Dict[int, float],
):
    edges_df = pd.read_csv(graph_data["edge_file_path"])
    mapping_df = pd.read_csv(graph_data["node_graph_mapping_file_path"])
    features_df = pd.read_csv(graph_data["features_file_path"])

    ground_truth_df = features_df.copy()
    features_df = semi_supervised_set(ground_truth_df, hide_frac={0: frac[0], 1: frac[1]})
    ground_truth_df = (
        ground_truth_df.rename(columns={"node_id": "graph_id"})[["graph_id", "is_outlier"]].sort_values("graph_id").reset_index(drop=True)
    )
    return edges_df, mapping_df, features_df, ground_truth_df
