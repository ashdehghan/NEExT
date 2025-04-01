import numpy as np
import pandas as pd
from typing import Dict


def semi_supervised_set(
    features_df: pd.DataFrame,
    hide_frac: Dict[int, float],
    col: str = "is_outlier",
    seed: int = 42,
):
    """Fot a graph with known labels randomly hide a fraction of labels"""
    _features_df = features_df.copy()
    np.random.seed(seed)

    for _cls, frac in hide_frac.items():
        if frac == 0:
            continue
        mask = _features_df[col] == _cls
        drop_indices = np.random.choice(_features_df[mask].index, size=int(len(_features_df[mask]) * frac), replace=False)
        _features_df.loc[drop_indices, col] = -1

    return _features_df


def load_abcdo_data(
    name: str = "abcdo_data_1000_200_0.1",
    hide_frac: Dict[int, float] = {0: 1, 1: 1},
):
    path = f"https://raw.githubusercontent.com/CptQuak/graph_data/refs/heads/main/simulated/{name}/"

    edges_df = pd.read_csv(path + 'edges.csv')
    mapping_df = pd.read_csv(path + 'graph_mapping.csv')
    features_df = pd.read_csv(path + 'features.csv').sort_values("node_id").reset_index(drop=True)
    ground_truth_df = features_df[["node_id", "is_outlier"]].rename(columns={"node_id": "graph_id"}).copy()

    features_df = semi_supervised_set(features_df, hide_frac)
    return edges_df, mapping_df, features_df, ground_truth_df


def load_pygod_data(
    name: str = "gen_100",
    outlier_mode: str = "any",
    hide_frac: Dict[int, float] = {0: 1, 1: 1},
):
    from pygod.utils import load_data

    dataset = load_data(name)

    if outlier_mode == "any":
        y_target = np.where(dataset.y > 0, 1, 0)
    elif outlier_mode == "structural":
        y_target = np.where((dataset.y == 2) | (dataset.y == 3), 1, 0)
    elif outlier_mode == "contextual":
        y_target = np.where((dataset.y == 1) | (dataset.y == 3), 1, 0)

    edges_df = pd.DataFrame(dataset.edge_index.T)
    edges_df.columns = ["src_node_id", "dest_node_id"]

    features_df = pd.DataFrame()
    features_df = pd.DataFrame(dataset.x)
    features_df.columns = [f'x_{i}' for i in range(dataset.x.shape[1])]

    features_df["is_outlier"] = y_target
    features_df.insert(0, "node_id", list(range(len(features_df))))

    ground_truth_df = pd.DataFrame()
    ground_truth_df["graph_id"] = list(range(len(features_df)))
    ground_truth_df["is_outlier"] = y_target

    features_df = semi_supervised_set(features_df, hide_frac)

    mapping_df = pd.DataFrame()
    mapping_df["node_id"] = list(range(len(features_df)))
    mapping_df["graph_id"] = 0

    return edges_df, mapping_df, features_df, ground_truth_df
