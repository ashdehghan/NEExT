import logging
from typing import List

import pandas as pd
from pydantic import BaseModel, Field

from NEExT.collections import GraphCollection
from NEExT.features import Features
from NEExT.graphs import Graph

# Set up logger
logger = logging.getLogger(__name__)


class NodeFeatureConfig(BaseModel):
    """Configuration for node feature computation"""

    feature_list: List[str]
    normalize_features: bool = True
    show_progress: bool = Field(default=True)
    n_jobs: int = 1


class NodeFeatures:
    """
    Gather native node attributes (the columns of the node table) across a collection of graphs.

    Rows are emitted per graph in ``graph.nodes`` order and each node's values are looked up by
    node id, so the result is aligned whatever order the attributes were loaded in. Every node
    must carry every requested attribute, and the selected columns must be numeric; otherwise a
    ``ValueError`` names the offending nodes or column. ``feature_list=["all"]`` selects every
    attribute present on the first graph's nodes. With ``normalize_features=True`` (the default)
    the selected columns are standardised over all rows of the collection.

    Attributes:
        graph_collection (GraphCollection): Collection of graphs to process
        config (NodeFeatureConfig): Configuration for feature computation

    Example:
        >>> node_features = NodeFeatures(
        ...     graph_collection=collection,
        ...     feature_list=["Age"],
        ...     n_jobs=1
        ... )
        >>> features_df = node_features.compute()
    """

    def __init__(
        self, graph_collection: GraphCollection, feature_list: List[str], normalize_features: bool = True, show_progress: bool = True, n_jobs: int = 1
    ):
        """Initialize the NodeFeatures processor."""
        self.graph_collection = graph_collection

        self.config = NodeFeatureConfig(feature_list=feature_list, normalize_features=normalize_features, show_progress=show_progress, n_jobs=n_jobs)
        self.features_df = None

    def compute(self) -> Features:
        """Compute all requested features for all graphs."""
        graphs = self.graph_collection.graphs
        features_df = self._get_known_graph_features(graphs)

        # Get feature columns (excluding node_id and graph_id)
        feature_columns = [col for col in features_df.columns if col not in ["node_id", "graph_id"]]

        features = Features(features_df, feature_columns)
        if self.config.normalize_features and feature_columns:
            features.normalize("StandardScaler")
        self.features_df = features.features_df
        return features

    def _resolve_feature_list(self, graphs: List[Graph]) -> List[str]:
        if any(name.lower() == "all" for name in self.config.feature_list):
            seen: List[str] = []
            for graph in graphs:
                for attrs in graph.node_attributes.values():
                    for key in attrs:
                        if key not in seen:
                            seen.append(key)
                if seen:
                    break
            extra = [name for name in self.config.feature_list if name.lower() != "all" and name not in seen]
            return seen + extra
        return list(self.config.feature_list)

    def _get_known_graph_features(self, graphs: List[Graph]) -> pd.DataFrame:
        feature_list = self._resolve_feature_list(graphs)
        out_features = []

        for graph in graphs:
            rows = []
            missing_nodes = []
            missing_columns = set()
            for node in graph.nodes:
                attrs = graph.node_attributes.get(node)
                if attrs is None:
                    missing_nodes.append(node)
                    continue
                absent = [col for col in feature_list if col not in attrs]
                if absent:
                    missing_columns.update(absent)
                    continue
                rows.append([attrs[col] for col in feature_list])
            if missing_nodes:
                raise ValueError(f"Graph {graph.graph_id}: {len(missing_nodes)} node(s) have no attributes, e.g. node_id in {missing_nodes[:5]}")
            if missing_columns:
                raise ValueError(f"Graph {graph.graph_id}: requested node feature(s) {sorted(missing_columns)} are missing on some nodes")

            features_df = pd.DataFrame(rows, columns=feature_list)
            features_df.insert(0, "graph_id", graph.graph_id)
            features_df.insert(0, "node_id", list(graph.nodes))
            out_features.append(features_df)

        out_df = pd.concat(out_features, axis=0, ignore_index=True)
        for col in feature_list:
            if not pd.api.types.is_numeric_dtype(out_df[col]):
                raise ValueError(f"Node feature {col!r} is not numeric (dtype {out_df[col].dtype}); encode it before using it as a feature")
        return out_df
