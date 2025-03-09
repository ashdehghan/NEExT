import logging
from multiprocessing import Pool
from typing import Dict, List, Literal, Optional, Union

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from NEExT.collections import GraphCollection
from NEExT.features import Features
from NEExT.graphs import Graph
from NEExT.helper_functions import get_all_neighborhoods_ig, get_all_neighborhoods_nx, get_nodes_x_hops_away

# Set up logger
logger = logging.getLogger(__name__)

class NodeFeatureConfig(BaseModel):
    """Configuration for node feature computation"""
    feature_list: List[str]
    normalize_features: bool = True
    show_progress: bool = Field(default=True)
    n_jobs: int = -1
    

class NodeFeatures:
    """
    A class for computing node features across a collection of graphs.
    
    This class provides parallel computation of various node features for all nodes
    in a graph collection. It supports both NetworkX and iGraph backends and includes
    multiple feature computation methods like PageRank, centrality measures, and 
    structural embeddings.

    Features are computed with neighborhood aggregation, meaning for each feature,
    we compute values for the node itself and its neighborhood up to a specified depth.

    Attributes:
        graph_collection (GraphCollection): Collection of graphs to process
        config (NodeFeatureConfig): Configuration for feature computation
        available_features (Dict): Dictionary mapping feature names to computation methods

    Example:
        >>> node_features = NodeFeatures(
        ...     graph_collection=collection,
        ...     feature_list=["page_rank", "degree_centrality"],
        ...     n_jobs=-1
        ... )
        >>> features_df = node_features.compute()
    """
    
    def __init__(
        self,
        graph_collection: GraphCollection,
        feature_list: List[str],
        normalize_features: bool = True,
        show_progress: bool = True,
        n_jobs: int = -1
    ):
        """Initialize the NodeFeatures processor."""
        self.graph_collection = graph_collection
        
        self.config = NodeFeatureConfig(
            feature_list=feature_list,
            normalize_features=normalize_features,
            show_progress=show_progress,
            n_jobs=n_jobs
        )
        self.features_df = None

    def compute(self) -> Features:
        """Compute all requested features for all graphs."""
        # Process each graph sequentially
        graphs = self.graph_collection.graphs
        # if self.config.show_progress:
        #     graphs = tqdm(graphs, desc="Computing node features")
        
        features_df = self._get_known_graph_features(graphs)
        
        # Get feature columns (excluding node_id and graph_id)
        feature_columns = [col for col in features_df.columns 
                        if col not in ['node_id', 'graph_id']]
        
        return Features(features_df, feature_columns)
    
    def _get_known_graph_features(self, graphs: List[Graph]):
        out_features = []

        for graph in graphs:
            features_df = pd.DataFrame([node_attributes for _, node_attributes in graph.node_attributes.items()])

            features = [col for col in features_df.columns if col in self.config.feature_list] 

            features_df = features_df[features]
            features_df["node_id"] = graph.nodes
            features_df["graph_id"] = graph.graph_id

            out_features.append(features_df)

        return pd.concat(out_features, axis=0, ignore_index=True)