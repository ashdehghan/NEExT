from typing import Optional, Union, List, Dict, Literal
from pathlib import Path
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .io import GraphIO
from .graph_collection import GraphCollection
from .node_features import NodeFeatures
from .features import Features
from .graph_embeddings import GraphEmbeddings
import numpy as np
from .embeddings import Embeddings
from .feature_importance import FeatureImportance

class NEExT:
    """
    Main interface class for the NEExT framework.
    
    This class maintains the state of various components and provides
    a unified interface for users to interact with the framework.
    
    Attributes:
        logger: Logger instance for the framework
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the NEExT framework.
        
        Args:
            log_level: Initial logging level (default: "INFO")
        """
        # Initialize logger
        self.logger = logging.getLogger("NEExT")
        self.logger.setLevel(logging.INFO)
        
        # Create console handler with formatting
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Initialize components
        self.graph_io = GraphIO(logger=self.logger)
        
        self.logger.info("NEExT framework initialized")

    def set_log_level(self, level: str) -> None:
        """
        Set the logging level for the framework.

        Args:
            level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        if level.upper() not in level_map:
            self.logger.error(f"Invalid log level: {level}")
            raise ValueError(f"Invalid log level. Choose from: {', '.join(level_map.keys())}")
        
        log_level = level_map[level.upper()]
        self.logger.setLevel(log_level)
        self.logger.info(f"Log level set to: {level}")

    def read_from_csv(
        self,
        edges_path: Union[str, Path],
        node_graph_mapping_path: Union[str, Path],
        graph_label_path: Optional[Union[str, Path]] = None,
        node_features_path: Optional[Union[str, Path]] = None,
        edge_features_path: Optional[Union[str, Path]] = None,
        graph_type: str = "networkx",
        reindex_nodes: bool = True,
        filter_largest_component: bool = True,
        node_sample_rate: float = 1.0
    ) -> GraphCollection:
        """
        Read graph data from CSV files and return a graph collection.

        Args:
            edges_path: Path to edges CSV file (src_node_id, dest_node_id)
            node_graph_mapping_path: Path to node-graph mapping CSV file (node_id, graph_id)
            graph_label_path: Optional path to graph labels CSV file (graph_id, graph_label)
            node_features_path: Optional path to node features CSV file
            edge_features_path: Optional path to edge features CSV file
            graph_type: Backend to use ("networkx" or "igraph"). Defaults to "networkx"
            reindex_nodes: Whether to reindex nodes to start from 0 (default: True)
            filter_largest_component: Whether to keep only the largest connected 
                                    component of each graph (default: True)
            node_sample_rate: Rate at which to sample nodes from each graph (default: 1.0).
                            Must be between 0 and 1.
                                    
        Returns:
            GraphCollection: Collection of graphs loaded from CSV files
        """
        self.logger.info("Reading graph data from CSV files")
        self.logger.debug(f"Edges path: {edges_path}")
        self.logger.debug(f"Node-graph mapping path: {node_graph_mapping_path}")
        self.logger.debug(f"Reindex nodes: {reindex_nodes}")
        self.logger.debug(f"Filter largest component: {filter_largest_component}")
        self.logger.debug(f"Node sample rate: {node_sample_rate}")
        
        try:
            graph_collection = self.graph_io.read_from_csv(
                edges_path=edges_path,
                node_graph_mapping_path=node_graph_mapping_path,
                graph_label_path=graph_label_path,
                node_features_path=node_features_path,
                edge_features_path=edge_features_path,
                graph_type=graph_type,
                reindex_nodes=reindex_nodes,
                filter_largest_component=filter_largest_component,
                node_sample_rate=node_sample_rate
            )
            self.logger.info("Successfully loaded graph collection")
            self.logger.debug(f"Loaded {len(graph_collection.graphs)} graphs")
            return graph_collection
        except Exception as e:
            self.logger.error(f"Failed to read CSV files: {str(e)}")
            raise

    def get_collection_info(self, graph_collection: GraphCollection) -> dict:
        """
        Get basic information about a graph collection.
        
        This method is deprecated. Use graph_collection.describe() instead.
        
        Args:
            graph_collection: The graph collection to get information about
            
        Returns:
            dict: Dictionary containing collection information
        """
        self.logger.warning("get_collection_info is deprecated. Use graph_collection.describe() instead.")
        info = graph_collection.describe()
        self.logger.debug(f"Collection info: {info}")
        return info

    def compute_node_features(
        self,
        graph_collection: GraphCollection,
        feature_list: List[str],
        feature_vector_length: int = 3,
        normalize_features: bool = True,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Compute node features for all graphs in the collection.

        Args:
            graph_collection: Collection of graphs to compute features for
            feature_list: List of features to compute (e.g., ["page_rank", "degree_centrality"])
            feature_vector_length: Length of feature vector for each node (default: 3)
            normalize_features: Whether to normalize features across all nodes (default: True)
            show_progress: Whether to show progress bars during computation (default: True)

        Returns:
            pd.DataFrame: DataFrame containing computed features for all nodes
        """
        self.logger.info(f"Computing node features: {feature_list}")
        
        node_features = NodeFeatures(
            graph_collection=graph_collection,
            feature_list=feature_list,
            feature_vector_length=feature_vector_length,
            normalize_features=normalize_features,
            show_progress=show_progress
        )
        
        features = node_features.compute()
        self.logger.info(f"Computed features for {len(features.features_df)} nodes")
        
        return features

    def compute_graph_embeddings(
        self,
        graph_collection: GraphCollection,
        features: Features,
        embedding_algorithm: str,
        embedding_dimension: int,
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
        memory_size: str = "4G"
    ) -> Embeddings:
        """
        Compute graph embeddings based on node features.
        
        Args:
            graph_collection: Collection of graphs to compute embeddings for
            features: Features object containing node features
            embedding_algorithm: Algorithm to use for embedding computation
            embedding_dimension: Dimension of the output embeddings
            feature_columns: Specific feature columns to use (default: all)
            random_state: Random seed for reproducibility
            memory_size: Memory limit for algorithms that support it
            
        Returns:
            Embeddings: Embeddings object containing computed embeddings
        """
        self.logger.info(f"Computing graph embeddings using {embedding_algorithm}")
        
        graph_embeddings = GraphEmbeddings(
            graph_collection=graph_collection,
            features=features,
            embedding_algorithm=embedding_algorithm,
            embedding_dimension=embedding_dimension,
            feature_columns=feature_columns,
            random_state=random_state,
            memory_size=memory_size
        )
        
        embeddings = graph_embeddings.compute()
        self.logger.info(f"Computed embeddings for {len(embeddings.embeddings_df)} graphs")
        
        return embeddings

    def train_ml_model(
        self,
        graph_collection: GraphCollection,
        embeddings: Embeddings,
        model_type: Literal["classifier", "regressor"],
        balance_dataset: bool = False,
        sample_size: int = 5,
        n_jobs: int = -1,
        parallel_backend: str = "process"
    ) -> Dict:
        """
        Train and evaluate a machine learning model using graph embeddings.
        
        Args:
            graph_collection: Collection of graphs with labels
            embeddings: Embeddings object containing graph embeddings
            model_type: Type of model to train ("classifier" or "regressor")
            balance_dataset: Whether to balance the dataset for classification (default: False)
            sample_size: Number of training/testing iterations (default: 5)
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            parallel_backend: Parallelization backend ("process" or "thread")
            
        Returns:
            Dict: Dictionary containing model information and evaluation metrics
        """
        self.logger.info(f"Training {model_type} model on graph embeddings")
        
        from .ml_models import MLModels
        
        ml_models = MLModels(
            graph_collection=graph_collection,
            embeddings=embeddings,
            model_type=model_type,
            balance_dataset=balance_dataset,
            sample_size=sample_size,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend
        )
        
        results = ml_models.compute()
        
        if model_type == "classifier":
            self.logger.info(f"Model trained with average accuracy: {np.mean(results['accuracy']):.4f}")
        else:
            self.logger.info(f"Model trained with average RMSE: {np.mean(results['rmse']):.4f}")
        
        return results

    def compute_feature_importance(
        self,
        graph_collection: GraphCollection,
        features: Features,
        feature_importance_algorithm: str,
        embedding_algorithm: str = "approx_wasserstein",
        random_state: int = 42,
        n_iterations: int = 5
    ) -> pd.DataFrame:
        """
        Compute feature importance for graph embeddings.
        
        Args:
            graph_collection: Collection of graphs to analyze
            features: Features object containing node features
            feature_importance_algorithm: Algorithm to use for importance analysis
                ("supervised_greedy", "supervised_fast", "unsupervised")
            embedding_algorithm: Algorithm to use for embedding computation
            random_state: Random seed for reproducibility
            n_iterations: Number of iterations for computing average performance
            
        Returns:
            pd.DataFrame: DataFrame containing feature importance results
        """
        self.logger.info(f"Computing feature importance using {feature_importance_algorithm}")
        
        feature_importance = FeatureImportance(
            graph_collection=graph_collection,
            features=features,
            algorithm=feature_importance_algorithm,
            embedding_algorithm=embedding_algorithm,
            random_state=random_state,
            n_iterations=n_iterations
        )
        
        results_df = feature_importance.compute()
        self.logger.info("Feature importance analysis completed")
        
        return results_df
