from typing import List, Dict, Optional, Union, Literal
import pandas as pd
import numpy as np
import scipy.sparse
from pydantic import BaseModel, Field, validator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
from .graph_collection import GraphCollection
from .features import Features
from .embeddings import Embeddings

class GraphEmbeddingConfig(BaseModel):
    """Configuration for graph embedding computation"""
    embedding_algorithm: str
    embedding_dimension: int
    feature_columns: Optional[List[str]] = None
    random_state: int = 42
    memory_size: str = "4G"

    @validator('embedding_algorithm')
    def validate_embedding_algorithm(cls, v):
        valid_algorithms = ["approx_wasserstein", "wasserstein", "sinkhornvectorizer"]
        if v not in valid_algorithms:
            raise ValueError(f"Embedding algorithm must be one of {valid_algorithms}")
        return v


class GraphEmbeddings:
    """
    A class for computing graph embeddings based on node features.
    
    This class provides methods to compute graph-level embeddings from node-level
    features using various embedding algorithms.
    
    Attributes:
        graph_collection (GraphCollection): Collection of graphs to process
        features (Features): Features object containing node features
        config (GraphEmbeddingConfig): Configuration for embedding computation
        available_algorithms (Dict): Dictionary mapping algorithm names to computation methods
    """
    
    def __init__(
        self,
        graph_collection: GraphCollection,
        features: Features,
        embedding_algorithm: str,
        embedding_dimension: int,
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
        memory_size: str = "4G"
    ):
        """Initialize the GraphEmbeddings processor."""
        self.config = GraphEmbeddingConfig(
            embedding_algorithm=embedding_algorithm,
            embedding_dimension=embedding_dimension,
            feature_columns=feature_columns or features.feature_columns,
            random_state=random_state,
            memory_size=memory_size
        )
        self.graph_collection = graph_collection
        self.features = features
        
        # Define available embedding algorithms
        self.available_algorithms = {
            "approx_wasserstein": self._compute_approx_wasserstein,
            "wasserstein": self._compute_wasserstein,
            "sinkhornvectorizer": self._compute_sinkhorn
        }
        
        # Import vectorizers only when needed
        try:
            import vectorizers
            self.vectorizers = vectorizers
        except ImportError:
            raise ImportError(
                "The 'vectorizers' package is required for graph embeddings. "
                "Install it with: pip install vectorizers"
            )

    def compute(self) -> Embeddings:
        """
        Compute graph embeddings based on node features.
        
        Returns:
            Embeddings: Embeddings object containing computed embeddings
        """
        # Get embedding algorithm
        if self.config.embedding_algorithm not in self.available_algorithms:
            valid_algorithms = list(self.available_algorithms.keys())
            raise ValueError(f"Unknown algorithm: {self.config.embedding_algorithm}. Valid algorithms: {valid_algorithms}")
        
        embedding_func = self.available_algorithms[self.config.embedding_algorithm]
        
        # Compute embeddings
        embeddings_df = embedding_func(self.features.features_df)
        
        # Get embedding column names
        embedding_columns = [f"emb_{i}" for i in range(self.config.embedding_dimension)]
        
        return Embeddings(
            embeddings_df=embeddings_df,
            embedding_name=self.config.embedding_algorithm,
            embedding_columns=embedding_columns
        )

    def _prepare_incidence_matrix(self, node_features_df: pd.DataFrame):
        """
        Prepare incidence matrix and feature matrix for embedding computation.
        
        Args:
            node_features_df: DataFrame containing node features
            
        Returns:
            Tuple: (incidence_matrix, feature_matrix, graph_ids)
        """
        # Select feature columns
        if self.config.feature_columns:
            feature_cols = self.config.feature_columns
        else:
            feature_cols = [col for col in node_features_df.columns 
                          if col not in ['node_id', 'graph_id']]
        
        # Create sparse incidence matrix (graphs x nodes)
        graph_ids = sorted(node_features_df['graph_id'].unique())
        node_ids = node_features_df['node_id'].values
        graph_id_to_idx = {gid: i for i, gid in enumerate(graph_ids)}
        
        # Create row and column indices for sparse matrix
        rows = [graph_id_to_idx[gid] for gid in node_features_df['graph_id']]
        cols = range(len(node_ids))
        
        # Create sparse incidence matrix
        incidence_matrix = scipy.sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(len(graph_ids), len(node_ids))
        )
        
        # Create feature matrix
        feature_matrix = node_features_df[feature_cols].values
        
        return incidence_matrix, feature_matrix, graph_ids

    def _compute_approx_wasserstein(self, node_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute graph embeddings using approximate Wasserstein distance.
        
        Args:
            node_features_df: DataFrame containing node features
            
        Returns:
            pd.DataFrame: DataFrame containing graph embeddings with graph_id as first column
        """
        # Prepare data
        incidence_matrix, feature_matrix, graph_ids = self._prepare_incidence_matrix(node_features_df)
        
        # Compute embeddings
        embeddings = self.vectorizers.ApproximateWassersteinVectorizer(
            random_state=self.config.random_state,
            n_components=self.config.embedding_dimension
        ).fit_transform(incidence_matrix, vectors=feature_matrix)
        
        # Create DataFrame with graph_id as first column
        embeddings_df = pd.DataFrame()
        embeddings_df['graph_id'] = graph_ids
        
        # Add embedding columns
        for i in range(self.config.embedding_dimension):
            embeddings_df[f"emb_{i}"] = embeddings[:, i]
        
        return embeddings_df

    def _compute_wasserstein(self, node_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute graph embeddings using exact Wasserstein distance.
        
        Args:
            node_features_df: DataFrame containing node features
            
        Returns:
            pd.DataFrame: DataFrame containing graph embeddings with graph_id as first column
        """
        # Prepare data
        incidence_matrix, feature_matrix, graph_ids = self._prepare_incidence_matrix(node_features_df)
        
        # Compute embeddings
        embeddings = self.vectorizers.WassersteinVectorizer(
            memory_size=self.config.memory_size,
            random_state=self.config.random_state,
            n_components=self.config.embedding_dimension
        ).fit_transform(incidence_matrix, vectors=feature_matrix)
        
        # Create DataFrame with graph_id as first column
        embeddings_df = pd.DataFrame()
        embeddings_df['graph_id'] = graph_ids
        
        # Add embedding columns
        for i in range(self.config.embedding_dimension):
            embeddings_df[f"emb_{i}"] = embeddings[:, i]
        
        return embeddings_df

    def _compute_sinkhorn(self, node_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute graph embeddings using Sinkhorn distance.
        
        Args:
            node_features_df: DataFrame containing node features
            
        Returns:
            pd.DataFrame: DataFrame containing graph embeddings with graph_id as first column
        """
        # Prepare data
        incidence_matrix, feature_matrix, graph_ids = self._prepare_incidence_matrix(node_features_df)
        
        # Compute embeddings
        embeddings = self.vectorizers.SinkhornVectorizer(
            memory_size=self.config.memory_size,
            random_state=self.config.random_state,
            n_components=self.config.embedding_dimension
        ).fit_transform(incidence_matrix, vectors=feature_matrix)
        
        # Create DataFrame with graph_id as first column
        embeddings_df = pd.DataFrame()
        embeddings_df['graph_id'] = graph_ids
        
        # Add embedding columns
        for i in range(self.config.embedding_dimension):
            embeddings_df[f"emb_{i}"] = embeddings[:, i]
        
        return embeddings_df
