"""Converter for transforming NEExT graphs to DGL format and back."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    import torch
    import dgl
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    logger.warning("DGL not available. Install with: pip install 'NEExT[dgl]'")


class DGLConverterConfig(BaseModel):
    """Configuration for NEExT to DGL conversion."""
    
    node_feature_columns: Optional[List[str]] = Field(
        default=None,
        description="Specific node feature columns to include. If None, all features are included."
    )
    edge_feature_columns: Optional[List[str]] = Field(
        default=None,
        description="Specific edge feature columns to include."
    )
    device: str = Field(
        default="cpu",
        description="Device to place the DGL graphs on ('cpu' or 'cuda')"
    )
    dtype: str = Field(
        default="float32",
        description="Data type for features ('float32' or 'float64')"
    )


class DGLConverter:
    """Converts between NEExT GraphCollection and DGL graph formats."""
    
    def __init__(self, config: Optional[DGLConverterConfig] = None):
        """
        Initialize the DGL converter.
        
        Args:
            config: Configuration for the converter. If None, uses defaults.
        """
        if not DGL_AVAILABLE:
            raise ImportError("DGL is not installed. Install with: pip install 'NEExT[dgl]'")
        
        self.config = config or DGLConverterConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set dtype
        self.torch_dtype = torch.float32 if self.config.dtype == "float32" else torch.float64
        
        # Set device
        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)
    
    def networkx_to_dgl(
        self,
        nx_graph,
        node_features: Optional[pd.DataFrame] = None,
        graph_id: Optional[int] = None
    ) -> dgl.DGLGraph:
        """
        Convert a single NetworkX graph to DGL format.
        
        Args:
            nx_graph: NetworkX graph object
            node_features: DataFrame with node features (must have 'node_id' and 'graph_id' columns)
            graph_id: Graph ID to filter features for this specific graph
            
        Returns:
            DGL graph with node features if provided
        """
        # Create DGL graph from NetworkX
        dgl_graph = dgl.from_networkx(nx_graph)
        
        # Move to specified device
        dgl_graph = dgl_graph.to(self.device)
        
        # Add node features if provided
        if node_features is not None and not node_features.empty:
            # Filter features for this graph if graph_id is provided
            if graph_id is not None:
                graph_features = node_features[node_features['graph_id'] == graph_id].copy()
            else:
                graph_features = node_features.copy()
            
            if not graph_features.empty:
                # Sort by node_id to ensure correct ordering
                graph_features = graph_features.sort_values('node_id')
                
                # Get feature columns (exclude node_id and graph_id)
                feature_cols = [col for col in graph_features.columns 
                              if col not in ['node_id', 'graph_id']]
                
                # Apply column filtering if specified
                if self.config.node_feature_columns:
                    feature_cols = [col for col in feature_cols 
                                  if col in self.config.node_feature_columns]
                
                if feature_cols:
                    # Convert features to tensor
                    feature_array = graph_features[feature_cols].values
                    feature_tensor = torch.tensor(
                        feature_array,
                        dtype=self.torch_dtype,
                        device=self.device
                    )
                    
                    # Add features to graph
                    dgl_graph.ndata['feat'] = feature_tensor
                    
                    # Store feature column names for later reference
                    dgl_graph.feature_columns = feature_cols
        
        return dgl_graph
    
    def to_dgl_graphs(
        self,
        graph_collection,
        features_df: Optional[pd.DataFrame] = None
    ) -> List[dgl.DGLGraph]:
        """
        Convert NEExT GraphCollection to list of DGL graphs.
        
        Args:
            graph_collection: NEExT GraphCollection object
            features_df: Optional DataFrame with node features
            
        Returns:
            List of DGL graphs with features
        """
        dgl_graphs = []
        
        for graph in graph_collection.graphs:
            # Get NetworkX representation
            nx_graph = graph.G
            
            # Convert to DGL
            dgl_graph = self.networkx_to_dgl(
                nx_graph,
                node_features=features_df,
                graph_id=graph.graph_id
            )
            
            # Store graph metadata
            dgl_graph.graph_id = graph.graph_id
            if hasattr(graph, 'graph_label') and graph.graph_label is not None:
                dgl_graph.label = graph.graph_label
            
            dgl_graphs.append(dgl_graph)
        
        self.logger.info(f"Converted {len(dgl_graphs)} graphs to DGL format")
        return dgl_graphs
    
    def to_dgl_batch(
        self,
        graph_collection,
        features_df: Optional[pd.DataFrame] = None
    ) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        """
        Convert NEExT GraphCollection to a batched DGL graph.
        
        Args:
            graph_collection: NEExT GraphCollection object
            features_df: Optional DataFrame with node features
            
        Returns:
            Tuple of (batched DGL graph, labels tensor)
        """
        # First convert to individual DGL graphs
        dgl_graphs = self.to_dgl_graphs(graph_collection, features_df)
        
        # Extract labels if available
        labels = []
        for g in dgl_graphs:
            if hasattr(g, 'label'):
                labels.append(g.label)
            else:
                labels.append(0)  # Default label
        
        # Batch the graphs
        batched_graph = dgl.batch(dgl_graphs)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        self.logger.info(f"Created batched graph with {batched_graph.num_nodes()} total nodes")
        return batched_graph, labels_tensor
    
    def from_dgl_node_embeddings(
        self,
        dgl_graphs: List[dgl.DGLGraph],
        embeddings: torch.Tensor,
        embedding_type: str = "node"
    ) -> pd.DataFrame:
        """
        Convert DGL node embeddings back to NEExT DataFrame format.
        
        Args:
            dgl_graphs: List of DGL graphs
            embeddings: Tensor of node embeddings
            embedding_type: Type of embeddings ("node" or "graph")
            
        Returns:
            DataFrame with embeddings in NEExT format
        """
        embeddings_np = embeddings.cpu().numpy()
        
        if embedding_type == "node":
            # Node-level embeddings
            rows = []
            node_offset = 0
            
            for dgl_graph in dgl_graphs:
                num_nodes = dgl_graph.num_nodes()
                graph_id = dgl_graph.graph_id if hasattr(dgl_graph, 'graph_id') else 0
                
                for i in range(num_nodes):
                    row = {'node_id': i, 'graph_id': graph_id}
                    for j, val in enumerate(embeddings_np[node_offset + i]):
                        row[f'embedding_{j}'] = val
                    rows.append(row)
                
                node_offset += num_nodes
            
            return pd.DataFrame(rows)
        
        else:  # graph-level embeddings
            rows = []
            for i, dgl_graph in enumerate(dgl_graphs):
                graph_id = dgl_graph.graph_id if hasattr(dgl_graph, 'graph_id') else i
                row = {'graph_id': graph_id}
                for j, val in enumerate(embeddings_np[i]):
                    row[f'embedding_{j}'] = val
                rows.append(row)
            
            return pd.DataFrame(rows)
    
    def from_dgl_graph_embeddings(
        self,
        dgl_graphs: List[dgl.DGLGraph],
        embeddings: torch.Tensor
    ) -> pd.DataFrame:
        """
        Convert DGL graph embeddings back to NEExT DataFrame format.
        
        This is a convenience method that calls from_dgl_node_embeddings
        with embedding_type="graph".
        
        Args:
            dgl_graphs: List of DGL graphs
            embeddings: Tensor of graph embeddings
            
        Returns:
            DataFrame with embeddings in NEExT format
        """
        return self.from_dgl_node_embeddings(dgl_graphs, embeddings, embedding_type="graph")