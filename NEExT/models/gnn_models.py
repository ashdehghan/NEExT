"""Graph Neural Network model implementations using DGL."""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import dgl
    import dgl.nn.pytorch as dglnn
    from dgl.nn import glob as dgl_glob
    
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    logger.warning("DGL not available. Install with: pip install 'NEExT[dgl]'")


class BaseGNNModel(nn.Module):
    """Base class for all GNN models."""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.0):
        """
        Initialize base GNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
    def forward(self, g: 'dgl.DGLGraph', features: 'torch.Tensor') -> 'torch.Tensor':
        """Forward pass through the model."""
        raise NotImplementedError
        
    def get_node_embeddings(self, g: 'dgl.DGLGraph', features: 'torch.Tensor') -> 'torch.Tensor':
        """
        Extract node-level embeddings (before pooling).
        
        Args:
            g: DGL graph
            features: Node features
            
        Returns:
            Node embeddings tensor
        """
        return self.forward(g, features)
    
    def get_graph_embeddings(
        self,
        g: 'dgl.DGLGraph',
        features: 'torch.Tensor',
        pooling: str = 'mean'
    ) -> 'torch.Tensor':
        """
        Extract graph-level embeddings (after pooling).
        
        Args:
            g: DGL graph or batch of graphs
            features: Node features
            pooling: Pooling method ('mean', 'sum', 'max')
            
        Returns:
            Graph embeddings tensor
        """
        node_embeds = self.get_node_embeddings(g, features)
        
        # Store embeddings and use DGL's readout
        with g.local_scope():
            g.ndata['h'] = node_embeds
            if pooling == 'mean':
                return dgl.mean_nodes(g, 'h')
            elif pooling == 'sum':
                return dgl.sum_nodes(g, 'h')
            elif pooling == 'max':
                return dgl.max_nodes(g, 'h')
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")


class GCNModel(BaseGNNModel):
    """Graph Convolutional Network implementation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        dropout: float = 0.0,
        activation: str = 'relu'
    ):
        """
        Initialize GCN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
            activation: Activation function ('relu', 'elu', 'leaky_relu')
        """
        super().__init__(input_dim, hidden_dims, output_dim, dropout)
        
        # Build layer dimensions
        dims = [input_dim] + hidden_dims + [output_dim]
        
        # Create GCN layers
        self.convs = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.convs.append(dglnn.GraphConv(dims[i], dims[i+1], allow_zero_in_degree=True))
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Activation
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu
    
    def forward(self, g: 'dgl.DGLGraph', features: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass through GCN.
        
        Args:
            g: DGL graph
            features: Node features
            
        Returns:
            Node embeddings
        """
        h = features
        
        for i, conv in enumerate(self.convs):
            h = conv(g, h)
            
            # Apply activation and dropout except for last layer
            if i < len(self.convs) - 1:
                h = self.activation(h)
                if self.dropout_layer is not None:
                    h = self.dropout_layer(h)
        
        return h


class GATModel(BaseGNNModel):
    """Graph Attention Network implementation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        activation: str = 'elu'
    ):
        """
        Initialize GAT model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            num_heads: Number of attention heads
            dropout: Feature dropout rate
            attn_dropout: Attention dropout rate
            activation: Activation function
        """
        super().__init__(input_dim, hidden_dims, output_dim, dropout)
        
        self.num_heads = num_heads
        
        # Build layer dimensions
        dims = [input_dim] + hidden_dims + [output_dim]
        
        # Create GAT layers
        self.convs = nn.ModuleList()
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                # Hidden layers with multiple heads
                self.convs.append(
                    dglnn.GATConv(
                        dims[i],
                        dims[i+1],
                        num_heads=num_heads,
                        feat_drop=dropout,
                        attn_drop=attn_dropout,
                        allow_zero_in_degree=True
                    )
                )
            else:
                # Last layer with single head
                self.convs.append(
                    dglnn.GATConv(
                        dims[i] * num_heads if i > 0 else dims[i],
                        dims[i+1],
                        num_heads=1,
                        feat_drop=dropout,
                        attn_drop=attn_dropout,
                        allow_zero_in_degree=True
                    )
                )
        
        # Activation
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.elu
    
    def forward(self, g: 'dgl.DGLGraph', features: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass through GAT.
        
        Args:
            g: DGL graph
            features: Node features
            
        Returns:
            Node embeddings
        """
        h = features
        
        for i, conv in enumerate(self.convs):
            h = conv(g, h)
            
            if i < len(self.convs) - 1:
                # Flatten multi-head outputs
                h = h.flatten(1)
                h = self.activation(h)
            else:
                # Last layer - average over heads
                h = h.mean(1)
        
        return h


class GraphSAGEModel(BaseGNNModel):
    """GraphSAGE implementation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        aggregator_type: str = 'mean',
        dropout: float = 0.0,
        activation: str = 'relu'
    ):
        """
        Initialize GraphSAGE model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            aggregator_type: Type of aggregator ('mean', 'gcn', 'pool', 'lstm')
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__(input_dim, hidden_dims, output_dim, dropout)
        
        self.aggregator_type = aggregator_type
        
        # Build layer dimensions
        dims = [input_dim] + hidden_dims + [output_dim]
        
        # Create GraphSAGE layers
        self.convs = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.convs.append(
                dglnn.SAGEConv(
                    dims[i],
                    dims[i+1],
                    aggregator_type=aggregator_type
                )
            )
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Activation
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu
    
    def forward(self, g: 'dgl.DGLGraph', features: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass through GraphSAGE.
        
        Args:
            g: DGL graph
            features: Node features
            
        Returns:
            Node embeddings
        """
        h = features
        
        for i, conv in enumerate(self.convs):
            h = conv(g, h)
            
            # Apply activation and dropout except for last layer
            if i < len(self.convs) - 1:
                h = self.activation(h)
                if self.dropout_layer is not None:
                    h = self.dropout_layer(h)
        
        return h


class GINModel(BaseGNNModel):
    """Graph Isomorphism Network implementation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        num_layers: Optional[int] = None,
        eps: float = 0.0,
        dropout: float = 0.0,
        activation: str = 'relu'
    ):
        """
        Initialize GIN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            num_layers: Number of GIN layers (if None, uses len(hidden_dims) + 1)
            eps: Initial epsilon value
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__(input_dim, hidden_dims, output_dim, dropout)
        
        # Build layer dimensions
        dims = [input_dim] + hidden_dims + [output_dim]
        
        # Create GIN layers
        self.convs = nn.ModuleList()
        for i in range(len(dims) - 1):
            # Create MLP for each GIN layer
            mlp = nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU() if activation == 'relu' else nn.ELU(),
                nn.Linear(dims[i+1], dims[i+1])
            )
            
            self.convs.append(
                dglnn.GINConv(
                    mlp,
                    aggregator_type='sum',
                    init_eps=eps,
                    learn_eps=True
                )
            )
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Activation
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            self.activation = F.relu
    
    def forward(self, g: 'dgl.DGLGraph', features: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass through GIN.
        
        Args:
            g: DGL graph
            features: Node features
            
        Returns:
            Node embeddings
        """
        h = features
        
        for i, conv in enumerate(self.convs):
            h = conv(g, h)
            
            # Apply activation and dropout except for last layer
            if i < len(self.convs) - 1:
                h = self.activation(h)
                if self.dropout_layer is not None:
                    h = self.dropout_layer(h)
        
        return h


class GNNModelFactory:
    """Factory for creating GNN models."""
    
    # Registry of available models
    _models = {
        "GCN": GCNModel,
        "GAT": GATModel,
        "GraphSAGE": GraphSAGEModel,
        "GIN": GINModel
    }
    
    @classmethod
    def create_model(
        cls,
        architecture: str,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        **kwargs
    ) -> BaseGNNModel:
        """
        Create a GNN model.
        
        Args:
            architecture: Model architecture name
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            **kwargs: Additional model-specific parameters
            
        Returns:
            GNN model instance
        """
        if not DGL_AVAILABLE:
            raise ImportError("DGL is not installed. Install with: pip install 'NEExT[dgl]'")
        
        if architecture not in cls._models:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Available: {list(cls._models.keys())}"
            )
        
        model_class = cls._models[architecture]
        
        # Filter kwargs for model-specific parameters
        import inspect
        sig = inspect.signature(model_class.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        return model_class(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            **filtered_kwargs
        )
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        """
        Register a custom GNN model.
        
        Args:
            name: Name for the model
            model_class: Model class (must inherit from BaseGNNModel)
        """
        if not issubclass(model_class, BaseGNNModel):
            raise ValueError("Model class must inherit from BaseGNNModel")
        
        cls._models[name] = model_class
        logger.info(f"Registered custom GNN model: {name}")