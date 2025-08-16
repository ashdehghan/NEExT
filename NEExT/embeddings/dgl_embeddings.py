"""DGL-based graph neural network embeddings for NEExT."""

import logging
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from tqdm import tqdm

from NEExT.embeddings.embeddings import Embeddings

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import dgl
    
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    logger.warning("DGL not available. Install with: pip install 'NEExT[dgl]'")


class GNNArchitectureConfig(BaseModel):
    """Configuration for GNN architecture."""
    
    architecture: Literal["GCN", "GAT", "GraphSAGE", "GIN"] = Field(
        default="GCN",
        description="GNN architecture to use"
    )
    hidden_dims: List[int] = Field(
        default=[64, 32],
        description="Hidden layer dimensions"
    )
    output_dim: int = Field(
        default=16,
        description="Output embedding dimension"
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout rate"
    )
    activation: str = Field(
        default="relu",
        description="Activation function"
    )
    
    # Architecture-specific parameters
    gat_num_heads: int = Field(
        default=4,
        ge=1,
        description="Number of attention heads for GAT"
    )
    gat_attn_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Attention dropout for GAT"
    )
    graphsage_aggregator: str = Field(
        default="mean",
        description="Aggregator type for GraphSAGE"
    )
    gin_eps: float = Field(
        default=0.0,
        description="Initial epsilon value for GIN"
    )
    
    @validator('hidden_dims')
    def validate_hidden_dims(cls, v):
        if len(v) == 0:
            raise ValueError("hidden_dims must have at least one dimension")
        return v


class GNNTrainingConfig(BaseModel):
    """Configuration for GNN training."""
    
    epochs: int = Field(
        default=200,
        ge=1,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=0.01,
        gt=0,
        description="Learning rate"
    )
    weight_decay: float = Field(
        default=5e-4,
        ge=0,
        description="Weight decay for regularization"
    )
    early_stopping_patience: int = Field(
        default=10,
        ge=1,
        description="Patience for early stopping"
    )
    train_ratio: float = Field(
        default=0.8,
        gt=0,
        lt=1,
        description="Ratio of data for training"
    )
    val_ratio: float = Field(
        default=0.1,
        gt=0,
        lt=1,
        description="Ratio of data for validation"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    verbose: bool = Field(
        default=True,
        description="Whether to show training progress"
    )


class GNNEmbeddingConfig(BaseModel):
    """Combined configuration for GNN embeddings."""
    
    architecture: GNNArchitectureConfig = Field(
        default_factory=GNNArchitectureConfig,
        description="GNN architecture configuration"
    )
    training: GNNTrainingConfig = Field(
        default_factory=GNNTrainingConfig,
        description="Training configuration"
    )
    device: str = Field(
        default="cpu",
        description="Device to use ('cpu' or 'cuda')"
    )
    task_type: Literal["node_embedding", "graph_embedding"] = Field(
        default="graph_embedding",
        description="Type of embedding task"
    )
    pooling_method: Literal["mean", "sum", "max"] = Field(
        default="mean",
        description="Pooling method for graph embeddings"
    )


class GNNEmbeddings:
    """
    GNN-based embeddings that integrate with NEExT's embedding pipeline.
    
    This class provides methods to compute graph or node embeddings using
    various GNN architectures (GCN, GAT, GraphSAGE, GIN) through DGL.
    """
    
    def __init__(
        self,
        graph_collection,
        features,
        config: Optional[GNNEmbeddingConfig] = None,
        converter_config: Optional['DGLConverterConfig'] = None
    ):
        """
        Initialize GNN embeddings.
        
        Args:
            graph_collection: NEExT GraphCollection
            features: NEExT Features object containing node features
            config: GNN embedding configuration
            converter_config: DGL converter configuration
        """
        if not DGL_AVAILABLE:
            raise ImportError("DGL is not installed. Install with: pip install 'NEExT[dgl]'")
        
        # Import converters and models only when DGL is available
        from NEExT.converters.dgl_converter import DGLConverter, DGLConverterConfig
        from NEExT.models.gnn_models import GNNModelFactory, BaseGNNModel
        
        self.graph_collection = graph_collection
        self.features = features
        self.config = config or GNNEmbeddingConfig()
        self.converter = DGLConverter(converter_config or DGLConverterConfig())
        self.GNNModelFactory = GNNModelFactory
        self.BaseGNNModel = BaseGNNModel
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set device
        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Set random seed for reproducibility
        torch.manual_seed(self.config.training.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.training.random_state)
        
        self.model = None
        self.training_history = []
    
    def _build_gnn_model(self, input_dim: int) -> 'BaseGNNModel':
        """
        Build the GNN model based on configuration.
        
        Args:
            input_dim: Input feature dimension
            
        Returns:
            GNN model instance
        """
        arch_config = self.config.architecture
        
        # Prepare model-specific kwargs
        kwargs = {
            'dropout': arch_config.dropout,
            'activation': arch_config.activation
        }
        
        # Add architecture-specific parameters
        if arch_config.architecture == "GAT":
            kwargs['num_heads'] = arch_config.gat_num_heads
            kwargs['attn_dropout'] = arch_config.gat_attn_dropout
        elif arch_config.architecture == "GraphSAGE":
            kwargs['aggregator_type'] = arch_config.graphsage_aggregator
        elif arch_config.architecture == "GIN":
            kwargs['eps'] = arch_config.gin_eps
        
        model = self.GNNModelFactory.create_model(
            architecture=arch_config.architecture,
            input_dim=input_dim,
            hidden_dims=arch_config.hidden_dims,
            output_dim=arch_config.output_dim,
            **kwargs
        )
        
        return model.to(self.device)
    
    def _train_unsupervised(
        self,
        model: 'BaseGNNModel',
        dgl_graphs: List['dgl.DGLGraph']
    ) -> 'BaseGNNModel':
        """
        Train GNN model in unsupervised manner (using reconstruction loss).
        
        Args:
            model: GNN model to train
            dgl_graphs: List of DGL graphs
            
        Returns:
            Trained model
        """
        # Get input feature dimension from first graph
        input_feature_dim = dgl_graphs[0].ndata['feat'].shape[1]
        
        # Create decoder once before training loop
        decoder = nn.Linear(
            self.config.architecture.output_dim,
            input_feature_dim
        ).to(self.device)
        
        # Create optimizer for both model and decoder
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(decoder.parameters()),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Split graphs for training and validation
        n_graphs = len(dgl_graphs)
        n_train = int(n_graphs * self.config.training.train_ratio)
        n_val = int(n_graphs * self.config.training.val_ratio)
        
        indices = torch.randperm(n_graphs)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        
        train_graphs = [dgl_graphs[i] for i in train_indices]
        val_graphs = [dgl_graphs[i] for i in val_indices] if n_val > 0 else []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        iterator = tqdm(range(self.config.training.epochs), desc="Training GNN") \
            if self.config.training.verbose else range(self.config.training.epochs)
        
        for epoch in iterator:
            model.train()
            decoder.train()
            train_losses = []
            
            for g in train_graphs:
                features = g.ndata['feat']
                
                # Forward pass
                embeddings = model(g, features)
                
                # Reconstruction using the shared decoder
                reconstructed = decoder(embeddings)
                loss = F.mse_loss(reconstructed, features)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            if val_graphs:
                model.eval()
                decoder.eval()
                val_losses = []
                
                with torch.no_grad():
                    for g in val_graphs:
                        features = g.ndata['feat']
                        embeddings = model(g, features)
                        reconstructed = decoder(embeddings)
                        loss = F.mse_loss(reconstructed, features)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.training.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                self.training_history.append({
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                })
            else:
                self.training_history.append({
                    'epoch': epoch,
                    'train_loss': avg_train_loss
                })
        
        return model
    
    def compute(self) -> Embeddings:
        """
        Compute GNN-based embeddings.
        
        Returns:
            Embeddings object containing computed embeddings
        """
        self.logger.info(f"Computing GNN embeddings using {self.config.architecture.architecture}")
        
        # Convert to DGL format
        dgl_graphs = self.converter.to_dgl_graphs(
            self.graph_collection,
            self.features.features_df
        )
        
        # Determine input dimension
        if dgl_graphs[0].ndata and 'feat' in dgl_graphs[0].ndata:
            input_dim = dgl_graphs[0].ndata['feat'].shape[1]
        else:
            # Use number of feature columns minus node_id and graph_id
            input_dim = len(self.features.feature_columns)
        
        # Build model
        self.model = self._build_gnn_model(input_dim)
        
        # Train model
        self.model = self._train_unsupervised(self.model, dgl_graphs)
        
        # Extract embeddings
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for g in dgl_graphs:
                features = g.ndata['feat']
                
                if self.config.task_type == "node_embedding":
                    # Get node embeddings
                    node_embeds = self.model.get_node_embeddings(g, features)
                    all_embeddings.append(node_embeds)
                else:
                    # Get graph embeddings
                    graph_embeds = self.model.get_graph_embeddings(
                        g, features, pooling=self.config.pooling_method
                    )
                    # Ensure it's 1D for each graph
                    # Only squeeze if we have a batch dimension of size 1
                    if graph_embeds.dim() > 1 and graph_embeds.shape[0] == 1:
                        graph_embeds = graph_embeds.squeeze(0)
                    elif graph_embeds.dim() > 1:
                        # If we have multiple dimensions but first isn't 1, flatten
                        graph_embeds = graph_embeds.flatten()
                    all_embeddings.append(graph_embeds)
        
        # Convert to numpy and create DataFrame
        if self.config.task_type == "node_embedding":
            # Stack all node embeddings
            embeddings_tensor = torch.cat(all_embeddings, dim=0)
            embeddings_df = self.converter.from_dgl_node_embeddings(
                dgl_graphs, embeddings_tensor, embedding_type="node"
            )
        else:
            # Stack graph embeddings
            embeddings_tensor = torch.stack(all_embeddings)
            embeddings_df = self.converter.from_dgl_graph_embeddings(
                dgl_graphs, embeddings_tensor
            )
        
        # Get the actual embedding columns from the dataframe
        current_emb_cols = [col for col in embeddings_df.columns if col.startswith('embedding_')]
        
        # Create new embedding column names
        embedding_columns = [
            f"gnn_{self.config.architecture.architecture.lower()}_emb_{i}"
            for i in range(len(current_emb_cols))
        ]
        
        # Rename embedding columns to match expected format
        rename_dict = {old: new for old, new in zip(current_emb_cols, embedding_columns)}
        embeddings_df = embeddings_df.rename(columns=rename_dict)
        
        # Create Embeddings object
        embeddings = Embeddings(
            embeddings_df=embeddings_df,
            embedding_name=f"gnn_{self.config.architecture.architecture.lower()}",
            embedding_columns=embedding_columns
        )
        
        self.logger.info(
            f"Computed {len(embeddings_df)} embeddings with dimension "
            f"{self.config.architecture.output_dim}"
        )
        
        return embeddings
    
    def get_model(self) -> Optional['BaseGNNModel']:
        """
        Get the trained GNN model.
        
        Returns:
            Trained GNN model or None if not trained yet
        """
        return self.model
    
    def get_training_history(self) -> List[Dict]:
        """
        Get training history.
        
        Returns:
            List of dictionaries containing training metrics per epoch
        """
        return self.training_history