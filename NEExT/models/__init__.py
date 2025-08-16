"""Models module for NEExT framework."""

try:
    from .gnn_models import (
        BaseGNNModel,
        GCNModel,
        GATModel,
        GraphSAGEModel,
        GINModel,
        GNNModelFactory
    )
    
    __all__ = [
        "BaseGNNModel",
        "GCNModel",
        "GATModel",
        "GraphSAGEModel",
        "GINModel",
        "GNNModelFactory"
    ]
    GNN_MODELS_AVAILABLE = True
except ImportError:
    # GNN models are optional, requiring DGL
    GNN_MODELS_AVAILABLE = False
    __all__ = []