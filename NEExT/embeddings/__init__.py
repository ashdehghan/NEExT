from NEExT.embeddings.embeddings import Embeddings
from NEExT.embeddings.graph_embeddings import GraphEmbeddings

# Optional DGL embeddings - only import if explicitly needed
DGL_EMBEDDINGS_AVAILABLE = False

def _import_dgl_embeddings():
    """Import DGL embeddings only when needed."""
    global DGL_EMBEDDINGS_AVAILABLE
    try:
        from NEExT.embeddings.dgl_embeddings import (
            GNNEmbeddings,
            GNNEmbeddingConfig,
            GNNArchitectureConfig,
            GNNTrainingConfig
        )
        DGL_EMBEDDINGS_AVAILABLE = True
        return GNNEmbeddings, GNNEmbeddingConfig, GNNArchitectureConfig, GNNTrainingConfig
    except ImportError:
        return None, None, None, None