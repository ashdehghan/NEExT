# DGL Integration for NEExT

This document describes the Deep Graph Library (DGL) integration with NEExT, which enables Graph Neural Network (GNN) embeddings alongside NEExT's traditional graph analysis capabilities.

## Overview

The DGL integration adds state-of-the-art Graph Neural Network capabilities to NEExT, allowing users to:
- Generate learned graph embeddings using GNN architectures (GCN, GAT, GraphSAGE, GIN)
- Combine traditional graph features with deep learning approaches
- Leverage GPU acceleration for large-scale graph processing
- Create hybrid embeddings combining geometric (Wasserstein) and learned (GNN) representations

## Installation

DGL support is optional and can be installed as an extra dependency:

```bash
# Install NEExT with DGL support
pip install 'NEExT[dgl]'

# Or install DGL separately
pip install torch dgl
```

**Note**: DGL requires PyTorch. For GPU support, ensure you have CUDA-compatible versions installed.

## Architecture

The DGL integration follows NEExT's modular design principles:

```
NEExT GraphCollection
         ↓
   Node Features
         ↓
   [DGL Converter]
         ↓
   GNN Processing
         ↓
   Graph Embeddings
         ↓
   ML Pipeline
```

### New Components

1. **Converters** (`NEExT/converters/`)
   - `DGLConverter`: Converts between NEExT and DGL graph formats
   - `DGLConverterConfig`: Configuration for conversion process

2. **GNN Models** (`NEExT/models/`)
   - `GCNModel`: Graph Convolutional Network
   - `GATModel`: Graph Attention Network
   - `GraphSAGEModel`: GraphSAGE implementation
   - `GINModel`: Graph Isomorphism Network
   - `GNNModelFactory`: Factory for creating GNN models

3. **DGL Embeddings** (`NEExT/embeddings/`)
   - `GNNEmbeddings`: Main class for computing GNN-based embeddings
   - `GNNEmbeddingConfig`: Configuration for GNN embeddings
   - `GNNArchitectureConfig`: Architecture-specific settings
   - `GNNTrainingConfig`: Training hyperparameters

## Usage

### Basic GNN Embeddings

```python
from NEExT import NEExT

# Initialize NEExT
nxt = NEExT()

# Load graphs
graph_collection = nxt.load_from_networkx(nx_graphs)

# Compute node features
features = nxt.compute_node_features(
    graph_collection,
    feature_list=["pagerank", "degree_centrality"]
)

# Generate GNN embeddings
gnn_embeddings = nxt.compute_gnn_embeddings(
    graph_collection=graph_collection,
    features=features,
    architecture="GCN",  # or "GAT", "GraphSAGE", "GIN"
    hidden_dims=[64, 32],
    output_dim=16,
    epochs=200,
    device="cuda"  # Use GPU if available
)

# Use embeddings for classification
results = nxt.train_ml_model(
    graph_collection,
    gnn_embeddings,
    model_type="classifier"
)
```

### Architecture-Specific Parameters

#### Graph Attention Network (GAT)
```python
gnn_embeddings = nxt.compute_gnn_embeddings(
    graph_collection=graph_collection,
    features=features,
    architecture="GAT",
    gat_num_heads=8,        # Number of attention heads
    gat_attn_dropout=0.1,   # Attention dropout
    dropout=0.2
)
```

#### GraphSAGE
```python
gnn_embeddings = nxt.compute_gnn_embeddings(
    graph_collection=graph_collection,
    features=features,
    architecture="GraphSAGE",
    graphsage_aggregator="mean",  # or "gcn", "pool", "lstm"
    dropout=0.2
)
```

#### Graph Isomorphism Network (GIN)
```python
gnn_embeddings = nxt.compute_gnn_embeddings(
    graph_collection=graph_collection,
    features=features,
    architecture="GIN",
    gin_eps=0.0,  # Initial epsilon value
    dropout=0.2
)
```

### Hybrid Embeddings

Combine traditional and GNN embeddings for enhanced performance:

```python
# Compute Wasserstein embeddings
wasserstein_emb = nxt.compute_graph_embeddings(
    graph_collection,
    features,
    embedding_algorithm="approx_wasserstein",
    embedding_dimension=16
)

# Compute GNN embeddings
gnn_emb = nxt.compute_gnn_embeddings(
    graph_collection,
    features,
    architecture="GAT",
    output_dim=16
)

# Combine embeddings
hybrid_emb = wasserstein_emb + gnn_emb

# Use hybrid embeddings
results = nxt.train_ml_model(
    graph_collection,
    hybrid_emb,
    model_type="classifier"
)
```

### Advanced Configuration

For fine-grained control, use configuration objects directly:

```python
from NEExT.embeddings.dgl_embeddings import (
    GNNEmbeddings,
    GNNEmbeddingConfig,
    GNNArchitectureConfig,
    GNNTrainingConfig
)

# Configure architecture
arch_config = GNNArchitectureConfig(
    architecture="GCN",
    hidden_dims=[128, 64, 32],
    output_dim=16,
    dropout=0.2,
    activation="elu"
)

# Configure training
train_config = GNNTrainingConfig(
    epochs=300,
    learning_rate=0.005,
    weight_decay=1e-4,
    early_stopping_patience=20,
    train_ratio=0.8,
    val_ratio=0.1
)

# Create embeddings
config = GNNEmbeddingConfig(
    architecture=arch_config,
    training=train_config,
    device="cuda",
    pooling_method="mean"
)

gnn_embeddings = GNNEmbeddings(
    graph_collection,
    features,
    config=config
).compute()
```

## Supported GNN Architectures

### Graph Convolutional Network (GCN)
- **Paper**: Kipf & Welling, 2017
- **Use Case**: General-purpose, good baseline
- **Characteristics**: Simple, efficient, works well on homophilic graphs

### Graph Attention Network (GAT)
- **Paper**: Veličković et al., 2018
- **Use Case**: When edge importance varies
- **Characteristics**: Attention mechanism, multi-head support, interpretable

### GraphSAGE
- **Paper**: Hamilton et al., 2017
- **Use Case**: Inductive learning, large graphs
- **Characteristics**: Sampling-based, scalable, multiple aggregators

### Graph Isomorphism Network (GIN)
- **Paper**: Xu et al., 2019
- **Use Case**: Maximum expressive power
- **Characteristics**: Theoretically powerful, good for graph-level tasks

## Performance Considerations

### GPU Acceleration
```python
# Check CUDA availability
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use GPU for training
gnn_embeddings = nxt.compute_gnn_embeddings(
    graph_collection,
    features,
    device=device
)
```

### Memory Management
For large graphs, consider:
- Reducing batch size
- Using smaller hidden dimensions
- Enabling gradient checkpointing
- Using node sampling (future feature)

### Training Tips
1. **Learning Rate**: Start with 0.01, reduce if training is unstable
2. **Hidden Dimensions**: [64, 32] or [128, 64] are good defaults
3. **Epochs**: 200-300 for small graphs, may need more for larger ones
4. **Early Stopping**: Enabled by default to prevent overfitting

## Backwards Compatibility

The DGL integration maintains full backwards compatibility:
- All existing NEExT functionality remains unchanged
- DGL is an optional dependency
- Traditional embeddings (Wasserstein) continue to work
- Existing ML pipeline supports GNN embeddings seamlessly

## Examples

See `test_dgl_example.py` for a complete working example demonstrating:
- Loading graphs with NEExT
- Computing traditional features
- Generating GNN embeddings with different architectures
- Creating hybrid embeddings
- Comparing performance across embedding types

## Troubleshooting

### ImportError: DGL not found
```bash
pip install torch dgl
# or
pip install 'NEExT[dgl]'
```

### CUDA/GPU Issues
```python
# Force CPU usage
gnn_embeddings = nxt.compute_gnn_embeddings(
    graph_collection,
    features,
    device="cpu"
)
```

### Memory Issues
- Reduce `hidden_dims` size
- Decrease `epochs`
- Use smaller `output_dim`
- Process graphs in smaller batches

### Convergence Issues
- Adjust `learning_rate` (try 0.001 or 0.1)
- Increase `epochs`
- Try different `architecture`
- Normalize features before embedding

## Future Enhancements

Planned improvements for the DGL integration:
- [ ] Mini-batch training with graph sampling
- [ ] Pre-trained model support
- [ ] Additional GNN architectures (SGC, APPNP, etc.)
- [ ] Graph data augmentation
- [ ] Distributed training support
- [ ] AutoML for architecture selection
- [ ] Temporal graph support
- [ ] Heterogeneous graph support

## Contributing

Contributions to improve the DGL integration are welcome! Please:
1. Follow NEExT's coding style
2. Add tests for new features
3. Update documentation
4. Ensure backwards compatibility

## References

- [DGL Documentation](https://docs.dgl.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [GCN Paper](https://arxiv.org/abs/1609.02907)
- [GAT Paper](https://arxiv.org/abs/1710.10903)
- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216)
- [GIN Paper](https://arxiv.org/abs/1810.00826)