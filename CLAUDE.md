# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test, and Development Commands

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=NEExT

# Run specific test file
pytest tests/test_node_sampling.py

# Run tests with verbose output
pytest -v --tb=short
```

### Code Quality
```bash
# Format code (150 char line length)
black .

# Sort imports
isort .

# Run linting (ruff configured for E/W/F/I/B/C4/UP rules)
ruff check .

# Type checking
mypy .

# Run all formatters and linters before committing
black . && isort . && ruff check .
```

### Documentation
```bash
# Build documentation
cd docs && make html
# or
make docs

# Clean documentation build
cd docs && make clean
# or
make clean
```

### Installation
```bash
# Basic installation
pip install -e .

# Development installation with all tools
pip install -e ".[dev]"

# Install with specific components
pip install -e ".[dev,test,docs,experiments]"
```

## High-Level Architecture

NEExT is a graph machine learning framework with a clean modular architecture designed for experimentation with graph embeddings and node-level analysis.

### Core Data Flow Pipeline
```
CSV/URLs → GraphIO → GraphCollection → Features → Embeddings → ML Models
                          ↓
                    EgonetCollection → Features → Embeddings → Outlier Detection
```

### Key Architectural Components

#### 1. **Graph Representation Layer** (`graphs/`, `collections/`)
- **Graph**: Base class wrapping NetworkX/iGraph backends with unified interface
- **Egonet**: Extends Graph, represents k-hop neighborhood subgraphs with mappings to original graph
- **GraphCollection**: Manages multiple graphs, handles sampling, provides batch operations
- **EgonetCollection**: Extends GraphCollection, decomposes graphs into node-centered egonets for node-level tasks

#### 2. **Feature Computation Layer** (`features/`)
- **StructuralNodeFeatures**: Computes graph-theoretic features (PageRank, centrality metrics, clustering)
- Supports k-hop neighborhood aggregation (features computed at different distances)
- Plugin architecture for custom features via registration system
- Parallel computation with configurable backends (ProcessPoolExecutor/ThreadPoolExecutor)

#### 3. **Embedding Layer** (`embeddings/`)
- **GraphEmbeddings**: Creates graph-level representations using:
  - Wasserstein distance-based embeddings (approximate/exact)
  - Sinkhorn vectorizer
  - Distribution-based approaches preserving structural properties
- Works on both GraphCollection and EgonetCollection

#### 4. **ML Pipeline** (`ml_models/`, `datasets/`)
- **GraphDataset**: Packages embeddings for sklearn-compatible ML
- **MLModels**: Classification/regression with XGBoost, handles imbalanced data
- **FeatureImportance**: Analyzes feature contributions

### Key Design Patterns

1. **Dual Backend Strategy**: NetworkX (flexibility) vs iGraph (performance) with transparent switching
2. **Egonet Decomposition**: Transforms node-level problems → graph-level problems by creating one subgraph per node
3. **Neighborhood Aggregation**: Features computed at k-hop distances capture multi-scale structural information
4. **Pydantic Validation**: All configurations use Pydantic models for runtime validation

### Critical Implementation Details

#### Egonet Processing Flow
1. Single graph → Extract k-hop neighborhoods for each node
2. Each neighborhood becomes independent Egonet object with:
   - Original graph/node ID mappings
   - Positional features (distance from ego center)
   - Preserved node/edge attributes
3. Collection of egonets processed as regular graphs for embeddings/ML

#### Feature Computation
- Default features: PageRank, degree/closeness/betweenness/eigenvector centrality, clustering coefficient, LSME
- βstar feature: Community-aware node metric (from 2023 paper)
- Custom features: Register via `my_feature_methods` parameter with specific DataFrame format

#### Memory Management
- Node sampling controls memory on large graphs
- Sparse matrices for incidence matrix computation
- Lazy graph initialization
- Batch processing with configurable parallelization

### Working with the Codebase

#### Entry Points
- `NEExT.framework.NEExT`: Main user interface class
- `test.py`: Example usage showing complete pipeline
- `experiments/`: Various experimental setups for research

#### Important Files
- `NEExT/collections/egonet_collection.py`: Core egonet decomposition logic
- `NEExT/features/structural_node_features.py`: Feature computation engine
- `NEExT/embeddings/graph_embeddings.py`: Wasserstein embedding implementation
- `NEExT/helper_functions.py`: Utility functions for graph operations

#### Configuration
- Graph backend: Set `graph_type="networkx"` or `"igraph"` in `read_from_csv()`
- Parallelization: Configure via `n_jobs` parameter in feature/embedding computation
- Node sampling: Use `node_sample_rate` parameter for large graphs

### Common Workflows

#### Basic Graph Classification
```python
nxt = NEExT()
graph_collection = nxt.read_from_csv(edges_path, node_graph_mapping_path, graph_label_path)
features = nxt.compute_node_features(graph_collection, feature_list=["all"])
embeddings = nxt.compute_graph_embeddings(graph_collection, features, embedding_algorithm="approx_wasserstein")
results = nxt.train_ml_model(graph_collection, embeddings, model_type="classifier")
```

#### Node Outlier Detection via Egonets
```python
from NEExT.collections import EgonetCollection
egonet_collection = EgonetCollection(egonet_feature_target="is_outlier")
egonet_collection.compute_k_hop_egonets(graph_collection, k_hop=2)
features = StructuralNodeFeatures(egonet_collection).compute()
embeddings = EmbeddingBuilder(egonet_collection, features).compute()
# Each embedding now represents one node's neighborhood
```

### Testing Approach
The codebase uses pytest with fixtures. Tests are in `tests/` directory. Always run tests after modifications to core components.