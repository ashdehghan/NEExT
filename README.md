# NEExT: Network Embedding Experimentation Toolkit

NEExT is a powerful Python framework for graph analysis, embedding computation, and machine learning on graph-structured data. It provides a unified interface for working with different graph backends (NetworkX and iGraph), computing node features, generating graph embeddings, and training machine learning models.

## 📚 Documentation

Detailed documentation is available in the `docs` directory. Build it locally or visit the online documentation at [NEExT Documentation](https://neext.readthedocs.io/en/latest/).

## 🌟 Features

- **Flexible Graph Handling**
  - Support for both NetworkX and iGraph backends
  - Automatic graph reindexing and largest component filtering
  - Node sampling capabilities for large graphs
  - Rich attribute support for nodes and edges

- **Comprehensive Node Features**
  - PageRank
  - Degree Centrality
  - Closeness Centrality
  - Betweenness Centrality
  - Eigenvector Centrality
  - Clustering Coefficient
  - Local Efficiency
  - LSME (Local Structural Motif Embeddings)

- **Graph Embeddings**
  - Approximate Wasserstein
  - Exact Wasserstein
  - Sinkhorn Vectorizer
  - Graph Neural Network (GCN, GraphSAGE, GIN; pure PyTorch, install with `NEExT[gnn]`)
  - Customizable embedding dimensions

- **Machine Learning Integration**
  - Classification and regression support
  - Dataset balancing options
  - Cross-validation with customizable splits
  - Feature importance analysis

### Custom Node Feature Functions

NEExT allows you to define and compute your own custom node feature functions alongside the built-in ones. This provides great flexibility for experimenting with novel graph metrics.

**Defining a Custom Feature Function:**

Your custom feature function must adhere to the following structure:

1.  **Input**: It must accept a single argument, which will be a `graph` object. This object provides access to the graph's structure (nodes, edges) and properties (e.g., `graph.nodes`, `graph.graph_id`, `graph.G` which is the underlying NetworkX or iGraph object).
2.  **Output**: It must return a `pandas.DataFrame` with the following specific columns in order:
    *   `"node_id"`: Identifiers for the nodes for which features are computed.
    *   `"graph_id"`: The identifier of the graph to which these nodes belong.
    *   One or more feature columns: These columns should contain the computed feature values. The naming convention for these columns should ideally follow the pattern `your_feature_name_0`, `your_feature_name_1`, etc., if your feature has multiple components or is expanded over hops (though a single feature column like `your_feature_name` is also acceptable).

**Example:**

Here's how you can define a simple custom feature function and use it:

```python
import networkx as nx
import pandas as pd

# 1. Define your custom feature function
# Works from scripts, modules, and Jupyter notebook cells — the function is
# shipped to workers via cloudpickle, so top-level notebook definitions are fine.
# Avoid closing over unpicklable objects (open file handles, live DB connections, etc.).
def my_node_degree_squared(graph):
    nodes = list(graph.nodes) # or range(graph.G.vcount()) for igraph if nodes are 0-indexed
    graph_id = graph.graph_id
    
    if hasattr(graph.G, 'degree'): # Handles both NetworkX and iGraph
        if isinstance(graph.G, nx.Graph): # NetworkX
            degrees = [graph.G.degree(n) for n in nodes]
        else: # iGraph
            degrees = graph.G.degree(nodes)
    else:
        raise TypeError("Graph object does not have a degree method.")
        
    degree_squared_values = [d**2 for d in degrees]
    
    df = pd.DataFrame({
        'node_id': nodes,
        'graph_id': graph_id,
        'degree_sq_0': degree_squared_values
    })
    # Ensure the correct column order
    return df[['node_id', 'graph_id', 'degree_sq_0']]

# 2. Prepare the list of custom feature methods
my_feature_methods = [
    {"feature_name": "my_degree_squared", "feature_function": my_node_degree_squared}
]

# 3. Pass it to compute_node_features
# Initialize NEExT and load your graph_collection as shown in the Quick Start
# nxt = NEExT()
# graph_collection = nxt.read_from_csv(...)

features = nxt.compute_node_features(
    graph_collection=graph_collection,
    feature_list=["page_rank", "my_degree_squared"], # Include your custom feature name
    feature_vector_length=3, # Applies to built-in features that use it
    my_feature_methods=my_feature_methods
)

print(features.features_df.head())
```

When you include `"my_degree_squared"` in the `feature_list` and provide `my_feature_methods`, NEExT will automatically register and compute your custom function. If `"all"` is in `feature_list`, your custom registered function will also be included in the computation.

**Parallel execution controls:**

By default, `compute_node_features()` uses `n_jobs=1`, so feature computation runs sequentially on a single CPU. To opt into graph-level parallel execution, pass `n_jobs=2`, `4`, or `-1`. When parallel execution is enabled, `parallel_backend="loky"` remains the default process backend. It is notebook-safe for custom functions because joblib can serialize them with cloudpickle, but it may spend substantial time serializing large graph objects and feature functions.

For serialization-heavy workloads, try `parallel_backend="threading"`. Threads avoid sending graph objects to worker processes, which can be faster when pickling dominates runtime, but GIL-bound Python code may still scale poorly. Benchmark expensive custom features with `n_jobs=1`, `2`, `4`, and `-1` before choosing production settings.

Advanced joblib options can be passed through with `joblib_kwargs`, for example `joblib_kwargs={"idle_worker_timeout": 120}` for the loky backend or `joblib_kwargs={"timeout": 300}`. These are tuning controls for scheduling and worker behavior, not guaranteed fixes for memory pressure.

## 📦 Installation

### Basic Installation
```bash
pip install NEExT
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/ashdehghan/NEExT.git
cd NEExT

# Install with development dependencies
pip install -e ".[dev]"

# Install with development and Workbench MCP dependencies
pip install -e ".[dev,workbench-mcp]"
```

### Additional Components
```bash
# For building documentation
pip install -e ".[docs]"

# For running experiments
pip install -e ".[experiments]"

# For the local web workbench
pip install -e ".[workbench]"

# For the local web workbench with MCP client integration
pip install -e ".[workbench-mcp]"

# Install all components
pip install -e ".[all]"
```

### Local Workbench
```bash
# From an installed package
neext-workbench

# From a development checkout
make neext-workbench
```

The workbench starts a localhost-only FastAPI server and serves the browser UI from the same process. Projects and artifacts are stored under `~/NEExT-Workbench` by default; override the location with `NEEXT_WORKBENCH_HOME` or `neext-workbench --workspace <path>`.

Workbench MCP integration requires the `workbench-mcp` extra in the Python environment that launches Workbench. Use a project virtual environment instead of relying on a global or Anaconda base environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[dev,workbench-mcp]"
make neext-workbench
```

#### Connecting MCP clients

Enable MCP from **Home → Settings → Agentic**, then copy the snippet for your client:

- **Claude Desktop** connects to a local server only over **stdio**. Its snippet launches the exact Python interpreter running Workbench (`<python> -m NEExT.workbench.mcp_cli --workspace <path>`) and is dropped into `claude_desktop_config.json`. Claude Desktop's remote/HTTP connectors run from Anthropic's cloud and cannot reach `127.0.0.1`, so stdio is the only local path.
- **Cursor, Claude Code, MCP Inspector, and other local clients** connect to the Streamable HTTP endpoint at `http://127.0.0.1:8765/mcp` with an `Authorization: Bearer <token>` header.
- **macOS only:** Claude Desktop runs stdio servers sandboxed and cannot read `~/Desktop`, `~/Documents`, or `~/Downloads`. If the Python environment lives in one of those folders, Workbench hides the Claude Desktop snippet and shows remediation — recreate the venv outside those folders (e.g. `~/neext-dev/.venv`) or grant Claude Full Disk Access. Windows and Linux have no such restriction (and Claude Desktop is not available on Linux — use a local HTTP client there).

## 🚀 Quick Start

### Basic Usage

```python
from NEExT import NEExT

# Initialize the framework
nxt = NEExT()
nxt.set_log_level("INFO")

# Load graph data
graph_collection = nxt.read_from_csv(
    edges_path="edges.csv",
    node_graph_mapping_path="node_graph_mapping.csv",
    graph_label_path="graph_labels.csv",
    reindex_nodes=True,
    filter_largest_component=True,
    graph_type="igraph"
)

# Compute node features
features = nxt.compute_node_features(
    graph_collection=graph_collection,
    feature_list=["all"],
    feature_vector_length=3
)

# Compute graph embeddings
embeddings = nxt.compute_graph_embeddings(
    graph_collection=graph_collection,
    features=features,
    embedding_algorithm="approx_wasserstein",
    embedding_dimension=3
)

# Or learn embeddings with a graph neural network (requires NEExT[gnn]):
# embeddings = nxt.compute_graph_embeddings(
#     graph_collection=graph_collection,
#     features=features,
#     embedding_algorithm="gnn",
#     architecture="GCN",          # or "GraphSAGE", "GIN"
#     embedding_dimension=16,
# )

# Train a classifier
model_results = nxt.train_ml_model(
    graph_collection=graph_collection,
    embeddings=embeddings,
    model_type="classifier",
    sample_size=50
)
```

### Working with Large Graphs

NEExT supports node sampling for handling large graphs:

```python
# Load graphs with 70% of nodes
graph_collection = nxt.read_from_csv(
    edges_path="edges.csv",
    node_graph_mapping_path="node_graph_mapping.csv",
    node_sample_rate=0.7  # Use 70% of nodes
)
```

### Feature Importance Analysis

```python
# Compute feature importance
importance_df = nxt.compute_feature_importance(
    graph_collection=graph_collection,
    features=features,
    feature_importance_algorithm="supervised_fast",
    embedding_algorithm="approx_wasserstein"
)
```

## 📊 Experiments

NEExT includes several pre-built experiments in the `examples/experiments` directory:

### Node Sampling Experiment
Investigates the effect of node sampling on classifier accuracy:
```bash
cd examples/experiments
python node_sampling_experiments.py
```

## 📝 Input File Formats

### edges.csv
```csv
src_node_id,dest_node_id
0,1
1,2
...
```

### node_graph_mapping.csv
```csv
node_id,graph_id
0,1
1,1
2,2
...
```

### graph_labels.csv
```csv
graph_id,graph_label
1,0
2,1
...
```

## 🛠️ Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=NEExT

# Run specific test file
pytest tests/test_node_sampling.py
```

### Building Documentation
```bash
cd docs
make html
```

### Code Style
The project uses several tools for code quality:
```bash
# Format code
black .

# Sort imports
isort .

# Check style
flake8 .

# Type checking
mypy .
```

### Publishing to PyPI

NEExT uses direct local publication through the root `Makefile`. GitHub Releases
do not publish the package.

Before publishing:

1. Update `__version__` in `NEExT/__init__.py`. `pyproject.toml` and the docs
   derive the package version from that file.
2. Commit all release changes on `main`.
3. Create `.env` with `PYPI_API_TOKEN=pypi-...`.
4. Run the validation and publish flow. For a one-command release that pushes
   `main`, pushes the tag, and publishes:

```bash
make release-check
make deploy
```

`make deploy` verifies the working tree and token, builds with `uv build`, pushes
`main`, creates and pushes the `release_v_<version>` tag, and publishes with
`uv publish`.

If `main` has already been pushed separately, use:

```bash
make publish-only
```

`make publish-only` requires local `main` to match `origin/main`, builds and
publishes the package, and creates the local `release_v_<version>` tag without
pushing commits or tags.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Ash Dehghan - [ash.dehghan@gmail.com](mailto:ash.dehghan@gmail.com)

## 🙏 Acknowledgments

- NetworkX team for the graph algorithms
- iGraph team for the efficient graph operations
- Scikit-learn team for machine learning components

## 📧 Contact

For questions and support:
- GitHub Issues: [NEExT Issues](https://github.com/ashdehghan/NEExT/issues)

## 🔄 Version History

- 0.1.0
  - Initial release
  - Basic graph operations
  - Node feature computation
  - Graph embeddings
  - Machine learning integration
