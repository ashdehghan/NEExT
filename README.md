<div align="center">

# NEExT

**Network Embedding Experimentation Toolkit**

An open-source Python framework for network science and graph machine learning.

[![PyPI version](https://img.shields.io/pypi/v/NEExT?color=1062a2)](https://pypi.org/project/NEExT/)
[![Python versions](https://img.shields.io/pypi/pyversions/NEExT?color=1062a2)](https://pypi.org/project/NEExT/)
[![License: MIT](https://img.shields.io/pypi/l/NEExT?color=1062a2)](https://github.com/ashdehghan/NEExT/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/NEExT?color=1062a2)](https://pypi.org/project/NEExT/)
[![Docs](https://img.shields.io/badge/docs-neext.app-c0541d)](https://www.neext.app/docs)

[**Documentation**](https://www.neext.app/docs) · [**Website**](https://www.neext.app) · [**Issues**](https://github.com/ashdehghan/NEExT/issues)

![NEExT Workbench](https://www.neext.app/readme/workbench.png)

</div>

## What is NEExT?

NEExT is an experimentation framework for graph and network data. It takes you from a
collection of graphs to predictive and scientific results through one pipeline you can
inspect and reproduce at every step:

> **Graphs → Features → Embeddings → Evidence**

Load graphs from CSV, pandas, or NetworkX into a unified `GraphCollection`; compute
structural node features (or write your own in plain Python); turn them into graph-level
embeddings with Wasserstein/Sinkhorn optimal transport or a GNN; then train classifiers
or regressors and read feature importance to see which structure drives the result. It's
built on the standard scientific Python stack — NumPy, pandas, scikit-learn, XGBoost,
NetworkX, iGraph — and works the same in a script, a notebook, or the Workbench.

There are two ways to use NEExT:

- **The Library** — a lightweight Python package for scripting and notebook workflows.
- **The Workbench** — a local, desktop-style GUI over the same NEExT workflows, with
  versioned artifacts and job tracking. It runs entirely on your machine (`127.0.0.1`,
  no accounts, no uploads) and is **MCP-native**, so an agent like Claude can drive it.

## Installation

```bash
pip install NEExT
```

Optional extras:

```bash
pip install "NEExT[gnn]"            # Graph Neural Network embeddings (pure PyTorch)
pip install "NEExT[workbench-mcp]"  # local Workbench + MCP integration
```

See the [docs](https://www.neext.app/docs) for the full list of extras.

## Quick start

```python
from NEExT import NEExT

nxt = NEExT()

# Load a collection of graphs from CSV
graph_collection = nxt.read_from_csv(
    edges_path="edges.csv",
    node_graph_mapping_path="node_graph_mapping.csv",
    graph_label_path="graph_labels.csv",
)

# Features → Embeddings → Evidence
features = nxt.compute_node_features(graph_collection, feature_list=["all"])
embeddings = nxt.compute_graph_embeddings(
    graph_collection, features, embedding_algorithm="approx_wasserstein"
)
results = nxt.train_ml_model(graph_collection, embeddings, model_type="classifier")
```

Custom features, GNN embeddings, large-graph sampling, feature importance, and the full
API are covered in the [documentation](https://www.neext.app/docs).

## The Workbench

The NEExT Workbench is a local, single-user FastAPI + React application that exposes
the NEExT workflows — datasets, features, embeddings, models, and analysis — as a
desktop-style UI. Everything stays on your machine, and it speaks MCP, so you can drive
the whole pipeline from an MCP client.

```bash
neext-workbench          # installed package
make neext-workbench     # from a development checkout
```

Then open **http://127.0.0.1:8765**. Projects are stored under `~/NEExT-Workbench` by
default (override with `NEEXT_WORKBENCH_HOME` or `neext-workbench --workspace <path>`).
The full Workbench tour, including MCP client setup, lives in the
[documentation](https://www.neext.app/docs).

## Learn more

- 📚 **Documentation** — guides, API reference, and the Workbench tour: [neext.app/docs](https://www.neext.app/docs)
- 🌐 **Website** — [neext.app](https://www.neext.app)
- 🐛 **Issues & support** — [github.com/ashdehghan/NEExT/issues](https://github.com/ashdehghan/NEExT/issues)

## License

NEExT is released under the [MIT License](LICENSE). Created and maintained by
[Ash Dehghan](mailto:ash.dehghan@gmail.com).
