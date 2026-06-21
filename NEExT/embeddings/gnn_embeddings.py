"""Pure-PyTorch graph neural network embeddings for NEExT.

This module provides graph-level embeddings learned by a graph neural network
(GCN, GraphSAGE, or GIN) trained in an unsupervised, autoencoder fashion
(node-feature reconstruction). It depends only on PyTorch -- no DGL or
PyTorch Geometric -- so it installs reliably across platforms. NEExT graphs are
small, so each graph is processed with a dense adjacency matrix.

The public contract matches ``GraphEmbeddings``: ``compute()`` returns an
``Embeddings`` object whose ``embeddings_df`` has ``graph_id`` plus
``emb_0 .. emb_{D-1}`` columns, where ``D`` is the requested embedding
dimension.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from NEExT.embeddings.embeddings import Embeddings

logger = logging.getLogger(__name__)

ARCHITECTURES = ("GCN", "GraphSAGE", "GIN")

# Each graph is processed with a dense adjacency matrix (O(n^2) memory), which is
# appropriate for NEExT's small graphs. Warn past this node count so a very large
# graph does not silently blow up memory.
DENSE_ADJACENCY_WARN_NODES = 5000


def _require_torch():
    """Import torch lazily with a clear, actionable error message."""
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401

        return torch, nn
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError("PyTorch is required for GNN embeddings. Install it with: " "pip install 'NEExT[gnn]'") from exc


class GNNEmbeddings:
    """Compute graph-level embeddings with a pure-PyTorch GNN.

    Args:
        graph_collection: Collection of graphs to embed.
        features: Node features (``features_df`` with ``node_id``/``graph_id``).
        architecture: One of ``"GCN"``, ``"GraphSAGE"``, ``"GIN"``.
        embedding_dimension: Output embedding dimension (the GNN ``output_dim``).
        random_state: Seed for reproducibility.
        hidden_dims: Hidden layer dimensions (default ``[64, 32]``).
        epochs: Number of training epochs (default ``100``).
        learning_rate: Adam learning rate (default ``0.01``).
        weight_decay: Adam weight decay (default ``5e-4``).
        early_stopping_patience: Epochs without val improvement before stopping.
        train_ratio: Fraction of graphs used for training.
        val_ratio: Fraction of graphs used for validation/early stopping.
        pooling: Node-to-graph pooling method (``"mean"``, ``"sum"``, ``"max"``).
        device: ``"cpu"`` or ``"cuda"``.
        verbose: Whether to log per-epoch training loss.
    """

    def __init__(
        self,
        graph_collection,
        features,
        architecture: str = "GCN",
        embedding_dimension: int = 16,
        random_state: int = 42,
        hidden_dims: Optional[list[int]] = None,
        epochs: int = 100,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        dropout: float = 0.0,
        early_stopping_patience: int = 10,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        pooling: str = "mean",
        device: str = "cpu",
        verbose: bool = False,
    ):
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown GNN architecture '{architecture}'. " f"Available: {list(ARCHITECTURES)}")
        if pooling not in ("mean", "sum", "max"):
            raise ValueError(f"Unknown pooling method '{pooling}'.")
        if int(embedding_dimension) < 1:
            raise ValueError("embedding_dimension must be >= 1")
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError("dropout must be between 0 and 1")

        self.graph_collection = graph_collection
        self.features = features
        self.architecture = architecture
        self.embedding_dimension = int(embedding_dimension)
        self.random_state = int(random_state)
        self.hidden_dims = list(hidden_dims) if hidden_dims else [64, 32]
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.early_stopping_patience = int(early_stopping_patience)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.pooling = pooling
        self.verbose = verbose

        self._torch, self._nn = _require_torch()
        if device == "cuda" and not self._torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = self._torch.device(device)

        self.feature_columns = list(self.features.feature_columns)
        if not self.feature_columns:
            raise ValueError("GNN embeddings require at least one feature column.")

    # ------------------------------------------------------------------
    # Graph -> tensors (backend-agnostic: uses graph.nodes / graph.edges)
    # ------------------------------------------------------------------
    def _build_graph_tensors(self) -> list[dict]:
        """Convert each graph into (feature matrix, propagation operator) tensors.

        For each graph we use only the nodes that have feature rows (this keeps
        the adjacency and feature matrices aligned even if features were computed
        on a sampled subset of nodes) and build the induced subgraph over them.
        """
        torch = self._torch
        features_df = self.features.features_df
        # Group feature rows by graph for fast lookup. (dict(groupby) misfires
        # because DataFrameGroupBy exposes a string ``keys`` attribute.)
        grouped = {gid: sub for gid, sub in features_df.groupby("graph_id")}  # noqa: C416

        tensors: list[dict] = []
        for graph in self.graph_collection.graphs:
            graph_id = graph.graph_id
            sub = grouped.get(graph_id)
            if sub is None or sub.empty:
                raise ValueError(f"No node features found for graph_id={graph_id!r}; " "compute features before GNN embeddings.")
            # Stable node ordering by node_id; map node_id -> matrix index.
            sub = sub.sort_values("node_id")
            node_ids = sub["node_id"].tolist()
            index_of = {nid: i for i, nid in enumerate(node_ids)}
            n = len(node_ids)

            if n > DENSE_ADJACENCY_WARN_NODES:
                logger.warning(
                    "Graph %r has %d nodes; the GNN builds a dense %dx%d adjacency "
                    "matrix (O(n^2) memory). Consider node sampling for very large graphs.",
                    graph_id,
                    n,
                    n,
                    n,
                )

            x = sub[self.feature_columns].to_numpy(dtype=np.float32)
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)

            # Dense, symmetric adjacency over the induced node set.
            adj = torch.zeros((n, n), dtype=torch.float32, device=self.device)
            for src, dst in graph.edges:
                i = index_of.get(src)
                j = index_of.get(dst)
                if i is None or j is None:
                    continue  # edge to a node without features (e.g. sampled out)
                adj[i, j] = 1.0
                adj[j, i] = 1.0

            tensors.append(
                {
                    "graph_id": graph_id,
                    "x": x_t,
                    "op": self._propagation_operator(adj),
                }
            )
        return tensors

    def _propagation_operator(self, adj):
        """Build the per-graph propagation matrix for the chosen architecture."""
        torch = self._torch
        n = adj.shape[0]
        eye = torch.eye(n, dtype=torch.float32, device=self.device)

        if self.architecture == "GCN":
            # Symmetric-normalized adjacency with self-loops: D^-1/2 (A+I) D^-1/2.
            a_hat = adj + eye
            deg = a_hat.sum(dim=1)
            d_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
            return d_inv_sqrt.unsqueeze(1) * a_hat * d_inv_sqrt.unsqueeze(0)
        if self.architecture == "GraphSAGE":
            # Row-normalized neighbor mean (no self-loop); isolated nodes -> 0 row.
            deg = adj.sum(dim=1, keepdim=True)
            return adj / deg.clamp(min=1.0)
        # GIN: raw neighbor sum (no self-loop).
        return adj

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _build_model(self, input_dim: int):
        torch, nn = self._torch, self._nn
        architecture = self.architecture
        dropout = self.dropout
        dims = [input_dim] + self.hidden_dims + [self.embedding_dimension]

        class _GNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.architecture = architecture
                self.dropout_layer = nn.Dropout(dropout)
                if architecture == "GIN":
                    self.eps = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(len(dims) - 1)])
                    self.layers = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(dims[i], dims[i + 1]),
                                nn.ReLU(),
                                nn.Linear(dims[i + 1], dims[i + 1]),
                            )
                            for i in range(len(dims) - 1)
                        ]
                    )
                    # GIN's unnormalized sum aggregation otherwise blows up
                    # magnitudes and collapses the embeddings under the
                    # reconstruction objective. LayerNorm after each layer is the
                    # variable-size-graph-safe analogue of the BatchNorm the
                    # original GIN paper uses, and restores healthy variance.
                    self.norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(len(dims) - 1)])
                elif architecture == "GraphSAGE":
                    self.self_lin = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
                    self.neigh_lin = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
                else:  # GCN
                    self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

            def forward(self, x, op):
                h = x
                n_layers = len(dims) - 1
                for i in range(n_layers):
                    if self.architecture == "GCN":
                        h = self.layers[i](torch.matmul(op, h))
                    elif self.architecture == "GraphSAGE":
                        h = self.self_lin[i](h) + self.neigh_lin[i](torch.matmul(op, h))
                    else:  # GIN
                        agg = (1.0 + self.eps[i]) * h + torch.matmul(op, h)
                        h = self.norms[i](self.layers[i](agg))
                    if i < n_layers - 1:
                        h = torch.relu(h)
                        h = self.dropout_layer(h)
                return h

        return _GNN().to(self.device)

    # ------------------------------------------------------------------
    # Training (unsupervised node-feature reconstruction)
    # ------------------------------------------------------------------
    def _train(self, model, graphs: list[dict], input_dim: int):
        torch, nn = self._torch, self._nn
        decoder = nn.Linear(self.embedding_dimension, input_dim).to(self.device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(decoder.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        n = len(graphs)
        perm = torch.randperm(n)
        n_train = max(1, int(n * self.train_ratio))
        n_val = int(n * self.val_ratio)
        train_idx = perm[:n_train].tolist()
        val_idx = perm[n_train : n_train + n_val].tolist()
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]

        best_val = float("inf")
        patience = 0
        for epoch in range(self.epochs):
            model.train()
            decoder.train()
            losses = []
            for g in train_graphs:
                node_emb = model(g["x"], g["op"])
                recon = decoder(node_emb)
                loss = torch.nn.functional.mse_loss(recon, g["x"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if self.verbose:
                logger.info("GNN epoch %d train_loss=%.6f", epoch, float(np.mean(losses)))

            if val_graphs:
                model.eval()
                decoder.eval()
                with torch.no_grad():
                    val_losses = [torch.nn.functional.mse_loss(decoder(model(g["x"], g["op"])), g["x"]).item() for g in val_graphs]
                avg_val = float(np.mean(val_losses))
                if avg_val < best_val:
                    best_val = avg_val
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.early_stopping_patience:
                        if self.verbose:
                            logger.info("Early stopping at epoch %d", epoch)
                        break
        return model

    def _pool(self, node_emb):
        if self.pooling == "mean":
            return node_emb.mean(dim=0)
        if self.pooling == "sum":
            return node_emb.sum(dim=0)
        return node_emb.max(dim=0).values

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(self) -> Embeddings:
        """Train the GNN and return graph-level ``Embeddings``."""
        torch = self._torch
        torch.manual_seed(self.random_state)

        # NEExT pulls in other OpenMP runtimes (igraph/numba); on some platforms
        # (notably macOS) letting torch spawn its own OpenMP threads alongside
        # them can crash the process. NEExT graphs are small, so single-threaded
        # torch is appropriate here. Save/restore to avoid surprising callers.
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            graphs = self._build_graph_tensors()
            input_dim = len(self.feature_columns)

            model = self._build_model(input_dim)
            model = self._train(model, graphs, input_dim)

            model.eval()
            rows = []
            with torch.no_grad():
                for g in graphs:
                    node_emb = model(g["x"], g["op"])
                    graph_emb = self._pool(node_emb).cpu().numpy()
                    rows.append((g["graph_id"], graph_emb))
        finally:
            torch.set_num_threads(previous_threads)

        embedding_columns = [f"emb_{i}" for i in range(self.embedding_dimension)]
        data = {"graph_id": [gid for gid, _ in rows]}
        matrix = np.vstack([vec for _, vec in rows])
        for i, col in enumerate(embedding_columns):
            data[col] = matrix[:, i]
        embeddings_df = pd.DataFrame(data)

        return Embeddings(
            embeddings_df=embeddings_df,
            embedding_name=f"gnn_{self.architecture.lower()}",
            embedding_columns=embedding_columns,
        )
