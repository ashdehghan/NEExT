"""Manuscript pipeline figure for the Framework section (fig:framework-pipeline).

Redesigned 2026-09-12 around one idea per stage, read left to right:

  graph G with a few highlighted nodes  ->  one ego-net per node  ->  each
  ego-net becomes a cloud of member feature vectors in R^p  ->  each cloud
  becomes one vector z_v in R^D (the node embedding)  ->  any downstream
  model.

Three nodes are followed through the pipeline in three colours so the
"collection of distributions" idea is visible. All labels are symbolic (k, p,
D); nothing in the figure is tied to one experimental configuration. The
feature-space panel plots two real full-graph measurements of the toy
graph's nodes (degree and clustering coefficient), so the clouds are what
the method actually sees, not decoration.

Run from the repo root; outputs into this session's figures/.
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from lib.containment import plotstyle as ps

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

TRACK = ["#eb6834", "#2a78d6", "#1baf7a"]  # the three followed nodes (validated palette slots)
NODE_FILL = "#e4e2dc"
EDGE_INK = "#b9b7b0"


def toy_graph(seed=7, n=60, radius=0.19):
    rng = np.random.RandomState(seed)
    pos = rng.rand(n, 2)
    G = nx.random_geometric_graph(n, radius, pos={i: pos[i] for i in range(n)}, seed=seed)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    pos = {i: pos[i] for i in G.nodes}
    targets = [(0.22, 0.72), (0.75, 0.68), (0.5, 0.2)]
    centres = []
    for t in targets:
        cand = sorted(G.nodes, key=lambda i: np.hypot(*(pos[i] - t)))
        centres.append(next(c for c in cand if c not in centres))
    return G, pos, centres


def draw_edges(ax, G, pos, nodes=None, color=EDGE_INK, lw=0.5, alpha=1.0):
    for u, v in G.edges:
        if nodes is not None and (u not in nodes or v not in nodes):
            continue
        ax.plot(*zip(pos[u], pos[v]), color=color, lw=lw, alpha=alpha, zorder=1, solid_capstyle="round")


def draw_nodes(ax, pos, nodes, color, size, z=2, edge=ps.INK):
    xy = np.array([pos[u] for u in nodes])
    ax.scatter(xy[:, 0], xy[:, 1], s=size, color=color, edgecolors=edge, linewidths=0.4, zorder=z)


def stage_title(ax, text):
    fig = ax.figure
    ax._stage_title = text  # placed after layout by place_titles()


def place_titles(fig, gs, cols):
    for col, text in cols:
        bb = gs[:, col].get_position(fig)
        fig.text((bb.x0 + bb.x1) / 2, 0.93, text, fontsize=8, ha="center", va="center", color=ps.INK)


def main():
    ps.use_style()
    G, pos, centres = toy_graph()
    k = 1
    egonets = [set(nx.single_source_shortest_path_length(G, c, cutoff=k)) for c in centres]
    deg = dict(G.degree())
    clu = nx.clustering(G)

    fig = plt.figure(figsize=(ps.FULL_W, 2.55))
    gs = fig.add_gridspec(
        3, 4, width_ratios=[1.25, 0.8, 1.05, 0.95],
        left=0.01, right=0.99, top=0.86, bottom=0.08, wspace=0.45, hspace=0.35,
    )

    # Stage 1: the graph with three followed nodes
    ax = fig.add_subplot(gs[:, 0])
    draw_edges(ax, G, pos)
    draw_nodes(ax, pos, list(G.nodes), NODE_FILL, 11, z=2)
    for c, col in zip(centres, TRACK):
        draw_nodes(ax, pos, [c], col, 55, z=3)
    ax.set_aspect("equal"); ax.axis("off")
    stage_title(ax, "graph $G$")

    # Stage 2: one ego-net per node (three shown)
    for row, (c, members, col) in enumerate(zip(centres, egonets, TRACK)):
        ax = fig.add_subplot(gs[row, 1])
        draw_edges(ax, G, pos, nodes=members, color=col, lw=0.8, alpha=0.7)
        draw_nodes(ax, pos, [u for u in members if u != c], col, 16, z=2)
        draw_nodes(ax, pos, [c], col, 42, z=3)
        xy = np.array([pos[u] for u in members])
        pad = 0.05
        ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
        ax.set_aspect("equal"); ax.axis("off")
        if row == 0:
            stage_title(ax, "ego-nets $S_k(v)$")

    # Stage 3: each ego-net as a cloud of member feature vectors
    ax = fig.add_subplot(gs[:, 2])
    allx = np.array([deg[u] for u in G.nodes], dtype=float)
    ally = np.array([clu[u] for u in G.nodes], dtype=float)
    rng = np.random.RandomState(1)
    jitter = lambda a, s: a + rng.uniform(-s, s, size=len(a))
    ax.scatter(jitter(allx, 0.18), ally, s=7, color=NODE_FILL, edgecolors="none", zorder=1)
    for members, col in zip(egonets, TRACK):
        mx = np.array([deg[u] for u in members], dtype=float)
        my = np.array([clu[u] for u in members], dtype=float)
        ax.scatter(jitter(mx, 0.18), my, s=16, color=col, edgecolors=ps.INK, linewidths=0.3, alpha=0.9, zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    stage_title(ax, r"member features $\phi(u)\in\mathbb{R}^{p}$")
    ax.text(0.5, -0.04, "two of the $p$ feature axes; one cloud per ego-net", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.5, color=ps.MUTED)

    # Stage 4: one vector per node, then a downstream model
    ax = fig.add_subplot(gs[:, 3])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    D = 7
    for row, (members, col) in enumerate(zip(egonets, TRACK)):
        y0 = 0.84 - row * 0.19
        mx = np.mean([deg[u] for u in members]); my = np.mean([clu[u] for u in members])
        vals = np.abs(np.sin(np.arange(D) * (0.7 + 0.4 * row) + 3.0 * mx / 10 + my))
        for j in range(D):
            ax.add_patch(mpatches.Rectangle((0.08 + j * 0.10, y0), 0.09, 0.11,
                                            facecolor=col, alpha=0.25 + 0.7 * vals[j], edgecolor="none"))
        ax.text(0.03, y0 + 0.055, "$z_v$", ha="right", va="center", fontsize=7, color=col)
    ax.text(0.43, 0.36, r"one vector per node, $z_v\in\mathbb{R}^{D}$", ha="center", va="center", fontsize=7, color=ps.INK)
    ax.annotate("", xy=(0.43, 0.20), xytext=(0.43, 0.30),
                arrowprops=dict(arrowstyle="->", color=ps.INK, lw=0.9))
    ax.add_patch(mpatches.FancyBboxPatch((0.02, 0.04), 0.82, 0.13, boxstyle="round,pad=0.01",
                                         facecolor="white", edgecolor=ps.INK, lw=0.7, clip_on=False))
    ax.text(0.43, 0.105, "downstream model", ha="center", va="center", fontsize=6.5, color=ps.INK)
    stage_title(ax, "Wasserstein embedding")

    place_titles(fig, gs, [(0, "graph $G$"), (1, "ego-nets $S_k(v)$"),
                           (2, r"member features $\phi(u)\in\mathbb{R}^{p}$"), (3, "Wasserstein embedding")])

    # Arrows between stages, with the operation named on top
    for x, label in [(0.300, "extract"), (0.513, "measure"), (0.764, "embed")]:
        fig.text(x, 0.50, r"$\rightarrow$", fontsize=14, ha="center", va="center", color=ps.INK)
        fig.text(x, 0.60, label, fontsize=6.5, ha="center", va="bottom", color=ps.MUTED)

    ps.save(fig, SESSION_ROOT / "figures" / "framework_pipeline")
    print("saved framework_pipeline")


if __name__ == "__main__":
    main()
