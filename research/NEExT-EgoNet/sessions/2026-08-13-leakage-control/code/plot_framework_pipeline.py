"""Manuscript pipeline figure for the Framework section (fig:framework-pipeline).

One row, four stages with arrows: source graph with a highlighted center ->
the two bag constructions (census k-hop egonet / survey walk bag with
visit-weighted members) -> member features under the two scopes -> weighted
distribution -> Wasserstein embedding -> classifier. Colors follow the house
language: blue = local scope, orange = global scope, accent for the center;
bag membership itself is neutral ink so it cannot be confused with scope.

Run from the repo root; outputs into this session's figures/.
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from lib.containment import plotstyle as ps
from lib.nodeclass.figures import NODE_COLOR, SCOPE_COLOR

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

ACCENT = "#E07016"
BAG_FILL = "#B8B6B0"


def toy_graph(seed=7, n=46, radius=0.21):
    rng = np.random.RandomState(seed)
    pos = rng.rand(n, 2)
    G = nx.random_geometric_graph(n, radius, pos={i: pos[i] for i in range(n)}, seed=seed)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    center = min(G.nodes, key=lambda i: np.hypot(*(pos[i] - 0.48)))
    hops = nx.single_source_shortest_path_length(G, center)
    return G, {i: pos[i] for i in G.nodes}, center, hops


def draw_graph(ax, G, pos, center, hops, k=2, weights=None, title=""):
    members = {u for u, h in hops.items() if h <= k}
    for u, v in G.edges:
        inside = u in members and v in members
        ax.plot(*zip(pos[u], pos[v]), color=ps.INK if inside else ps.MUTED,
                lw=0.8 if inside else 0.4, alpha=0.9 if inside else 0.35, zorder=1)
    for u in G.nodes:
        if u == center:
            c, s, z = ACCENT, 90, 3
        elif u in members:
            c = BAG_FILL
            s = 26 if weights is None else 12 + 240 * weights.get(u, 0.0)
            z = 2
        else:
            c, s, z = "#E4E2DC", 14, 1
        ax.scatter(*pos[u], s=s, color=c, edgecolors=ps.INK, linewidths=0.4, zorder=z)
    ax.set_title(title, fontsize=9.5)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ps.use_style()
    G, pos, center, hops = toy_graph()

    # survey weights: actual short random walks with restart from the center
    rng = np.random.RandomState(3)
    counts = {center: 40}
    adj = {u: list(G.neighbors(u)) for u in G.nodes}
    for _ in range(40):
        u = center
        for _ in range(10):
            u = center if rng.rand() < 0.15 else adj[u][rng.randint(len(adj[u]))]
            counts[u] = counts.get(u, 0) + 1
    kept = {u: c for u, c in counts.items() if c >= 3}
    total = sum(kept.values())
    weights = {u: c / total for u, c in kept.items()}
    walk_hops = {u: 0 for u in kept}

    fig = plt.figure(figsize=(10.8, 3.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.15, 1.0, 1.05, 1.0],
                          left=0.01, right=0.99, top=0.86, bottom=0.04,
                          wspace=0.28, hspace=0.3)

    ax_src = fig.add_subplot(gs[:, 0])
    draw_graph(ax_src, G, pos, center, hops, k=-1,
               title="source graph $G$, center $v$\nlabel $y(v)$")

    ax_cen = fig.add_subplot(gs[0, 1])
    draw_graph(ax_cen, G, pos, center, hops, k=2, title="census: $S_k(v)$, equal mass")
    ax_sur = fig.add_subplot(gs[1, 1])
    draw_graph(ax_sur, G, pos, center, walk_hops, k=0, weights=weights,
               title="survey: walk bag, visit mass")

    # stage 3: member features under the two scopes (schematic strips)
    ax_feat = fig.add_subplot(gs[:, 2])
    rng2 = np.random.RandomState(1)
    for row, (scope, y0) in enumerate([("local", 0.58), ("global", 0.08)]):
        color = SCOPE_COLOR[scope]
        vals = rng2.rand(6, 11)
        for i in range(6):
            for j in range(11):
                ax_feat.add_patch(mpatches.Rectangle(
                    (0.08 + j * 0.075, y0 + i * 0.045), 0.068, 0.038,
                    facecolor=color, alpha=0.25 + 0.6 * vals[i, j], edgecolor="none"))
        ax_feat.text(0.5, y0 + 0.30, f"{scope} scope", color=color,
                     fontsize=9.5, ha="center")
    ax_feat.text(0.5, 0.985, "members $\\times$ 11 features $\\phi(u)$",
                 fontsize=9.5, ha="center", va="top")
    ax_feat.set_xlim(0, 1); ax_feat.set_ylim(0, 1); ax_feat.axis("off")

    # stage 4: distribution -> embedding -> classifier
    ax_out = fig.add_subplot(gs[:, 3])
    pts = rng2.randn(24, 2) * 0.09 + [0.30, 0.72]
    sz = 10 + 60 * rng2.rand(24)
    ax_out.scatter(pts[:, 0], pts[:, 1], s=sz, color=BAG_FILL,
                   edgecolors=ps.INK, linewidths=0.4)
    ax_out.text(0.30, 0.92, "weighted distribution $\\mu_v$", fontsize=9.5, ha="center")
    ax_out.annotate("", xy=(0.30, 0.44), xytext=(0.30, 0.56),
                    arrowprops=dict(arrowstyle="->", color=ps.INK, lw=1.0))
    emb = rng2.rand(11)
    for j in range(11):
        ax_out.add_patch(mpatches.Rectangle(
            (0.06 + j * 0.045, 0.32), 0.04, 0.09,
            facecolor=NODE_COLOR, alpha=0.25 + 0.65 * emb[j], edgecolor="none"))
    ax_out.text(0.30, 0.26, "Wasserstein embedding $\\in \\mathbb{R}^{D}$",
                fontsize=9.5, ha="center")
    ax_out.annotate("", xy=(0.30, 0.115), xytext=(0.30, 0.21),
                    arrowprops=dict(arrowstyle="->", color=ps.INK, lw=1.0))
    ax_out.text(0.30, 0.05, "XGBoost $\\rightarrow \\hat{y}(v)$", fontsize=10,
                ha="center", color=ACCENT)
    ax_out.set_xlim(0, 0.62); ax_out.set_ylim(0, 1); ax_out.axis("off")

    # inter-stage arrows on the figure canvas
    for x in [0.265, 0.505, 0.75]:
        fig.text(x, 0.47, "$\\rightarrow$", fontsize=17, ha="center", va="center",
                 color=ps.INK)

    ps.save(fig, SESSION_ROOT / "figures" / "framework_pipeline")
    print("saved framework_pipeline")


if __name__ == "__main__":
    main()
