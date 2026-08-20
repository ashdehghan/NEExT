"""Compact per-dataset slide figures (exp1s/exp2s/exp3s) with the extended
k range, replacing the originals whose inline render scripts were not
preserved. Same visual grammar as the deck's existing s-series: floor box,
node-features box (green, dashed reference line), local/global box pairs
per k; no walk columns (the deck discusses walks in prose).

Run from the repo root; outputs into this session's figures/.
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from lib.containment import plotstyle as ps
from lib.nodeclass.figures import LOCAL_TAG, GLOBAL_TAG, NODE_COLOR, SCOPE_COLOR

import matplotlib.pyplot as plt
import pandas as pd

SESSIONS = {
    "exp1s": ("2026-08-06-egonet-signal-airports-eu", 5),
    "exp2s": ("2026-08-06-egonet-signal-roman-empire", 5),
    "exp3s": ("2026-08-06-egonet-signal-wiki-chameleon", 5),
}
PAIR_OFF = 0.19
BOX_W = 0.3


def _box(ax, values, position, color, width=BOX_W):
    b = ax.boxplot([values], positions=[position], widths=width, showfliers=False, patch_artist=True, manage_ticks=False)
    b["boxes"][0].set_facecolor(color)
    b["boxes"][0].set_alpha(0.4)
    b["boxes"][0].set_edgecolor(color)
    b["medians"][0].set_color(ps.INK)


def render(name, session, kmax):
    df = pd.read_csv(RESEARCH_ROOT / "sessions" / session / "results.csv")
    df = df[df.status == "ok"]

    def vals(tag, method):
        v = df.loc[(df.feature_tag == tag) & (df.method == method), "f1_macro"].to_numpy()
        return v if len(v) else None

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    floor = vals(LOCAL_TAG, "permuted")
    node = vals(LOCAL_TAG, "node_struct")
    _box(ax, floor, 0.0, ps.FAMILY_COLOR["floor"])
    ax.axhline(float(floor.mean()), color=ps.MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
    _box(ax, node, 1.0, NODE_COLOR)
    ax.axhline(float(node.mean()), color=NODE_COLOR, linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
    ticks, labels = [0.0, 1.0], ["floor", "node"]
    for i, k in enumerate(range(1, kmax + 1)):
        c = 2.1 + i
        lv = vals(LOCAL_TAG, f"egonet_k{k}_wass")
        gv = vals(GLOBAL_TAG, f"egonet_k{k}_wass")
        if lv is not None:
            _box(ax, lv, c - PAIR_OFF, SCOPE_COLOR["local"])
        if gv is not None:
            _box(ax, gv, c + PAIR_OFF, SCOPE_COLOR["global"])
        ticks.append(c)
        labels.append(f"$k$={k}")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("macro-F1", fontsize=12)
    fig.tight_layout()
    ps.save(fig, SESSION_ROOT / "figures" / name)
    print("saved", name)


def main():
    ps.use_style()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
    for name, (session, kmax) in SESSIONS.items():
        render(name, session, kmax)


if __name__ == "__main__":
    main()
