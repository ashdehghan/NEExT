"""Board-sized figures: same style family as the manuscript, sized to be read.

Three renders:
- board_profiles: 3-panel mean±std k-profile summary (all worlds), full-width.
- board_overviews: the three whisper-ON worlds as stacked Layout-A box-plot
  rows sharing one legend — same colors/conventions as the manuscript
  overview figures, composed as one readable column.
- board_money: plateau k* vs measured reach, larger fonts.

Fonts are bumped for board display; aspect ratios are printed so scratchpad
boxes can match them exactly (no squish).

Run from the repo root so the project env (matplotlib) resolves.
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SESSION_ROOT.parents[1]))

from lib.containment import plotstyle as ps
from lib.nodeclass.figures import BOX_W, LOCAL_TAG, GLOBAL_TAG, NODE_COLOR, PAIR_OFFSET, SCOPE_COLOR

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

WORLDS = {
    "SYNTH_HR_L1_B1": "World 1 — small patches ($\\lambda$=0.10, reach 2.7)",
    "SYNTH_HR_L2_B1": "World 2 — medium patches ($\\lambda$=0.20, reach 5.1)",
    "SYNTH_HR_L3_B1": "World 3 — large patches ($\\lambda$=0.35, reach 8.2)",
}
REACH = {"SYNTH_HR_L1_B1": 2.68, "SYNTH_HR_L2_B1": 5.10, "SYNTH_HR_L3_B1": 8.22,
         "SYNTH_HR_L1_B0": 2.83, "SYNTH_HR_L2_B0": 5.44, "SYNTH_HR_L3_B0": 8.14}
COUPLING_STYLE = {"B1": "-", "B0": (0, (4, 2))}


def load():
    df = pd.read_csv(SESSION_ROOT / "results.csv")
    return df[df.status == "ok"]


def save(fig, name):
    w, h = fig.get_size_inches()
    ps.save(fig, SESSION_ROOT / "figures" / name)
    print(f"{name}: aspect {w / h:.3f}")


def board_profiles(df):
    kh = df[df.method.str.startswith("egonet_k")].copy()
    kh["k"] = kh.method.str.extract(r"k(\d)").astype(int)
    kh["scope"] = kh.feature_tag.map(lambda t: "global" if t.endswith("glob") else "local")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (cell, title) in zip(axes, WORLDS.items()):
        for B in ("B1", "B0"):
            ds = cell.replace("B1", B)
            for scope in ("global", "local"):
                g = kh[(kh.dataset == ds) & (kh.scope == scope)].groupby("k")["f1_macro"].agg(["mean", "std"])
                ax.errorbar(g.index, g["mean"], yerr=g["std"], color=SCOPE_COLOR[scope],
                            linestyle=COUPLING_STYLE[B], marker="o", ms=3.5, lw=1.4, capsize=2,
                            alpha=1.0 if B == "B1" else 0.6)
        floor = df[(df.dataset == cell) & (df.method == "permuted") & (df.feature_tag == LOCAL_TAG)]["f1_macro"].mean()
        ax.axhline(floor, color=ps.MUTED, lw=0.9, linestyle=(0, (2, 2)))
        ax.axvline(REACH[cell], color=ps.INK, lw=1.1, linestyle=(0, (5, 2)), alpha=0.55)
        ax.set_xlabel("$k$ (hops)")
        ax.set_xticks(range(1, 7))
        ax.set_xlim(0.6, 6.4)
        ax.set_title(title.split(" — ")[1], fontsize=11)
    axes[0].set_ylabel("macro-F1")
    handles = [
        plt.Line2D([], [], color=SCOPE_COLOR["global"], lw=1.6, label="Global (full-graph)"),
        plt.Line2D([], [], color=SCOPE_COLOR["local"], lw=1.6, label="Local (in-bag)"),
        plt.Line2D([], [], color=ps.INK, lw=1.6, label="whisper ON ($\\beta$=0.03)"),
        plt.Line2D([], [], color=ps.INK, lw=1.6, linestyle=(0, (4, 2)), alpha=0.6, label="whisper OFF ($\\beta$=0)"),
        plt.Line2D([], [], color=ps.INK, lw=1.1, linestyle=(0, (5, 2)), alpha=0.55, label="Measured reach"),
        plt.Line2D([], [], color=ps.MUTED, lw=0.9, linestyle=(0, (2, 2)), label="Random floor"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    save(fig, "board_profiles_v4")


def board_profiles_box(df):
    """Side-by-side panels like board_profiles, but small per-k box plots
    (over the 10 splits) with a trend line through each series' medians."""
    kh = df[df.method.str.startswith("egonet_k")].copy()
    kh["k"] = kh.method.str.extract(r"k(\d)").astype(int)
    kh["scope"] = kh.feature_tag.map(lambda t: "global" if t.endswith("glob") else "local")
    offsets = {("global", "B1"): -0.27, ("global", "B0"): -0.09, ("local", "B1"): +0.09, ("local", "B0"): +0.27}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (cell, title) in zip(axes, WORLDS.items()):
        for (scope, B), off in offsets.items():
            ds = cell.replace("B1", B)
            color = SCOPE_COLOR[scope]
            alpha = 0.4 if B == "B1" else 0.18
            medians = []
            for k in range(1, 7):
                vals = kh[(kh.dataset == ds) & (kh.scope == scope) & (kh.k == k)]["f1_macro"].to_numpy()
                b = ax.boxplot([vals], positions=[k + off], widths=0.15, showfliers=False,
                               patch_artist=True, zorder=3)
                b["boxes"][0].set_facecolor(color)
                b["boxes"][0].set_alpha(alpha)
                b["boxes"][0].set_edgecolor(color)
                b["medians"][0].set_color(ps.INK)
                for part in ("whiskers", "caps"):
                    for line in b[part]:
                        line.set_color(color)
                        line.set_alpha(0.4 if B == "B1" else 0.25)
                medians.append(float(pd.Series(vals).median()))
            ax.plot([k + off for k in range(1, 7)], medians, color=color, lw=1.2,
                    linestyle=COUPLING_STYLE[B], alpha=1.0 if B == "B1" else 0.55, zorder=2)
        floor = df[(df.dataset == cell) & (df.method == "permuted") & (df.feature_tag == LOCAL_TAG)]["f1_macro"].mean()
        ax.axhline(floor, color=ps.MUTED, lw=0.9, linestyle=(0, (2, 2)))
        ax.axvline(REACH[cell], color=ps.INK, lw=1.1, linestyle=(0, (5, 2)), alpha=0.55)
        ax.set_xlabel("$k$ (hops)")
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels([f"$k$={k}" for k in range(1, 7)])
        ax.set_xlim(0.45, 6.55)
        ax.set_title(title.split(" — ")[1], fontsize=11)
    axes[0].set_ylabel("macro-F1")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SCOPE_COLOR["global"], alpha=0.4,
                      edgecolor=SCOPE_COLOR["global"], label="Global (full-graph)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=SCOPE_COLOR["local"], alpha=0.4,
                      edgecolor=SCOPE_COLOR["local"], label="Local (in-bag)"),
        plt.Line2D([], [], color=ps.INK, lw=1.4, label="whisper ON ($\\beta$=0.03)"),
        plt.Line2D([], [], color=ps.INK, lw=1.4, linestyle=(0, (4, 2)), alpha=0.55, label="whisper OFF ($\\beta$=0)"),
        plt.Line2D([], [], color=ps.INK, lw=1.1, linestyle=(0, (5, 2)), alpha=0.55, label="Measured reach"),
        plt.Line2D([], [], color=ps.MUTED, lw=0.9, linestyle=(0, (2, 2)), label="Random floor"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    save(fig, "board_profiles_box")


def _box(ax, values, position, color):
    b = ax.boxplot([values], positions=[position], widths=BOX_W, showfliers=False, patch_artist=True)
    b["boxes"][0].set_facecolor(color)
    b["boxes"][0].set_alpha(0.35)
    b["boxes"][0].set_edgecolor(color)
    b["medians"][0].set_color(ps.INK)


def board_overviews(df):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.6), sharex=True)
    for ax, (cell, title) in zip(axes, WORLDS.items()):
        sub = df[df.dataset == cell]

        def vals(tag, method):
            return sub.loc[(sub.feature_tag == tag) & (sub.method == method), "f1_macro"].to_numpy()

        floor = vals(LOCAL_TAG, "permuted")
        node = vals(LOCAL_TAG, "node_struct")
        _box(ax, floor, 0.0, ps.FAMILY_COLOR["floor"])
        ax.axhline(floor.mean(), color=ps.MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
        _box(ax, node, 1.05, NODE_COLOR)
        ax.axhline(node.mean(), color=NODE_COLOR, linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
        for i, k in enumerate(range(1, 7)):
            center = 2.35 + i
            _box(ax, vals(LOCAL_TAG, f"egonet_k{k}_wass"), center - PAIR_OFFSET, SCOPE_COLOR["local"])
            _box(ax, vals(GLOBAL_TAG, f"egonet_k{k}_wass"), center + PAIR_OFFSET, SCOPE_COLOR["global"])
        ax.set_ylabel("macro-F1")
        ax.set_title(title, fontsize=11)
    axes[-1].set_xticks([0.0, 1.05] + [2.35 + i for i in range(6)])
    axes[-1].set_xticklabels(["Random\nfloor", "Node\nfeatures"] + [f"$k$={k}" for k in range(1, 7)])
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ps.FAMILY_COLOR["floor"], alpha=0.35,
                      edgecolor=ps.FAMILY_COLOR["floor"], label="Random floor"),
        plt.Rectangle((0, 0), 1, 1, facecolor=NODE_COLOR, alpha=0.35, edgecolor=NODE_COLOR, label="Node features"),
        plt.Rectangle((0, 0), 1, 1, facecolor=SCOPE_COLOR["local"], alpha=0.35,
                      edgecolor=SCOPE_COLOR["local"], label="Local (in-bag)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=SCOPE_COLOR["global"], alpha=0.35,
                      edgecolor=SCOPE_COLOR["global"], label="Global (full-graph)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "board_overviews_stack")


def board_money(df):
    kh = df[df.method.str.startswith("egonet_k")].copy()
    kh["k"] = kh.method.str.extract(r"k(\d)").astype(int)
    kh["scope"] = kh.feature_tag.map(lambda t: "global" if t.endswith("glob") else "local")

    def k_star(ds, scope):
        piv = kh[(kh.dataset == ds) & (kh.scope == scope)].pivot_table(index="split", columns="k", values="f1_macro")
        last = 1
        for k in range(2, 7):
            _, p = stats.ttest_rel(piv[k], piv[k - 1])
            if p < 0.05 and (piv[k] - piv[k - 1]).mean() > 0:
                last = k
        return last

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for scope in ("global", "local"):
        for B, marker, fill in (("B1", "o", "full"), ("B0", "s", "none")):
            cells = [f"SYNTH_HR_{L}_{B}" for L in ("L1", "L2", "L3")]
            ax.plot([REACH[c] for c in cells], [k_star(c, scope) for c in cells],
                    marker=marker, ms=7, lw=1.4, color=SCOPE_COLOR[scope],
                    linestyle="-" if B == "B1" else (0, (4, 2)), fillstyle=fill,
                    label=f"{scope}, whisper {'ON' if B == 'B1' else 'OFF'}")
    lim = [1.5, 9]
    ax.plot(lim, lim, color=ps.MUTED, lw=1.0, zorder=0)
    ax.text(4.55, 4.0, "perfect tracking ($k^*$=reach)", fontsize=9, color=ps.MUTED, rotation=33)
    ax.axhline(6, color=ps.GRID, lw=0.8)
    ax.text(1.75, 6.1, "sweep limit $k$=6", fontsize=9, color=ps.MUTED)
    ax.set_xlabel("measured homophily reach (hops)")
    ax.set_ylabel("plateau $k^*$ (last significant gain)")
    ax.set_xlim(*lim)
    ax.set_ylim(1.5, 6.8)
    ax.set_yticks(range(2, 7))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    save(fig, "board_money_v4")


def main():
    df = load()
    (SESSION_ROOT / "figures").mkdir(exist_ok=True)
    with plt.rc_context({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10}):
        ps.use_style()
        plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                             "ytick.labelsize": 10, "legend.fontsize": 10})
        board_profiles_box(df)


if __name__ == "__main__":
    main()
