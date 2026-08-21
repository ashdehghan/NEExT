"""Figure PROTOTYPES for the baselines section (brainstorm pass with Ash,
2026-08-21). Everything regenerates from outputs/combined_results.csv,
results.csv and the leakage-control separation CSVs. Run from the repo root;
writes figures/protos/*.png|pdf.

A  overview_boxes      2x2 panels; floor / node features / best bag as BOXES + guide lines, then the 6 embeddings
B  structure_scatter   one panel: shared-split lift (x) vs region-holdout lift at B=2 (y), one point per method x dataset
C  retention_bars      holdout lift at B=0 and B=2 as a fraction of the shared-split lift, per method, per dataset
D  anchors_boxes       synthetic beta=0 / beta=0.03 worlds: shared-split macro-F1 per method (same layout as A)
E  sensitivity         seeds 42/43/44 and D=128 per method on Airports + Chameleon (dot plot)
"""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(SESSION_ROOT / "code"))

from lib.containment import plotstyle as ps  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402

OUT = SESSION_ROOT / "figures" / "protos"
SEP_DIR = RESEARCH_ROOT / "sessions" / "2026-08-13-leakage-control" / "outputs"
METHODS = {
    "deepwalk": ("proximity", "DeepWalk"),
    "node2vec_bfs": ("proximity", "n2v BFS"),
    "node2vec_dfs": ("proximity", "n2v DFS"),
    "struc2vec": ("structural", "struc2vec"),
    "role2vec": ("structural", "Role2Vec"),
    "graphwave": ("structural", "GraphWave"),
}
FAM = {"proximity": ps.FAMILY_COLOR["emb_proximity"], "structural": ps.FAMILY_COLOR["emb_structural"]}
REF = {"floor": ps.FAMILY_COLOR["floor"], "node": ps.FAMILY_COLOR["egonet_walk"], "bag": ps.FAMILY_COLOR["egonet_hop"]}
TITLES = {"airports": "(a) Air Traffic -- Europe", "roman": "(b) Roman Empire", "chameleon": "(c) Wikipedia -- Chameleon", "amazon": "(d) Amazon Ratings"}
DSETS = ["airports", "roman", "chameleon", "amazon"]
TAG = "d16-s42"


def _box(ax, vals, x, color, w=0.55, hatch=None):
    b = ax.boxplot([vals], positions=[x], widths=w, showfliers=False, patch_artist=True, zorder=3)
    p = b["boxes"][0]
    p.set_facecolor(color)
    p.set_alpha(0.35)
    p.set_edgecolor(color)
    if hatch:
        p.set_hatch(hatch)
    b["medians"][0].set_color(ps.INK)
    for part in ("whiskers", "caps"):
        for line in b[part]:
            line.set_color(color)


def _set_ylim(ax, arrays, pad=0.05, headroom=0.16):
    """Explicit limits from the plotted data (boxes + whiskers never exceed the
    data range) with extra headroom at the top for the in-axes panel tag."""
    allv = np.concatenate([np.asarray(a, float) for a in arrays if len(a)])
    lo, hi = allv.min(), allv.max()
    rng = hi - lo
    ax.set_ylim(lo - pad * rng, hi + headroom * rng)


def ref_series(comb, key):
    d = comb[comb.dataset_key == key]
    best = C.DATASETS[key]["expected_best_bag"]
    if best is None:  # synthetic anchors: best bag = argmax over the session's bag rows
        bags = d[d.family.isin(["bag_hop", "bag_walk"])]
        best = bags.groupby(["method", "feature_tag"])["f1_macro"].mean().idxmax() if len(bags) else ("none", "none")
    bm, bt = best
    return {
        "floor": d.loc[(d.method == "permuted") & (d.feature_tag == "fall-vl1"), "f1_macro"].to_numpy(),
        "node": d.loc[(d.method == "node_struct") & (d.feature_tag == "fall-vl1"), "f1_macro"].to_numpy(),
        "bag": d.loc[(d.method == bm) & (d.feature_tag == bt), "f1_macro"].to_numpy(),
    }


def proto_A(comb):
    ps.use_style()
    fig, axes = plt.subplots(2, 2, figsize=(ps.FULL_W, 5.2))
    for ax, key in zip(axes.ravel(), DSETS):
        d = comb[comb.dataset_key == key]
        refs = ref_series(comb, key)
        plotted = [refs[n] for n in REF]
        # reference boxes at x = 0, 1, 2
        for x, (name, color) in enumerate(REF.items()):
            _box(ax, refs[name], x, color, hatch="////" if name == "floor" else None)
        ax.axhline(refs["floor"].mean(), color=REF["floor"], lw=0.8, ls=(0, (4, 3)), zorder=0)
        ax.axhline(refs["bag"].mean(), color=REF["bag"], lw=0.8, ls=(0, (4, 3)), zorder=0)
        ax.axhline(refs["node"].mean(), color=REF["node"], lw=0.8, ls=(0, (1, 1.6)), zorder=0)
        ax.axvline(2.75, color=ps.GRID, lw=0.8, zorder=0)
        ax.axvline(6.25, color=ps.GRID, lw=0.8, zorder=0)
        for i, (m, (fam, _)) in enumerate(METHODS.items()):
            vals = d.loc[(d.method == m) & (d.feature_tag == TAG), "f1_macro"].to_numpy()
            plotted.append(vals)
            _box(ax, vals, 3.5 + i, FAM[fam])
        ax.set_xticks([0, 1, 2] + [3.5 + i for i in range(6)])
        ax.set_xticklabels(["floor", "node feat.", "best bag"] + [lab for _, lab in METHODS.values()], rotation=35, ha="right", fontsize=6.5)
        ax.set_xlim(-0.6, 9.1)
        _set_ylim(ax, plotted)
        ps.panel_tag(ax, TITLES[key])
    for ax in axes[:, 0]:
        ax.set_ylabel("macro-F1")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=REF["floor"], alpha=0.35, edgecolor=REF["floor"], hatch="////", label="Permutation floor"),
        plt.Rectangle((0, 0), 1, 1, facecolor=REF["node"], alpha=0.35, edgecolor=REF["node"], label="Node features"),
        plt.Rectangle((0, 0), 1, 1, facecolor=REF["bag"], alpha=0.35, edgecolor=REF["bag"], label="Best egonet bag"),
        plt.Rectangle((0, 0), 1, 1, facecolor=FAM["proximity"], alpha=0.35, edgecolor=FAM["proximity"], label="Proximity embedding"),
        plt.Rectangle((0, 0), 1, 1, facecolor=FAM["structural"], alpha=0.35, edgecolor=FAM["structural"], label="Structural embedding"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
    fig.tight_layout()
    ps.save(fig, OUT / "A_overview_boxes")


def _lift(sep, method, tag, B):
    ok = sep[(sep.status == "ok") & (sep.buffer_hops == B)]
    m = ok[(ok.method == method) & (ok.feature_tag == tag)].set_index("split")["f1_macro"]
    f = ok[(ok.method == f"permuted_{method}") & (ok.feature_tag == tag)].set_index("split")["f1_macro"]
    idx = m.index.intersection(f.index)
    return (m.loc[idx] - f.loc[idx]).to_numpy()


def holdout_points(comb):
    """Rows: dataset, method, family, shared_lift, lift_B0, lift_B2 (embeddings + bag + node)."""
    rows = []
    for key in ("amazon", "roman", "chameleon"):
        emb = pd.read_csv(SEP_DIR / f"separation_{key}_emb.csv")
        base = pd.read_csv(SEP_DIR / f"separation_{key}.csv")
        d = comb[comb.dataset_key == key]
        for m, (fam, lab) in METHODS.items():
            sh = d.loc[(d.method == m) & (d.feature_tag == TAG), "f1_macro"].mean() - d.loc[(d.method == f"permuted_{m}") & (d.feature_tag == TAG), "f1_macro"].mean()
            rows.append({"dataset": key, "method": lab, "family": fam, "shared": sh,
                         "B0": _lift(emb, m, TAG, 0).mean(), "B2": _lift(emb, m, TAG, 2).mean()})
        bm, bt = C.DATASETS[key]["expected_best_bag"]
        sh = d.loc[(d.method == bm) & (d.feature_tag == bt), "f1_macro"].mean() - d.loc[(d.method == "permuted") & (d.feature_tag == bt), "f1_macro"].mean()
        rows.append({"dataset": key, "method": "best bag", "family": "bag", "shared": sh, "B0": _lift(base, bm, bt, 0).mean(), "B2": _lift(base, bm, bt, 2).mean()})
        sh = d.loc[(d.method == "node_struct") & (d.feature_tag == "fall-vl1"), "f1_macro"].mean() - d.loc[(d.method == "permuted") & (d.feature_tag == "fall-vl1"), "f1_macro"].mean()
        rows.append({"dataset": key, "method": "node features", "family": "node", "shared": sh, "B0": _lift(base, "node_struct", "fall-vl1", 0).mean(), "B2": _lift(base, "node_struct", "fall-vl1", 2).mean()})
    return pd.DataFrame(rows)


MARK = {"amazon": "o", "roman": "s", "chameleon": "^"}
COL = {"proximity": FAM["proximity"], "structural": FAM["structural"], "bag": REF["bag"], "node": REF["node"]}


def proto_B(comb):
    pts = holdout_points(comb)
    ps.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 3.0))
    for ax, B, keys in ((axes[0], "B0", ("amazon", "roman", "chameleon")), (axes[1], "B2", ("amazon", "roman"))):
        d = pts[pts.dataset.isin(keys)]
        lim = max(d["shared"].max(), d[B].max()) + 0.03
        ax.plot([0, lim], [0, lim], color=ps.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax.axhline(0, color=ps.AXIS, lw=0.8, zorder=0)
        ax.fill_between([0, lim], [0, 0], [0, lim], color=ps.GRID, alpha=0.35, zorder=0)
        for _, r in d.iterrows():
            ax.scatter(r["shared"], r[B], marker=MARK[r["dataset"]], s=34, facecolor=COL[r["family"]], edgecolor=ps.INK, lw=0.4, zorder=3)
        ax.set_xlabel("shared-split lift over floor")
        ax.set_xlim(-0.01, lim)
        ax.set_ylim(min(-0.03, d[B].min() - 0.02), lim)
        ax.text(lim * 0.45, lim * 0.55, "structure\n(lift survives)", fontsize=7, color=ps.MUTED, ha="center")
        ax.text(lim * 0.8, lim * 0.08, "position\n(lift dies)", fontsize=7, color=ps.MUTED, ha="center")
        ps.panel_tag(ax, f"$B={B[1]}$" + ("" if B == "B0" else " (Chameleon not separable)"))
    axes[0].set_ylabel("region-holdout lift over paired floor")
    handles = [plt.Line2D([], [], marker=MARK[k], ls="", color=ps.INK, label=lab) for k, lab in (("amazon", "Amazon"), ("roman", "Roman"), ("chameleon", "Chameleon"))]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor=COL[f], edgecolor=ps.INK, lw=0.4, label=lab) for f, lab in (("proximity", "proximity"), ("structural", "structural"), ("bag", "best bag"), ("node", "node features"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=7, frameon=False)
    fig.tight_layout()
    ps.save(fig, OUT / "B_structure_scatter")


def proto_C(comb):
    pts = holdout_points(comb)
    ps.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_W, 2.6))
    for ax, key in zip(axes, ("amazon", "roman", "chameleon")):
        d = pts[pts.dataset == key].reset_index(drop=True)
        x = np.arange(len(d))
        for i, r in d.iterrows():
            ax.bar(i - 0.18, r["shared"], 0.34, color=COL[r["family"]], alpha=0.25, edgecolor=COL[r["family"]], lw=0.6)
            ax.bar(i + 0.18, r["B2"] if key != "chameleon" else r["B0"], 0.34, color=COL[r["family"]], alpha=0.8, edgecolor=COL[r["family"]], lw=0.6)
        ax.axhline(0, color=ps.AXIS, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(d["method"], rotation=45, ha="right", fontsize=6.5)
        ps.panel_tag(ax, {"amazon": "Amazon ($B$=2)", "roman": "Roman ($B$=2)", "chameleon": "Chameleon ($B$=0)"}[key])
    axes[0].set_ylabel("macro-F1 lift over paired floor")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=ps.MUTED, alpha=0.25, edgecolor=ps.MUTED, label="shared splits"),
               plt.Rectangle((0, 0), 1, 1, facecolor=ps.MUTED, alpha=0.8, edgecolor=ps.MUTED, label="region holdout")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    ps.save(fig, OUT / "C_retention_bars")


def proto_D(comb):
    ps.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.7), sharey=True)
    for ax, key, title in zip(axes, ("synth_b0", "synth_b1"), ("$\\beta=0$: no structure--label signal", "$\\beta=0.03$: faint planted trace")):
        d = comb[comb.dataset_key == key]
        refs = ref_series(comb, key)
        plotted = [refs[n] for n in REF if len(refs[n])]
        for x, (name, color) in enumerate(REF.items()):
            if len(refs[name]):
                _box(ax, refs[name], x, color, hatch="////" if name == "floor" else None)
        ax.axhline(refs["floor"].mean(), color=REF["floor"], lw=0.8, ls=(0, (4, 3)), zorder=0)
        ax.axvline(2.75, color=ps.GRID, lw=0.8, zorder=0)
        ax.axvline(6.25, color=ps.GRID, lw=0.8, zorder=0)
        for i, (m, (fam, _)) in enumerate(METHODS.items()):
            vals = d.loc[(d.method == m) & (d.feature_tag == TAG), "f1_macro"].to_numpy()
            plotted.append(vals)
            _box(ax, vals, 3.5 + i, FAM[fam])
        ax.set_xticks([0, 1, 2] + [3.5 + i for i in range(6)])
        ax.set_xticklabels(["floor", "node feat.", "best bag"] + [lab for _, lab in METHODS.values()], rotation=35, ha="right", fontsize=6.5)
        ax.set_xlim(-0.6, 9.1)
        _set_ylim(ax, plotted)
        ps.panel_tag(ax, title)
    axes[0].set_ylabel("macro-F1 (shared splits)")
    fig.tight_layout()
    ps.save(fig, OUT / "D_anchors_boxes")


def proto_E():
    r = pd.read_csv(SESSION_ROOT / "results.csv")
    r = r[(r.status == "ok") & (~r.method.str.startswith("permuted"))]
    g = r.groupby(["dataset_key", "method", "dim", "embed_seed"])["f1_macro"].mean().reset_index()
    ps.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.6))
    for ax, key in zip(axes, ("airports", "chameleon")):
        d = g[g.dataset_key == key]
        for i, (m, (fam, lab)) in enumerate(METHODS.items()):
            s16 = d[(d.method == m) & (d.dim == 16)]
            for _, row in s16.iterrows():
                off = {42: -0.12, 43: 0.0, 44: 0.12}[int(row.embed_seed)]
                ax.scatter(i + off, row.f1_macro, s=22, facecolor=FAM[fam], edgecolor=ps.INK, lw=0.4, zorder=3)
            d128 = d[(d.method == m) & (d.dim == 128)]
            if len(d128):
                ax.scatter(i + 0.3, d128.f1_macro.iloc[0], s=34, marker="D", facecolor="white", edgecolor=FAM[fam], lw=1.0, zorder=3)
        ax.set_xticks(range(6))
        ax.set_xticklabels([lab for _, lab in METHODS.values()], rotation=35, ha="right", fontsize=6.5)
        ax.set_title(TITLES[key], fontsize=8, loc="left")
    axes[0].set_ylabel("macro-F1 (split mean)")
    handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor=ps.MUTED, markeredgecolor=ps.INK, label="seeds 42 / 43 / 44, $D$=16"),
               plt.Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor=ps.MUTED, label="$D$=128, seed 42")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    ps.save(fig, OUT / "E_sensitivity")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comb = pd.read_csv(SESSION_ROOT / "outputs" / "combined_results.csv")
    proto_A(comb)
    proto_B(comb)
    proto_C(comb)
    proto_D(comb)
    proto_E()
    print("written", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
