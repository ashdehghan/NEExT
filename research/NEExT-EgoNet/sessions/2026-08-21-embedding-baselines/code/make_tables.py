"""Render the baselines tables as LaTeX fragments from the stats CSVs — no
number in the manuscript is hand-typed.

outputs/tables/baselines_shared.tex   family x method x dataset, \accf{mean}{std}{acc},
                                      markers vs best bag (dagger: above, p<.05; ddagger: below)
outputs/tables/baselines_holdout.tex  lift over paired floor at B in {0,2,4}
Run from the repo root after stats_tests.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SESSION_ROOT / "code"))

import config as C  # noqa: E402

OUT = SESSION_ROOT / "outputs"
DATASET_ORDER = ["airports", "roman", "chameleon", "amazon"]
DATASET_HEAD = {"airports": "Airports", "roman": "Roman", "chameleon": "Chameleon", "amazon": "Amazon"}
METHOD_LABEL = {
    "deepwalk": "DeepWalk", "node2vec_bfs": "node2vec (BFS)", "node2vec_dfs": "node2vec (DFS)",
    "struc2vec": "struc2vec", "role2vec": "Role2Vec", "graphwave": "GraphWave",
    "walklets": "Walklets", "netmf": "NetMF",
}
FAMILY_LABEL = {"proximity": "Proximity", "structural": "Structural"}


def accf(mean, std, acc):
    if not np.isfinite(mean):
        return "---"
    return f"\\accf{{{mean:.3f}}}{{{std:.3f}}}{{{acc:.3f}}}"


def shared_table(tag="d16-s42"):
    st = pd.read_csv(OUT / "stats_shared_splits.csv")
    comb = pd.read_csv(OUT / "combined_results.csv")
    st = st[st.feature_tag == tag]
    datasets = [k for k in DATASET_ORDER if (st.dataset_key == k).any()]
    lines = []
    lines.append("\\begin{tabular}{|l|l|" + "c|" * len(datasets) + "}")
    lines.append("  \\hline")
    lines.append("  \\textbf{Family} & \\textbf{Method} & " + " & ".join(f"\\textbf{{{DATASET_HEAD[k]}}}" for k in datasets) + " \\\\")
    lines.append("  \\hline")

    def ref_row(label, method, tag_of):
        cells = []
        for k in datasets:
            t = tag_of(k)
            d = comb[(comb.dataset_key == k) & (comb.method == method) & (comb.feature_tag == t)]
            cells.append(accf(d.f1_macro.mean(), d.f1_macro.std(ddof=1), d.accuracy.mean()) if len(d) else "---")
        return f"  \\multirow{{3}}{{*}}{{References}} & {label} & " + " & ".join(cells) + " \\\\" if label.startswith("Random") else f"   & {label} & " + " & ".join(cells) + " \\\\"

    lines.append(ref_row("Random floor (permuted)", "permuted", lambda k: "fall-vl1"))
    lines.append("  \\cline{2-" + str(2 + len(datasets)) + "}")
    lines.append(ref_row("Node features", "node_struct", lambda k: "fall-vl1"))
    lines.append("  \\cline{2-" + str(2 + len(datasets)) + "}")
    # best bag row: method differs per dataset
    cells = []
    for k in datasets:
        bm, bt = C.DATASETS[k]["expected_best_bag"]
        d = comb[(comb.dataset_key == k) & (comb.method == bm) & (comb.feature_tag == bt)]
        cells.append(accf(d.f1_macro.mean(), d.f1_macro.std(ddof=1), d.accuracy.mean()))
    lines.append("   & Best egonet bag$^{a}$ & " + " & ".join(cells) + " \\\\")
    lines.append("  \\hline")
    for fam in ("proximity", "structural"):
        methods = [m for m in C.METHODS_BY_FAMILY[fam] if (st.method == m).any()]
        for i, m in enumerate(methods):
            cells = []
            for k in datasets:
                r = st[(st.dataset_key == k) & (st.method == m)]
                if r.empty:
                    cells.append("---")
                    continue
                r = r.iloc[0]
                mark = ""
                if np.isfinite(r.p_bestbag) and r.p_bestbag < 0.05:
                    mark = "$^{\\dagger}$" if r.diff_bestbag > 0 else "$^{\\ddagger}$"
                cells.append(accf(r.f1_mean, r.f1_std, r.acc_mean) + mark)
            head = f"  \\multirow{{{len(methods)}}}{{*}}{{{FAMILY_LABEL[fam]}}}" if i == 0 else "   "
            lines.append(f"{head} & {METHOD_LABEL.get(m, m)} & " + " & ".join(cells) + " \\\\")
            if i < len(methods) - 1:
                lines.append("  \\cline{2-" + str(2 + len(datasets)) + "}")
        lines.append("  \\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def holdout_table():
    path = OUT / "stats_region_holdout.csv"
    if not path.exists():
        return None
    h = pd.read_csv(path)
    datasets = [k for k in ("amazon", "roman", "chameleon") if (h.dataset_key == k).any()]
    buffers = [0, 2, 4]
    lines = ["\\begin{tabular}{|l|l|" + "c|" * len(buffers) + "}", "  \\hline",
             "  \\textbf{Dataset} & \\textbf{Method} & " + " & ".join(f"$B={b}$" for b in buffers) + " \\\\", "  \\hline"]
    for k in datasets:
        d = h[h.dataset_key == k]
        methods = list(d.method.unique())
        for i, m in enumerate(methods):
            cells = []
            for b in buffers:
                r = d[(d.method == m) & (d.buffer_hops == b)]
                if r.empty or not np.isfinite(r.iloc[0].lift_mean):
                    cells.append("---")
                else:
                    r = r.iloc[0]
                    star = "$^{*}$" if np.isfinite(r.p_vs_zero) and r.p_vs_zero < 0.05 else ""
                    cells.append(f"${r.lift_mean:+.3f} \\pm {r.lift_std:.3f}${star}")
            head = f"  \\multirow{{{len(methods)}}}{{*}}{{{DATASET_HEAD[k]}}}" if i == 0 else "   "
            lines.append(f"{head} & {METHOD_LABEL.get(m, m)} & " + " & ".join(cells) + " \\\\")
            if i < len(methods) - 1:
                lines.append("  \\cline{2-" + str(2 + len(buffers)) + "}")
        lines.append("  \\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "tables" / "baselines_shared.tex").write_text(shared_table())
    print((OUT / "tables" / "baselines_shared.tex").read_text())
    t = holdout_table()
    if t:
        (OUT / "tables" / "baselines_holdout.tex").write_text(t)
        print(t)


if __name__ == "__main__":
    main()
