"""Figure: region-holdout lift vs buffer — best proximity, best structural,
best bag, node features — on Amazon, Roman, Chameleon. Reads the leakage-control
session's separation_*.csv and this session's combined_results.csv (for the
shared-split references). Run from the repo root."""

import sys
from pathlib import Path

SESSION_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = SESSION_ROOT.parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(SESSION_ROOT / "code"))

import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from lib.containment import plotstyle as ps  # noqa: E402
from lib.nodeclass.figures import separation_families_figure  # noqa: E402

SEP_DIR = RESEARCH_ROOT / "sessions" / "2026-08-13-leakage-control" / "outputs"
PANELS = {"amazon": ("Amazon Ratings", [0, 1, 2, 4, 6, 8, 10]), "roman": ("Roman Empire", [0, 1, 2, 4, 6, 8, 10]), "chameleon": ("Wikipedia -- Chameleon", [0, 1, 2])}
FAMILIES = {"proximity": ps.FAMILY_COLOR["emb_proximity"], "structural": ps.FAMILY_COLOR["emb_structural"]}


def shared_lift(comb, key, method, tag):
    d = comb[comb.dataset_key == key]
    m = d.loc[(d.method == method) & (d.feature_tag == tag), "f1_macro"].mean()
    floor_name = f"permuted_{method}" if method in C.CORE_METHODS or method in C.OPTIONAL_METHODS else "permuted"
    floor_tag = tag if floor_name != "permuted" else "fall-vl1"
    f = d.loc[(d.method == floor_name) & (d.feature_tag == floor_tag), "f1_macro"].mean()
    return float(m - f)


def main(tag="d16-s42"):
    comb = pd.read_csv(SESSION_ROOT / "outputs" / "combined_results.csv")
    panels = []
    legend = [(FAMILIES["proximity"], "best proximity"), (FAMILIES["structural"], "best structural"),
              (ps.FAMILY_COLOR["egonet_hop"], "best egonet bag"), (ps.FAMILY_COLOR["egonet_walk"], "node features")]
    for key, (title, buffers) in PANELS.items():
        emb_path = SEP_DIR / f"separation_{key}_emb.csv"
        base_path = SEP_DIR / f"separation_{key}.csv"
        if not (emb_path.exists() and base_path.exists()):
            print(f"skip {key}: missing separation CSV")
            continue
        emb = pd.read_csv(emb_path)
        base = pd.read_csv(base_path)
        series = []
        d = comb[(comb.dataset_key == key) & (comb.feature_tag == tag)]
        for fam in ("proximity", "structural"):
            cands = d[d.family == fam].groupby("method")["f1_macro"].mean()
            if cands.empty:
                continue
            best = cands.idxmax()
            series.append((emb, best, tag, FAMILIES[fam], best, shared_lift(comb, key, best, tag)))
        bm, bt = C.DATASETS[key]["expected_best_bag"]
        series.append((base, bm, bt, ps.FAMILY_COLOR["egonet_hop"], "best bag", shared_lift(comb, key, bm, bt)))
        series.append((base, "node_struct", "fall-vl1", ps.FAMILY_COLOR["egonet_walk"], "node features", shared_lift(comb, key, "node_struct", "fall-vl1")))
        panels.append({"title": title, "buffers": buffers, "series": series, "legend": legend})
    if not panels:
        return
    (SESSION_ROOT / "figures").mkdir(exist_ok=True)
    stem = separation_families_figure(panels, SESSION_ROOT / "figures" / "baselines_separation")
    print(f"written {stem}.pdf/.png")


if __name__ == "__main__":
    main()
