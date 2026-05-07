"""Benchmark custom node feature performance in NEExT.

This is an experiment harness, not public NEExT API. It builds deterministic
synthetic graph families, runs custom feature functions through
NEExT.compute_node_features(), and writes per-run/per-feature timing data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

nx = None
np = None
pd = None
NEExT = None


FeatureFunction = Callable[[object], object]


def load_dependencies() -> None:
    global NEExT, np, nx, pd

    if NEExT is not None:
        return

    try:
        import networkx as _nx
        import numpy as _np
        import pandas as _pd

        from NEExT import NEExT as _NEExT
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency while running the benchmark harness. "
            "Create a virtual environment and install the project first, for example: "
            "python3 -m pip install -e '.[dev]'"
        ) from exc

    nx = _nx
    np = _np
    pd = _pd
    NEExT = _NEExT


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    sizes: List[int]
    avg_degree: int
    triangle_probability: float


PRESETS: Dict[str, DatasetPreset] = {
    "smoke": DatasetPreset(
        name="smoke",
        sizes=[40, 60, 90],
        avg_degree=6,
        triangle_probability=0.10,
    ),
    "quick": DatasetPreset(
        name="quick",
        sizes=[80, 90, 110, 140, 220, 350, 800],
        avg_degree=8,
        triangle_probability=0.12,
    ),
    "mixed": DatasetPreset(
        name="mixed",
        sizes=[90, 100, 120, 150, 180, 250, 400, 700, 1500, 2500],
        avg_degree=8,
        triangle_probability=0.14,
    ),
    "stress": DatasetPreset(
        name="stress",
        sizes=[120, 150, 200, 300, 500, 1000, 2000, 3500, 5000],
        avg_degree=10,
        triangle_probability=0.16,
    ),
}


def _nodes_to_process(graph) -> List[int]:
    return list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)


def _as_networkx(graph) -> nx.Graph:
    if isinstance(graph.G, nx.Graph):
        return graph.G
    return nx.Graph(graph.G.get_edgelist())


def cheap_degree_feature(graph) -> pd.DataFrame:
    """Cheap baseline: mostly framework, serialization, and DataFrame overhead."""
    G = _as_networkx(graph)
    nodes = _nodes_to_process(graph)
    degrees = dict(G.degree(nodes))
    feature_values = np.array([degrees.get(node, 0) for node in nodes], dtype=float)
    df = pd.DataFrame(
        {
            "node_id": nodes,
            "graph_id": graph.graph_id,
            "cheap_degree_0": feature_values,
            "cheap_degree_1": feature_values**2,
        }
    )
    return df[["node_id", "graph_id", "cheap_degree_0", "cheap_degree_1"]]


def triangle_feature(graph) -> pd.DataFrame:
    """Medium baseline: whole-graph triangle and clustering work."""
    G = _as_networkx(graph)
    nodes = _nodes_to_process(graph)
    triangles = nx.triangles(G)
    clustering = nx.clustering(G)
    df = pd.DataFrame(
        {
            "node_id": nodes,
            "graph_id": graph.graph_id,
            "triangle_count_0": [float(triangles.get(node, 0)) for node in nodes],
            "triangle_clustering_0": [float(clustering.get(node, 0.0)) for node in nodes],
        }
    )
    return df[["node_id", "graph_id", "triangle_count_0", "triangle_clustering_0"]]


def ego_graphlet_proxy_feature(graph) -> pd.DataFrame:
    """Expensive graphlet proxy: per-node radius-2 ego graph pattern counts."""
    G = _as_networkx(graph)
    nodes = _nodes_to_process(graph)
    ego_nodes = []
    ego_edges = []
    ego_triangles = []
    ego_wedges = []

    for node in nodes:
        ego = nx.ego_graph(G, node, radius=2)
        degrees = [degree for _, degree in ego.degree()]
        triangle_count = sum(nx.triangles(ego).values()) // 3
        wedge_count = sum(degree * (degree - 1) // 2 for degree in degrees) - 3 * triangle_count
        ego_nodes.append(float(ego.number_of_nodes()))
        ego_edges.append(float(ego.number_of_edges()))
        ego_triangles.append(float(triangle_count))
        ego_wedges.append(float(max(wedge_count, 0)))

    df = pd.DataFrame(
        {
            "node_id": nodes,
            "graph_id": graph.graph_id,
            "ego_graphlet_nodes_0": ego_nodes,
            "ego_graphlet_edges_0": ego_edges,
            "ego_graphlet_triangles_0": ego_triangles,
            "ego_graphlet_wedges_0": ego_wedges,
        }
    )
    return df[
        [
            "node_id",
            "graph_id",
            "ego_graphlet_nodes_0",
            "ego_graphlet_edges_0",
            "ego_graphlet_triangles_0",
            "ego_graphlet_wedges_0",
        ]
    ]


FEATURES: Dict[str, FeatureFunction] = {
    "cheap": cheap_degree_feature,
    "triangles": triangle_feature,
    "ego_graphlet": ego_graphlet_proxy_feature,
}


@dataclass
class TimedFeature:
    name: str
    fn: FeatureFunction
    event_dir: str
    run_id: str

    def __call__(self, graph) -> pd.DataFrame:
        start = time.perf_counter()
        status = "ok"
        error = ""
        df: Optional[pd.DataFrame] = None
        try:
            df = self.fn(graph)
            return df
        except Exception as exc:
            status = "error"
            error = repr(exc)
            raise
        finally:
            duration_s = time.perf_counter() - start
            event = {
                "run_id": self.run_id,
                "feature": self.name,
                "graph_id": getattr(graph, "graph_id", None),
                "nodes": len(getattr(graph, "nodes", [])),
                "edges": len(getattr(graph, "edges", [])),
                "duration_s": duration_s,
                "rows": len(df) if df is not None else 0,
                "cols": len(df.columns) if df is not None else 0,
                "pid": os.getpid(),
                "status": status,
                "error": error,
            }
            _write_event(Path(self.event_dir), event)


def _write_event(event_dir: Path, event: Dict) -> None:
    try:
        event_dir.mkdir(parents=True, exist_ok=True)
        event_name = f"{time.time_ns()}_{os.getpid()}_{event['feature']}_{event['graph_id']}.json"
        with (event_dir / event_name).open("w", encoding="utf-8") as f:
            json.dump(event, f, sort_keys=True)
    except Exception:
        # Timing instrumentation must not mask the feature computation result.
        pass


def make_reddit_like_graphs(preset: DatasetPreset, seed: int) -> List[nx.Graph]:
    graphs = []
    rng = np.random.default_rng(seed)
    m = max(1, preset.avg_degree // 2)
    for graph_id, n_nodes in enumerate(preset.sizes):
        graph_seed = int(rng.integers(0, 2**31 - 1))
        m_for_graph = min(m, max(1, n_nodes - 1))
        graph = nx.powerlaw_cluster_graph(
            n=n_nodes,
            m=m_for_graph,
            p=preset.triangle_probability,
            seed=graph_seed,
        )
        graph.graph["label"] = _size_label(n_nodes)
        graph.graph["source"] = f"synthetic_{preset.name}"
        graphs.append(graph)
    return graphs


def _size_label(n_nodes: int) -> int:
    if n_nodes < 200:
        return 0
    if n_nodes < 1000:
        return 1
    return 2


def build_collection(graphs: List[nx.Graph], node_sample_rate: float, seed: int):
    nxt = NEExT(log_level="WARNING")
    collection = nxt.load_from_networkx(
        nx_graphs=graphs,
        graph_type="networkx",
        reindex_nodes=True,
        filter_largest_component=False,
        node_sample_rate=1.0,
    )
    if node_sample_rate < 1.0:
        collection.node_sample_rate = node_sample_rate
        collection.sample_nodes(random_seed=seed)
    return nxt, collection


def graph_summary(collection, estimate_pickle: bool) -> List[Dict]:
    rows = []
    for graph in collection.graphs:
        nodes = len(graph.nodes)
        edges = len(graph.edges)
        row = {
            "graph_id": graph.graph_id,
            "nodes": nodes,
            "edges": edges,
            "density": (2.0 * edges / (nodes * (nodes - 1))) if nodes > 1 else 0.0,
            "sampled_nodes": len(graph.sampled_nodes) if graph.sampled_nodes is not None else nodes,
            "pickle_bytes": None,
        }
        if estimate_pickle:
            try:
                row["pickle_bytes"] = len(pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:
                row["pickle_bytes"] = None
        rows.append(row)
    return rows


def get_maxrss_kb(children: bool = False) -> Optional[int]:
    try:
        import resource

        who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
        return int(resource.getrusage(who).ru_maxrss)
    except Exception:
        return None


def fingerprint_features(df: pd.DataFrame) -> str:
    ordered = df.sort_values(["graph_id", "node_id"]).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_n_jobs(value: str) -> List[int]:
    return [int(item) for item in parse_csv(value)]


def selected_feature_sets(feature_names: List[str], include_combined: bool) -> List[List[str]]:
    feature_sets = [[name] for name in feature_names]
    if include_combined and len(feature_names) > 1:
        feature_sets.append(feature_names)
    return feature_sets


def run_one(
    *,
    nxt,
    collection,
    output_dir: Path,
    preset_name: str,
    backend: str,
    n_jobs: int,
    feature_names: List[str],
    repeat_index: int,
) -> Dict:
    run_id = f"{preset_name}_{backend}_n{n_jobs}_{'-'.join(feature_names)}_r{repeat_index}_{time.time_ns()}"
    event_dir = output_dir / "events" / run_id
    my_feature_methods = [
        {
            "feature_name": name,
            "feature_function": TimedFeature(
                name=name,
                fn=FEATURES[name],
                event_dir=str(event_dir),
                run_id=run_id,
            ),
        }
        for name in feature_names
    ]

    rss_self_before = get_maxrss_kb(children=False)
    rss_children_before = get_maxrss_kb(children=True)
    start = time.perf_counter()
    status = "ok"
    error = ""
    warning_messages: List[str] = []
    result_rows = 0
    result_cols = 0
    fingerprint = ""

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            features = nxt.compute_node_features(
                graph_collection=collection,
                feature_list=feature_names,
                feature_vector_length=3,
                normalize_features=False,
                show_progress=False,
                n_jobs=n_jobs,
                parallel_backend=backend,
                my_feature_methods=my_feature_methods,
            )
        warning_messages = [str(item.message) for item in captured]
        result_rows = len(features.features_df)
        result_cols = len(features.features_df.columns)
        fingerprint = fingerprint_features(features.features_df)
    except Exception as exc:
        status = "error"
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()

    duration_s = time.perf_counter() - start
    rss_self_after = get_maxrss_kb(children=False)
    rss_children_after = get_maxrss_kb(children=True)

    return {
        "run_id": run_id,
        "preset": preset_name,
        "backend": backend,
        "n_jobs": n_jobs,
        "features": ",".join(feature_names),
        "repeat": repeat_index,
        "graph_count": len(collection.graphs),
        "total_nodes": sum(len(graph.nodes) for graph in collection.graphs),
        "total_edges": sum(len(graph.edges) for graph in collection.graphs),
        "duration_s": duration_s,
        "rss_self_before_kb": rss_self_before,
        "rss_self_after_kb": rss_self_after,
        "rss_children_before_kb": rss_children_before,
        "rss_children_after_kb": rss_children_after,
        "result_rows": result_rows,
        "result_cols": result_cols,
        "fingerprint": fingerprint,
        "warnings_count": len(warning_messages),
        "warnings": " | ".join(warning_messages),
        "status": status,
        "error": error,
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_events(output_dir: Path) -> List[Dict]:
    events = []
    for event_path in sorted((output_dir / "events").glob("*/*.json")):
        try:
            with event_path.open("r", encoding="utf-8") as f:
                events.append(json.load(f))
        except Exception:
            continue
    return events


def validate_args(args) -> None:
    invalid_features = sorted(set(args.features) - set(FEATURES))
    if invalid_features:
        available = ", ".join(sorted(FEATURES))
        raise SystemExit(f"Unknown feature(s): {', '.join(invalid_features)}. Available: {available}")

    invalid_backends = sorted(set(args.backends) - {"loky", "threading"})
    if invalid_backends:
        raise SystemExit(f"Unknown backend(s): {', '.join(invalid_backends)}")

    if not 0 < args.node_sample_rate <= 1:
        raise SystemExit("--node-sample-rate must be in (0, 1]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--features", type=parse_csv, default=parse_csv("cheap,triangles,ego_graphlet"))
    parser.add_argument("--n-jobs", type=parse_n_jobs, default=parse_n_jobs("1,2"))
    parser.add_argument("--backends", type=parse_csv, default=parse_csv("loky"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--node-sample-rate", type=float, default=1.0)
    parser.add_argument("--include-combined", action="store_true")
    parser.add_argument("--estimate-pickle", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    load_dependencies()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preset = PRESETS[args.preset]
    graphs = make_reddit_like_graphs(preset, seed=args.seed)
    nxt, collection = build_collection(graphs, node_sample_rate=args.node_sample_rate, seed=args.seed)
    dataset_rows = graph_summary(collection, estimate_pickle=args.estimate_pickle)

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "preset": preset.name,
                "seed": args.seed,
                "node_sample_rate": args.node_sample_rate,
                "graphs": dataset_rows,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    summary_rows = []
    feature_sets = selected_feature_sets(args.features, include_combined=args.include_combined)
    for repeat_index in range(args.repeat):
        for backend in args.backends:
            for n_jobs in args.n_jobs:
                for feature_names in feature_sets:
                    row = run_one(
                        nxt=nxt,
                        collection=collection,
                        output_dir=output_dir,
                        preset_name=preset.name,
                        backend=backend,
                        n_jobs=n_jobs,
                        feature_names=feature_names,
                        repeat_index=repeat_index,
                    )
                    summary_rows.append(row)
                    print(
                        f"{row['status']:>5} backend={backend:<9} n_jobs={n_jobs:<3} "
                        f"features={row['features']:<30} duration={row['duration_s']:.3f}s "
                        f"warnings={row['warnings_count']}"
                    )

    write_csv(output_dir / "run_summary.csv", summary_rows)
    write_csv(output_dir / "feature_events.csv", collect_events(output_dir))
    print(f"\nWrote benchmark results to {output_dir}")
    return 1 if any(row["status"] != "ok" for row in summary_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
