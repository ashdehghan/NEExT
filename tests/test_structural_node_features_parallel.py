import logging

import networkx as nx
import pandas as pd
import pytest
from pydantic import ValidationError

import NEExT.features.structural_node_features as structural_module
from NEExT import NEExT
from NEExT.features import StructuralNodeFeatures


def _make_collection():
    graphs = [
        nx.path_graph(5),
        nx.cycle_graph(6),
        nx.star_graph(4),
    ]
    nxt = NEExT(log_level="WARNING")
    collection = nxt.load_from_networkx(
        nx_graphs=graphs,
        graph_type="networkx",
        reindex_nodes=True,
        filter_largest_component=False,
        node_sample_rate=1.0,
    )
    return nxt, collection


def _sort_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["graph_id", "node_id"]).reset_index(drop=True)


def test_parallel_backends_match_sequential_for_builtin_and_local_custom_features():
    nxt, collection = _make_collection()

    def local_degree_plus_one(graph):
        nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)
        values = [float(graph.G.degree(node) + 1) for node in nodes]
        df = pd.DataFrame(
            {
                "node_id": nodes,
                "graph_id": graph.graph_id,
                "local_degree_plus_one_0": values,
            }
        )
        return df[["node_id", "graph_id", "local_degree_plus_one_0"]]

    custom_methods = [{"feature_name": "local_degree_plus_one", "feature_function": local_degree_plus_one}]
    common_kwargs = {
        "graph_collection": collection,
        "feature_list": ["degree_centrality", "local_degree_plus_one"],
        "feature_vector_length": 2,
        "normalize_features": False,
        "show_progress": False,
        "my_feature_methods": custom_methods,
    }

    sequential = nxt.compute_node_features(**common_kwargs, n_jobs=1)
    loky = nxt.compute_node_features(**common_kwargs, n_jobs=2, parallel_backend="loky")
    threading = nxt.compute_node_features(**common_kwargs, n_jobs=2, parallel_backend="threading")

    pd.testing.assert_frame_equal(_sort_features(sequential.features_df), _sort_features(loky.features_df))
    pd.testing.assert_frame_equal(_sort_features(sequential.features_df), _sort_features(threading.features_df))
    assert sequential.feature_columns == loky.feature_columns == threading.feature_columns


def test_default_compute_node_features_is_sequential(monkeypatch):
    nxt, collection = _make_collection()

    def fail_parallel(*args, **kwargs):
        raise AssertionError("Parallel should not be used for default n_jobs=1")

    monkeypatch.setattr(structural_module, "Parallel", fail_parallel)

    features = nxt.compute_node_features(
        graph_collection=collection,
        feature_list=["degree_centrality"],
        feature_vector_length=2,
        normalize_features=False,
        show_progress=False,
    )

    assert not features.features_df.empty
    assert features.feature_columns == ["degree_centrality_0", "degree_centrality_1"]


def test_compute_node_features_preserves_positional_custom_methods_argument():
    nxt, collection = _make_collection()

    def positional_custom_feature(graph):
        nodes = list(graph.sampled_nodes if graph.sampled_nodes is not None else graph.nodes)
        df = pd.DataFrame(
            {
                "node_id": nodes,
                "graph_id": graph.graph_id,
                "positional_custom_0": [float(node) for node in nodes],
            }
        )
        return df[["node_id", "graph_id", "positional_custom_0"]]

    custom_methods = [{"feature_name": "positional_custom", "feature_function": positional_custom_feature}]

    features = nxt.compute_node_features(
        collection,
        ["positional_custom"],
        2,
        False,
        False,
        1,
        custom_methods,
    )

    assert features.feature_columns == ["positional_custom_0"]
    assert "positional_custom_0" in features.features_df.columns


def test_structural_node_features_preserves_positional_suffix_argument():
    _, collection = _make_collection()

    node_features = StructuralNodeFeatures(
        collection,
        ["degree_centrality"],
        2,
        False,
        False,
        1,
        "legacy",
    )
    features = node_features.compute()

    assert features.feature_columns == ["degree_centrality_0_legacy", "degree_centrality_1_legacy"]


def test_invalid_parallel_backend_raises_validation_error():
    _, collection = _make_collection()

    with pytest.raises(ValidationError, match="parallel_backend"):
        StructuralNodeFeatures(
            graph_collection=collection,
            feature_list=["degree_centrality"],
            parallel_backend="processes",
        )


def test_joblib_kwargs_reject_neext_owned_arguments():
    _, collection = _make_collection()

    with pytest.raises(ValidationError, match="NEExT-owned joblib arguments: backend"):
        StructuralNodeFeatures(
            graph_collection=collection,
            feature_list=["degree_centrality"],
            joblib_kwargs={"backend": "threading"},
        )


def test_valid_joblib_kwargs_pass_through_for_parallel_execution(monkeypatch):
    _, collection = _make_collection()
    original_parallel = structural_module.Parallel
    captured_kwargs = {}

    def capturing_parallel(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return original_parallel(*args, **kwargs)

    monkeypatch.setattr(structural_module, "Parallel", capturing_parallel)

    node_features = StructuralNodeFeatures(
        graph_collection=collection,
        feature_list=["degree_centrality"],
        feature_vector_length=2,
        normalize_features=False,
        show_progress=False,
        n_jobs=2,
        parallel_backend="threading",
        joblib_kwargs={"batch_size": 1, "pre_dispatch": "2*n_jobs"},
    )
    features = node_features.compute()

    assert not features.features_df.empty
    assert captured_kwargs["n_jobs"] == 2
    assert captured_kwargs["backend"] == "threading"
    assert captured_kwargs["batch_size"] == 1
    assert captured_kwargs["pre_dispatch"] == "2*n_jobs"


def test_profile_features_logs_records_without_changing_returned_features(caplog):
    nxt, collection = _make_collection()
    common_kwargs = {
        "graph_collection": collection,
        "feature_list": ["degree_centrality", "clustering_coefficient"],
        "feature_vector_length": 2,
        "normalize_features": False,
        "show_progress": False,
        "n_jobs": 2,
        "parallel_backend": "threading",
    }

    baseline = nxt.compute_node_features(**common_kwargs, profile_features=False)

    caplog.set_level(logging.INFO, logger="NEExT.features.structural_node_features")
    profiled = nxt.compute_node_features(**common_kwargs, profile_features=True)

    pd.testing.assert_frame_equal(_sort_features(baseline.features_df), _sort_features(profiled.features_df))
    assert baseline.feature_columns == profiled.feature_columns

    profile_records = [record.node_feature_profile for record in caplog.records if hasattr(record, "node_feature_profile")]
    assert len(profile_records) == len(collection.graphs) * len(common_kwargs["feature_list"])

    first_record = profile_records[0]
    assert first_record["graph_id"] in {graph.graph_id for graph in collection.graphs}
    assert first_record["feature"] in common_kwargs["feature_list"]
    assert first_record["backend"] == "threading"
    assert first_record["n_jobs"] == 2
    assert first_record["duration_s"] >= 0
    assert first_record["output_rows"] > 0
    assert first_record["output_columns"] > 0

    first_message = caplog.records[0].getMessage()
    assert "graph_id=" in first_message
    assert "feature=" in first_message
    assert "backend=threading" in first_message
    assert "n_jobs=2" in first_message
    assert "duration_s=" in first_message
