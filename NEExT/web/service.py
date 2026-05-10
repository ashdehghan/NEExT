"""Application service layer for the NEExT web workbench."""

from __future__ import annotations

from typing import Any

import pandas as pd

from NEExT import NEExT
from NEExT.web.project import ArtifactRecord, ProjectManager
from NEExT.web.serializers import model_result_payload


class WorkbenchService:
    def __init__(self, project: ProjectManager):
        self.project = project

    def import_dataset_from_paths(
        self,
        name: str,
        edges_path: str,
        node_graph_mapping_path: str,
        graph_label_path: str | None = None,
        node_features_path: str | None = None,
        edge_features_path: str | None = None,
        graph_type: str = "networkx",
        reindex_nodes: bool = True,
        filter_largest_component: bool = True,
        node_sample_rate: float = 1.0,
    ) -> ArtifactRecord:
        nxt = NEExT(log_level="WARNING")
        collection = nxt.read_from_csv(
            edges_path=edges_path,
            node_graph_mapping_path=node_graph_mapping_path,
            graph_label_path=graph_label_path,
            node_features_path=node_features_path,
            edge_features_path=edge_features_path,
            graph_type=graph_type,
            reindex_nodes=reindex_nodes,
            filter_largest_component=filter_largest_component,
            node_sample_rate=node_sample_rate,
        )
        return self.project.register_artifact(
            artifact_type="dataset",
            name=name,
            obj=collection,
            metadata={
                "source": "csv",
                "graph_type": graph_type,
                "num_graphs": len(collection.graphs),
                "has_labels": any(graph.graph_label is not None for graph in collection.graphs),
                "inputs": {
                    "edges_path": edges_path,
                    "node_graph_mapping_path": node_graph_mapping_path,
                    "graph_label_path": graph_label_path,
                    "node_features_path": node_features_path,
                    "edge_features_path": edge_features_path,
                },
            },
        )

    def generate_preset_dataset(self, name: str, preset: str, seed: int = 42, params: dict[str, Any] | None = None) -> ArtifactRecord:
        nxt = NEExT(log_level="WARNING")
        collection = nxt.generate_synthetic_graphs(preset=preset, seed=seed, **(params or {}))
        return self.project.register_artifact(
            artifact_type="dataset",
            name=name,
            obj=collection,
            metadata={
                "source": "synthetic_preset",
                "preset": preset,
                "seed": seed,
                "params": params or {},
                "num_graphs": len(collection.graphs),
                "has_labels": any(graph.graph_label is not None for graph in collection.graphs),
            },
        )

    def compute_features(
        self,
        dataset_id: str,
        name: str,
        feature_list: list[str],
        feature_vector_length: int = 3,
        normalize_features: bool = True,
        n_jobs: int = 1,
        parallel_backend: str = "loky",
    ) -> ArtifactRecord:
        collection = self.project.load_object(dataset_id)
        nxt = NEExT(log_level="WARNING")
        features = nxt.compute_node_features(
            graph_collection=collection,
            feature_list=feature_list,
            feature_vector_length=feature_vector_length,
            normalize_features=normalize_features,
            show_progress=False,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )
        return self.project.register_artifact(
            artifact_type="features",
            name=name,
            obj=features,
            dataframe=features.features_df,
            metadata={
                "feature_columns": features.feature_columns,
                "feature_list": feature_list,
                "feature_vector_length": feature_vector_length,
                "normalize_features": normalize_features,
                "n_jobs": n_jobs,
                "parallel_backend": parallel_backend,
            },
            parent_ids=[dataset_id],
        )

    def compute_embeddings(
        self,
        dataset_id: str,
        features_id: str,
        name: str,
        embedding_algorithm: str = "approx_wasserstein",
        embedding_dimension: int = 3,
        feature_columns: list[str] | None = None,
        random_state: int = 42,
        memory_size: str = "4G",
    ) -> ArtifactRecord:
        collection = self.project.load_object(dataset_id)
        features = self.project.load_object(features_id)
        nxt = NEExT(log_level="WARNING")
        embeddings = nxt.compute_graph_embeddings(
            graph_collection=collection,
            features=features,
            embedding_algorithm=embedding_algorithm,
            embedding_dimension=embedding_dimension,
            feature_columns=feature_columns,
            random_state=random_state,
            memory_size=memory_size,
        )
        return self.project.register_artifact(
            artifact_type="embeddings",
            name=name,
            obj=embeddings,
            dataframe=embeddings.embeddings_df,
            metadata={
                "embedding_algorithm": embedding_algorithm,
                "embedding_dimension": embedding_dimension,
                "embedding_columns": embeddings.embedding_columns,
                "feature_columns": feature_columns or features.feature_columns,
                "random_state": random_state,
                "memory_size": memory_size,
            },
            parent_ids=[dataset_id, features_id],
        )

    def train_model(
        self,
        dataset_id: str,
        embeddings_id: str,
        name: str,
        model_type: str = "classifier",
        balance_dataset: bool = False,
        sample_size: int = 5,
        n_jobs: int = -1,
        parallel_backend: str = "process",
    ) -> ArtifactRecord:
        collection = self.project.load_object(dataset_id)
        embeddings = self.project.load_object(embeddings_id)
        nxt = NEExT(log_level="WARNING")
        results = nxt.train_ml_model(
            graph_collection=collection,
            embeddings=embeddings,
            model_type=model_type,
            balance_dataset=balance_dataset,
            sample_size=sample_size,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )
        payload = model_result_payload(results)
        result_df = pd.DataFrame([payload])
        return self.project.register_artifact(
            artifact_type="model_run",
            name=name,
            obj=payload,
            dataframe=result_df,
            metadata={
                "model_type": model_type,
                "balance_dataset": balance_dataset,
                "sample_size": sample_size,
                "n_jobs": n_jobs,
                "parallel_backend": parallel_backend,
            },
            parent_ids=[dataset_id, embeddings_id],
        )

    def generate_python_export(self, name: str, artifact_id: str) -> ArtifactRecord:
        artifact = self.project.get_artifact(artifact_id)
        script = self._script_for_artifact(artifact)
        export_dir = self.project.exports_dir / artifact_id
        export_dir.mkdir(parents=True, exist_ok=True)
        script_path = export_dir / "reproduce.py"
        script_path.write_text(script, encoding="utf-8")
        return self.project.register_artifact(
            artifact_type="export",
            name=name,
            metadata={
                "export_path": str(script_path.relative_to(self.project.root)),
                "source_artifact_id": artifact_id,
                "format": "python",
            },
            parent_ids=[artifact_id],
        )

    def _script_for_artifact(self, artifact: dict[str, Any]) -> str:
        lines = [
            "from NEExT import NEExT",
            "",
            'nxt = NEExT(log_level="INFO")',
            "",
            f"# Reproduce artifact: {artifact['name']} ({artifact['id']})",
            "# This script is generated from the local NEExT web workbench manifest.",
            "# Fill in local data paths if this artifact depends on imported CSV files.",
            "",
        ]
        if artifact["type"] == "dataset" and artifact["metadata"].get("source") == "synthetic_preset":
            metadata = artifact["metadata"]
            lines.append(
                "graph_collection = nxt.generate_synthetic_graphs("
                f"preset={metadata['preset']!r}, seed={metadata.get('seed', 42)!r}, **{metadata.get('params', {})!r})"
            )
        elif artifact["type"] == "dataset":
            inputs = artifact["metadata"].get("inputs", {})
            lines.extend(
                [
                    "graph_collection = nxt.read_from_csv(",
                    f"    edges_path={inputs.get('edges_path')!r},",
                    f"    node_graph_mapping_path={inputs.get('node_graph_mapping_path')!r},",
                    f"    graph_label_path={inputs.get('graph_label_path')!r},",
                    f"    node_features_path={inputs.get('node_features_path')!r},",
                    f"    edge_features_path={inputs.get('edge_features_path')!r},",
                    ")",
                ]
            )
        else:
            lines.append("# Use the parent artifact metadata in manifest.json to reconstruct the full workflow.")
        lines.append("")
        return "\n".join(lines)
