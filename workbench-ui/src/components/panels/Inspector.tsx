import { useQuery } from "@tanstack/react-query";
import { api } from "../../api";
import type {
  DatasetCatalogEntry,
  DatasetGraphSummary,
  DatasetManifest,
  EmbeddingCatalogEntry,
  EmbeddingGraphDetail,
  EmbeddingManifest,
  FeatureCatalogEntry,
  FeatureGraphDetail,
  FeatureManifest,
  ModelAnalysis,
  ModelCatalogEntry,
  ModelManifest,
  ProjectManifest
} from "../../api";
import { EmptyState } from "../primitives/EmptyState";

interface InspectorProps {
  activeProjectId: string;
  project?: ProjectManifest;
  dataset?: DatasetManifest;
  exploreDataset?: DatasetManifest;
  exploreGraphSummary?: DatasetGraphSummary | null;
  exploreNodeId?: string;
  exploreNodeVisible?: boolean | null;
  catalogEntry?: DatasetCatalogEntry;
  catalogImportStatus?: string;
  feature?: FeatureManifest;
  featureDataset?: DatasetManifest;
  featureCatalogEntry?: FeatureCatalogEntry;
  selectedFeatureCatalogEntry?: FeatureCatalogEntry;
  exploreFeature?: FeatureManifest;
  exploreFeatureDataset?: DatasetManifest;
  exploreFeatureGraphId?: string;
  exploreFeatureGraphVisible?: boolean | null;
  embedding?: EmbeddingManifest;
  embeddingDataset?: DatasetManifest;
  embeddingFeatures: FeatureManifest[];
  embeddingCatalogEntry?: EmbeddingCatalogEntry;
  selectedEmbeddingCatalogEntry?: EmbeddingCatalogEntry;
  exploreEmbedding?: EmbeddingManifest;
  exploreEmbeddingDataset?: DatasetManifest;
  exploreEmbeddingFeatures: FeatureManifest[];
  exploreEmbeddingGraphId?: string;
  exploreEmbeddingGraphVisible?: boolean | null;
  model?: ModelManifest;
  modelDataset?: DatasetManifest;
  modelEmbeddings: EmbeddingManifest[];
  modelCatalogEntry?: ModelCatalogEntry;
  selectedModelCatalogEntry?: ModelCatalogEntry;
  exploreModel?: ModelManifest;
  exploreModelDataset?: DatasetManifest;
  exploreModelEmbeddings: EmbeddingManifest[];
  exploreModelIteration?: number | null;
}

function InspectorRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="inspector-row">
      <dt>{label}</dt>
      <dd className={mono ? "mono" : undefined}>{value}</dd>
    </div>
  );
}

function boolText(value: boolean): string {
  return value ? "Yes" : "No";
}

function featureTypeLabel(entry: FeatureCatalogEntry): string {
  return entry.type === "structural_node_feature" ? "Structural node feature" : entry.type;
}

function datasetStats(dataset: DatasetManifest) {
  return dataset.prepared_stats || dataset.stats || dataset.source_stats;
}

function taskLabel(taskType?: string): string {
  if (taskType === "classifier") return "Classifier";
  if (taskType === "regressor") return "Regressor";
  return "";
}

function metricLabel(metric: string): string {
  if (metric === "f1_score") return "F1 Score";
  if (metric === "rmse") return "RMSE";
  if (metric === "mae") return "MAE";
  return metric.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function metricValue(value: unknown): string {
  return typeof value === "number" ? value.toFixed(4) : value == null ? "" : String(value);
}

function inspectorValue(value: unknown): string {
  if (value == null) return "None";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(value);
  if (typeof value === "boolean") return boolText(value);
  return String(value);
}

export function Inspector({
  activeProjectId,
  project,
  dataset,
  exploreDataset,
  exploreGraphSummary,
  exploreNodeId = "",
  exploreNodeVisible = null,
  catalogEntry,
  catalogImportStatus,
  feature,
  featureDataset,
  featureCatalogEntry,
  selectedFeatureCatalogEntry,
  exploreFeature,
  exploreFeatureDataset,
  exploreFeatureGraphId = "",
  exploreFeatureGraphVisible = null,
  embedding,
  embeddingDataset,
  embeddingFeatures,
  embeddingCatalogEntry,
  selectedEmbeddingCatalogEntry,
  exploreEmbedding,
  exploreEmbeddingDataset,
  exploreEmbeddingFeatures,
  exploreEmbeddingGraphId = "",
  exploreEmbeddingGraphVisible = null,
  model,
  modelDataset,
  modelEmbeddings,
  modelCatalogEntry,
  selectedModelCatalogEntry,
  exploreModel,
  exploreModelDataset,
  exploreModelEmbeddings,
  exploreModelIteration = null
}: InspectorProps) {
  const description = project?.description.trim() || "None";
  const datasetDescription = dataset?.description.trim() || "None";
  const catalogDescription = catalogEntry?.description.trim() || "None";
  const featureDescription = feature?.description.trim() || "None";
  const featureMethodName = featureCatalogEntry?.name || feature?.source_feature_id || "";
  const featureCatalogDescription = selectedFeatureCatalogEntry?.description.trim() || "None";
  const embeddingDescription = embedding?.description.trim() || "None";
  const embeddingAlgorithmName = embeddingCatalogEntry?.name || embedding?.source_embedding_id || "";
  const embeddingFeatureNames = embeddingFeatures.length ? embeddingFeatures.map((item) => item.name).join(", ") : "Unknown features";
  const exploreEmbeddingFeatureNames = exploreEmbeddingFeatures.length
    ? exploreEmbeddingFeatures.map((item) => item.name).join(", ")
    : "Unknown features";
  const embeddingCatalogDescription = selectedEmbeddingCatalogEntry?.description.trim() || "None";
  const modelDescription = model?.description.trim() || "None";
  const modelAlgorithmName = modelCatalogEntry?.name || model?.source_model_id || "";
  const modelEmbeddingNames = modelEmbeddings.length ? modelEmbeddings.map((item) => item.name).join(", ") : "Unknown embeddings";
  const exploreModelEmbeddingNames = exploreModelEmbeddings.length
    ? exploreModelEmbeddings.map((item) => item.name).join(", ")
    : "Unknown embeddings";
  const modelCatalogDescription = selectedModelCatalogEntry?.description.trim() || "None";
  const modelPreview = useQuery({
    queryKey: ["projects", model?.project_id, "models", model?.id, "preview", "inspector"],
    queryFn: () => api.modelPreview(model!.project_id, model!.id),
    enabled: Boolean(model?.project_id && model?.id && model.status === "completed")
  });
  const exploreNodeDetail = useQuery({
    queryKey: [
      "projects",
      activeProjectId,
      "datasets",
      exploreDataset?.id,
      "analysis",
      "node",
      exploreGraphSummary?.graph_id,
      exploreNodeId,
      "inspector"
    ],
    queryFn: () => api.datasetNodeDetail(activeProjectId, exploreDataset!.id, exploreGraphSummary!.graph_id, exploreNodeId),
    enabled: Boolean(activeProjectId && exploreDataset?.id && exploreGraphSummary?.graph_id && exploreNodeId)
  });
  const exploreFeatureGraphDetail = useQuery<FeatureGraphDetail>({
    queryKey: [
      "projects",
      activeProjectId,
      "features",
      exploreFeature?.id,
      "analysis",
      "graph",
      exploreFeatureGraphId,
      "inspector"
    ],
    queryFn: () => api.featureGraphDetail(activeProjectId, exploreFeature!.id, exploreFeatureGraphId),
    enabled: Boolean(activeProjectId && exploreFeature?.id && exploreFeatureGraphId)
  });
  const exploreEmbeddingGraphDetail = useQuery<EmbeddingGraphDetail>({
    queryKey: [
      "projects",
      activeProjectId,
      "embeddings",
      exploreEmbedding?.id,
      "analysis",
      "graph",
      exploreEmbeddingGraphId,
      "inspector"
    ],
    queryFn: () => api.embeddingGraphDetail(activeProjectId, exploreEmbedding!.id, exploreEmbeddingGraphId),
    enabled: Boolean(activeProjectId && exploreEmbedding?.id && exploreEmbeddingGraphId)
  });
  const exploreModelAnalysis = useQuery<ModelAnalysis>({
    queryKey: [
      "projects",
      activeProjectId,
      "models",
      exploreModel?.id,
      "analysis",
      "iteration",
      exploreModelIteration,
      "inspector"
    ],
    queryFn: () => api.modelAnalysis(activeProjectId, exploreModel!.id),
    enabled: Boolean(activeProjectId && exploreModel?.id && exploreModel.status === "completed" && exploreModelIteration != null)
  });
  const exploreModelIterationRow = exploreModelAnalysis.data?.metrics.find(
    (row, rowIndex) => Number(row.iteration ?? rowIndex) === exploreModelIteration
  );

  return (
    <section className="panel inspector-panel">
      <div className="panel-header">
        <span>Inspector</span>
      </div>
      <div className="panel-body">
        {exploreModel && exploreModelIteration != null ? (
          <div className="inspector-details">
            <h3>Model Iteration Details</h3>
            <dl>
              <InspectorRow label="Model" value={exploreModel.name} />
              <InspectorRow label="Source Dataset" value={exploreModelDataset?.name || "Unknown dataset"} />
              <InspectorRow label="Source Embeddings" value={exploreModelEmbeddingNames} />
              <InspectorRow label="Iteration" value={String(exploreModelIteration)} mono />
              <InspectorRow label="Task Type" value={taskLabel(String(exploreModel.operation.params.task_type))} />
              {exploreModelAnalysis.isLoading ? <InspectorRow label="Status" value="Loading" /> : null}
              {exploreModelAnalysis.error ? <InspectorRow label="Error" value={exploreModelAnalysis.error.message} /> : null}
              {exploreModelIterationRow
                ? exploreModel.expected_output.metrics.map((metric) => (
                    <InspectorRow key={metric} label={metricLabel(metric)} value={metricValue(exploreModelIterationRow[metric])} />
                  ))
                : null}
            </dl>
          </div>
        ) : exploreEmbedding && exploreEmbeddingGraphId ? (
          <div className="inspector-details">
            <h3>Embedding Graph Details</h3>
            <dl>
              <InspectorRow label="Embedding" value={exploreEmbedding.name} />
              <InspectorRow label="Source Dataset" value={exploreEmbeddingDataset?.name || "Unknown dataset"} />
              <InspectorRow label="Source Features" value={exploreEmbeddingFeatureNames} />
              <InspectorRow label="Graph ID" value={exploreEmbeddingGraphId} mono />
              <InspectorRow
                label="Plotted In Chart"
                value={exploreEmbeddingGraphVisible == null ? "Unknown" : boolText(Boolean(exploreEmbeddingGraphVisible))}
              />
              {exploreEmbeddingGraphVisible === false ? <InspectorRow label="Plot Sample" value="Graph is outside the plotted sample" /> : null}
              {exploreEmbeddingGraphDetail.data ? (
                <InspectorRow label="Graph Label" value={inspectorValue(exploreEmbeddingGraphDetail.data.graph_label)} />
              ) : null}
              {exploreEmbeddingGraphDetail.data
                ? Object.entries(exploreEmbeddingGraphDetail.data.embedding_values).map(([name, value]) => (
                    <InspectorRow key={name} label={name} value={inspectorValue(value)} />
                  ))
                : null}
              {exploreEmbeddingGraphDetail.error ? <InspectorRow label="Error" value={exploreEmbeddingGraphDetail.error.message} /> : null}
            </dl>
          </div>
        ) : exploreFeature && exploreFeatureGraphId ? (
          <div className="inspector-details">
            <h3>Feature Graph Details</h3>
            <dl>
              <InspectorRow label="Feature" value={exploreFeature.name} />
              <InspectorRow label="Source Dataset" value={exploreFeatureDataset?.name || "Unknown dataset"} />
              <InspectorRow label="Graph ID" value={exploreFeatureGraphId} mono />
              <InspectorRow
                label="Plotted In Chart"
                value={exploreFeatureGraphVisible == null ? "Unknown" : boolText(Boolean(exploreFeatureGraphVisible))}
              />
              {exploreFeatureGraphVisible === false ? <InspectorRow label="Plot Sample" value="Graph is outside the plotted sample" /> : null}
              {exploreFeatureGraphDetail.isLoading ? <InspectorRow label="Status" value="Loading" /> : null}
              {exploreFeatureGraphDetail.error ? <InspectorRow label="Error" value={exploreFeatureGraphDetail.error.message} /> : null}
              {exploreFeatureGraphDetail.data ? (
                <InspectorRow label="Graph Label" value={inspectorValue(exploreFeatureGraphDetail.data.graph_label)} />
              ) : null}
              {exploreFeatureGraphDetail.data ? (
                <InspectorRow label="Node Count" value={String(exploreFeatureGraphDetail.data.node_count)} />
              ) : null}
              {exploreFeatureGraphDetail.data ? <InspectorRow label="Aggregation" value="Mean" /> : null}
              {exploreFeatureGraphDetail.data
                ? Object.entries(exploreFeatureGraphDetail.data.feature_values).map(([name, value]) => (
                    <InspectorRow key={name} label={`${name} mean`} value={inspectorValue(value)} />
                  ))
                : null}
            </dl>
          </div>
        ) : exploreDataset && exploreGraphSummary && exploreNodeId ? (
          <div className="inspector-details">
            <h3>Dataset Node Details</h3>
            <dl>
              <InspectorRow label="Dataset" value={exploreDataset.name} />
              <InspectorRow label="Graph ID" value={exploreGraphSummary.graph_id} mono />
              <InspectorRow label="Node ID" value={exploreNodeId} mono />
              <InspectorRow
                label="Visible In Visual"
                value={exploreNodeVisible == null ? "Unknown" : boolText(Boolean(exploreNodeVisible))}
              />
              {exploreNodeVisible === false ? <InspectorRow label="Sampled Visual" value="Node is outside the sampled visual" /> : null}
              {exploreNodeDetail.isLoading ? <InspectorRow label="Status" value="Loading" /> : null}
              {exploreNodeDetail.error ? <InspectorRow label="Error" value={exploreNodeDetail.error.message} /> : null}
              {exploreNodeDetail.data ? <InspectorRow label="Degree" value={String(exploreNodeDetail.data.degree)} /> : null}
              {exploreNodeDetail.data ? <InspectorRow label="Graph Label" value={inspectorValue(exploreNodeDetail.data.graph_label)} /> : null}
              {exploreNodeDetail.data?.source_graph_id ? (
                <InspectorRow label="Source Graph ID" value={exploreNodeDetail.data.source_graph_id} mono />
              ) : null}
              {exploreNodeDetail.data?.source_node_id ? (
                <InspectorRow label="Source Node ID" value={exploreNodeDetail.data.source_node_id} mono />
              ) : null}
              {exploreNodeDetail.data
                ? Object.entries(exploreNodeDetail.data.feature_values).map(([name, value]) => (
                    <InspectorRow key={name} label={name} value={inspectorValue(value)} />
                  ))
                : null}
            </dl>
          </div>
        ) : exploreDataset && exploreGraphSummary ? (
          <div className="inspector-details">
            <h3>Dataset Graph Details</h3>
            <dl>
              <InspectorRow label="Dataset" value={exploreDataset.name} />
              <InspectorRow label="Graph ID" value={exploreGraphSummary.graph_id} mono />
              <InspectorRow label="Graph Label" value={inspectorValue(exploreGraphSummary.graph_label)} />
              <InspectorRow label="Nodes" value={String(exploreGraphSummary.node_count)} />
              <InspectorRow label="Edges" value={String(exploreGraphSummary.edge_count)} />
              <InspectorRow label="Dataset ID" value={exploreDataset.id} mono />
            </dl>
          </div>
        ) : model ? (
          <div className="inspector-details">
            <h3>Model Details</h3>
            <dl>
              <InspectorRow label="Name" value={model.name} />
              <InspectorRow label="Description" value={modelDescription} />
              <InspectorRow label="Status" value={model.status} />
              <InspectorRow label="Dataset" value={modelDataset?.name || "Unknown dataset"} />
              <InspectorRow label="Embeddings" value={modelEmbeddingNames} />
              <InspectorRow label="Algorithm" value={modelAlgorithmName} />
              <InspectorRow label="Algorithm ID" value={model.source_model_id} mono />
              <InspectorRow label="Task Type" value={taskLabel(String(model.operation.params.task_type))} />
              <InspectorRow label="Sample Size" value={String(model.operation.params.sample_size)} />
              <InspectorRow label="Test Size" value={String(model.operation.params.test_size)} />
              <InspectorRow label="Balance Dataset" value={boolText(Boolean(model.operation.params.balance_dataset))} />
              <InspectorRow label="Random Seed" value={String(model.operation.params.random_state)} />
              <InspectorRow label="Parallel Jobs" value={String(model.operation.params.n_jobs)} />
              <InspectorRow label="Parallel Backend" value={String(model.operation.params.parallel_backend)} />
              <InspectorRow label="Expected Metrics" value={model.expected_output.metrics.join(", ")} mono />
              <InspectorRow label="Operation ID" value={model.operation.operation_id} mono />
              <InspectorRow label="Operation Version" value={model.operation.operation_version} mono />
              {model.output_stats ? <InspectorRow label="Metric Count" value={String(model.output_stats.metric_count)} /> : null}
              {model.output_stats ? <InspectorRow label="Feature Count" value={String(model.output_stats.feature_count)} /> : null}
              {model.output_stats ? <InspectorRow label="Graph Count" value={String(model.output_stats.graph_count)} /> : null}
              {modelPreview.data
                ? model.expected_output.metrics.map((metric) => (
                    <InspectorRow
                      key={metric}
                      label={`${metricLabel(metric)} Mean`}
                      value={metricValue(modelPreview.data.summary[`${metric}_mean`])}
                    />
                  ))
                : null}
              {model.output_files ? <InspectorRow label="Metrics File" value={model.output_files.metrics} mono /> : null}
              {model.output_files ? <InspectorRow label="Model File" value={model.output_files.model} mono /> : null}
              {model.error ? <InspectorRow label="Error" value={model.error.message} /> : null}
              <InspectorRow label="Model ID" value={model.id} mono />
              <InspectorRow label="Project ID" value={model.project_id} mono />
              <InspectorRow label="Created" value={model.created_at} mono />
              <InspectorRow label="Updated" value={model.updated_at} mono />
            </dl>
          </div>
        ) : selectedModelCatalogEntry ? (
          <div className="inspector-details">
            <h3>Catalog Model Details</h3>
            <dl>
              <InspectorRow label="Name" value={selectedModelCatalogEntry.name} />
              <InspectorRow label="Description" value={modelCatalogDescription} />
              <InspectorRow label="Status" value="Available" />
              <InspectorRow label="Algorithm ID" value={selectedModelCatalogEntry.id} mono />
              <InspectorRow label="Output" value={selectedModelCatalogEntry.output} />
              <InspectorRow label="Operation ID" value={selectedModelCatalogEntry.operation_id} mono />
              <InspectorRow label="Operation Version" value={selectedModelCatalogEntry.operation_version} mono />
            </dl>
          </div>
        ) : embedding ? (
          <div className="inspector-details">
            <h3>Embedding Details</h3>
            <dl>
              <InspectorRow label="Name" value={embedding.name} />
              <InspectorRow label="Description" value={embeddingDescription} />
              <InspectorRow label="Status" value={embedding.status} />
              <InspectorRow label="Dataset" value={embeddingDataset?.name || "Unknown dataset"} />
              <InspectorRow label="Features" value={embeddingFeatureNames} />
              <InspectorRow label="Algorithm" value={embeddingAlgorithmName} />
              <InspectorRow label="Algorithm ID" value={embedding.source_embedding_id} mono />
              <InspectorRow label="Dimension" value={String(embedding.operation.params.embedding_dimension)} />
              <InspectorRow label="Random Seed" value={String(embedding.operation.params.random_state)} />
              <InspectorRow label="Memory Size" value={String(embedding.operation.params.memory_size)} />
              <InspectorRow label="Expected Columns" value={embedding.expected_output.columns.join(", ")} mono />
              <InspectorRow label="Operation ID" value={embedding.operation.operation_id} mono />
              <InspectorRow label="Operation Version" value={embedding.operation.operation_version} mono />
              {embedding.output_stats ? <InspectorRow label="Output Rows" value={String(embedding.output_stats.row_count)} /> : null}
              {embedding.output_files ? <InspectorRow label="Output File" value={embedding.output_files.embeddings} mono /> : null}
              {embedding.error ? <InspectorRow label="Error" value={embedding.error.message} /> : null}
              <InspectorRow label="Embedding ID" value={embedding.id} mono />
              <InspectorRow label="Project ID" value={embedding.project_id} mono />
              <InspectorRow label="Created" value={embedding.created_at} mono />
              <InspectorRow label="Updated" value={embedding.updated_at} mono />
            </dl>
          </div>
        ) : selectedEmbeddingCatalogEntry ? (
          <div className="inspector-details">
            <h3>Catalog Embedding Details</h3>
            <dl>
              <InspectorRow label="Name" value={selectedEmbeddingCatalogEntry.name} />
              <InspectorRow label="Description" value={embeddingCatalogDescription} />
              <InspectorRow label="Status" value="Available" />
              <InspectorRow label="Algorithm ID" value={selectedEmbeddingCatalogEntry.id} mono />
              <InspectorRow label="Output" value={selectedEmbeddingCatalogEntry.output} />
              <InspectorRow label="Operation ID" value={selectedEmbeddingCatalogEntry.operation_id} mono />
              <InspectorRow label="Operation Version" value={selectedEmbeddingCatalogEntry.operation_version} mono />
            </dl>
          </div>
        ) : feature ? (
          <div className="inspector-details">
            <h3>Feature Details</h3>
            <dl>
              <InspectorRow label="Name" value={feature.name} />
              <InspectorRow label="Description" value={featureDescription} />
              <InspectorRow label="Status" value={feature.status} />
              <InspectorRow label="Source Dataset" value={featureDataset?.name || "Unknown dataset"} />
              <InspectorRow label="Feature Method" value={featureMethodName} />
              <InspectorRow label="Expected Columns" value={feature.expected_output.columns.join(", ")} mono />
              <InspectorRow label="Operation ID" value={feature.operation.operation_id} mono />
              <InspectorRow label="Operation Version" value={feature.operation.operation_version} mono />
              <InspectorRow label="Feature Vector Length" value={String(feature.operation.params.feature_vector_length)} />
              <InspectorRow label="Normalize Features" value={boolText(feature.operation.params.normalize_features)} />
              <InspectorRow label="Parallel Jobs" value={String(feature.operation.params.n_jobs)} />
              <InspectorRow label="Parallel Backend" value={feature.operation.params.parallel_backend} />
              {feature.output_stats ? <InspectorRow label="Output Rows" value={String(feature.output_stats.row_count)} /> : null}
              {feature.output_files ? <InspectorRow label="Output File" value={feature.output_files.features} mono /> : null}
              {feature.error ? <InspectorRow label="Error" value={feature.error.message} /> : null}
              <InspectorRow label="Feature ID" value={feature.id} mono />
              <InspectorRow label="Project ID" value={feature.project_id} mono />
              <InspectorRow label="Created" value={feature.created_at} mono />
              <InspectorRow label="Updated" value={feature.updated_at} mono />
            </dl>
          </div>
        ) : selectedFeatureCatalogEntry ? (
          <div className="inspector-details">
            <h3>Catalog Feature Details</h3>
            <dl>
              <InspectorRow label="Name" value={selectedFeatureCatalogEntry.name} />
              <InspectorRow label="Description" value={featureCatalogDescription} />
              <InspectorRow label="Status" value="Available" />
              <InspectorRow label="Type" value={featureTypeLabel(selectedFeatureCatalogEntry)} />
              <InspectorRow label="Output" value={selectedFeatureCatalogEntry.output} />
              <InspectorRow label="Source Type" value={selectedFeatureCatalogEntry.source_type} />
              <InspectorRow label="Operation ID" value={selectedFeatureCatalogEntry.operation_id} mono />
              <InspectorRow label="Operation Version" value={selectedFeatureCatalogEntry.operation_version} mono />
              <InspectorRow label="Feature ID" value={selectedFeatureCatalogEntry.id} mono />
            </dl>
          </div>
        ) : dataset ? (
          <div className="inspector-details">
            <h3>Dataset Details</h3>
            <dl>
              <InspectorRow label="Name" value={dataset.name} />
              <InspectorRow label="Description" value={datasetDescription} />
              <InspectorRow label="Status" value={dataset.status} />
              <InspectorRow label="Dataset ID" value={dataset.id} mono />
              <InspectorRow label="Project ID" value={dataset.project_id} mono />
              <InspectorRow label="Source" value={dataset.source_catalog_id} />
              <InspectorRow label="Source Name" value={dataset.source_name || dataset.source_catalog_id} />
              <InspectorRow label="Source Type" value={dataset.source_graph_shape === "single_graph" ? "Single Graph" : "Collection"} />
              <InspectorRow label="Graph Backend" value={String(dataset.operation.params.graph_type)} />
              <InspectorRow label="Reindex Nodes" value={boolText(Boolean(dataset.operation.params.reindex_nodes))} />
              {dataset.source_graph_shape === "single_graph" ? (
                <>
                  <InspectorRow label="K-Hop" value={String(dataset.operation.params.k_hop)} />
                  <InspectorRow label="Node Selection" value={String(dataset.operation.params.node_selection).replace(/_/g, " ")} />
                  <InspectorRow label="Sample Fraction" value={String(dataset.operation.params.sample_fraction)} />
                  <InspectorRow label="Random Seed" value={String(dataset.operation.params.random_seed)} />
                  <InspectorRow label="Target Attribute" value={String(dataset.operation.params.target_node_attribute || "None")} />
                </>
              ) : (
                <InspectorRow label="Filter Largest Component" value={boolText(Boolean(dataset.operation.params.filter_largest_component))} />
              )}
              <InspectorRow label="Graph Shape" value={dataset.graph_shape} />
              <InspectorRow label="Storage" value={dataset.storage_format} />
              <InspectorRow label="Graphs" value={String(datasetStats(dataset).graph_count)} />
              <InspectorRow label="Nodes" value={String(datasetStats(dataset).node_count)} />
              <InspectorRow label="Edges" value={String(datasetStats(dataset).edge_count)} />
              <InspectorRow label="Graph Labels" value={boolText(datasetStats(dataset).has_graph_labels)} />
              <InspectorRow label="Node Features" value={boolText(datasetStats(dataset).has_node_features)} />
              <InspectorRow label="Edge Features" value={boolText(datasetStats(dataset).has_edge_features)} />
              {dataset.raw_data_files ? <InspectorRow label="Raw Nodes" value={dataset.raw_data_files.nodes} mono /> : null}
              {dataset.prepared_data_files ? <InspectorRow label="Prepared Nodes" value={dataset.prepared_data_files.nodes} mono /> : null}
              {dataset.mapping_files ? <InspectorRow label="Node Mapping" value={dataset.mapping_files.node_mapping} mono /> : null}
              {dataset.error ? <InspectorRow label="Error" value={dataset.error.message} /> : null}
              <InspectorRow label="Created" value={dataset.created_at} mono />
              <InspectorRow label="Updated" value={dataset.updated_at} mono />
            </dl>
          </div>
        ) : catalogEntry ? (
          <div className="inspector-details">
            <h3>Catalog Dataset Details</h3>
            <dl>
              <InspectorRow label="Name" value={catalogEntry.name} />
              <InspectorRow label="Description" value={catalogDescription} />
              <InspectorRow label="Status" value={catalogImportStatus || "Not configured"} />
              <InspectorRow label="Source" value={catalogEntry.source} />
              <InspectorRow label="Domain" value={catalogEntry.domain} />
              <InspectorRow label="Type" value={catalogEntry.source_graph_shape === "single_graph" ? "Single Graph" : "Collection"} />
              <InspectorRow label="Graph Shape" value={catalogEntry.graph_shape} />
              <InspectorRow label="Graphs" value={String(catalogEntry.graph_count)} />
              <InspectorRow label="Nodes" value={String(catalogEntry.node_count)} />
              <InspectorRow label="Edges" value={String(catalogEntry.edge_count)} />
              {catalogEntry.source_graph_shape === "single_graph" ? (
                <InspectorRow
                  label="Node Attributes"
                  value={catalogEntry.node_attribute_columns.length ? catalogEntry.node_attribute_columns.join(", ") : "None"}
                />
              ) : null}
              <InspectorRow label="Graph Labels" value={boolText(catalogEntry.has_graph_labels)} />
              <InspectorRow label="Node Features" value={boolText(catalogEntry.has_node_features)} />
              <InspectorRow label="Edge Features" value={boolText(catalogEntry.has_edge_features)} />
            </dl>
          </div>
        ) : project ? (
          <div className="inspector-details">
            <h3>Project Details</h3>
            <dl>
              <InspectorRow label="Name" value={project.name} />
              <InspectorRow label="Description" value={description} />
              <InspectorRow label="Status" value="Active project" />
              <InspectorRow label="Project ID" value={project.id} mono />
              <InspectorRow label="Created" value={project.created_at} mono />
              <InspectorRow label="Updated" value={project.updated_at} mono />
            </dl>
          </div>
        ) : (
          <EmptyState compact>No active project.</EmptyState>
        )}
      </div>
    </section>
  );
}
