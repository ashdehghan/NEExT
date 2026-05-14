import type { DatasetCatalogEntry, DatasetManifest, FeatureCatalogEntry, FeatureManifest, ProjectManifest } from "../../api";
import { EmptyState } from "../primitives/EmptyState";

interface InspectorProps {
  project?: ProjectManifest;
  dataset?: DatasetManifest;
  catalogEntry?: DatasetCatalogEntry;
  catalogImportStatus?: string;
  feature?: FeatureManifest;
  featureDataset?: DatasetManifest;
  featureCatalogEntry?: FeatureCatalogEntry;
  selectedFeatureCatalogEntry?: FeatureCatalogEntry;
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

export function Inspector({
  project,
  dataset,
  catalogEntry,
  catalogImportStatus,
  feature,
  featureDataset,
  featureCatalogEntry,
  selectedFeatureCatalogEntry
}: InspectorProps) {
  const description = project?.description.trim() || "None";
  const datasetDescription = dataset?.description.trim() || "None";
  const catalogDescription = catalogEntry?.description.trim() || "None";
  const featureDescription = feature?.description.trim() || "None";
  const featureMethodName = featureCatalogEntry?.name || feature?.source_feature_id || "";
  const featureCatalogDescription = selectedFeatureCatalogEntry?.description.trim() || "None";

  return (
    <section className="panel inspector-panel">
      <div className="panel-header">
        <span>Inspector</span>
      </div>
      <div className="panel-body">
        {feature ? (
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
              <InspectorRow label="Graph Backend" value={String(dataset.operation.params.graph_type)} />
              <InspectorRow label="Reindex Nodes" value={boolText(Boolean(dataset.operation.params.reindex_nodes))} />
              <InspectorRow label="Filter Largest Component" value={boolText(Boolean(dataset.operation.params.filter_largest_component))} />
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
              <InspectorRow label="Graph Shape" value={catalogEntry.graph_shape} />
              <InspectorRow label="Graphs" value={String(catalogEntry.graph_count)} />
              <InspectorRow label="Nodes" value={String(catalogEntry.node_count)} />
              <InspectorRow label="Edges" value={String(catalogEntry.edge_count)} />
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
