import { BarChart3, Box, Database, FolderOpen, Sigma } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { DatasetManifest, EmbeddingManifest, FeatureManifest, ModelManifest, ProjectManifest } from "../../api";
import { FcIcon, type FcIconName } from "../primitives/FcIcon";

interface SelectionPanelProps {
  project?: ProjectManifest;
  datasets: DatasetManifest[];
  features: FeatureManifest[];
  embeddings: EmbeddingManifest[];
  models: ModelManifest[];
  selectedDatasetId: string;
  selectedFeatureId: string;
  selectedEmbeddingId: string;
  selectedModelId: string;
  onSelectDataset: (datasetId: string) => void;
  onSelectFeature: (featureId: string) => void;
  onSelectEmbedding: (embeddingId: string) => void;
  onSelectModel: (modelId: string) => void;
}

const SHELL_SECTIONS: {
  title: string;
  kind: "datasets" | "features" | "embeddings" | "models";
  fcIcon: FcIconName;
  itemIcon: LucideIcon;
  color: string;
  emptyLabel: string;
}[] = [
  {
    title: "Datasets",
    kind: "datasets",
    fcIcon: "datasets",
    itemIcon: Database,
    color: "var(--neext-blue)",
    emptyLabel: "No datasets"
  },
  {
    title: "Features",
    kind: "features",
    fcIcon: "features",
    itemIcon: Sigma,
    color: "var(--neext-orange)",
    emptyLabel: "No features"
  },
  {
    title: "Embeddings",
    kind: "embeddings",
    fcIcon: "embeddings",
    itemIcon: Box,
    color: "var(--neext-green)",
    emptyLabel: "No embeddings"
  },
  {
    title: "Models",
    kind: "models",
    fcIcon: "models",
    itemIcon: BarChart3,
    color: "var(--neext-purple)",
    emptyLabel: "No models"
  }
];

export function SelectionPanel({
  project,
  datasets,
  features,
  embeddings,
  models,
  selectedDatasetId,
  selectedFeatureId,
  selectedEmbeddingId,
  selectedModelId,
  onSelectDataset,
  onSelectFeature,
  onSelectEmbedding,
  onSelectModel
}: SelectionPanelProps) {
  return (
    <section className="panel selection-panel">
      <div className="panel-header">
        <span>Selection</span>
      </div>
      <div className="panel-body panel-body-flush">
        <section className="sel-section" data-kind="project">
          <header className="sel-header">
            <span className="sel-header-title">
              <FcIcon name="projects" size={16} />
              Project
            </span>
            <span className="sel-count">{project ? 1 : 0}</span>
          </header>
          <div className="sel-list">
            {project ? (
              <div className="sel-item is-active" aria-current="true">
                <span className="sel-item-icon" style={{ color: "var(--neext-blue)" }}>
                  <FolderOpen />
                </span>
                <span className="sel-item-name">{project.name}</span>
                <span className="sel-item-sub">Active</span>
              </div>
            ) : (
              <div className="sel-empty">No project</div>
            )}
          </div>
        </section>

        {SHELL_SECTIONS.map((section) => {
          const ItemIcon = section.itemIcon;
          const isDatasetSection = section.kind === "datasets";
          const isFeatureSection = section.kind === "features";
          const isEmbeddingSection = section.kind === "embeddings";
          const isModelSection = section.kind === "models";
          const count = isDatasetSection ? datasets.length : isFeatureSection ? features.length : isEmbeddingSection ? embeddings.length : models.length;
          return (
            <section className="sel-section" data-kind={section.kind} key={section.title}>
              <header className="sel-header">
                <span className="sel-header-title">
                  <FcIcon name={section.fcIcon} size={16} />
                  {section.title}
                </span>
                <span className="sel-count">{count}</span>
              </header>
              <div className="sel-list">
                {isDatasetSection && datasets.length ? (
                  datasets.map((dataset) => (
                    <button
                      key={dataset.id}
                      type="button"
                      className={`sel-item ${dataset.id === selectedDatasetId ? "is-active" : ""}`}
                      aria-current={dataset.id === selectedDatasetId ? "true" : undefined}
                      onClick={() => onSelectDataset(dataset.id)}
                    >
                      <span className="sel-item-icon" style={{ color: section.color }}>
                        <ItemIcon />
                      </span>
                      <span className="sel-item-name">{dataset.name}</span>
                      <span className="sel-item-sub">Dataset</span>
                    </button>
                  ))
                ) : isFeatureSection && features.length ? (
                  features.map((feature) => (
                    <button
                      key={feature.id}
                      type="button"
                      className={`sel-item ${feature.id === selectedFeatureId ? "is-active" : ""}`}
                      aria-current={feature.id === selectedFeatureId ? "true" : undefined}
                      onClick={() => onSelectFeature(feature.id)}
                    >
                      <span className="sel-item-icon" style={{ color: section.color }}>
                        <ItemIcon />
                      </span>
                      <span className="sel-item-name">{feature.name}</span>
                      <span className="sel-item-sub">Feature</span>
                    </button>
                  ))
                ) : isEmbeddingSection && embeddings.length ? (
                  embeddings.map((embedding) => (
                    <button
                      key={embedding.id}
                      type="button"
                      className={`sel-item ${embedding.id === selectedEmbeddingId ? "is-active" : ""}`}
                      aria-current={embedding.id === selectedEmbeddingId ? "true" : undefined}
                      onClick={() => onSelectEmbedding(embedding.id)}
                    >
                      <span className="sel-item-icon" style={{ color: section.color }}>
                        <ItemIcon />
                      </span>
                      <span className="sel-item-name">{embedding.name}</span>
                      <span className="sel-item-sub">Embedding</span>
                    </button>
                  ))
                ) : isModelSection && models.length ? (
                  models.map((model) => (
                    <button
                      key={model.id}
                      type="button"
                      className={`sel-item ${model.id === selectedModelId ? "is-active" : ""}`}
                      aria-current={model.id === selectedModelId ? "true" : undefined}
                      onClick={() => onSelectModel(model.id)}
                    >
                      <span className="sel-item-icon" style={{ color: section.color }}>
                        <ItemIcon />
                      </span>
                      <span className="sel-item-name">{model.name}</span>
                      <span className="sel-item-sub">Model</span>
                    </button>
                  ))
                ) : (
                  <div className="sel-empty">{section.emptyLabel}</div>
                )}
              </div>
            </section>
          );
        })}
      </div>
    </section>
  );
}
