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
  fcIcon: FcIconName;
  itemIcon: LucideIcon;
  color: string;
}[] = [
  { title: "Datasets", fcIcon: "datasets", itemIcon: Database, color: "var(--neext-blue)" },
  { title: "Features", fcIcon: "features", itemIcon: Sigma, color: "var(--neext-orange)" },
  { title: "Embeddings", fcIcon: "embeddings", itemIcon: Box, color: "var(--neext-green)" },
  { title: "Models", fcIcon: "models", itemIcon: BarChart3, color: "var(--neext-purple)" }
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
        <section className="sel-section">
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
            ) : null}
          </div>
        </section>

        {SHELL_SECTIONS.map((section) => {
          const ItemIcon = section.itemIcon;
          const isDatasetSection = section.title === "Datasets";
          const isFeatureSection = section.title === "Features";
          const isEmbeddingSection = section.title === "Embeddings";
          const isModelSection = section.title === "Models";
          const count = isDatasetSection ? datasets.length : isFeatureSection ? features.length : isEmbeddingSection ? embeddings.length : models.length;
          return (
            <section className="sel-section" key={section.title}>
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
                  <div className="sel-item sel-item-shell" aria-hidden="true">
                    <span className="sel-item-icon" style={{ color: section.color }}>
                      <ItemIcon />
                    </span>
                    <span className="sel-item-name" />
                    <span className="sel-item-sub" />
                  </div>
                )}
              </div>
            </section>
          );
        })}
      </div>
    </section>
  );
}
