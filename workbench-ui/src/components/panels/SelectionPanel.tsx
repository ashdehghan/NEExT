import { BarChart3, Box, Database, FolderOpen, Sigma } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { DatasetManifest, FeatureManifest, ProjectManifest } from "../../api";
import { FcIcon, type FcIconName } from "../primitives/FcIcon";

interface SelectionPanelProps {
  project?: ProjectManifest;
  datasets: DatasetManifest[];
  features: FeatureManifest[];
  selectedDatasetId: string;
  selectedFeatureId: string;
  onSelectDataset: (datasetId: string) => void;
  onSelectFeature: (featureId: string) => void;
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
  selectedDatasetId,
  selectedFeatureId,
  onSelectDataset,
  onSelectFeature
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
          const count = isDatasetSection ? datasets.length : isFeatureSection ? features.length : 0;
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
