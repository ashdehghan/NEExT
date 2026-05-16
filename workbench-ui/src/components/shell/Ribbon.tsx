import { ToolButton } from "./ToolButton";
import type { FcIconName } from "../primitives/FcIcon";
import type { MainTab } from "../../types";

export type RibbonCommand = "import" | "create" | "projects" | "settings" | "help" | "library" | "explore" | MainTab;

interface RibbonProps {
  activeTab: MainTab;
  activeCommand: string;
  onCommand: (command: RibbonCommand) => void;
}

type ToolDef = {
  label: string;
  icon: FcIconName;
  command: RibbonCommand;
};

type GroupDef = { label: string; tools: ToolDef[] };

const STRUCTURAL_TOOLS = {
  import: { label: "Import", icon: "import", command: "import" },
  library: { label: "Library", icon: "library", command: "library" },
  create: { label: "Create", icon: "create", command: "create" }
} as const satisfies Record<string, ToolDef>;

const GROUPS: Record<MainTab, GroupDef[]> = {
  home: [
    {
      label: "Project Management",
      tools: [
        STRUCTURAL_TOOLS.import,
        STRUCTURAL_TOOLS.create,
        { label: "Projects", icon: "projects", command: "projects" }
      ]
    },
    {
      label: "App Management",
      tools: [
        { label: "Settings", icon: "settings", command: "settings" },
        { label: "Help", icon: "help", command: "help" }
      ]
    }
  ],
  datasets: [
    {
      label: "Dataset Management",
      tools: [
        STRUCTURAL_TOOLS.import,
        STRUCTURAL_TOOLS.library,
        STRUCTURAL_TOOLS.create,
        { label: "Datasets", icon: "datasets", command: "datasets" }
      ]
    },
    {
      label: "Dataset Analysis",
      tools: [{ label: "Explore", icon: "explore", command: "explore" }]
    }
  ],
  features: [
    {
      label: "Feature Management",
      tools: [
        STRUCTURAL_TOOLS.import,
        STRUCTURAL_TOOLS.library,
        STRUCTURAL_TOOLS.create,
        { label: "Features", icon: "features", command: "features" }
      ]
    },
    {
      label: "Feature Analysis",
      tools: [{ label: "Explore", icon: "explore", command: "explore" }]
    }
  ],
  embeddings: [
    {
      label: "Embedding Management",
      tools: [
        STRUCTURAL_TOOLS.import,
        STRUCTURAL_TOOLS.library,
        STRUCTURAL_TOOLS.create,
        { label: "Embeddings", icon: "embeddings", command: "embeddings" }
      ]
    },
    {
      label: "Embedding Analysis",
      tools: [{ label: "Explore", icon: "explore", command: "explore" }]
    }
  ],
  models: [
    {
      label: "Model Management",
      tools: [
        STRUCTURAL_TOOLS.import,
        STRUCTURAL_TOOLS.library,
        STRUCTURAL_TOOLS.create,
        { label: "Models", icon: "models", command: "models" }
      ]
    },
    {
      label: "Model Analysis",
      tools: [{ label: "Explore", icon: "explore", command: "explore" }]
    }
  ]
};

export function Ribbon({ activeTab, activeCommand, onCommand }: RibbonProps) {
  return (
    <section className="ribbon" aria-label="Ribbon commands">
      {GROUPS[activeTab].map((group) => (
        <div className="tool-group" key={group.label}>
          <div className="tool-items">
            {group.tools.map((tool) => (
              <ToolButton
                key={`${group.label}-${tool.label}`}
                icon={tool.icon}
                label={tool.label}
                command={String(tool.command)}
                active={tool.command === activeCommand}
                onClick={() => onCommand(tool.command)}
              />
            ))}
          </div>
          <div className="group-label">{group.label}</div>
        </div>
      ))}
    </section>
  );
}
