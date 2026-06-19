import type { MainTab } from "./types";
import type { RibbonCommand } from "./components/shell/Ribbon";

// Persisted slice of navigation state so a browser refresh restores where the
// user was (Space + Ribbon Command + active project + selected artifact + open
// Explore view), instead of resetting to Home -> Projects every time.
export interface PersistedNav {
  topTab: MainTab;
  command: RibbonCommand;
  activeProjectId: string;
  selectedDatasetId: string;
  selectedFeatureId: string;
  selectedEmbeddingId: string;
  selectedModelId: string;
  configureDatasetCatalogId: string;
  configureFeatureCatalogId: string;
  configureEmbeddingCatalogId: string;
  configureModelCatalogId: string;
  exploreDatasetId: string;
  exploreFeatureId: string;
  exploreEmbeddingId: string;
  exploreModelId: string;
  commandWindowHeight: number;
  commandWindowCollapsed: boolean;
}

const STORAGE_KEY = "neext-workbench:nav:v1";

export function readNav(): Partial<PersistedNav> | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as Partial<PersistedNav>) : null;
  } catch {
    // localStorage unavailable (private mode / disabled) or corrupt JSON.
    return null;
  }
}

export function writeNav(nav: PersistedNav): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(nav));
  } catch {
    // localStorage unavailable — persistence is best-effort.
  }
}

// Tracks the MCP UI-state id we last applied, so a stale server-side UI state
// is not re-applied on every page refresh (which would clobber the restored
// localStorage navigation). A genuinely new MCP-driven state still applies.
const MCP_APPLIED_KEY = "neext-workbench:mcp-applied:v1";

export function readAppliedMcpUiStateId(): string {
  try {
    return window.localStorage.getItem(MCP_APPLIED_KEY) ?? "";
  } catch {
    return "";
  }
}

export function writeAppliedMcpUiStateId(id: string): void {
  try {
    window.localStorage.setItem(MCP_APPLIED_KEY, id);
  } catch {
    // localStorage unavailable — best-effort.
  }
}
