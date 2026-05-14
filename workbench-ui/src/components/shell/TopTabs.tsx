import { RefreshCw } from "lucide-react";
import type { MainTab } from "../../types";

interface TopTabsProps {
  activeTab: MainTab;
  onSelect: (tab: MainTab) => void;
  onRefresh: () => void;
}

const TABS: MainTab[] = ["home", "datasets", "features", "embeddings", "models"];

export function TopTabs({ activeTab, onSelect, onRefresh }: TopTabsProps) {
  return (
    <header className="tabs-bar">
      <button className="tabs-home" title="NEExT Home" onClick={() => onSelect("home")}>
        <span className="logo-tile" aria-hidden />
      </button>
      <nav className="tabs" aria-label="Workbench tabs">
        {TABS.map((tab) => (
          <button
            key={tab}
            type="button"
            className={`tab${activeTab === tab ? " is-active" : ""}`}
            onClick={() => onSelect(tab)}
          >
            {tab.toUpperCase()}
          </button>
        ))}
      </nav>
      <div className="tabs-quick">
        <button type="button" title="Refresh" onClick={onRefresh} aria-label="Refresh">
          <RefreshCw />
        </button>
      </div>
    </header>
  );
}
