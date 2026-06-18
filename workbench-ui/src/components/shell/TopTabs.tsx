import type { MainTab } from "../../types";

interface TopTabsProps {
  activeTab: MainTab;
  onSelect: (tab: MainTab) => void;
}

const TABS: MainTab[] = ["home", "datasets", "features", "embeddings", "models"];

export function TopTabs({ activeTab, onSelect }: TopTabsProps) {
  return (
    <header className="tabs-bar">
      <button className="tabs-home" title="NEExT Home" onClick={() => onSelect("home")}>
        <span className="tabs-brand">
          <svg className="tabs-brand-mark" viewBox="0 0 32 32" width="22" height="22" fill="none" aria-hidden>
            <circle cx="7" cy="9" r="3" fill="currentColor" />
            <circle cx="25" cy="7" r="2.4" fill="currentColor" opacity="0.75" />
            <circle cx="16" cy="18" r="3.4" fill="currentColor" />
            <circle cx="6" cy="25" r="2.4" fill="currentColor" opacity="0.75" />
            <circle cx="26" cy="24" r="2.8" fill="currentColor" />
            <path d="M7 9l9 9M25 7l-9 11M16 18l-10 7M16 18l10 6" stroke="currentColor" strokeWidth="1.3" opacity="0.5" />
          </svg>
          <span className="tabs-brand-name">NEExT</span>
        </span>
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
    </header>
  );
}
