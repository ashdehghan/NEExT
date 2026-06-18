/**
 * ChartCard — a reusable, publication-quality wrapper for an ECharts plot.
 *
 * Provides consistent scientific styling (title, headline metrics, a controls slot for knobs) and a
 * three-dot export menu that downloads the plot as PNG / high-res PNG / SVG so a user can drop the
 * figure straight into a manuscript. Also renders a running/error overlay so a card can show
 * in-progress state in place.
 *
 * The card is presentational: the caller owns the ECharts instance and passes its ref so the export
 * menu can reach it. The chart container itself goes in `children`.
 */
import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import * as echarts from "echarts";
import { RunningState } from "../primitives/RunningState";
import { exportChart, type ChartExportFormat } from "./exportChart";

type EChartsInstance = ReturnType<typeof echarts.init>;

interface ChartCardProps {
  title: string;
  /** Headline metrics / description line under the title. */
  subtitle?: ReactNode;
  /** Knobs and toggles rendered in the header's control area. */
  controls?: ReactNode;
  /** Live ECharts instance ref, used by the export menu. */
  chartRef: RefObject<EChartsInstance | null>;
  /** Base filename for exports (without extension). */
  exportName?: string;
  running?: boolean;
  runningLabel?: string;
  runningDetail?: string;
  error?: string;
  /** Shown instead of the chart when there's nothing to plot. */
  empty?: ReactNode;
  /** The chart container element. */
  children: ReactNode;
  className?: string;
}

const EXPORT_OPTIONS: { format: ChartExportFormat; label: string }[] = [
  { format: "png", label: "Download PNG" },
  { format: "png2x", label: "Download PNG (2×)" },
  { format: "svg", label: "Download SVG (vector)" }
];

export function ChartCard({
  title,
  subtitle,
  controls,
  chartRef,
  exportName,
  running,
  runningLabel,
  runningDetail,
  error,
  empty,
  children,
  className
}: ChartCardProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!menuOpen) return;
    const onPointerDown = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [menuOpen]);

  const handleExport = (format: ChartExportFormat) => {
    setMenuOpen(false);
    const chart = chartRef.current;
    if (chart) exportChart(chart, exportName ?? title, format);
  };

  const exportDisabled = Boolean(running || error || empty);

  return (
    <section className={["chart-card", className].filter(Boolean).join(" ")}>
      <header className="chart-card-head">
        <div className="chart-card-titles">
          <h4 className="chart-card-title">{title}</h4>
          {subtitle ? <div className="chart-card-subtitle">{subtitle}</div> : null}
        </div>
        <div className="chart-card-actions">
          {controls ? <div className="chart-card-controls">{controls}</div> : null}
          <div className="chart-card-menu" ref={menuRef}>
            <button
              type="button"
              className="icon-btn chart-card-menu-btn"
              aria-haspopup="menu"
              aria-expanded={menuOpen}
              aria-label="Export chart"
              title="Export chart"
              disabled={exportDisabled}
              onClick={() => setMenuOpen((open) => !open)}
            >
              <KebabIcon />
            </button>
            {menuOpen ? (
              <div className="chart-card-menu-list" role="menu">
                {EXPORT_OPTIONS.map((option) => (
                  <button
                    key={option.format}
                    type="button"
                    role="menuitem"
                    className="chart-card-menu-item"
                    onClick={() => handleExport(option.format)}
                  >
                    <DownloadIcon />
                    <span>{option.label}</span>
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        </div>
      </header>
      <div className="chart-card-body">
        {children}
        {error ? (
          <div className="chart-card-overlay">
            <p className="error-text">{error}</p>
          </div>
        ) : running ? (
          <div className="chart-card-overlay">
            <RunningState label={runningLabel ?? "Computing…"} detail={runningDetail} compact />
          </div>
        ) : empty ? (
          <div className="chart-card-overlay chart-card-overlay-empty">{empty}</div>
        ) : null}
      </div>
    </section>
  );
}

function KebabIcon() {
  return (
    <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden>
      <circle cx="8" cy="3" r="1.4" />
      <circle cx="8" cy="8" r="1.4" />
      <circle cx="8" cy="13" r="1.4" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden>
      <path d="M8 2.5v7.5M5 7l3 3 3-3" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3 12.5h10" strokeLinecap="round" />
    </svg>
  );
}
