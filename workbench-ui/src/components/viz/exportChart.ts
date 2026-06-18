/**
 * Publication-quality chart export helpers.
 *
 * Charts are rendered with the default canvas renderer (best for interactivity + raster export).
 * - PNG: `getDataURL` straight off the live canvas instance, with a configurable pixel ratio for
 *   high-resolution figures.
 * - SVG: re-render the chart's current option into a throwaway offscreen SVG-renderer instance and
 *   grab the markup. This works for any chart without the chart itself knowing about export.
 */
import * as echarts from "echarts";

type EChartsInstance = ReturnType<typeof echarts.init>;

function triggerDownload(href: string, filename: string) {
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

export function downloadChartPng(chart: EChartsInstance, filename: string, pixelRatio = 2): void {
  const url = chart.getDataURL({ type: "png", pixelRatio, backgroundColor: "#ffffff" });
  triggerDownload(url, filename);
}

export function downloadChartSvg(chart: EChartsInstance, filename: string): void {
  const option = chart.getOption();
  const rect = chart.getDom().getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));

  const holder = document.createElement("div");
  holder.style.position = "absolute";
  holder.style.left = "-99999px";
  holder.style.top = "0";
  holder.style.width = `${width}px`;
  holder.style.height = `${height}px`;
  document.body.appendChild(holder);

  const tmp = echarts.init(holder, null, { renderer: "svg", width, height });
  try {
    tmp.setOption(option);
    const svg = holder.querySelector("svg");
    if (svg) {
      const blob = new Blob([svg.outerHTML], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      triggerDownload(url, filename);
      URL.revokeObjectURL(url);
    }
  } finally {
    tmp.dispose();
    holder.remove();
  }
}

export type ChartExportFormat = "png" | "png2x" | "svg";

export function exportChart(chart: EChartsInstance, baseName: string, format: ChartExportFormat): void {
  const safe = baseName.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^[._-]+|[._-]+$/g, "") || "chart";
  switch (format) {
    case "png":
      return downloadChartPng(chart, `${safe}.png`, 1);
    case "png2x":
      return downloadChartPng(chart, `${safe}@2x.png`, 2);
    case "svg":
      return downloadChartSvg(chart, `${safe}.svg`);
  }
}
