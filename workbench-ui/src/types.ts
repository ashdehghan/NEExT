export type MainTab = "home" | "datasets" | "features" | "embeddings" | "models";

export const MAIN_TABS: MainTab[] = ["home", "datasets", "features", "embeddings", "models"];

export function titleCase(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}
