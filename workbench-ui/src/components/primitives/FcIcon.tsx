import type { IconType } from "react-icons";
import {
  FcBarChart,
  FcComboChart,
  FcDatabase,
  FcDataSheet,
  FcEmptyTrash,
  FcFullTrash,
  FcLibrary,
  FcOpenedFolder,
  FcPackage,
  FcPlus,
  FcSettings,
  FcUpload
} from "react-icons/fc";

export const FC_ICONS = {
  import: FcUpload,
  create: FcPlus,
  library: FcLibrary,
  projects: FcOpenedFolder,
  datasets: FcDatabase,
  features: FcDataSheet,
  embeddings: FcPackage,
  models: FcBarChart,
  explore: FcComboChart,
  trash: FcEmptyTrash,
  trashFull: FcFullTrash,
  settings: FcSettings
} as const satisfies Record<string, IconType>;

export type FcIconName = keyof typeof FC_ICONS;

interface FcIconProps {
  name: FcIconName;
  size?: number;
  className?: string;
  title?: string;
}

export function FcIcon({ name, size = 24, className, title }: FcIconProps) {
  const Icon = FC_ICONS[name];
  return <Icon className={["fc-icon", className].filter(Boolean).join(" ")} size={size} title={title} aria-hidden />;
}
