import { forwardRef } from "react";
import { FcIcon, type FcIconName } from "../primitives/FcIcon";

interface ToolButtonProps {
  icon: FcIconName;
  label: string;
  onClick?: () => void;
  active?: boolean;
  command?: string;
}

export const ToolButton = forwardRef<HTMLButtonElement, ToolButtonProps>(function ToolButton(
  { icon, label, onClick, active, command },
  ref
) {
  return (
    <button
      ref={ref}
      type="button"
      className={`tool-button${active ? " is-active" : ""}`}
      data-command={command}
      aria-pressed={Boolean(active)}
      onClick={onClick}
      title={label}
      aria-label={label}
    >
      <span className="tool-glyph">
        <FcIcon name={icon} size={32} />
      </span>
      <span className="tool-label">{label}</span>
    </button>
  );
});
