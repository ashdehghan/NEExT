import type { ReactNode } from "react";

interface ConfirmDialogProps {
  title: string;
  message: string;
  confirmLabel: string;
  cancelLabel?: string;
  busy?: boolean;
  /** Label shown on the confirm button while busy (defaults to "Deleting"). */
  busyLabel?: string;
  /** Disable the confirm button (e.g. until a type-to-confirm value matches). */
  confirmDisabled?: boolean;
  error?: string;
  children?: ReactNode;
  onCancel: () => void;
  onConfirm: () => void;
}

export function ConfirmDialog({
  title,
  message,
  confirmLabel,
  cancelLabel = "Cancel",
  busy = false,
  busyLabel = "Deleting",
  confirmDisabled = false,
  error,
  children,
  onCancel,
  onConfirm
}: ConfirmDialogProps) {
  return (
    <div className="confirm-backdrop">
      <section className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
        <header className="confirm-head">
          <h3 id="confirm-title">{title}</h3>
        </header>
        <div className="confirm-body">
          <p>{message}</p>
          {children}
          {error ? <p className="error-text">{error}</p> : null}
        </div>
        <footer className="confirm-foot">
          <button type="button" className="btn" onClick={onCancel} disabled={busy}>
            {cancelLabel}
          </button>
          <button type="button" className="btn btn-danger" onClick={onConfirm} disabled={busy || confirmDisabled}>
            {busy ? busyLabel : confirmLabel}
          </button>
        </footer>
      </section>
    </div>
  );
}
