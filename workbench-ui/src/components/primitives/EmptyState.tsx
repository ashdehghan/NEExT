interface EmptyStateProps {
  children: React.ReactNode;
  icon?: React.ReactNode;
  compact?: boolean;
}

export function EmptyState({ children, icon, compact }: EmptyStateProps) {
  return (
    <div className={`empty-state${compact ? " is-compact" : ""}`}>
      {icon && <div className="empty-icon">{icon}</div>}
      <div>{children}</div>
    </div>
  );
}
