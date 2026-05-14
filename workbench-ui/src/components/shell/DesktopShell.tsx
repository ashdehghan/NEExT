interface DesktopShellProps {
  topTabs: React.ReactNode;
  ribbon: React.ReactNode;
  left: React.ReactNode;
  center: React.ReactNode;
  right: React.ReactNode;
  statusBar: React.ReactNode;
}

export function DesktopShell({
  topTabs,
  ribbon,
  left,
  center,
  right,
  statusBar
}: DesktopShellProps) {
  return (
    <div className="shell">
      {topTabs}
      {ribbon}
      <main className="desktop">
        <aside className="desktop-left">{left}</aside>
        <section className="desktop-center">{center}</section>
        <aside className="desktop-right">{right}</aside>
      </main>
      {statusBar}
    </div>
  );
}
