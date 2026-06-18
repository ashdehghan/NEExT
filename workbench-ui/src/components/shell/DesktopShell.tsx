interface DesktopShellProps {
  topTabs: React.ReactNode;
  ribbon: React.ReactNode;
  left: React.ReactNode;
  center: React.ReactNode;
  right: React.ReactNode;
  statusBar: React.ReactNode;
  commandWindowHeight: number;
  commandWindowCollapsed: boolean;
}

export function DesktopShell({
  topTabs,
  ribbon,
  left,
  center,
  right,
  statusBar,
  commandWindowHeight,
  commandWindowCollapsed
}: DesktopShellProps) {
  return (
    <div className="shell">
      {topTabs}
      {ribbon}
      <main className="desktop">
        <aside className="desktop-left">{left}</aside>
        <section
          className="desktop-center"
          style={{
            gridTemplateRows: commandWindowCollapsed ? "minmax(0, 1fr) 28px" : `minmax(0, 1fr) ${commandWindowHeight}px`
          }}
        >
          {center}
        </section>
        <aside className="desktop-right">{right}</aside>
      </main>
      {statusBar}
    </div>
  );
}
