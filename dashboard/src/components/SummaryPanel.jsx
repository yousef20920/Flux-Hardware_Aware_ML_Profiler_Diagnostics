import { formatMicroseconds } from '../utils/parseTrace';

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

function ClassificationBar({ label, pct, tone }) {
  return (
    <div className="class-row">
      <div className="class-label">
        <span className={`dot dot-${tone}`} />
        {label}
      </div>
      <div className="class-meter">
        <div className={`class-meter-fill fill-${tone}`} style={{ width: `${Math.max(0, pct)}%` }} />
      </div>
      <div className="class-value">{pct.toFixed(1)}%</div>
    </div>
  );
}

export default function SummaryPanel({ parsed }) {
  if (!parsed || parsed.events.length === 0) {
    return (
      <div className="panel-body-empty">
        <h3>No Trace Loaded</h3>
        <p>Load a trace to view operation distribution and hardware behavior.</p>
      </div>
    );
  }

  const { stats, lanes } = parsed;

  return (
    <div className="summary-panel-content">
      <div className="stats-grid">
        <StatCard label="Total Ops" value={stats.totalOps} />
        <StatCard label="Threads" value={lanes.length} />
        <StatCard label="Wall Time" value={formatMicroseconds(stats.wallTimeUs)} />
        <StatCard label="Active Time" value={formatMicroseconds(stats.activeTimeUs)} />
        <StatCard label="Idle Time" value={formatMicroseconds(stats.idleTimeUs)} />
        <StatCard label="Utilization" value={`${stats.utilizationPct.toFixed(1)}%`} />
      </div>

      <section className="summary-section">
        <h3>Bound Classification</h3>
        <ClassificationBar
          label="Compute-bound"
          tone="compute"
          pct={stats.classificationPct['compute-bound'] ?? 0}
        />
        <ClassificationBar
          label="Memory-bound"
          tone="memory"
          pct={stats.classificationPct['memory-bound'] ?? 0}
        />
        <ClassificationBar
          label="Unknown"
          tone="unknown"
          pct={stats.classificationPct.unknown ?? 0}
        />
      </section>

      <section className="summary-section">
        <h3>Top Operations</h3>
        <div className="op-list">
          {stats.topOps.slice(0, 8).map((item) => (
            <div className="op-row" key={item.opName}>
              <div className="op-name">{item.opName}</div>
              <div className="op-metrics">
                <span>{item.count} calls</span>
                <span>{formatMicroseconds(item.meanUs)} avg</span>
                <span>{item.sharePct.toFixed(1)}%</span>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
