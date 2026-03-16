import { formatMicroseconds } from '../utils/parseTrace';

function SectionTitle({ children }) {
  return <div className="summary-section-title">{children}</div>;
}

function StatCard({ label, value, desc }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {desc && <div className="stat-desc">{desc}</div>}
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

function formatBytes(value) {
  const bytes = Number(value) || 0;
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(2)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KiB`;
  return `${bytes} B`;
}

export default function SummaryPanel({ parsed }) {
  if (!parsed || parsed.events.length === 0) {
    return (
      <div className="panel-body-empty">
        <div className="empty-icon">📊</div>
        <h3>No Trace Loaded</h3>
        <p>Load a trace to see how your model spent its time.</p>
      </div>
    );
  }

  const { stats } = parsed;
  const gpu = stats.gpu || {};
  const memTotals = gpu?.memory?.totals || {};

  return (
    <div className="summary-panel-content">
      <div>
        <SectionTitle>Time Breakdown</SectionTitle>
        <div className="stats-grid">
          <StatCard
            label="Total Duration"
            value={formatMicroseconds(stats.wallTimeUs)}
            desc="Wall-clock time for the full trace"
          />
          <StatCard
            label="Working Time"
            value={formatMicroseconds(stats.activeTimeUs)}
            desc="Time with ops running"
          />
          <StatCard
            label="Idle Time"
            value={formatMicroseconds(stats.idleTimeUs)}
            desc="Gaps between ops — lower is better"
          />
          <StatCard
            label="Efficiency"
            value={`${stats.utilizationPct.toFixed(1)}%`}
            desc="% of time spent doing work"
          />
        </div>
      </div>

      {gpu.available !== false && (
        <div>
          <SectionTitle>GPU Activity</SectionTitle>
          <div className="stats-grid">
            <StatCard
              label="GPU Operations"
              value={(gpu.cuda_ops ?? 0).toLocaleString()}
              desc="Ops that ran on the GPU"
            />
            <StatCard
              label="GPU Time"
              value={formatMicroseconds(gpu.cuda_time_us ?? 0)}
              desc="Total time on GPU"
            />
            <StatCard
              label="Core Usage"
              value={`${(Number(gpu.sm_utilization_estimate_pct) || 0).toFixed(1)}%`}
              desc="Estimated GPU core utilization"
            />
            <StatCard
              label="Memory Load"
              value={`${(Number(gpu.memory_bandwidth_pressure_estimate_pct) || 0).toFixed(1)}%`}
              desc="GPU memory bandwidth pressure"
            />
          </div>
        </div>
      )}

      <div>
        <SectionTitle>What Was the GPU Doing?</SectionTitle>
        <ClassificationBar
          label="Math-Heavy"
          tone="compute"
          pct={stats.classificationPct['compute-bound'] ?? 0}
        />
        <ClassificationBar
          label="Memory-Heavy"
          tone="memory"
          pct={stats.classificationPct['memory-bound'] ?? 0}
        />
        <ClassificationBar
          label="Other"
          tone="unknown"
          pct={stats.classificationPct.unknown ?? 0}
        />
        <div style={{ marginTop: 8, fontSize: '0.75rem', color: 'var(--text-3)', lineHeight: 1.5 }}>
          Math-heavy ops (matrix multiply, convolution) drive computation.
          Memory-heavy ops wait on data reads/writes.
        </div>
      </div>

      <div>
        <SectionTitle>Top Operations by Time</SectionTitle>
        <div className="op-list">
          {stats.topOps.slice(0, 8).map((item) => (
            <div
              className="op-row"
              key={item.opName}
              style={{ '--share': `${item.sharePct}%` }}
            >
              <div className="op-row-main">
                <div className="op-name">{item.opName.replace('aten::', '')}</div>
                <div className="op-metrics">
                  <span>{item.count}×</span>
                  <span>{formatMicroseconds(item.meanUs)} avg</span>
                  <span>{item.sharePct.toFixed(1)}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {(memTotals.peak_allocated_bytes > 0 || memTotals.peak_reserved_bytes > 0) && (
        <div>
          <SectionTitle>GPU Memory</SectionTitle>
          <div className="memory-grid">
            <div className="memory-item">
              <div className="memory-item-label">Peak Memory Used</div>
              <div className="memory-item-value">{formatBytes(memTotals.peak_allocated_bytes)}</div>
            </div>
            <div className="memory-item">
              <div className="memory-item-label">Peak Memory Reserved</div>
              <div className="memory-item-value">{formatBytes(memTotals.peak_reserved_bytes)}</div>
            </div>
            <div className="memory-item">
              <div className="memory-item-label">Memory Change</div>
              <div className="memory-item-value">{formatBytes(memTotals.allocated_delta_bytes)}</div>
            </div>
            <div className="memory-item">
              <div className="memory-item-label">Reserved Change</div>
              <div className="memory-item-value">{formatBytes(memTotals.reserved_delta_bytes)}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
