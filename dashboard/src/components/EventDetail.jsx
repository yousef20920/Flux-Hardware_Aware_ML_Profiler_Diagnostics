import { formatMicroseconds } from '../utils/parseTrace';

const CLASSIFICATION_INFO = {
  'compute-bound': {
    label: 'Math-Heavy',
    tone: 'compute',
    desc: 'This op does heavy computation — matrix multiplies, convolutions. This is expected for neural network layers.',
  },
  'memory-bound': {
    label: 'Memory-Heavy',
    tone: 'memory',
    desc: 'This op is bottlenecked by memory reads/writes. Common in normalization and activation layers.',
  },
  unknown: {
    label: 'Unclassified',
    tone: 'unknown',
    desc: 'No specific classification. Likely a utility or less common operation.',
  },
};

export default function EventDetail({ event }) {
  if (!event) {
    return (
      <div className="panel-body-empty">
        <div className="empty-icon">🔍</div>
        <h3>Nothing Selected</h3>
        <p>Click any bar in the timeline to inspect it here.</p>
      </div>
    );
  }

  const args = event.args || {};
  const info = CLASSIFICATION_INFO[event.classification] ?? CLASSIFICATION_INFO.unknown;

  return (
    <div className="event-detail-content">
      <div className="event-name-banner">
        <div className="event-name-label">Operation</div>
        <div className="event-name-value">{event.name}</div>
        <span className={`classification-badge ${info.tone}`}>{info.label}</span>
        <div className="classification-desc">{info.desc}</div>
      </div>

      <div className="detail-grid">
        <div className="detail-item">
          <div className="detail-item-label">Duration</div>
          <div className="detail-item-value">{formatMicroseconds(event.dur)}</div>
        </div>
        <div className="detail-item">
          <div className="detail-item-label">Start Time</div>
          <div className="detail-item-value">{formatMicroseconds(event.ts)}</div>
        </div>
        <div className="detail-item">
          <div className="detail-item-label">End Time</div>
          <div className="detail-item-value">{formatMicroseconds(event.end)}</div>
        </div>
        <div className="detail-item">
          <div className="detail-item-label">Thread</div>
          <div className="detail-item-value">{event.tid}</div>
        </div>
      </div>

      {Object.keys(args).length > 0 && (
        <div className="detail-args">
          <div className="detail-args-label">Raw Metadata</div>
          <pre>{JSON.stringify(args, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
