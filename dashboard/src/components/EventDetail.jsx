import { formatMicroseconds } from '../utils/parseTrace';

function DetailRow({ label, value }) {
  return (
    <div className="detail-row">
      <div className="detail-label">{label}</div>
      <div className="detail-value">{value}</div>
    </div>
  );
}

export default function EventDetail({ event }) {
  if (!event) {
    return (
      <div className="panel-body-empty compact">
        <h3>Event Inspector</h3>
        <p>Click any bar in the timeline to inspect operation details.</p>
      </div>
    );
  }

  const args = event.args || {};

  return (
    <div className="event-detail-content">
      <h3>Event Inspector</h3>
      <DetailRow label="Operation" value={event.name} />
      <DetailRow label="Classification" value={event.classification} />
      <DetailRow label="Duration" value={formatMicroseconds(event.dur)} />
      <DetailRow label="Start" value={formatMicroseconds(event.ts)} />
      <DetailRow label="End" value={formatMicroseconds(event.end)} />
      <DetailRow label="Thread" value={event.tid} />
      <DetailRow label="Process" value={event.pid} />
      <div className="detail-args">
        <div className="detail-label">Raw Args</div>
        <pre>{JSON.stringify(args, null, 2)}</pre>
      </div>
    </div>
  );
}
