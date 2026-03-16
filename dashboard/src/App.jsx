import { useEffect, useMemo, useState } from 'react';
import EventDetail from './components/EventDetail';
import SummaryPanel from './components/SummaryPanel';
import Timeline from './components/Timeline';
import TraceLoader from './components/TraceLoader';
import { parseTracePayload } from './utils/parseTrace';

export default function App() {
  const [parsed, setParsed] = useState(null);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [sourceLabel, setSourceLabel] = useState('No source loaded');
  const [error, setError] = useState('');

  function handleLoad(payload, source) {
    try {
      const next = parseTracePayload(payload);
      setParsed(next);
      setSelectedEvent(next.events[0] ?? null);
      setSourceLabel(source || 'Local file');
      setError('');
    } catch (loadError) {
      setError(loadError.message || 'Failed to parse trace payload.');
    }
  }

  useEffect(() => {
    // Attempt to auto-load /trace.json for the local serve workflow.
    let active = true;

    async function bootstrap() {
      try {
        const response = await fetch('/trace.json', { cache: 'no-store' });
        if (!response.ok) {
          return;
        }
        const payload = await response.json();
        if (active) {
          handleLoad(payload, '/trace.json');
        }
      } catch (_error) {
        // No-op: manual file loading is still available.
      }
    }

    bootstrap();

    return () => {
      active = false;
    };
  }, []);

  const statusLabel = useMemo(() => {
    if (!parsed || parsed.events.length === 0) {
      return 'Waiting for trace input';
    }
    return `${parsed.stats.totalOps} ops loaded`;
  }, [parsed]);

  return (
    <div className="app-shell">
      <header className="hero reveal">
        <div>
          <p className="eyebrow">Flux Local Dashboard</p>
          <h1>Hardware-Aware ML Timeline Explorer</h1>
          <p className="hero-subtitle">
            Inspect execution overlap, identify memory pressure, and trace bottlenecks per thread.
          </p>
        </div>
        <div className="hero-meta">
          <div>
            <span className="meta-label">Source</span>
            <span className="meta-value mono">{sourceLabel}</span>
          </div>
          <div>
            <span className="meta-label">Status</span>
            <span className="meta-value">{statusLabel}</span>
          </div>
        </div>
      </header>

      {error ? <div className="error-banner reveal">{error}</div> : null}

      <main className="layout-grid">
        <section className="panel reveal delay-1">
          <h2>Trace Input</h2>
          <TraceLoader onLoad={handleLoad} onError={setError} />
        </section>

        <section className="panel panel-wide reveal delay-2">
          <h2>Execution Timeline</h2>
          <Timeline parsed={parsed} selectedEvent={selectedEvent} onSelect={setSelectedEvent} />
        </section>

        <section className="panel reveal delay-3">
          <h2>Performance Summary</h2>
          <SummaryPanel parsed={parsed} />
        </section>

        <section className="panel reveal delay-4">
          <EventDetail event={selectedEvent} />
        </section>
      </main>
    </div>
  );
}
