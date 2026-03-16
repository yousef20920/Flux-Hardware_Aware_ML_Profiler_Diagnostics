import { useEffect, useMemo, useRef, useState } from 'react';
import EventDetail from './components/EventDetail';
import SummaryPanel from './components/SummaryPanel';
import Timeline from './components/Timeline';
import TraceLoader from './components/TraceLoader';
import { formatMicroseconds, parseTracePayload } from './utils/parseTrace';

function HeroCard({ label, value, sub, badge, accentColor, delayClass }) {
  return (
    <div className={`hero-card reveal ${delayClass || ''}`} style={{ '--card-accent': accentColor }}>
      <div className="hero-label">{label}</div>
      <div className="hero-value">{value}</div>
      {sub && <div className="hero-sub">{sub}</div>}
      {badge && <span className={`hero-badge ${badge.tone}`}>{badge.text}</span>}
    </div>
  );
}

export default function App() {
  const [parsed, setParsed] = useState(null);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [sourceLabel, setSourceLabel] = useState('');
  const [error, setError] = useState('');
  const [showLoader, setShowLoader] = useState(false);
  const [isParsing, setIsParsing] = useState(false);
  const workerRef = useRef(null);
  const workerRequestIdRef = useRef(0);
  const latestAppliedRequestRef = useRef(0);

  useEffect(() => {
    if (typeof Worker === 'undefined') {
      return undefined;
    }
    const worker = new Worker(new URL('./workers/parseTraceWorker.js', import.meta.url), {
      type: 'module'
    });
    workerRef.current = worker;
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  function applyParsedTrace(next, source) {
    setParsed(next);
    setSelectedEvent(next.events[0] ?? null);
    setSourceLabel(source || 'Local file');
    setError('');
    setShowLoader(false);
  }

  function parseTraceText(traceText) {
    const payload = JSON.parse(traceText);
    if (!payload || !Array.isArray(payload.traceEvents)) {
      throw new Error('Trace JSON must contain a traceEvents array.');
    }
    return parseTracePayload(payload);
  }

  async function handleLoadText(traceText, source) {
    const requestId = ++workerRequestIdRef.current;
    setIsParsing(true);
    setError('');
    try {
      const worker = workerRef.current;
      if (!worker) {
        const next = parseTraceText(traceText);
        latestAppliedRequestRef.current = requestId;
        applyParsedTrace(next, source);
        return;
      }

      const next = await new Promise((resolve, reject) => {
        const onMessage = (event) => {
          const data = event.data || {};
          if (data.id !== requestId) {
            return;
          }
          worker.removeEventListener('message', onMessage);
          worker.removeEventListener('error', onError);
          if (data.ok) {
            resolve(data.parsed);
            return;
          }
          reject(new Error(data.error || 'Failed to parse trace payload.'));
        };
        const onError = (event) => {
          worker.removeEventListener('message', onMessage);
          worker.removeEventListener('error', onError);
          reject(new Error(event?.message || 'Worker failed while parsing trace.'));
        };
        worker.addEventListener('message', onMessage);
        worker.addEventListener('error', onError);
        worker.postMessage({ id: requestId, traceText });
      });

      if (requestId >= latestAppliedRequestRef.current) {
        latestAppliedRequestRef.current = requestId;
        applyParsedTrace(next, source);
      }
    } catch (loadError) {
      setError(loadError?.message || 'Failed to parse trace payload.');
    } finally {
      if (requestId >= latestAppliedRequestRef.current) {
        setIsParsing(false);
      }
    }
  }

  useEffect(() => {
    let active = true;
    async function bootstrap() {
      try {
        const response = await fetch('/trace.json', { cache: 'no-store' });
        if (!response.ok) return;
        const traceText = await response.text();
        if (active) {
          await handleLoadText(traceText, '/trace.json');
        }
      } catch (_err) {}
    }
    bootstrap();
    return () => {
      active = false;
    };
  }, []);

  const hasTrace = Boolean(parsed && parsed.events.length > 0);

  const efficiencyBadge = useMemo(() => {
    if (!parsed) return null;
    const pct = parsed.stats.utilizationPct;
    if (pct >= 70) return { text: '✓ Healthy', tone: 'good' };
    if (pct >= 40) return { text: '~ Moderate', tone: 'warn' };
    return { text: '⚠ High Idle', tone: 'bad' };
  }, [parsed]);

  const gpuBadge = useMemo(() => {
    if (!parsed) return null;
    const ops = parsed.stats.gpu?.cuda_ops ?? 0;
    if (ops > 0) return { text: 'GPU Active', tone: 'good' };
    return { text: 'CPU Only', tone: 'neutral' };
  }, [parsed]);

  return (
    <div className="app-shell">
      <header className="header">
        <div className="header-brand">
          <div className="brand-icon">⚡</div>
          <div>
            <div className="brand-name">Flux</div>
            <div className="brand-tagline">ML Performance Profiler</div>
          </div>
        </div>
        <div className="header-right">
          {hasTrace && sourceLabel && (
            <div className="header-source">
              <span className="source-dot" />
              {sourceLabel}
            </div>
          )}
          {hasTrace && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => setShowLoader((v) => !v)}
            >
              {showLoader ? 'Cancel' : 'Load New Trace'}
            </button>
          )}
        </div>
      </header>

      <div className="main-content">
        {error && <div className="error-banner reveal">{error}</div>}

        {(!hasTrace || showLoader) && (
          <div className="welcome-wrapper reveal">
            {!hasTrace && (
              <div className="welcome-hero">
                <div className="welcome-icon">⚡</div>
                <h1 className="welcome-title">Flux</h1>
                <p className="welcome-sub">
                  Drop a trace file to see exactly where your PyTorch model<br />
                  spends its time — and what's slowing it down.
                </p>
              </div>
            )}
            <TraceLoader onLoadText={handleLoadText} onError={setError} />
          </div>
        )}

        {isParsing && (
          <div className="parsing-overlay reveal">
            <div className="parsing-card">
              <div className="parsing-spinner" />
              <div className="parsing-title">Parsing Trace</div>
              <div className="parsing-subtitle">
                Processing large trace data in a background worker...
              </div>
            </div>
          </div>
        )}

        {hasTrace && !showLoader && (
          <>
            <div className="hero-metrics">
              <HeroCard
                label="Total Run Time"
                value={formatMicroseconds(parsed.stats.wallTimeUs)}
                sub="How long the full trace took"
                accentColor="var(--accent)"
                delayClass=""
              />
              <HeroCard
                label="Efficiency"
                value={`${parsed.stats.utilizationPct.toFixed(1)}%`}
                sub="Time spent actively working"
                badge={efficiencyBadge}
                accentColor="var(--compute)"
                delayClass="delay-1"
              />
              <HeroCard
                label="Operations"
                value={parsed.stats.totalOps.toLocaleString()}
                sub={`Across ${parsed.stats.cpuLaneCount} CPU thread${parsed.stats.cpuLaneCount !== 1 ? 's' : ''}`}
                accentColor="#818cf8"
                delayClass="delay-2"
              />
              <HeroCard
                label="GPU Operations"
                value={(parsed.stats.gpu?.cuda_ops ?? 0).toLocaleString()}
                sub={`${parsed.stats.gpuLaneCount} GPU stream${parsed.stats.gpuLaneCount !== 1 ? 's' : ''} active`}
                badge={gpuBadge}
                accentColor="var(--memory)"
                delayClass="delay-3"
              />
            </div>

            <div className="layout-grid">
              <div className="panel panel-wide reveal delay-1">
                <div className="panel-header">
                  <h2>Execution Timeline</h2>
                  <p>Zoom and scroll to explore operations over time. Click any bar to inspect it.</p>
                </div>
                <div className="panel-body">
                  <Timeline parsed={parsed} selectedEvent={selectedEvent} onSelect={setSelectedEvent} />
                </div>
              </div>

              <div className="panel reveal delay-2">
                <div className="panel-header">
                  <h2>Performance Summary</h2>
                  <p>Where time and resources were spent.</p>
                </div>
                <div className="panel-body">
                  <SummaryPanel parsed={parsed} />
                </div>
              </div>

              <div className="panel reveal delay-3">
                <div className="panel-header">
                  <h2>Event Inspector</h2>
                  <p>Click any bar in the timeline to see details.</p>
                </div>
                <div className="panel-body">
                  <EventDetail event={selectedEvent} />
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
