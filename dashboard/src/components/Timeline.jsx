import { useEffect, useMemo, useRef, useState } from 'react';
import { formatMicroseconds } from '../utils/parseTrace';

const LEFT_GUTTER = 160;
const TOP_AXIS = 44;
const LANE_HEIGHT = 38;
const LANE_GAP = 10;
const BOTTOM_PADDING = 24;
const RIGHT_PADDING = 40;

const EVENT_COLORS = {
  'compute-bound': '#1aa56d',
  'memory-bound': '#e35343',
  unknown: '#8795a7'
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function niceTickStep(target) {
  if (target <= 0) {
    return 1;
  }
  const power = 10 ** Math.floor(Math.log10(target));
  const candidates = [1, 2, 5, 10].map((n) => n * power);
  return candidates.find((candidate) => candidate >= target) ?? 10 * power;
}

function formatTick(us) {
  if (us >= 1_000_000) {
    return `${(us / 1_000_000).toFixed(2)} s`;
  }
  if (us >= 1000) {
    return `${(us / 1000).toFixed(2)} ms`;
  }
  return `${Math.round(us)} us`;
}

export default function Timeline({ parsed, selectedEvent, onSelect }) {
  const canvasRef = useRef(null);
  const scrollRef = useRef(null);
  const hitboxesRef = useRef([]);

  const [viewportWidth, setViewportWidth] = useState(900);
  const [zoom, setZoom] = useState(0.12);

  useEffect(() => {
    const target = scrollRef.current;
    if (!target) {
      return undefined;
    }

    const updateSize = () => setViewportWidth(target.clientWidth || 900);
    updateSize();

    const observer = new ResizeObserver(updateSize);
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    // Reset scroll when loading a new trace so the user starts at the first events.
    if (scrollRef.current) {
      scrollRef.current.scrollLeft = 0;
    }
  }, [parsed?.bounds?.startUs]);

  const laneCount = parsed?.lanes?.length ?? 0;
  const totalHeight = useMemo(
    () => TOP_AXIS + laneCount * (LANE_HEIGHT + LANE_GAP) + BOTTOM_PADDING,
    [laneCount]
  );

  const rangeUs = parsed?.bounds?.rangeUs ?? 0;
  const totalWidth = useMemo(() => {
    const chartWidth = LEFT_GUTTER + rangeUs * zoom + RIGHT_PADDING;
    return Math.max(viewportWidth, chartWidth);
  }, [rangeUs, viewportWidth, zoom]);

  function applyZoom(nextZoom, anchorPx = null) {
    const scroller = scrollRef.current;
    if (!scroller) {
      setZoom(clamp(nextZoom, 0.01, 1.2));
      return;
    }

    const bounded = clamp(nextZoom, 0.01, 1.2);
    const oldZoom = zoom;
    const anchor =
      anchorPx ?? scroller.scrollLeft + scroller.clientWidth / 2 - LEFT_GUTTER;
    const anchorTimeUs = Math.max(0, anchor / oldZoom);

    setZoom(bounded);

    requestAnimationFrame(() => {
      const nextScroll = anchorTimeUs * bounded - (scroller.clientWidth / 2 - LEFT_GUTTER);
      scroller.scrollLeft = Math.max(0, nextScroll);
    });
  }

  useEffect(() => {
    const scroller = scrollRef.current;
    if (!scroller) {
      return undefined;
    }

    function handleWheel(event) {
      // Ctrl/cmd + wheel zooms around cursor, regular wheel keeps native scrolling.
      if (!(event.ctrlKey || event.metaKey)) {
        return;
      }
      event.preventDefault();
      const delta = event.deltaY > 0 ? 0.9 : 1.1;
      const rect = scroller.getBoundingClientRect();
      const anchor = scroller.scrollLeft + (event.clientX - rect.left) - LEFT_GUTTER;
      applyZoom(zoom * delta, anchor);
    }

    scroller.addEventListener('wheel', handleWheel, { passive: false });
    return () => scroller.removeEventListener('wheel', handleWheel);
  }, [zoom]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !parsed || parsed.events.length === 0) {
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(totalWidth * dpr);
    canvas.height = Math.floor(totalHeight * dpr);
    canvas.style.width = `${totalWidth}px`;
    canvas.style.height = `${totalHeight}px`;

    const ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    // Canvas background and chart frame.
    ctx.fillStyle = '#f7fafc';
    ctx.fillRect(0, 0, totalWidth, totalHeight);
    ctx.fillStyle = '#eef3f8';
    ctx.fillRect(LEFT_GUTTER, TOP_AXIS, totalWidth - LEFT_GUTTER, totalHeight - TOP_AXIS);

    // Top axis with readable tick spacing across large/small ranges.
    const chartWidth = totalWidth - LEFT_GUTTER - RIGHT_PADDING;
    const tickStepUs = niceTickStep((chartWidth / 8) / zoom);
    ctx.font = '12px IBM Plex Mono, monospace';
    ctx.fillStyle = '#5d6d83';
    ctx.strokeStyle = '#d6e1ec';
    ctx.lineWidth = 1;

    for (let tickUs = 0; tickUs <= rangeUs; tickUs += tickStepUs) {
      const x = LEFT_GUTTER + tickUs * zoom;
      ctx.beginPath();
      ctx.moveTo(x, TOP_AXIS - 6);
      ctx.lineTo(x, totalHeight - BOTTOM_PADDING + 4);
      ctx.stroke();
      ctx.fillText(formatTick(tickUs), x + 4, 16);
    }

    // Draw lanes and event rectangles, and cache hitboxes for click selection.
    const hitboxes = [];
    parsed.lanes.forEach((lane, laneIndex) => {
      const y = TOP_AXIS + laneIndex * (LANE_HEIGHT + LANE_GAP);

      ctx.fillStyle = laneIndex % 2 === 0 ? '#f2f6fb' : '#edf2f7';
      ctx.fillRect(LEFT_GUTTER, y, totalWidth - LEFT_GUTTER, LANE_HEIGHT);

      ctx.fillStyle = '#1f2e45';
      ctx.font = '600 12px Space Grotesk, sans-serif';
      ctx.fillText(`Thread ${lane.tid}`, 14, y + 16);
      ctx.font = '11px IBM Plex Mono, monospace';
      ctx.fillStyle = '#617188';
      ctx.fillText(`${lane.eventCount} events`, 14, y + 31);

      lane.events.forEach((event) => {
        const x = LEFT_GUTTER + (event.ts - parsed.bounds.startUs) * zoom;
        const width = Math.max(1.5, event.dur * zoom);
        const color = EVENT_COLORS[event.classification] ?? EVENT_COLORS.unknown;

        ctx.fillStyle = color;
        ctx.fillRect(x, y + 6, width, LANE_HEIGHT - 12);

        if (selectedEvent?.id === event.id) {
          ctx.lineWidth = 2;
          ctx.strokeStyle = '#111827';
          ctx.strokeRect(x, y + 6, width, LANE_HEIGHT - 12);
        }

        if (width > 68) {
          ctx.font = '11px IBM Plex Mono, monospace';
          ctx.fillStyle = '#f8fafc';
          ctx.fillText(event.name.replace('aten::', ''), x + 6, y + 24);
        }

        hitboxes.push({
          x,
          y: y + 6,
          width,
          height: LANE_HEIGHT - 12,
          event
        });
      });
    });

    hitboxesRef.current = hitboxes;
  }, [parsed, rangeUs, selectedEvent?.id, totalHeight, totalWidth, zoom]);

  function handleCanvasClick(event) {
    const x = event.nativeEvent.offsetX;
    const y = event.nativeEvent.offsetY;

    // Search from the latest drawn rectangle so overlap favors top-most bar.
    for (let index = hitboxesRef.current.length - 1; index >= 0; index -= 1) {
      const box = hitboxesRef.current[index];
      if (x >= box.x && x <= box.x + box.width && y >= box.y && y <= box.y + box.height) {
        onSelect(box.event);
        return;
      }
    }
  }

  if (!parsed || parsed.events.length === 0) {
    return (
      <div className="panel-body-empty">
        <h3>Timeline</h3>
        <p>Load a trace to render thread lanes and operation spans.</p>
      </div>
    );
  }

  return (
    <div className="timeline-shell">
      <div className="timeline-toolbar">
        <div className="zoom-controls">
          <button type="button" className="btn btn-ghost" onClick={() => applyZoom(zoom * 0.8)}>
            -
          </button>
          <input
            type="range"
            min="0.01"
            max="1.2"
            step="0.01"
            value={zoom}
            onChange={(event) => applyZoom(Number(event.target.value))}
            aria-label="Zoom"
          />
          <button type="button" className="btn btn-ghost" onClick={() => applyZoom(zoom * 1.25)}>
            +
          </button>
        </div>
        <div className="timeline-meta">
          <span>Range: {formatMicroseconds(parsed.bounds.rangeUs)}</span>
          <span>Ops: {parsed.stats.totalOps}</span>
          <span>Lanes: {parsed.lanes.length}</span>
        </div>
      </div>

      <div className="timeline-scroll" ref={scrollRef}>
        <canvas ref={canvasRef} onClick={handleCanvasClick} />
      </div>

      <div className="timeline-legend">
        <span>
          <i className="legend-swatch compute" /> Compute-bound
        </span>
        <span>
          <i className="legend-swatch memory" /> Memory-bound
        </span>
        <span>
          <i className="legend-swatch unknown" /> Unknown
        </span>
      </div>
    </div>
  );
}
