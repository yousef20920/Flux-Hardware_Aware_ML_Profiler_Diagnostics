const COMPUTE_HEAVY = new Set([
  'aten::mm',
  'aten::matmul',
  'aten::addmm',
  'aten::linear',
  'aten::convolution',
  'aten::conv1d',
  'aten::conv2d',
  'aten::conv3d'
]);

const MEMORY_HEAVY = new Set(['aten::relu', 'aten::layer_norm', 'aten::batch_norm']);

function toNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function classify(opName, args = {}) {
  if (typeof args.classification === 'string') {
    return args.classification;
  }
  if (MEMORY_HEAVY.has(opName)) {
    return 'memory-bound';
  }
  if (COMPUTE_HEAVY.has(opName)) {
    return 'compute-bound';
  }
  return 'unknown';
}

function normalizeEvent(event, index) {
  const args = event?.args ?? {};
  const ts = toNumber(event?.ts, 0);
  const dur = Math.max(0, toNumber(event?.dur, 0));
  const opName = String(event?.name ?? args?.op_name ?? `event-${index}`);
  const threadId = toNumber(event?.tid ?? args?.thread_id, 0);

  return {
    id: `${threadId}-${ts}-${index}`,
    name: opName,
    ts,
    dur,
    end: ts + dur,
    tid: threadId,
    pid: toNumber(event?.pid, 1),
    args,
    classification: classify(opName, args)
  };
}

function computeActiveTime(events) {
  if (!events.length) {
    return 0;
  }

  const sorted = [...events].sort((a, b) => a.ts - b.ts || a.end - b.end);
  let active = 0;
  let cursorStart = sorted[0].ts;
  let cursorEnd = sorted[0].end;

  for (let i = 1; i < sorted.length; i += 1) {
    const current = sorted[i];
    if (current.ts > cursorEnd) {
      active += Math.max(0, cursorEnd - cursorStart);
      cursorStart = current.ts;
      cursorEnd = current.end;
      continue;
    }
    cursorEnd = Math.max(cursorEnd, current.end);
  }

  active += Math.max(0, cursorEnd - cursorStart);
  return active;
}

export function parseTracePayload(payload) {
  const rawEvents = Array.isArray(payload?.traceEvents) ? payload.traceEvents : [];

  const events = rawEvents
    .filter((event) => event?.ph === 'X')
    .map((event, index) => normalizeEvent(event, index))
    .sort((a, b) => a.ts - b.ts || a.end - b.end);

  if (!events.length) {
    return {
      sourceSummary: payload?.summary ?? null,
      events: [],
      lanes: [],
      stats: {
        totalOps: 0,
        totalDurationUs: 0,
        wallTimeUs: 0,
        activeTimeUs: 0,
        idleTimeUs: 0,
        utilizationPct: 0,
        classifications: {},
        classificationPct: {},
        topOps: []
      },
      bounds: {
        startUs: 0,
        endUs: 0,
        rangeUs: 0
      }
    };
  }

  const startUs = events[0].ts;
  const endUs = Math.max(...events.map((event) => event.end));
  const rangeUs = Math.max(0, endUs - startUs);

  const laneMap = new Map();
  const opStats = new Map();
  const classifications = {
    'compute-bound': 0,
    'memory-bound': 0,
    unknown: 0
  };

  let totalDurationUs = 0;

  for (const event of events) {
    totalDurationUs += event.dur;

    if (!laneMap.has(event.tid)) {
      laneMap.set(event.tid, []);
    }
    laneMap.get(event.tid).push(event);

    if (!opStats.has(event.name)) {
      opStats.set(event.name, { opName: event.name, count: 0, totalUs: 0 });
    }
    const op = opStats.get(event.name);
    op.count += 1;
    op.totalUs += event.dur;

    if (!Object.prototype.hasOwnProperty.call(classifications, event.classification)) {
      classifications[event.classification] = 0;
    }
    classifications[event.classification] += 1;
  }

  const lanes = [...laneMap.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([tid, laneEvents]) => ({
      tid,
      events: laneEvents,
      totalDurationUs: laneEvents.reduce((sum, item) => sum + item.dur, 0),
      eventCount: laneEvents.length
    }));

  const topOps = [...opStats.values()]
    .map((item) => ({
      ...item,
      meanUs: item.count > 0 ? item.totalUs / item.count : 0,
      sharePct: totalDurationUs > 0 ? (item.totalUs / totalDurationUs) * 100 : 0
    }))
    .sort((a, b) => b.totalUs - a.totalUs);

  const activeTimeUs = computeActiveTime(events);
  const idleTimeUs = Math.max(0, rangeUs - activeTimeUs);
  const utilizationPct = rangeUs > 0 ? (activeTimeUs / rangeUs) * 100 : 0;

  const totalEvents = events.length;
  const classificationPct = Object.fromEntries(
    Object.entries(classifications).map(([key, value]) => [
      key,
      totalEvents > 0 ? (value / totalEvents) * 100 : 0
    ])
  );

  return {
    sourceSummary: payload?.summary ?? null,
    events,
    lanes,
    stats: {
      totalOps: events.length,
      totalDurationUs,
      wallTimeUs: rangeUs,
      activeTimeUs,
      idleTimeUs,
      utilizationPct,
      classifications,
      classificationPct,
      topOps
    },
    bounds: {
      startUs,
      endUs,
      rangeUs
    }
  };
}

export function formatMicroseconds(value) {
  const numeric = toNumber(value, 0);
  if (numeric >= 1000) {
    return `${(numeric / 1000).toFixed(2)} ms`;
  }
  return `${numeric.toFixed(2)} us`;
}
