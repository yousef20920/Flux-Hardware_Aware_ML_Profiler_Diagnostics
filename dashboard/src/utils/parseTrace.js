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

const TRANSFER_HEAVY = new Set(['aten::copy_', 'aten::_to_copy', 'aten::to', 'cudaMemcpy']);

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

function classifyGpuBucket(opName) {
  if (TRANSFER_HEAVY.has(opName) || opName.includes('copy') || opName.includes('to')) {
    return 'transfer';
  }
  if (MEMORY_HEAVY.has(opName)) {
    return 'memory';
  }
  if (COMPUTE_HEAVY.has(opName)) {
    return 'compute';
  }
  return 'other';
}

function normalizeEvent(event, index) {
  const args = event?.args ?? {};
  const ts = toNumber(event?.ts, 0);
  const dur = Math.max(0, toNumber(event?.dur, 0));
  const opName = String(event?.name ?? args?.op_name ?? `event-${index}`);
  const threadId = toNumber(event?.tid ?? args?.thread_id, 0);
  const isCuda = Boolean(args?.is_cuda);
  const deviceId = toNumber(args?.device_id, -1);
  const streamId = toNumber(args?.stream_id, -1);
  const cudaElapsedUs = toNumber(args?.cuda_elapsed_us, -1);

  const laneType = isCuda && deviceId >= 0 && streamId >= 0 ? 'gpu' : 'cpu';
  const laneId = laneType === 'gpu' ? `gpu-${deviceId}-${streamId}` : `cpu-${threadId}`;
  const laneLabel =
    laneType === 'gpu' ? `GPU ${deviceId} / Stream ${streamId}` : `CPU Thread ${threadId}`;

  return {
    id: `${laneId}-${ts}-${index}`,
    name: opName,
    ts,
    dur,
    end: ts + dur,
    tid: threadId,
    pid: toNumber(event?.pid, 1),
    args,
    classification: classify(opName, args),
    isCuda,
    deviceId,
    streamId,
    cudaElapsedUs,
    laneType,
    laneId,
    laneLabel
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

function buildLaneSortKey(laneType, laneEvents) {
  if (laneType === 'gpu') {
    const first = laneEvents[0];
    return [1, first.deviceId, first.streamId];
  }
  return [0, laneEvents[0]?.tid ?? 0, 0];
}

function deriveGpuSummary(events, wallTimeUs, sourceGpuSummary) {
  const cudaEvents = events.filter((event) => event.isCuda);
  const cudaOps = cudaEvents.length;
  let cudaTimeUs = 0;
  let computeTimeUs = 0;
  let memoryTimeUs = 0;
  let transferTimeUs = 0;

  for (const event of cudaEvents) {
    const effectiveUs = event.cudaElapsedUs > 0 ? event.cudaElapsedUs : event.dur;
    cudaTimeUs += effectiveUs;

    const bucket = classifyGpuBucket(event.name);
    if (bucket === 'compute') {
      computeTimeUs += effectiveUs;
    } else if (bucket === 'memory') {
      memoryTimeUs += effectiveUs;
    } else if (bucket === 'transfer') {
      transferTimeUs += effectiveUs;
    }
  }

  const gpuActivityPct = wallTimeUs > 0 ? Math.min(100, (cudaTimeUs / wallTimeUs) * 100) : 0;
  const computeSharePct = cudaTimeUs > 0 ? (computeTimeUs / cudaTimeUs) * 100 : 0;
  const memorySharePct = cudaTimeUs > 0 ? (memoryTimeUs / cudaTimeUs) * 100 : 0;
  const transferSharePct = cudaTimeUs > 0 ? (transferTimeUs / cudaTimeUs) * 100 : 0;

  const derived = {
    available: cudaOps > 0,
    cuda_ops: cudaOps,
    cuda_time_us: Number(cudaTimeUs.toFixed(2)),
    gpu_activity_pct: Number(gpuActivityPct.toFixed(2)),
    sm_utilization_estimate_pct: Number((gpuActivityPct * (computeSharePct / 100)).toFixed(2)),
    memory_bandwidth_pressure_estimate_pct: Number(
      (gpuActivityPct * (memorySharePct / 100)).toFixed(2)
    ),
    h2d_transfer_pressure_estimate_pct: Number(
      (gpuActivityPct * (transferSharePct / 100)).toFixed(2)
    ),
    compute_share_pct: Number(computeSharePct.toFixed(2)),
    memory_share_pct: Number(memorySharePct.toFixed(2)),
    transfer_share_pct: Number(transferSharePct.toFixed(2)),
    memory: {
      cuda_available: false,
      device_count: 0,
      devices: [],
      totals: {
        allocated_start_bytes: 0,
        allocated_end_bytes: 0,
        allocated_delta_bytes: 0,
        reserved_start_bytes: 0,
        reserved_end_bytes: 0,
        reserved_delta_bytes: 0,
        peak_allocated_bytes: 0,
        peak_reserved_bytes: 0
      }
    }
  };

  if (sourceGpuSummary && typeof sourceGpuSummary === 'object') {
    return {
      ...derived,
      ...sourceGpuSummary,
      memory: sourceGpuSummary.memory ?? derived.memory
    };
  }
  return derived;
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
      sourceMetadata: payload?.metadata ?? null,
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
        topOps: [],
        cpuLaneCount: 0,
        gpuLaneCount: 0,
        gpu: deriveGpuSummary([], 0, payload?.summary?.gpu ?? null)
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

    if (!laneMap.has(event.laneId)) {
      laneMap.set(event.laneId, {
        id: event.laneId,
        tid: event.tid,
        type: event.laneType,
        label: event.laneLabel,
        events: [],
        sortKey: buildLaneSortKey(event.laneType, [event])
      });
    }
    laneMap.get(event.laneId).events.push(event);

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

  const lanes = [...laneMap.values()]
    .map((lane) => ({
      ...lane,
      totalDurationUs: lane.events.reduce((sum, item) => sum + item.dur, 0),
      eventCount: lane.events.length
    }))
    .sort((a, b) => {
      if (a.sortKey[0] !== b.sortKey[0]) {
        return a.sortKey[0] - b.sortKey[0];
      }
      if (a.sortKey[1] !== b.sortKey[1]) {
        return a.sortKey[1] - b.sortKey[1];
      }
      return a.sortKey[2] - b.sortKey[2];
    });

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

  const gpuLaneCount = lanes.filter((lane) => lane.type === 'gpu').length;
  const cpuLaneCount = lanes.filter((lane) => lane.type === 'cpu').length;
  const gpu = deriveGpuSummary(events, rangeUs, payload?.summary?.gpu ?? null);

  return {
    sourceSummary: payload?.summary ?? null,
    sourceMetadata: payload?.metadata ?? null,
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
      topOps,
      cpuLaneCount,
      gpuLaneCount,
      gpu
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
