import { parseTracePayload } from '../utils/parseTrace';

self.onmessage = (event) => {
  const data = event.data || {};
  const id = data.id;
  const traceText = data.traceText;

  try {
    const payload = JSON.parse(traceText);
    if (!payload || !Array.isArray(payload.traceEvents)) {
      throw new Error('Trace JSON must contain a traceEvents array.');
    }

    const parsed = parseTracePayload(payload);
    self.postMessage({ id, ok: true, parsed });
  } catch (error) {
    self.postMessage({
      id,
      ok: false,
      error: error?.message || 'Failed to parse trace payload.'
    });
  }
};
