from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

Record = Dict[str, Any]


def _trace_args_from_record(record: Record) -> Dict[str, Any]:
    keys = (
        "op_name",
        "duration_us",
        "classification",
        "classification_reason",
        "bottleneck_ratio",
    )
    args: Dict[str, Any] = {}
    for key in keys:
        if key in record:
            args[key] = record[key]
    return args


def records_to_trace_events(records: Iterable[Record], pid: int = 1) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for record in records:
        start_us = int(record.get("start_us", 0))
        duration_us = int(record.get("duration_us", 0))
        event = {
            "name": str(record.get("op_name", "unknown")),
            "cat": "ml-op",
            "ph": "X",
            "ts": start_us,
            "dur": max(duration_us, 0),
            "pid": int(record.get("pid", pid)),
            "tid": int(record.get("thread_id", 0)),
            "args": _trace_args_from_record(record),
        }
        events.append(event)
    events.sort(key=lambda x: (x["ts"], x["dur"]))
    return events


def export_trace(
    records: Iterable[Record],
    output_path: str,
    summary: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"traceEvents": records_to_trace_events(records)}
    if summary:
        payload["summary"] = summary
    if metadata:
        payload["metadata"] = metadata

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
