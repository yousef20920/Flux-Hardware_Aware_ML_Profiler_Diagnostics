from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

Record = Dict[str, Any]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_records(records: Iterable[Record]) -> List[Record]:
    normalized: List[Record] = []
    for record in records:
        item = dict(record)
        item["op_name"] = str(item.get("op_name", "unknown"))
        item["start_us"] = _as_int(item.get("start_us"))
        item["end_us"] = _as_int(item.get("end_us"))
        item["duration_us"] = max(0, _as_int(item.get("duration_us")))
        item["thread_id"] = _as_int(item.get("thread_id"))
        normalized.append(item)

    normalized.sort(key=lambda x: (x["start_us"], x["end_us"]))
    return normalized


def aggregate_records(records: Iterable[Record]) -> Dict[str, Any]:
    items = normalize_records(records)
    if not items:
        return {
            "total_ops": 0,
            "total_duration_us": 0,
            "wall_time_us": 0,
            "idle_time_us": 0,
            "utilization_pct": 0.0,
            "ops": [],
            "idle_gaps": [],
        }

    op_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"count": 0, "total_us": 0, "min_us": 0, "max_us": 0}
    )
    total_duration = 0

    for item in items:
        duration = item["duration_us"]
        total_duration += duration
        stat = op_stats[item["op_name"]]
        stat["count"] += 1
        stat["total_us"] += duration
        if stat["count"] == 1:
            stat["min_us"] = duration
            stat["max_us"] = duration
        else:
            stat["min_us"] = min(stat["min_us"], duration)
            stat["max_us"] = max(stat["max_us"], duration)

    first_start = items[0]["start_us"]
    last_end = max(item["end_us"] for item in items)
    wall_time = max(0, last_end - first_start)

    idle_gaps: List[Record] = []
    idle_time = 0
    cursor = items[0]["end_us"]
    for item in items[1:]:
        if item["start_us"] > cursor:
            gap = item["start_us"] - cursor
            idle_time += gap
            idle_gaps.append(
                {
                    "start_us": cursor,
                    "end_us": item["start_us"],
                    "duration_us": gap,
                }
            )
        cursor = max(cursor, item["end_us"])

    utilization_pct = 0.0
    if wall_time > 0:
        utilization_pct = round(((wall_time - idle_time) / wall_time) * 100.0, 2)

    ops = []
    for op_name, stat in op_stats.items():
        mean_us = stat["total_us"] / stat["count"]
        pct_total = (stat["total_us"] / total_duration * 100.0) if total_duration > 0 else 0.0
        ops.append(
            {
                "op_name": op_name,
                "count": stat["count"],
                "total_us": stat["total_us"],
                "mean_us": round(mean_us, 2),
                "min_us": stat["min_us"],
                "max_us": stat["max_us"],
                "pct_total_duration": round(pct_total, 2),
            }
        )

    ops.sort(key=lambda x: x["total_us"], reverse=True)

    return {
        "total_ops": len(items),
        "total_duration_us": total_duration,
        "wall_time_us": wall_time,
        "idle_time_us": idle_time,
        "utilization_pct": utilization_pct,
        "ops": ops,
        "idle_gaps": idle_gaps,
    }
