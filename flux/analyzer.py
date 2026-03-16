from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

Record = Dict[str, Any]

COMPUTE_HEAVY_OPS = {
    "aten::mm",
    "aten::matmul",
    "aten::addmm",
    "aten::linear",
    "aten::convolution",
    "aten::conv1d",
    "aten::conv2d",
    "aten::conv3d",
}

MEMORY_HEAVY_OPS = {
    "aten::relu",
    "aten::layer_norm",
    "aten::batch_norm",
}


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _estimate_compute_time_us(flops: Optional[float], peak_flops_tflops: float) -> Optional[float]:
    if flops is None or peak_flops_tflops <= 0:
        return None
    return (flops / (peak_flops_tflops * 1e12)) * 1e6


def _estimate_memory_time_us(
    bytes_moved: Optional[float], memory_bandwidth_gbps: float
) -> Optional[float]:
    if bytes_moved is None or memory_bandwidth_gbps <= 0:
        return None
    return (bytes_moved / (memory_bandwidth_gbps * 1e9)) * 1e6


def classify_record(
    record: Record,
    peak_flops_tflops: float = 10.0,
    memory_bandwidth_gbps: float = 300.0,
) -> Record:
    item = dict(record)
    op_name = str(item.get("op_name", "unknown"))
    flops = _as_float(item.get("flops"))
    bytes_moved = _as_float(item.get("bytes_moved"))

    compute_time_us = _estimate_compute_time_us(flops, peak_flops_tflops)
    memory_time_us = _estimate_memory_time_us(bytes_moved, memory_bandwidth_gbps)

    if compute_time_us is not None and memory_time_us is not None:
        if compute_time_us >= memory_time_us:
            classification = "compute-bound"
            bottleneck_us = compute_time_us
        else:
            classification = "memory-bound"
            bottleneck_us = memory_time_us

        other_us = min(compute_time_us, memory_time_us)
        intensity = float("inf") if other_us == 0 else bottleneck_us / max(other_us, 1e-9)
        reason = "model-based"
    elif op_name in MEMORY_HEAVY_OPS:
        classification = "memory-bound"
        intensity = 1.0
        reason = "op-heuristic"
    elif op_name in COMPUTE_HEAVY_OPS:
        classification = "compute-bound"
        intensity = 1.0
        reason = "op-heuristic"
    else:
        classification = "unknown"
        intensity = 0.0
        reason = "insufficient-metadata"

    item["classification"] = classification
    item["classification_reason"] = reason
    item["bottleneck_ratio"] = round(float(intensity), 3)
    return item


def analyze_records(
    records: Iterable[Record],
    peak_flops_tflops: float = 10.0,
    memory_bandwidth_gbps: float = 300.0,
) -> Dict[str, Any]:
    analyzed: List[Record] = []
    counts: Counter[str] = Counter()

    for record in records:
        item = classify_record(
            record,
            peak_flops_tflops=peak_flops_tflops,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
        )
        analyzed.append(item)
        counts[item["classification"]] += 1

    total = len(analyzed)
    percentages = {
        key: round((value / total) * 100.0, 2) if total > 0 else 0.0
        for key, value in counts.items()
    }

    return {
        "records": analyzed,
        "summary": {
            "total_ops": total,
            "counts": dict(counts),
            "percentages": percentages,
        },
    }
