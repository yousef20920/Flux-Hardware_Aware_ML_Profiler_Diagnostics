from __future__ import annotations

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

TRANSFER_OP_HINTS = {
    "aten::copy_",
    "aten::_to_copy",
    "aten::to",
    "cudaMemcpy",
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _classify_gpu_op(op_name: str) -> str:
    if op_name in TRANSFER_OP_HINTS:
        return "transfer"
    if "copy" in op_name or "to" in op_name:
        return "transfer"
    if op_name in MEMORY_HEAVY_OPS:
        return "memory"
    if op_name in COMPUTE_HEAVY_OPS:
        return "compute"
    return "other"


def _empty_memory_telemetry() -> Dict[str, Any]:
    return {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "totals": {
            "allocated_start_bytes": 0,
            "allocated_end_bytes": 0,
            "allocated_delta_bytes": 0,
            "reserved_start_bytes": 0,
            "reserved_end_bytes": 0,
            "reserved_delta_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        },
    }


def analyze_gpu_records(
    records: Iterable[Record],
    wall_time_us: Optional[float] = None,
    memory_telemetry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    memory = memory_telemetry or _empty_memory_telemetry()
    cuda_records: List[Record] = []
    per_device: Dict[int, Dict[str, float]] = {}

    total_cuda_time_us = 0.0
    compute_time_us = 0.0
    memory_time_us = 0.0
    transfer_time_us = 0.0

    for record in records:
        if not bool(record.get("is_cuda", False)):
            continue

        cuda_records.append(record)
        op_name = str(record.get("op_name", "unknown"))
        duration_us = _as_float(record.get("duration_us"), 0.0)
        cuda_elapsed_us = _as_float(record.get("cuda_elapsed_us"), -1.0)
        effective_us = cuda_elapsed_us if cuda_elapsed_us > 0 else duration_us
        total_cuda_time_us += effective_us

        bucket = _classify_gpu_op(op_name)
        if bucket == "compute":
            compute_time_us += effective_us
        elif bucket == "memory":
            memory_time_us += effective_us
        elif bucket == "transfer":
            transfer_time_us += effective_us

        device_id = int(record.get("device_id", -1))
        if device_id >= 0:
            device_stat = per_device.setdefault(
                device_id,
                {
                    "device_id": device_id,
                    "op_count": 0.0,
                    "cuda_time_us": 0.0,
                    "compute_time_us": 0.0,
                    "memory_time_us": 0.0,
                    "transfer_time_us": 0.0,
                },
            )
            device_stat["op_count"] += 1.0
            device_stat["cuda_time_us"] += effective_us
            if bucket == "compute":
                device_stat["compute_time_us"] += effective_us
            elif bucket == "memory":
                device_stat["memory_time_us"] += effective_us
            elif bucket == "transfer":
                device_stat["transfer_time_us"] += effective_us

    wall = max(0.0, float(wall_time_us or 0.0))
    gpu_activity_pct = min(100.0, (total_cuda_time_us / wall) * 100.0) if wall > 0 else 0.0
    compute_share_pct = (
        (compute_time_us / total_cuda_time_us) * 100.0 if total_cuda_time_us > 0 else 0.0
    )
    memory_share_pct = (
        (memory_time_us / total_cuda_time_us) * 100.0 if total_cuda_time_us > 0 else 0.0
    )
    transfer_share_pct = (
        (transfer_time_us / total_cuda_time_us) * 100.0 if total_cuda_time_us > 0 else 0.0
    )

    sm_utilization_estimate_pct = min(100.0, gpu_activity_pct * (compute_share_pct / 100.0))
    memory_bandwidth_pressure_estimate_pct = min(
        100.0, gpu_activity_pct * (memory_share_pct / 100.0)
    )
    h2d_transfer_pressure_estimate_pct = min(
        100.0, gpu_activity_pct * (transfer_share_pct / 100.0)
    )

    devices = [
        {
            "device_id": int(item["device_id"]),
            "op_count": int(item["op_count"]),
            "cuda_time_us": round(float(item["cuda_time_us"]), 2),
            "compute_share_pct": round(
                (float(item["compute_time_us"]) / float(item["cuda_time_us"])) * 100.0, 2
            )
            if float(item["cuda_time_us"]) > 0
            else 0.0,
            "memory_share_pct": round(
                (float(item["memory_time_us"]) / float(item["cuda_time_us"])) * 100.0, 2
            )
            if float(item["cuda_time_us"]) > 0
            else 0.0,
            "transfer_share_pct": round(
                (float(item["transfer_time_us"]) / float(item["cuda_time_us"])) * 100.0, 2
            )
            if float(item["cuda_time_us"]) > 0
            else 0.0,
        }
        for _, item in sorted(per_device.items(), key=lambda pair: pair[0])
    ]

    return {
        "available": bool(cuda_records) or bool(memory.get("cuda_available", False)),
        "cuda_ops": len(cuda_records),
        "cuda_time_us": round(total_cuda_time_us, 2),
        "gpu_activity_pct": round(gpu_activity_pct, 2),
        "sm_utilization_estimate_pct": round(sm_utilization_estimate_pct, 2),
        "memory_bandwidth_pressure_estimate_pct": round(
            memory_bandwidth_pressure_estimate_pct, 2
        ),
        "h2d_transfer_pressure_estimate_pct": round(
            h2d_transfer_pressure_estimate_pct, 2
        ),
        "compute_share_pct": round(compute_share_pct, 2),
        "memory_share_pct": round(memory_share_pct, 2),
        "transfer_share_pct": round(transfer_share_pct, 2),
        "device_breakdown": devices,
        "memory": memory,
    }
