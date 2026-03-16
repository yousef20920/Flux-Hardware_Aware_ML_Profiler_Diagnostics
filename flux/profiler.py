from __future__ import annotations

from contextlib import ContextDecorator
from typing import Any, Dict, List

import torch

from . import _C

Record = Dict[str, Any]


class FluxProfiler(ContextDecorator):
    """Context manager for collecting ATen timing records."""

    def __init__(self, clear_on_start: bool = True, timing_mode: str = "auto") -> None:
        self.clear_on_start = clear_on_start
        self.timing_mode = timing_mode
        self._running = False
        self._records: List[Record] = []
        self._gpu_memory_telemetry: Dict[str, Any] = {}
        self._gpu_memory_start: Dict[int, Dict[str, int]] = {}

    @property
    def records(self) -> List[Record]:
        return list(self._records)

    @property
    def gpu_memory_telemetry(self) -> Dict[str, Any]:
        return dict(self._gpu_memory_telemetry)

    @property
    def is_running(self) -> bool:
        return self._running

    def _capture_gpu_memory_start(self) -> None:
        self._gpu_memory_start = {}
        if not torch.cuda.is_available():
            return

        try:
            device_count = int(torch.cuda.device_count())
        except Exception:
            return

        for device_id in range(device_count):
            try:
                torch.cuda.reset_peak_memory_stats(device_id)
                self._gpu_memory_start[device_id] = {
                    "allocated_start_bytes": int(torch.cuda.memory_allocated(device_id)),
                    "reserved_start_bytes": int(torch.cuda.memory_reserved(device_id)),
                }
            except Exception:
                # Skip devices that are visible but not queryable in this runtime.
                continue

    def _capture_gpu_memory_stop(self) -> Dict[str, Any]:
        if not torch.cuda.is_available():
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

        devices: List[Dict[str, Any]] = []
        totals = {
            "allocated_start_bytes": 0,
            "allocated_end_bytes": 0,
            "allocated_delta_bytes": 0,
            "reserved_start_bytes": 0,
            "reserved_end_bytes": 0,
            "reserved_delta_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }

        try:
            device_count = int(torch.cuda.device_count())
        except Exception:
            return {
                "cuda_available": False,
                "device_count": 0,
                "devices": [],
                "totals": totals,
            }

        for device_id in range(device_count):
            try:
                start = self._gpu_memory_start.get(
                    device_id, {"allocated_start_bytes": 0, "reserved_start_bytes": 0}
                )
                allocated_end = int(torch.cuda.memory_allocated(device_id))
                reserved_end = int(torch.cuda.memory_reserved(device_id))
                peak_allocated = int(torch.cuda.max_memory_allocated(device_id))
                peak_reserved = int(torch.cuda.max_memory_reserved(device_id))
                device_name = torch.cuda.get_device_name(device_id)
            except Exception:
                continue

            item = {
                "device_id": device_id,
                "device_name": device_name,
                "allocated_start_bytes": int(start["allocated_start_bytes"]),
                "allocated_end_bytes": allocated_end,
                "allocated_delta_bytes": allocated_end - int(start["allocated_start_bytes"]),
                "reserved_start_bytes": int(start["reserved_start_bytes"]),
                "reserved_end_bytes": reserved_end,
                "reserved_delta_bytes": reserved_end - int(start["reserved_start_bytes"]),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
            }

            for key in totals:
                totals[key] += int(item[key])

            devices.append(item)

        return {
            "cuda_available": bool(devices),
            "device_count": device_count,
            "devices": devices,
            "totals": totals,
        }

    def start(self) -> None:
        if _C is None:
            raise RuntimeError(
                "Flux C++ extension is not available. Run `pip install -e .` first."
            )
        if self._running:
            return
        if self.timing_mode not in {"auto", "cpu", "cuda"}:
            raise RuntimeError("Invalid timing_mode. Expected one of: auto, cpu, cuda.")
        _C.flux_set_timing_mode(self.timing_mode)
        if self.clear_on_start:
            _C.flux_clear_records()
        # Capture start memory and reset peaks so this run's peak is isolated.
        self._capture_gpu_memory_start()
        _C.flux_start()
        self._running = True

    def stop(self) -> List[Record]:
        if not self._running:
            return list(self._records)
        _C.flux_stop()
        self._records = list(_C.flux_get_records())
        self._gpu_memory_telemetry = self._capture_gpu_memory_stop()
        self._running = False
        return list(self._records)

    def __enter__(self) -> "FluxProfiler":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False
