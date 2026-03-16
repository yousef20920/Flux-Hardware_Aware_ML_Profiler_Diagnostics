from __future__ import annotations

from contextlib import ContextDecorator
from typing import Any, Dict, List

from . import _C

Record = Dict[str, Any]


class FluxProfiler(ContextDecorator):
    """Context manager for collecting ATen timing records."""

    def __init__(self, clear_on_start: bool = True) -> None:
        self.clear_on_start = clear_on_start
        self._running = False
        self._records: List[Record] = []

    @property
    def records(self) -> List[Record]:
        return list(self._records)

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if _C is None:
            raise RuntimeError(
                "Flux C++ extension is not available. Run `pip install -e .` first."
            )
        if self._running:
            return
        if self.clear_on_start:
            _C.flux_clear_records()
        _C.flux_start()
        self._running = True

    def stop(self) -> List[Record]:
        if not self._running:
            return list(self._records)
        _C.flux_stop()
        self._records = list(_C.flux_get_records())
        self._running = False
        return list(self._records)

    def __enter__(self) -> "FluxProfiler":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False
