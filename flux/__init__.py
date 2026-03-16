"""
Flux package bootstrap.

Exposes the compiled extension entrypoints and the Python profiler interface.
"""

try:
    import torch  # Ensure libtorch dynamic libraries are loaded first.
    from . import _C as _C
except ImportError:
    _C = None

from .profiler import FluxProfiler

__all__ = ["_C", "FluxProfiler"]
