"""
Flux package bootstrap.

Currently exposes the compiled extension entrypoints.
"""

try:
    import torch  # Ensure libtorch dynamic libraries are loaded first.
    from . import _C as _C
except ImportError:
    _C = None

__all__ = ["_C"]
