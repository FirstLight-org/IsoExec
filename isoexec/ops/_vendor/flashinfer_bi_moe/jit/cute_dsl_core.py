# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Modified for IsoExec: FlashInfer's on-disk CuTe-DSL module cache (flashinfer/jit/cute_dsl_core.py, JitSpecCuteDsl) is
# not vendored; kernels compile once per process and are kept in memory, keyed by (module, kernel); a compile
# requested inside CUDA graph capture is refused.
"""In-process cache of compiled CuTe-DSL kernels (the two entry points ``batch_invariant/jit.py`` uses)."""

from __future__ import annotations

import re
import threading
from typing import Any, Callable, Dict, Sequence, Tuple

_CACHE: Dict[Tuple[str, str], Any] = {}
_LOCK = threading.Lock()


def sanitize_symbol_name(name: str) -> str:
    """Sanitize a name into a valid C symbol / filename component."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)


def build_and_load_cute_dsl_kernel(module_name: str, kernel_name: str, compile_fn: Callable[[], Any],
                                   extra_key_files: Sequence[str] = (), *, enable_tvm_ffi: bool = True) -> Any:
    key = (module_name, kernel_name)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is None:
            import torch
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"CuTe-DSL kernel {module_name}/{kernel_name} is not compiled: compile every variant "
                                   "at placement (warm-up), never inside CUDA graph capture")
            hit = _CACHE[key] = compile_fn()
    return hit


def compiled_kernels() -> Tuple[Tuple[str, str], ...]:
    """The (module, kernel) names compiled in this process (warm-up accounting)."""
    with _LOCK:
        return tuple(_CACHE)
