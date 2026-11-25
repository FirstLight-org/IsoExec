"""One logical floating default for model construction and graph capture."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from typing import Any, cast

import torch
from torch.fx.experimental import _config as symbolic_config

CAPTURE_DEFAULT_DTYPE = torch.float32
_LOCK = RLock()


@contextmanager
def capture_defaults() -> Iterator[None]:
    """Pin dtype-less author factories, restoring the caller's native context.

    Explicit component/weight dtypes are unaffected. Process construction
    settings must remain quiescent during capture; the lock serializes IsoExec
    construction/capture calls, not unrelated framework threads.
    """
    # Default export specializes backed sizes away from 0/1 even when the
    # declared range includes them. Retain those sizes in shape reasoning so
    # a shape-dependent branch must prove that it covers the declared domain.
    with _LOCK, cast(Any, symbolic_config).patch(backed_size_oblivious=True):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(CAPTURE_DEFAULT_DTYPE)
            yield
        finally:
            torch.set_default_dtype(previous)


__all__ = ["CAPTURE_DEFAULT_DTYPE", "capture_defaults"]
