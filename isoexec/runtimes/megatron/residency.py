"""Give a colocated trainer's driver-held local memory back when the trainer offloads.

``isoexec.core.local_memory.install_autotune_guard`` removes the common trigger (autotune sweeps
launching spilling candidates). A kernel whose *selected* config needs a large per-thread stack
still raises the context's local-memory reservation for the rest of the process, and a colocated
engine then runs its generation phase beside it. A worker records ``baseline()`` when it builds its
model and calls ``release(baseline)`` after each offload, lowering the stack limit to what it was
then; the next launch that needs more grows it again by itself.
"""

from __future__ import annotations

from isoexec.core.local_memory import release_local_memory, stack_limit


def baseline() -> int | None:
    """The current per-thread stack limit, or None (with a note) where it cannot be read."""
    try:
        return stack_limit()
    except (RuntimeError, OSError) as exc:
        print(f"[ISOEXEC-LMEM] offload release NOT installed: {exc}", flush=True)
        return None


def release(limit: int | None) -> None:
    """Lower the stack limit to ``limit`` after a trainer offload; no-op without a baseline."""
    if limit is not None:
        release_local_memory(limit, reason="after trainer offload")


__all__ = ["baseline", "release"]
