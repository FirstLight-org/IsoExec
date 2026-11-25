"""SM90-only entry of the vocab-parallel sampling head (``ops.shard_sample``).

A separate module so the logprobs facade can name it as an SM90-only export: an arch whose
logprobs package has no ``shard`` module refuses it with ``ArchSelectionError`` and keeps the
gathered-vocabulary sampler.
"""

from __future__ import annotations

from .ops import shard_sample


def build():
    """Compile and load the head's extension now (the engine calls this at model init, not mid-step)."""
    from ._shard_extension import extension

    return extension()


__all__ = ["build", "shard_sample"]
