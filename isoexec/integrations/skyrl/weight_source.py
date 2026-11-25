"""SkyRL's weight-source entry point; the stream itself is framework-neutral (``isoexec.sync.weight_source``)."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import torch
from vllm.distributed.weight_transfer.base import WeightSource

from isoexec.sync.weight_source import IsoExecWeightSource


class RepeatedWeightSource(WeightSource):
    """``source``'s stream twice in one update round, for an engine that verifies weight updates.

    The engine copies each Parameter on the first pass and compares the bytes it wrote with the second
    (``VllmModel.finish_weight_update``). vLLM's IPC engine sends whatever ``iter_full`` yields in one round;
    SkyRL's NCCL sender ships the names from ``metadata``, so this source is for IPC only.
    """

    def __init__(self, source: WeightSource) -> None:
        self._source = source

    @cached_property
    def metadata(self):
        return self._source.metadata

    def iter_full(self):
        yield from self._source.iter_full()
        yield from self._source.iter_full()


def build_weight_source(module: Any, dtype: torch.dtype, backend: str, *, verify: bool = False) -> WeightSource:
    """Select the native transport only after validating its update lifecycle."""
    if backend not in {"nccl", "ipc"}:
        raise ValueError(f"IsoExec supports nccl or ipc weight sync, got {backend!r}")
    if verify and backend != "ipc":
        raise ValueError("IsoExec verifies weight updates over ipc only")
    source = IsoExecWeightSource(module, dtype)
    return RepeatedWeightSource(source) if verify else source


__all__ = ["IsoExecWeightSource", "RepeatedWeightSource", "build_weight_source"]
