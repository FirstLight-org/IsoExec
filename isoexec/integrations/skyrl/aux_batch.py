"""Preserve synthetic-row identity through SkyRL's native batch iterators."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import sys
from typing import Any

import torch

VALID_ROWS = "isoexec_aux_valid_rows"
_ACTIVE: ContextVar[list[torch.Tensor] | None] = ContextVar("isoexec_aux_batches", default=None)


def install_padding_tracking() -> None:
    """Mark padding at its producer, before DP slicing or token-based reordering.

    SkyRL preserves tensor fields through both operations. A policy loss mask
    cannot identify synthetic rows: real overlong samples can be loss-masked too.
    The adapter changes no SkyRL source files or batching algorithm.
    """
    from skyrl.backends.skyrl_train import training_batch

    original = training_batch.pad_training_input_batch
    if getattr(original, "_isoexec_aux_padding", False):
        return

    @wraps(original)
    def pad(batch, pad_size):
        valid = batch.get(VALID_ROWS)
        if valid is None:
            valid = torch.ones(batch.batch_size, dtype=torch.bool, device=batch["sequences"].device)
        result = original(batch, pad_size)
        result[VALID_ROWS] = torch.cat((valid, valid.new_zeros(pad_size)))
        return result

    setattr(pad, "_isoexec_aux_padding", True)
    # Existing modules may have imported the function by value. Future imports
    # see the wrapper directly from training_batch.
    for name, module in tuple(sys.modules.items()):
        if module is not None and name.startswith("skyrl."):
            for attribute, value in tuple(vars(module).items()):
                if value is original:
                    setattr(module, attribute, pad)


def _install_iterator_observer() -> None:
    from skyrl.backends.skyrl_train.workers.worker_utils import BaseBatchIterator

    original = BaseBatchIterator.batch_to_experience
    if getattr(original, "_isoexec_aux_batches", False):
        return

    @wraps(original)
    def convert(batch):
        experience = original(batch)
        observed = _ACTIVE.get()
        if observed is not None:
            valid = batch.get(VALID_ROWS)
            if valid is None:
                valid = torch.ones(batch.batch_size, dtype=torch.bool, device=batch["sequences"].device)
            if (batch.metadata or {}).get("is_padding_batch", False):
                valid = torch.zeros_like(valid)
            observed.append(valid)
        return experience

    setattr(convert, "_isoexec_aux_batches", True)
    BaseBatchIterator.batch_to_experience = staticmethod(convert)


def install_worker(worker: Any) -> None:
    """Enable row observation for the actor's registered training method."""
    if getattr(worker, "_isoexec_aux_batches_installed", False):
        return
    _install_iterator_observer()
    worker._isoexec_aux_batches_installed = True


@contextmanager
def worker_microbatches(worker: Any) -> Iterator[None]:
    """Keep row observations within the method Ray actually registered."""
    if not getattr(worker, "_isoexec_aux_batches_installed", False):
        yield
        return
    if _ACTIVE.get() is not None:
        raise RuntimeError("Auxiliary batch metadata cannot overlap a training invocation")
    token = _ACTIVE.set([])
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def annotate_microbatches(micro_batches: Any, *, required: bool = False) -> Any:
    """Join row identity with the worker's uniformly padded microbatch dictionaries."""
    observed = _ACTIVE.get()
    if observed is None:
        if required:
            raise RuntimeError("Auxiliary training requires the registered IsoExec policy worker")
        return micro_batches
    if len(observed) != len(micro_batches):
        raise RuntimeError("Native batch conversion and auxiliary microbatch counts differ")
    batches = []
    for batch, valid in zip(micro_batches, observed, strict=True):
        rows = batch["attention_mask"].shape[0]
        if rows < valid.numel():
            raise ValueError("Native microbatch contains fewer rows than its original batch")
        valid = torch.cat((valid, valid.new_zeros(rows - valid.numel())))
        batches.append({**batch, VALID_ROWS: valid})
    return batches
