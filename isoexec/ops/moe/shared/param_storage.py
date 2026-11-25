"""Parameter-storage bookkeeping: one mutation epoch, one sync boundary, one buffer allocator.

Packed experts and packed projections give model parameters a shared contiguous
layout so stacking becomes a free alias. Their layouts differ, but the bookkeeping
must be shared literally:

  * **One mutation epoch.** :data:`EPOCH` is bumped by every parameter mutation that can break an
    alias -- Parameter OBJECT replacement (:class:`WatchedParamDict`) and ``.data =`` / ``set_`` /
    ``resize_`` / ``detach_`` (:class:`DataWatch`). Each buffer records the epoch it last verified
    at, so "nothing changed" is an O(1) integer compare and the O(E) rescan runs only after a real
    rebind. Two epochs would mean a rebind raised through one module's watcher is invisible to the
    other's fast path, and the stale-alias fallback would stop firing -- so there is exactly one,
    and it lives here rather than in either consumer.
  * **One sync boundary.** :func:`bump_sync_epoch` is called once per weight sync by the engine's
    receiver. Everything that must be re-established at that boundary registers with
    :func:`on_sync_boundary`; registration happens when a module is imported, so a process that
    never built the layout it manages pays nothing. :data:`SYNC_EPOCH` is what rate-limits a
    re-fuse / re-pack to one attempt per sync.
  * **One buffer allocator.** :func:`alloc_param_buffer` is where these buffers come from, so
    "give the driver back what the first allocation stranded" (:func:`release_stranded_segments`)
    has a single trigger.

None of this names an op, a rounding schedule or an impl id: it is bookkeeping about where a
Parameter's bytes live. Runtime adapters inject allocation policy; this module
imports no runtime.

``common/`` may not import a runtime, and this module does not: the vLLM-specific allocation policy
(place the buffer in the sleepable CuMem pool so level-1 sleep can offload the expert weights) is
INJECTED with :func:`set_buffer_allocator` by ``runtimes/vllm/vllm_plugin.py`` at worker start,
before any model is built. Unregistered, the default is a plain ``torch.empty``, which is what a
trainer process gets today.
"""

from __future__ import annotations

import gc
import logging
from types import GetSetDescriptorType
from typing import Callable, cast

import torch

logger = logging.getLogger(__name__)

# PyTorch's stub describes this as a Tensor, but class access returns the
# descriptor. DataWatch is mixed into Tensor subclasses at runtime.
_TENSOR_DATA = cast(GetSetDescriptorType, torch.Tensor.data)

# ---------------------------------------------------------------------------------------------
# mutation epoch
# ---------------------------------------------------------------------------------------------

# Makes "nothing changed" observable in O(1) so the O(E) full scan runs only after an actual rebind.
# The two mutations that can break an alias are intercepted at their only entry points -- Parameter
# object replacement in ``mod._parameters`` (:class:`WatchedParamDict`) and ``param.data = ...`` /
# ``set_`` / ``resize_`` / ``detach_`` on the recorded object (:class:`DataWatch`) -- and both bump
# this. ``Tensor._version`` does NOT bump on a ``.data`` rebind, so version counters cannot stand in.
# The epoch is global, not per-module: a rebind anywhere forces one full scan per buffer on the next
# read, which is a fast-path miss and never a correctness hole.
EPOCH = 0


def bump_epoch() -> None:
    global EPOCH
    EPOCH += 1


class WatchedParamDict(dict):
    """``mod._parameters`` that reports being mutated. Same dict semantics, plus an epoch bump.

    Installed on each parameter's owning module at fuse/pack time. Every path that can swap a
    Parameter OBJECT -- ``nn.Module.__setattr__``, ``del mod.weight``, and
    ``vllm_worker._set_on_module``'s direct ``mod._parameters["weight"] = Parameter(...)`` -- goes
    through one of these methods.
    """

    __slots__ = ()

    def __setitem__(self, k, v):
        bump_epoch()
        dict.__setitem__(self, k, v)

    def __delitem__(self, k):
        bump_epoch()
        dict.__delitem__(self, k)

    def pop(self, *a):
        bump_epoch()
        return dict.pop(self, *a)

    def popitem(self):
        bump_epoch()
        return dict.popitem(self)

    def clear(self):
        bump_epoch()
        dict.clear(self)

    def update(self, *a, **kw):
        bump_epoch()
        dict.update(self, *a, **kw)

    def setdefault(self, k, d=None):
        bump_epoch()
        return dict.setdefault(self, k, d)


class DataWatch:
    """Mixin that makes a Parameter's storage rebinds observable.

    Mixed in by reassigning ``__class__`` on the aliased params -- the SAME objects, so identity in
    ``_parameters``, ``isinstance(p, nn.Parameter)``, ``requires_grad`` and any subclass attributes
    all survive, arithmetic still yields plain tensors, and unfusing/unpacking puts the original
    class back. The getter has to be redefined too: a setter-only property would shadow
    ``Tensor.data``'s getter and make reads raise.
    """

    __slots__ = ()

    @property
    def data(self):  # noqa: D102
        return _TENSOR_DATA.__get__(self)

    @data.setter
    def data(self, value):
        bump_epoch()
        _TENSOR_DATA.__set__(self, value)

    def set_(self, *a, **kw):  # noqa: D102
        bump_epoch()
        return cast(Callable[..., torch.Tensor], torch.Tensor.set_)(cast(torch.Tensor, self), *a, **kw)

    def resize_(self, *a, **kw):  # noqa: D102
        bump_epoch()
        return cast(Callable[..., torch.Tensor], torch.Tensor.resize_)(cast(torch.Tensor, self), *a, **kw)

    def detach_(self, *a, **kw):  # noqa: D102
        bump_epoch()
        return cast(Callable[..., torch.Tensor], torch.Tensor.detach_)(cast(torch.Tensor, self), *a, **kw)


# Built per concrete parameter class: Megatron and vLLM put Parameter SUBCLASSES on expert weights, and
# hard-coding ``nn.Parameter`` would leave those unwatched and silently disable fusing.
_WATCHED_CLASSES: dict[type, type] = {}


def watched_class(cls: type) -> type:
    if issubclass(cls, DataWatch):
        return cls
    w = _WATCHED_CLASSES.get(cls)
    if w is None:
        w = type(f"_DataWatched{cls.__name__}", (DataWatch, cls), {"_ix_unwatched": cls})
        _WATCHED_CLASSES[cls] = w
    return w


# ---------------------------------------------------------------------------------------------
# buffer allocation
# ---------------------------------------------------------------------------------------------

# Set when an allocation actually called into the allocator; see release_stranded_segments.
_PENDING_RELEASE = [False]

_ALLOCATOR: Callable[..., torch.Tensor] | None = None


def set_buffer_allocator(fn: Callable[..., torch.Tensor] | None) -> None:
    """Install the process's allocator for aliased parameter buffers. Idempotent.

    ``runtimes/vllm/vllm_plugin.py`` registers its ``alloc_sleepable`` callback here at
    worker start: in a vLLM process the buffer must come from the sleepable CuMem pool, or level-1
    sleep cannot offload the expert weights and they stay resident through the training window.
    That policy needs ``vllm.device_allocator.cumem``, which ``common/`` may not import -- hence the
    injection rather than a call. ``None`` restores the plain ``torch.empty`` default.
    """
    global _ALLOCATOR
    _ALLOCATOR = fn


def alloc_param_buffer(shape, dtype, device) -> torch.Tensor:
    """Allocate one aliased-parameter buffer, through the registered allocator if there is one."""
    _PENDING_RELEASE[0] = True
    if _ALLOCATOR is not None:
        return _ALLOCATOR(shape, dtype, device)
    return torch.empty(shape, dtype=dtype, device=device)


def buffer_signature(buf: torch.Tensor) -> tuple[
    int, int, int, tuple[int, ...], tuple[int, ...], torch.dtype, torch.device, torch.layout,
]:
    """Owner-local address and layout facts; allocation counters are statistics only."""
    storage = buf.untyped_storage()
    return (buf.data_ptr(), storage.data_ptr(), storage.nbytes(), tuple(buf.shape), tuple(buf.stride()),
            buf.dtype, buf.device, buf.layout)


def release_stranded_segments(force: bool = False) -> float:
    """Give the driver back the memory the first fuse/pack stranded in the caching allocator.

    Per-parameter tensors come from the allocator's small pool and cannot satisfy the aliased
    buffers' large-block requests, so the first allocation leaves a full copy's worth of small-pool
    segments reserved but unusable. Must be called at a sync boundary (:func:`bump_sync_epoch` does),
    never inside a forward. Returns the GiB returned to the driver.
    """
    if not (force or _PENDING_RELEASE[0]) or not torch.cuda.is_available():
        return 0.0
    _PENDING_RELEASE[0] = False
    before = torch.cuda.memory_reserved()
    gc.collect()
    torch.cuda.empty_cache()
    freed = (before - torch.cuda.memory_reserved()) / (1 << 30)
    logger.info("[isoexec-param-storage] released %.2f GiB of post-fuse allocator fragmentation", freed)
    return freed


# ---------------------------------------------------------------------------------------------
# sync boundary
# ---------------------------------------------------------------------------------------------

# Bumped once per weight sync. A broken alias is not re-established on the read path -- that would
# race an owner who legitimately took the storage -- so a re-fuse / re-pack is licensed once per
# weight sync announcing itself here. Without the limit, anything that breaks the alias permanently
# would turn every forward into a full re-copy, which is slower than the stack it replaces.
SYNC_EPOCH = 0

_SYNC_HOOKS: list[Callable[[], object]] = []
_SCOPED_SYNC_HOOKS: list[Callable[[torch.nn.Module], object]] = []
_SCOPED_PREPARE_HOOKS: list[Callable[[torch.nn.Module], object]] = []


def on_sync_boundary(fn: Callable[[], object]) -> Callable[[], object]:
    """Register ``fn`` to run inside every :func:`bump_sync_epoch`. Idempotent; returns ``fn``.

    Called at module import by each consumer, so a process that never imported (and therefore never
    built) a given layout runs none of its refresh work -- the property the previous
    ``sys.modules``-probe wiring had, without the string key.
    """
    if fn not in _SYNC_HOOKS:
        _SYNC_HOOKS.append(fn)
    return fn


def on_scoped_sync_boundary(
    fn: Callable[[torch.nn.Module], object],
) -> Callable[[torch.nn.Module], object]:
    """Register a refresher that visits only modules owned by the given root.

    A runtime stage can share a process with unrelated models. Refreshers receive the
    actual module tree, so ownership is determined by object identity rather than
    a name or by the process-wide packed-module registries.
    """
    if fn not in _SCOPED_SYNC_HOOKS:
        _SCOPED_SYNC_HOOKS.append(fn)
    return fn


def on_packing_prepare(fn: Callable[[torch.nn.Module], object]) -> Callable[[torch.nn.Module], object]:
    """Register provider-owned preparation before this owner's native admission."""
    if fn not in _SCOPED_PREPARE_HOOKS:
        _SCOPED_PREPARE_HOOKS.append(fn)
    return fn


def prepare_packed_for(root: torch.nn.Module) -> None:
    """Initialize owned packing at a quiescent boundary, without a global sync."""
    for fn in _SCOPED_PREPARE_HOOKS:
        fn(root)
    release_stranded_segments()


def bump_sync_epoch() -> int:
    """Announce that a weight sync finished: bump the epoch, then run every registered refresher.

    Called from the engine's weight-update paths, which are where Parameter objects get replaced.
    Hook exceptions PROPAGATE on purpose: the refreshers raise rather than move a buffer a captured
    CUDA graph may point at, and swallowing that would leave a replay reading foreign memory.
    """
    global SYNC_EPOCH
    SYNC_EPOCH += 1
    for fn in _SYNC_HOOKS:
        fn()
    # A sync boundary is guaranteed not to be inside a forward, so release fragmentation here.
    release_stranded_segments()
    return SYNC_EPOCH


def refresh_packed_for(root: torch.nn.Module) -> None:
    """Refresh only this owner's packed sites at its weight-update boundary.

    This does not advance ``SYNC_EPOCH``: doing so would license an unrelated
    model's next-forward repack. Scoped hooks explicitly refresh owned sites and
    record their attempts at the current epoch. The shared mutation epoch still
    detects storage replacement across both scoped and legacy consumers.
    """
    for fn in _SCOPED_SYNC_HOOKS:
        fn(root)
    release_stranded_segments()


__all__ = [
    "EPOCH",
    "SYNC_EPOCH",
    "DataWatch",
    "WatchedParamDict",
    "alloc_param_buffer",
    "buffer_signature",
    "bump_epoch",
    "bump_sync_epoch",
    "refresh_packed_for",
    "prepare_packed_for",
    "on_packing_prepare",
    "on_scoped_sync_boundary",
    "on_sync_boundary",
    "release_stranded_segments",
    "set_buffer_allocator",
    "watched_class",
]
