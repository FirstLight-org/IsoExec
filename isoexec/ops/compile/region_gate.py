"""Compile a region under a pinned Inductor/Dynamo configuration and serve it only at its admitted shape.
PinnedCompiledRegion preserves the admitted compiler configuration and shape key. Admission itself is a
ledger verdict (``bwd_compile``) written by an offline battery."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from isoexec.ops.compile.cfgpin import make_config_pin

# Pin inductor numerics for region compilation.
REGION_INDUCTOR_CONFIG: dict[str, Any] = {
    # Eager numerics.
    "emulate_precision_casts": True,  # per-op rounding preserved; fp fusion off; toolkit libdevice
    "eager_numerics.division_rounding": True,  # Triton's div.full vs eager's div_rn
    # No graph-rewriting passes.
    "use_pre_grad_passes": False,
    "use_joint_graph_passes": False,
    "use_post_grad_passes": False,
    # Determinism of codegen choices.
    "deterministic": True,  # stable codegen choices
    "fx_graph_cache": False,  # artifact identity comes from the ledger, not a mutable cache
    "fx_graph_remote_cache": False,
}


def shape_key_of(example_inputs: Sequence) -> str:
    import torch

    parts = []
    for t in example_inputs:
        if isinstance(t, torch.Tensor):
            parts.append(f"{tuple(t.shape)}:{t.dtype}:{tuple(t.stride())}")
        else:
            parts.append(repr(t))
    return ";".join(parts)


def _trace_aten(fn: Callable, example_inputs: Sequence):
    """Trace to an aten-level graph with fake tensors, so no device work happens at trace time."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    fakes = [fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in example_inputs]
    with fake_mode:
        return make_fx(fn, tracing_mode="real", _allow_non_fake_inputs=True)(*fakes)


# Dynamo counts recompiles by code object: fresh_frame isolates artifacts.
# This raised cap covers GraphModule targets without cloneable code objects; ledger size bounds compilation.
PINNED_DYNAMO_CONFIG: dict[str, Any] = {
    "recompile_limit": 4096,
    "accumulated_recompile_limit": 65536,
}


def pinned_dynamo_patch():
    """Context manager raising :data:`PINNED_DYNAMO_CONFIG` for a pinned compile.

    Returns a no-op context on any torch that does not expose these knobs, so the caller never
    has to branch (older spellings were ``cache_size_limit`` / ``accumulated_cache_size_limit``;
    both are patched when present).
    """
    import contextlib

    import torch._dynamo.config as dcfg

    keys = dict(PINNED_DYNAMO_CONFIG)
    # legacy spellings, patched too when this torch still carries them
    legacy = (
        ("recompile_limit", "cache_size_limit"),
        ("accumulated_recompile_limit", "accumulated_cache_size_limit"),
    )
    for new, old in legacy:
        if hasattr(dcfg, old):
            keys[old] = keys[new]
    keys = {k: v for k, v in keys.items() if hasattr(dcfg, k)}
    if not keys:
        return contextlib.nullcontext()
    return dcfg.patch(keys)


def fresh_frame(fn: Callable) -> Callable:
    """A callable with ``fn``'s exact bytecode, globals and closure but a distinct code object.

    This is what makes "one artifact per shape key" true at dynamo's granularity; see
    :data:`PINNED_DYNAMO_CONFIG`. Anything that is not a plain Python function is returned unchanged.
    """
    import types

    if not isinstance(fn, types.FunctionType):
        return fn
    try:
        clone = types.FunctionType(fn.__code__.replace(), fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
        clone.__kwdefaults__ = fn.__kwdefaults__
        clone.__qualname__ = fn.__qualname__
        clone.__dict__.update(fn.__dict__)
        return clone
    except Exception:  # noqa: BLE001 -- never let the clone break a compile
        return fn


def _dynamo_compile_count() -> int:
    """Frames dynamo has compiled successfully so far, or -1 if unreadable.

    This is exact only because :func:`fresh_frame` guarantees the compile cannot be served from an
    existing cache entry: with a shared code object a cache hit and a permanently-skipped frame are
    indistinguishable, since neither moves these counters.
    """
    try:
        from torch._dynamo.utils import counters

        return int(counters["frames"]["ok"])
    except Exception:  # noqa: BLE001 -- a probe must never break a compile
        return -1


def _is_dynamo_product(fn: Callable) -> bool:
    """True iff ``fn`` came out of ``torch.compile``; the engagement probe only speaks about dynamo,
    so a stubbed or monkeypatched compile is never called disengaged on its evidence."""
    return hasattr(fn, "_torchdynamo_orig_callable") or hasattr(fn, "_torchdynamo_inline")


class RegionShapeMismatch(RuntimeError):
    """A pinned artifact was invoked at a shape it was never admitted for."""


class PinnedCompiledRegion:
    """The only compiled-region artifact the gate hands out.

    Every ``__call__`` runs under the admitted inductor config patch, so a dynamo recompile
    regenerates code under the same numerics rather than ambient config; inputs whose shape key
    differs from the admitted one raise :class:`RegionShapeMismatch`, since one artifact serves one
    shape. The patch re-enters per call from Python, but under CUDA-graph replay the call happens
    once at capture.
    """

    __slots__ = ("_fn", "_cfg", "_pin", "shape_key", "site", "engaged")

    def __init__(self, fn: Callable, cfg: dict[str, Any], shape_key: str, site: str = "", engaged: bool = True):
        self._fn = fn
        self._cfg = dict(cfg)
        # Built once, entered per call -- see cfgpin.py. The per-call re-entry, and the
        # recompile guarantee above, are unchanged; only the rebuild of the patch object is removed.
        self._pin = make_config_pin(self._cfg)
        self.shape_key = shape_key
        self.site = site
        #: False when dynamo did not compile the region on the warm call: the artifact is then
        #: semantically eager and a caller must not report it as COMPILED.
        self.engaged = engaged

    def __call__(self, *args):
        key = shape_key_of(args)
        if key != self.shape_key:
            raise RegionShapeMismatch(
                f"pinned region {self.site or '<region>'} admitted at {self.shape_key} was "
                f"invoked at {key}; a recompile outside the admitted envelope is refused, not "
                f"served (defect 2)"
            )
        revert = self._pin()
        try:
            return self._fn(*args)
        finally:
            revert()


def compile_pinned(
    fn: Callable,
    example_inputs: Sequence,
    *,
    site: str = "",
    config: dict[str, Any] | None = None,
    decomps: dict | None = None,
) -> PinnedCompiledRegion:
    """Compile a fresh frame under pinned configuration and wrap the admitted artifact.
    Decomposition repairs remain on the graph target across recompiles; callers must already hold a ledger verdict.
    An unengaged wrapper produced no compiled frame and must be reported as eager."""
    import torch._inductor.config as icfg

    from isoexec.ops.compile.decomp import compile_rewritten

    cfg = dict(REGION_INDUCTOR_CONFIG if config is None else config)
    target = fresh_frame(fn) if not decomps else fn
    with icfg.patch(cfg), pinned_dynamo_patch():
        compiled, _ = compile_rewritten(target, example_inputs, decomps, dynamic=False)
        before = _dynamo_compile_count()
        compiled(*example_inputs)  # warm inside the pin: the first real compile happens here
        after = _dynamo_compile_count()
    # Only dynamo products are judged on dynamo's counter; a stubbed compile is left to the caller's
    # self-check.
    engaged = before < 0 or not _is_dynamo_product(compiled) or after > before
    return PinnedCompiledRegion(compiled, cfg, shape_key_of(example_inputs), site=site, engaged=engaged)
