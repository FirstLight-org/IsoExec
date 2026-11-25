"""FULL_AND_PIECEWISE on Model Runner V2: where the V4.1 components break vLLM's breakable graphs.

A breakable PIECEWISE graph is captured once per padded token count and replayed for every mixed batch of that
count, so nothing captured may depend on the batch's composition. What does in V4.1: the paged attention step (page
writes, compressor groups, index-score routes and schedules, top-k lists, the dual-rail plan), the CED decoder-row
plan, and the Engram n-gram history. Two break structures (``ISOEXEC_V41_PIECEWISE_BREAKS``, agreed over the TP group):

* ``component`` (default): every component call (embedding, encoder block, decoder entry, decoder block) is ONE
  eager break; the graph segments between them keep only the head stage.
* ``core``: the mature attention-core pattern (``runtimes/vllm/graph_breaks.py``): each encoder row breaks only at
  its attention core (``V41Layer._attention_core``); the decoder entry only at its projection into the pages and the
  step's decoder-row plan (``DecoderEntry._entry_core``); each decoder row, in graphs of at most
  ``DECODER_ALL_ROWS_MAX`` tokens, only at its attention core on the step's decoder rows
  (``DecoderBlock._decoder_core``: row gather, attention, scatter onto the tokens). The mHC boundaries, norms, Engram
  residual, router + experts (EP exchange included) and the decoder rows' non-attention work (on every token) are
  captured. The embedding (with the Engram hash) stays one break; so does each decoder block in larger graphs.

Every break writes into outputs allocated at capture (the addresses the next segment captured) and reruns on every
replay with that step's live forward context, plan and model-state buffers; non-tensor arguments are the capture-time
objects, so a break reads the step's composition only from the forward context / published plan, never from them.
Outside a breakable capture (eager steps, FULL decode capture / replay) components run inline, unchanged.
"""
from __future__ import annotations

import os
import time

import torch
from torch.utils import _pytree as pytree

#: ``ISOEXEC_V41_PIECEWISE_BREAKS``: ``component`` (every V4.1 component one eager break) or ``core`` (attention cores
#: and the Engram hash only). Launch structure only: the same kernels on the same operands.
BREAKS_ENV = "ISOEXEC_V41_PIECEWISE_BREAKS"
BREAK_MODES = ("component", "core")
DEFAULT_MODE = "component"

#: Capture sizes up to which the decoder rows run their non-attention work on every token inside the captured segments
#: (the decoder-row selection then lives in the attention break); larger PIECEWISE graphs keep each decoder block one
#: eager break on its decoder rows (the wasted decoder work on rows below the segment start stays bounded).
DECODER_ALL_ROWS_MAX = 256

#: Host seconds and calls per break body (``impl.__qualname__``), capture and replay alike (evidence: light checks).
BREAK_HOST: dict = {}

_MODE: dict = {}


def requested_mode() -> str:
    v = os.environ.get(BREAKS_ENV, DEFAULT_MODE).strip().lower() or DEFAULT_MODE
    if v not in BREAK_MODES:
        raise ValueError(f"{BREAKS_ENV}={v!r}: expected one of {BREAK_MODES}")
    return v


def agree_mode(group=None) -> str:
    """The break structure of this process, agreed once over ``group`` (collective at TP > 1: every rank calls it at
    the same point, outside capture): ``core`` only when every rank asks for it."""
    if "mode" not in _MODE:
        mine = requested_mode()
        from .block import _group_agrees
        _MODE["mode"] = "core" if _group_agrees(group, mine == "core") else "component"
    return _MODE["mode"]


def mode() -> str:
    return _MODE.get("mode", DEFAULT_MODE)


def capturing() -> bool:
    try:
        from isoexec.runtimes.vllm import graph_breaks
    except ImportError:
        return False
    return graph_breaks.piecewise_capturing()


def core_capturing() -> bool:
    """A breakable PIECEWISE segment is recording and the agreed structure is ``core``."""
    return mode() == "core" and capturing()


def run(module, impl, args, kwargs):
    """``impl(*args, **kwargs)``, as an eager break while a breakable PIECEWISE segment is recorded."""
    if not capturing():
        return impl(*args, **kwargs)
    outs, spec = pytree.tree_flatten(module.abstract_output(*args, **kwargs))
    _component_break(impl, args, kwargs, _inputs(), outs)
    return pytree.tree_unflatten(outs, spec)


def core(impl, args, outs):
    """``impl(*args)`` as one eager break writing ``outs`` (tensors allocated by the caller while recording: the
    addresses the next segment reads). ``impl`` returns a tensor or tuple of tensors of ``outs``' shapes, or None
    (a break that only writes pages / host state)."""
    # vLLM's break convention (``eager_break_during_capture``): the replay holds weak references, so it pins no
    # graph-pool slot across capture sizes; the graphs own the memory
    from vllm.utils.torch_utils import weak_ref_tensor

    def weak(t):
        return weak_ref_tensor(t) if isinstance(t, torch.Tensor) else t
    _component_break(impl, tuple(weak(a) for a in args), {}, _inputs(), [weak(o) for o in outs])
    return outs


def _inputs():
    from isoexec.runtimes.vllm._inputs import current_engine_inputs
    return current_engine_inputs()          # the model state's persistent buffers (Engram lookback): kept


def _run_into(impl, args, kwargs, inputs, outs):
    from isoexec.runtimes.vllm._inputs import engine_inputs
    t0 = time.perf_counter()
    with engine_inputs(inputs or {}):
        got = impl(*args, **kwargs)
    flat = [] if got is None else pytree.tree_flatten(got)[0]
    if len(flat) != len(outs):
        raise RuntimeError(f"eager break output arity {len(flat)} != the abstract output's {len(outs)}")
    for out, value in zip(outs, flat):
        if isinstance(out, torch.Tensor) and isinstance(value, torch.Tensor):
            if out.shape != value.shape or out.dtype != value.dtype:
                raise RuntimeError(f"eager break output {tuple(value.shape)} {value.dtype} != the abstract "
                                   f"{tuple(out.shape)} {out.dtype}")
            if out is not value:
                out.copy_(value)
    key = getattr(impl, "__qualname__", repr(impl))
    rec = BREAK_HOST.get(key)
    if rec is None:
        rec = BREAK_HOST[key] = [0, 0.0]
    rec[0] += 1
    rec[1] += time.perf_counter() - t0


def _component_break(impl, args, kwargs, inputs, outs):
    from isoexec.runtimes.vllm.graph_breaks import _eager_break
    global _BREAK
    if _BREAK is None:
        _BREAK = _eager_break(_run_into)
    _BREAK(impl, args, kwargs, inputs, outs)


_BREAK = None

__all__ = ["BREAKS_ENV", "BREAK_HOST", "DECODER_ALL_ROWS_MAX", "agree_mode", "capturing", "core", "core_capturing",
           "mode", "requested_mode", "run"]
