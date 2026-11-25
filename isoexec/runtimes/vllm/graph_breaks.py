"""Eager break points for vLLM's breakable PIECEWISE CUDA graphs (``VLLM_USE_BREAKABLE_CUDAGRAPH=1``).

Under ``cudagraph_mode=FULL_AND_PIECEWISE`` Model Runner V2 captures one breakable graph per padded
token count and replays it for every mixed (prefill-bearing) batch with that count. The captured
segments therefore must not depend on batch composition. Everything that does -- each softmax
attention core (KV-cache write plus the paged kernel, which read slot mappings, block tables,
``query_start_loc``/``seq_lens`` and host ``max_query_len``/``max_seq_len``) and the whole
state-bearing part of each GDN layer (q/k/v concatenation, conv, recurrent core, the decode/prefill
split, slot ids, ``has_initial_state``, host ``cu_seqlens``) -- runs here, eagerly, on every replay,
reading the live ForwardContext. Each break writes into a caller-allocated output whose address the
following graph segment captured. Everything between breaks (embedding, norms, GEMMs, rope over the
static positions buffer, gates, row-parallel reduction, residuals) is a row-wise function of
static-address buffers.

Outside a breakable capture (eager steps, FULL decode capture and replay) the decorated functions
run inline, and the GDN layer keeps its direct path, so FULL_DECODE_ONLY graphs are unchanged.
"""

from __future__ import annotations

import functools
import sys

import torch

#: vLLM's breakable capture lives here; until something imports it, no breakable capture can be active.
_BREAKABLE = "vllm.compilation.breakable_cudagraph"

#: Host counters (evidence only): calls of each break body, including replays.
STATS: dict[str, int] = {"attention": 0, "gdn": 0}


def piecewise_capturing() -> bool:
    """True while a breakable PIECEWISE segment is being recorded on this thread.

    Mirrors ``eager_break_during_capture``'s own test, so a caller that allocates an output for a
    break does so exactly when the decorated call will break.
    """
    breakable = sys.modules.get(_BREAKABLE)
    if breakable is None:
        return False
    capture = breakable.BreakableCUDAGraphCapture.current()
    if capture is None or not capture._capturing:
        return False
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    return not (is_forward_context_available()
                and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL)


def _eager_break(fn):
    """``eager_break_during_capture(fn)``, applied on first call.

    Importing vLLM's breakable capture at module load pulls in ``vllm.config``; deferring it keeps
    this module importable wherever ``_gdn``/``_binding`` are (CPU tests stub parts of vLLM). A
    capture can only be recording once vLLM has imported that module, so until then calls run inline.
    """
    wrapped = None

    @functools.wraps(fn)
    def call(*args, **kwargs):
        nonlocal wrapped
        if wrapped is None:
            breakable = sys.modules.get(_BREAKABLE)
            if breakable is None:  # nothing can be capturing: run inline, as the decorator would
                return fn(*args, **kwargs)
            wrapped = breakable.eager_break_during_capture(fn)
        return wrapped(*args, **kwargs)

    return call


@_eager_break
def attention_core_into(layer: torch.nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        out: torch.Tensor) -> None:
    """KV-cache write and paged attention for ``layer`` into ``out`` (``PagedAttentionBinding.run``)."""
    from .bindings import PagedAttentionBinding

    STATS["attention"] += 1
    PagedAttentionBinding.run(layer, q, k, v, out=out)


@_eager_break
def gdn_core_into(layer: torch.nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  alpha: torch.Tensor, beta: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor,
                  conv_weight: torch.Tensor, out: torch.Tensor) -> None:
    """One GDN layer's state-bearing core (``GDNStateLayer.run_core``) into ``out`` (``[T, Hv, Dv]``).

    ``run_core`` returns a fresh ``[T, Hv, Dv]`` tensor with padding rows zeroed; the copy is bit-exact.
    """
    STATS["gdn"] += 1
    out.copy_(layer.run_core(q, k, v, alpha, beta, A_log, dt_bias, conv_weight=conv_weight))


__all__ = ["STATS", "attention_core_into", "gdn_core_into", "piecewise_capturing"]
