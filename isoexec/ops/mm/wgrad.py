"""Trainer weight gradients with gradient-accumulation fusion (Megatron's ``main_grad`` contract).

Megatron's DDP gives every trainable parameter an fp32 ``main_grad`` view into its grad buffer and
a backward hook that runs ``main_grad += param.grad`` unless ``param.grad_added_to_main_grad`` is
set. Unfused, a linear layer's weight gradient is one bf16 GEMM output (a full write and read of
``[N, K]`` bf16) followed by that fp32 ``add_`` (a read-modify-write of ``[N, K]`` fp32) per
micro-batch. :func:`weight_grad` instead accumulates the GEMM straight into ``main_grad`` -- on sm90
with ``ops/mm/mm_bwd_sm90`` (an fp32 add in the CUTLASS epilogue, one rounding of ``main_grad +
chain``), elsewhere with cuBLAS ``addmm(out_dtype=fp32, beta=1)`` -- and hands autograd a reusable
dummy gradient, exactly as Megatron's own ``gradient_accumulation_fusion`` does, so the DDP hook
still fires (bucket readiness for overlapped reduction) but skips its add.

Backward only: no forward value, logprob or engine bit depends on any of this. The gradient is
deterministic run to run (one fixed chain per element, no split-K, no atomics). On sm90 every
compiled tile configuration computes the same element program, so a TP shard of the gradient is
bitwise the matching block of the unsharded one.

Eligibility (all must hold, checked per call; otherwise the unfused ``aten.mm`` path runs and the
DDP hook adds as before):

* the weight is a ``Parameter`` that Megatron's DDP has given ``grad_added_to_main_grad`` and an
  fp32 ``main_grad`` of the weight's shape, or a packed operand whose members are adjacent rows of
  one grad bucket (``ops/mm/packed_proj`` records that as :data:`SINK_ATTR`);
* it is not marked :data:`OPT_OUT_ATTR` -- a weight that also receives a gradient from a path that
  does not accumulate into ``main_grad`` (a tied embedding/output weight) must opt out, because the
  hook skips ``param.grad`` once the flag is set (Megatron's contract, Megatron's caveat);
* ``ISOEXEC_WGRAD_FUSION`` is not ``0``.

The flag is re-armed per backward: once a parameter's grad accumulator has run, a node post-hook
registered after DDP's resets ``grad_added_to_main_grad``, so a later unfused contribution in a later
backward (an alias that fell back to the ``cat`` path, say) is never dropped by a stale flag.

An unfused contribution in the SAME backward (the weight also reaches a plain ``F.linear`` / ``aten::mm``
elsewhere -- MTP sharing the head, a LoRA path, a future site) is summed by autograd with the
placeholder before the accumulator sees it. The placeholder is all zeros, so that sum is exactly the
unfused contribution; a hook on the parameter sees a gradient that is not the placeholder and clears
the flag, so DDP's hook adds it to ``main_grad`` as it would unfused. Nothing is dropped, and a
placeholder that reaches ``main_grad`` some other way (a DDP hook ordered after the re-arm, Megatron's
``zero_out_wgrad`` add) adds zeros.
"""

from __future__ import annotations

import os

import torch

__all__ = ["COUNTS", "EXPERT_PARAMS_ATTR", "OPT_OUT_ATTR", "SINK_ATTR", "enabled", "expert_sink",
           "expert_weight_grads", "fusable", "install_column_assembler", "is_placeholder", "weight_grad"]

#: On a stacked expert operand (``[E, rows, K]``): per expert, the parameters whose rows it stacks, in order.
EXPERT_PARAMS_ATTR = "_isoexec_expert_params"

#: On a parameter: never fuse its weight gradient (it also receives an unfused gradient).
OPT_OUT_ATTR = "_isoexec_no_wgrad_fusion"
#: On a packed operand tensor: ``(main_grad_view, member_params)`` -- the members' fp32 main_grads
#: are adjacent rows of one buffer, so the packed gradient accumulates there in one GEMM.
SINK_ATTR = "_isoexec_wgrad_sink"

_ARMED_ATTR = "_isoexec_wgrad_rearm"
_DUMMIES: dict = {}
_DUMMY_STORAGES: set = set()  # storage base of every cached placeholder (what a fused-only gradient views)
_GRAD_ACCS: list = []  # keep the accumulators our re-arm hooks hang on alive, as DDP does
#: Same-backward unfused contributions to a fused weight that the parameter hook handed to DDP's add.
COUNTS = {"unfused_in_same_backward": 0}


def enabled() -> bool:
    return os.environ.get("ISOEXEC_WGRAD_FUSION", "1") != "0"


def _param_sink(w: torch.Tensor):
    if not isinstance(w, torch.nn.Parameter) or getattr(w, OPT_OUT_ATTR, False):
        return None
    if not hasattr(w, "grad_added_to_main_grad"):
        return None
    mg = getattr(w, "main_grad", None)
    if mg is None or mg.dtype is not torch.float32 or mg.shape != w.shape or not mg.is_cuda:
        return None
    return mg, (w,)


def fusable(w: torch.Tensor):
    """``(main_grad, params)`` when ``w``'s gradient may be accumulated into ``main_grad``, else ``None``.

    Called in the GEMM's ``autograd.Function.forward`` (graph construction), so the re-arm hooks are
    registered before the backward that needs them. Invariant: the gradient autograd hands a fused
    parameter's accumulator is the zeros placeholder plus whatever unfused contributions the same backward
    made; the parameter hook (see :func:`_arm`) gives those to DDP's add, so none is dropped.
    """
    if not enabled():
        return None
    sink = getattr(w, SINK_ATTR, None)
    if sink is None:
        sink = _param_sink(w)
    if sink is None:
        return None
    for p in sink[1]:
        _arm(p)
    return sink


def _arm(p: torch.nn.Parameter) -> None:
    """Register (once) a node post-hook on ``p``'s grad accumulator that clears the fusion flag.

    Node post-hooks run in registration order, and DDP registered its hook at construction, so this
    one runs after DDP has consumed the flag for this backward.
    """
    if getattr(p, _ARMED_ATTR, False):
        return
    with torch.enable_grad():  # callers run inside autograd.Function.forward, where grad mode is off
        acc = p.expand_as(p).grad_fn.next_functions[0][0]

    def rearm(*_unused, p=p):
        p.grad_added_to_main_grad = False

    def unfused_contribution(grad, p=p):
        # Runs on the summed gradient before the accumulator (and so before DDP's hook). A fused-only
        # gradient is a placeholder view; anything else carries an unfused contribution of this backward
        # (zeros + g == g), which DDP must add rather than skip.
        if getattr(p, "grad_added_to_main_grad", False) and not is_placeholder(grad):
            p.grad_added_to_main_grad = False
            COUNTS["unfused_in_same_backward"] += 1
        return None

    p.register_hook(unfused_contribution)
    acc.register_hook(rearm)
    _GRAD_ACCS.append(acc)
    setattr(p, _ARMED_ATTR, True)


def _dummy(shape, dtype, device) -> torch.Tensor:
    """A reusable all-zeros gradient placeholder; ``detach()`` gives autograd a fresh tensor it can steal (no
    copy). Zeros, so the placeholder is harmless wherever it lands (summed with an unfused contribution, or
    added to ``main_grad`` by a hook that ignores the flag). Autograd never writes into it: it accumulates in
    place only into a tensor whose storage it holds alone, and the cache keeps a second reference."""
    key = (tuple(shape), dtype, device)
    buf = _DUMMIES.get(key)
    if buf is None:
        buf = _DUMMIES[key] = torch.zeros(shape, dtype=dtype, device=device)
        _DUMMY_STORAGES.add(buf.untyped_storage().data_ptr())
    return buf.detach()


def is_placeholder(grad) -> bool:
    """Whether ``grad`` is (a view of) a cached placeholder, i.e. carries no unfused contribution."""
    return grad is not None and grad.untyped_storage().data_ptr() in _DUMMY_STORAGES


def _accumulate(dy: torch.Tensor, x: torch.Tensor, main_grad: torch.Tensor) -> None:
    from isoexec.ops.mm.mm_bwd_sm90 import ops as sm90

    if sm90.available() and sm90.supported(dy, x, main_grad):
        sm90.accumulate(dy, x, main_grad)
        return
    # cuBLAS beta=1 into fp32 (deterministic; same contract, different chain than sm90's).
    torch.ops.aten.addmm.dtype_out(main_grad, dy.t(), x, torch.float32, out=main_grad)


def weight_grad(dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, sink=None) -> torch.Tensor:
    """The gradient autograd should receive for ``w`` in ``y = x @ w.T`` (``dy[T, N]``, ``x[T, K]``).

    With a ``sink`` from :func:`fusable`, accumulate ``dy^T @ x`` into its fp32 ``main_grad``, mark
    its parameters' ``grad_added_to_main_grad`` and return a dummy; otherwise the unfused bf16
    ``aten.mm`` gradient.
    """
    if sink is not None and enabled():
        main_grad, params = sink
        if main_grad.shape == (dy.shape[1], x.shape[1]):
            _accumulate(dy, x, main_grad)
            for p in params:
                p.grad_added_to_main_grad = True
            return _dummy(w.shape, w.dtype, w.device)
    return torch.ops.aten.mm.out(dy.t(), x, out=torch.empty(dy.shape[1], x.shape[1], dtype=dy.dtype, device=dy.device))


def _assemble_columns(grads):
    from isoexec.ops.mm.mm_bwd_sm90 import ops as sm90

    if not sm90.assemble_supported(grads):
        return None
    return sm90.assemble_columns(grads)


def install_column_assembler() -> bool:
    """Route ``core/linear.split_columns``'s backward through the one-launch sm90 assembly kernel.

    Backward only, and the same per-element program as the eager loop it replaces (``g + 0``, one
    bf16 round, ``-0 -> +0``), so gradients are bit-identical. Returns whether it was installed.
    """
    from isoexec.core import linear
    from isoexec.ops.mm.mm_bwd_sm90 import ops as sm90

    if not sm90.available():
        return False
    linear.set_column_assembler(_assemble_columns)
    return True


# ------------------------------------------------------------------------------------- MoE experts
def expert_sink(stacked: torch.Tensor):
    """Per expert, the ``(param, main_grad)`` pairs of a stacked expert operand when every one of them
    may be accumulated into directly (sm90 only: the grouped kernel), else ``None``. Call in forward."""
    if not enabled():
        return None
    experts = getattr(stacked, EXPERT_PARAMS_ATTR, None)
    if not experts:
        return None
    from isoexec.ops.mm.mm_bwd_sm90 import ops as sm90

    if not stacked.is_cuda or not sm90.available():
        return None
    out = []
    for params in experts:
        pairs = []
        for p in params:
            sink = _param_sink(p)
            if sink is None:
                return None
            pairs.append((p, sink[0]))
        out.append(tuple(pairs))
    for pairs in out:
        for p, _ in pairs:
            _arm(p)
    return tuple(out)


def expert_weight_grads(dy: torch.Tensor, b: torch.Tensor, ends: torch.Tensor, stacked: torch.Tensor, sink):
    """The gradient for a stacked expert operand ``stacked[E, rows, K]`` of a grouped GEMM whose expert e
    computed ``y_e = b_e @ stacked[e].T`` over token rows ``ends[e-1]:ends[e]``.

    With a ``sink`` from :func:`expert_sink`, accumulate every expert's (and every member's) block of
    ``dy^T @ b`` into its fp32 ``main_grad`` in one grouped launch, mark the parameters and return a
    dummy; otherwise ``None`` (the caller runs its own grouped GEMM).
    """
    from isoexec.ops.mm.mm_bwd_sm90 import ops as sm90

    if sink is None or not enabled() or not sm90.grouped_supported(dy, b):
        return None
    groups = []
    for e, pairs in enumerate(sink):
        col = 0
        for _p, mg in pairs:
            groups.append((e, col, mg))
            col += mg.shape[0]
    try:
        sm90.grouped_accumulate(dy, b, ends, groups)
    except sm90.UnsupportedShape:
        return None
    for pairs in sink:
        for p, _ in pairs:
            p.grad_added_to_main_grad = True
    return _dummy(stacked.shape, stacked.dtype, stacked.device)
