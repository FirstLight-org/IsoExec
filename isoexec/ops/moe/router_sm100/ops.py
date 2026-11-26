"""Python entry points for the SM100 DeepSeek-class routers.

Two routing names, each one numerical identity (DESIGN.md "Numerical program"):

* ``sigmoid_noaux_bias_select_grouped`` -- DeepSeek V3-class (and GLM-5-class, GLM-4.7-Flash at
  ``n_group = topk_group = 1``): sigmoid scores, selection on ``score + bias`` limited to the top
  ``topk_group`` of ``n_group`` groups by top-2-sum, weights from the unbiased scores.
* ``sqrtsoftplus_bias_select`` -- V4-class: ``sqrt(softplus)`` scores, ungrouped selection on
  ``score + bias``, optional shared-as-routed slots and token mask.

Both return ``(indices int32 [T, k+S], weights fp32 [T, k+S])`` with the routed slots in ASCENDING
GLOBAL EXPERT ID -- the table ``moe_sm100``'s dispatch (``ops.route`` / ``ops.dispatch``) and
``moe_fp8_sm100`` consume, with ``routed_scaling_factor`` already applied, exactly where
``moe_sm100.ops.route_mechanics`` applies it.

``route_select_only`` runs the selection over caller-supplied fp32 logits. ``route_fused`` runs the
router GEMM (bf16 operands, tcgen05 fp32 accumulation, see DESIGN.md "The logits program") and the
same selection in one kernel; ``return_logits=True`` also writes the fp32 logits it selected on.
``route_select_tree`` runs the selection over the logits program's fp32 segment roots (the tree folded in
the selection launch), and ``route`` is ``route_fused`` for inference that takes that form (the segments on
DeepGEMM): the same logits program, so the same table. ``route_mega`` is the same program in one launch on the
vendored DeepGEMM MegaGate (the segments in tensor memory, the tree in its epilogue and gate, ``select.cuh`` as the
gate); not yet ``route``'s default.

Schedule (``isoexec/ops/schedule_policy.py``). The logits program is a tensor-core sum over the hidden axis: rule 3
with the reducer's block 16 and no published shortest segment, so ONE segment (``declared_segments``); every path
(fused, segment roots + selection, MegaGate) takes that count. The renormaliser is a sum over the K data-selected
experts: rule 2, one segment per selected expert, a chain in ascending expert id from +0 (``select.cuh``
``selected_sum``).
"""

from __future__ import annotations

import functools

import torch

from isoexec.ops import schedule_policy
from isoexec.ops.moe.router_sm100._extension import extension

#: The logits reducer: one ``tcgen05.mma`` consumes 16 hidden terms (hardware); it publishes no shortest segment
#: (policy_validation/sparse_design/ONEPAGE.md, Inputs: "Tensor-core sums (GEMM, the MoE router, ...) publish
#: nothing and are one segment per shard"; measured cost of one segment against four:
#: policy_validation/moe_megamoe_sm100/ROUND_B.md, Round F, build/r1.log).
LOGITS_REDUCER_BLOCK = 16
LOGITS_REDUCER_SHORTEST = None
#: Rule 2 for the renormaliser (``select.cuh`` ``selected_sum``): one segment per selected expert, chain from +0.
RENORM_SCHEDULE = schedule_policy.selected()

GROUPED = "sigmoid_noaux_bias_select_grouped"
SQRTSOFTPLUS = "sqrtsoftplus_bias_select"
_MODE = {GROUPED: 0, SQRTSOFTPLUS: 1}


def _check(t, *, dtype, ndim, name, device=None):
    if not isinstance(t, torch.Tensor) or t.dtype != dtype or t.dim() != ndim or not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA {dtype} tensor with {ndim} dims")
    if device is not None and t.device != device:
        raise ValueError(f"{name} must be on {device}")


def _args(routing, E, top_k, n_group, topk_group, expert_bias, num_shared, mask, T, device,
          hash_routed=False):
    if routing not in _MODE:
        raise ValueError(f"routing must be one of {sorted(_MODE)}, got {routing!r}")
    if hash_routed:
        raise ValueError("token-id hash routing layers (num_hash_layers / tid2eid) are not served")
    if routing == GROUPED:
        if expert_bias is None:
            raise ValueError(f"{GROUPED} requires the checkpoint's e_score_correction_bias")
        if num_shared or mask is not None:
            raise ValueError(f"{GROUPED} has no shared-as-routed slots and no token mask")
    elif n_group != 1 or topk_group != 1:
        raise ValueError(f"{SQRTSOFTPLUS} is ungrouped")
    if expert_bias is not None:
        _check(expert_bias, dtype=torch.float32, ndim=1, name="expert_bias", device=device)
        if expert_bias.numel() != E:
            raise ValueError("expert_bias must be fp32 [num_experts]")
        expert_bias = expert_bias.contiguous()
    if mask is not None:
        _check(mask, dtype=torch.bool, ndim=1, name="mask", device=device)
        if mask.numel() != T:
            raise ValueError("mask must be bool [T]")
        mask = mask.contiguous()
    return expert_bias, mask


def route_select_only(logits: torch.Tensor, *, routing: str, top_k: int, n_group: int = 1,
                      topk_group: int = 1, expert_bias: torch.Tensor | None = None,
                      routed_scale: float = 1.0, num_shared: int = 0,
                      mask: torch.Tensor | None = None):
    """``(indices int32 [T, k+S], weights fp32 [T, k+S])`` from fp32 ``logits [T, E]``."""
    _check(logits, dtype=torch.float32, ndim=2, name="logits")
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    T, E = logits.shape
    bias, mask = _args(routing, E, top_k, n_group, topk_group, expert_bias, num_shared, mask, T,
                       logits.device)
    ks = top_k + num_shared
    idx = torch.empty((T, ks), dtype=torch.int32, device=logits.device)
    w = torch.empty((T, ks), dtype=torch.float32, device=logits.device)
    extension().select(logits, bias, mask, _MODE[routing], top_k, num_shared, n_group,
                       topk_group, float(routed_scale), idx, w)
    return idx, w


@functools.lru_cache(maxsize=None)
def declared_segments(H: int) -> int:
    """The logits program's segment count: the schedule policy's G for the hidden axis (rule 3; the axis is not
    sharded, d_max = 1). 1 for every H. Cached per H: the policy is evaluated once, never on the per-call path."""
    return schedule_policy.reduce(H, b=LOGITS_REDUCER_BLOCK, min_segment=LOGITS_REDUCER_SHORTEST).G


@functools.lru_cache(None)
def mega_gate_segments() -> int:
    """The segment count compiled into the vendored MegaGate (``layout/mega_gate.cuh`` ``kNumSegments``), read from
    the header the kernels are built from. ``mega_serves`` refuses the one-launch path unless it equals the
    policy's count, so a header that drifts cannot run another schedule."""
    import re
    from pathlib import Path

    header = (Path(__file__).resolve().parents[2] / "_vendor" / "deep_gemm" / "include" / "deep_gemm" / "layout"
              / "mega_gate.cuh").read_text()
    return int(re.search(r"static constexpr uint32_t kNumSegments = (\d+);", header).group(1))


def route_select_tree(parts: torch.Tensor, *, routing: str, top_k: int, n_group: int = 1,
                      topk_group: int = 1, expert_bias: torch.Tensor | None = None,
                      routed_scale: float = 1.0, num_shared: int = 0,
                      mask: torch.Tensor | None = None):
    """``route_select_only`` on logits given as their FP32 segment roots ``parts [S, T, E]`` (S =
    ``declared_segments(H)`` = 1): each logit is the logits program's segment tree of its roots (folded in the
    selection launch), then the same selection."""
    _check(parts, dtype=torch.float32, ndim=3, name="parts")
    parts = parts.contiguous()
    T, E = parts.shape[1], parts.shape[2]
    bias, mask = _args(routing, E, top_k, n_group, topk_group, expert_bias, num_shared, mask, T,
                       parts.device)
    ks = top_k + num_shared
    idx = torch.empty((T, ks), dtype=torch.int32, device=parts.device)
    w = torch.empty((T, ks), dtype=torch.float32, device=parts.device)
    extension().select_tree(parts, bias, mask, _MODE[routing], top_k, num_shared, n_group,
                            topk_group, float(routed_scale), idx, w)
    return idx, w


#: Token rows up to which ``route`` evaluates the logits program as segment GEMMs on DeepGEMM (one CTA per
#: segment and expert tile; ``mm/deepgemm_sm100/segtree``) + the tree-folding selection, instead of the fused
#: cluster kernel. Same logits program (segments, chains, tree), so the same table at every row count. Measured at
#: least as fast as the fused kernel at every row count (isolated: 9.8 vs 10.4 us at 128 rows, 12.5 vs 17.8 at 1024,
#: 18.1 vs 26.9 at 2048; in situ the fused kernel's 12-CTA cluster loses more under concurrent work).
SEGTREE_ROWS = 1 << 30


#: rows where the one-launch ``route_mega`` is the measured default (B200, E 384, H 5120, r4o: 9.4 / 10.0 / 10.0 / 10.0 /
#: 10.6 / 12.1 / 46.7 us at T = 1 / 8 / 32 / 64 / 128 / 512 / 8192 vs the two-launch 9.4 / 10.7 / 10.7 / 10.2 / 10.8 /
#: 12.2 / 55.5; the two-launch path stays at 1024 < T <= 4096: 17.9 vs 18.7 at 2048). Bits are identical either way
#: (``tests/test_route_mega_gpu.py``), so the choice is placement only.
MEGA_SKIP = (1024, 4096)
_MEGA_WARM: set = set()


def _mega_ok(x, router_w, kw) -> bool:
    T, H = x.shape
    if kw.get("return_logits") or not (0 < T) or (MEGA_SKIP[0] < T <= MEGA_SKIP[1]):
        return False
    if not (x.is_cuda and x.dtype == router_w.dtype == torch.bfloat16 and mega_serves(router_w.shape[0], H)):
        return False
    key = (torch.cuda.current_stream(x.device).cuda_stream, T)
    if torch.cuda.is_current_stream_capturing():
        # route_mega's first call per (stream, shape) compiles and creates the stream's barrier workspace: never inside
        # a capture -- an unwarmed shape takes the two-launch path (the same bits)
        return key in _MEGA_WARM
    _MEGA_WARM.add(key)
    return True


def route(x: torch.Tensor, router_w: torch.Tensor, **kw):
    """The router (inference). ``route_mega`` (one launch) where it is the measured default (``MEGA_SKIP``); else up to
    ``SEGTREE_ROWS`` rows the logits program's segments run as DeepGEMM segment GEMMs and the selection folds the tree;
    otherwise (and for ``return_logits``) the fused kernel. Every path gives the same bits."""
    if _mega_ok(x, router_w, kw):
        return route_mega(x, router_w, **kw)
    T, H = x.shape
    segments = declared_segments(H)
    if (0 < T <= SEGTREE_ROWS and not kw.get("return_logits")
            and x.dtype == router_w.dtype == torch.bfloat16 and x.is_cuda):
        from isoexec.ops.mm.deepgemm_sm100 import segtree
        if segtree.serves(router_w.shape[0], H, segments):
            parts = segtree.partials(x.contiguous(), router_w.contiguous(), segments)
            return route_select_tree(parts, **kw)
    return route_fused(x, router_w, **kw)


def route_fused(x: torch.Tensor, router_w: torch.Tensor, *, routing: str, top_k: int,
                n_group: int = 1, topk_group: int = 1, expert_bias: torch.Tensor | None = None,
                routed_scale: float = 1.0, num_shared: int = 0, mask: torch.Tensor | None = None,
                return_logits: bool = False):
    """The router GEMM and the selection in one kernel.

    ``x [T, H]`` and ``router_w [E, H]`` are BF16. The native host selects the
    measured tile and declared segment count.
    """
    _check(x, dtype=torch.bfloat16, ndim=2, name="x")
    _check(router_w, dtype=torch.bfloat16, ndim=2, name="router_w", device=x.device)
    if torch.is_grad_enabled() and (x.requires_grad or router_w.requires_grad):
        raise ValueError("router_sm100 is forward-only")
    if x.stride(1) != 1:
        x = x.contiguous()
    if router_w.stride(1) != 1:
        router_w = router_w.contiguous()
    T, H = x.shape
    E = router_w.size(0)
    if router_w.size(1) != H:
        raise ValueError("x and router_w disagree on the hidden size")
    bias, mask = _args(routing, E, top_k, n_group, topk_group, expert_bias, num_shared, mask, T,
                       x.device)
    ks = top_k + num_shared
    idx = torch.empty((T, ks), dtype=torch.int32, device=x.device)
    w = torch.empty((T, ks), dtype=torch.float32, device=x.device)
    logits = torch.empty((T, E), dtype=torch.float32, device=x.device) if return_logits else None
    extension().fused(x, router_w, bias, mask, _MODE[routing], top_k, num_shared, n_group,
                      topk_group, float(routed_scale), idx, w, logits, declared_segments(H))
    return (idx, w, logits) if return_logits else (idx, w)


def mega_serves(num_experts: int, hidden: int) -> bool:
    """True when ``route_mega`` serves the weight ``[E, H]``: the MegaGate is compiled for the schedule policy's
    segment count (``mega_gate_segments() == declared_segments(H)``), whole 64-wide K blocks in each segment, and
    DeepGEMM MegaGate's expert range (``E <= 512``, ``E % 4 == 0``)."""
    if hidden <= 0:
        return False
    segments = declared_segments(hidden)
    return (mega_gate_segments() == segments and hidden % (64 * segments) == 0
            and 2 <= num_experts <= 512 and num_experts % 4 == 0)


def route_mega(x: torch.Tensor, router_w: torch.Tensor, *, routing: str, top_k: int, n_group: int = 1,
               topk_group: int = 1, expert_bias: torch.Tensor | None = None, routed_scale: float = 1.0,
               num_shared: int = 0, mask: torch.Tensor | None = None, num_split_k: int = 0,
               num_mma_ctas: int = 0):
    """``route(x, router_w, ...)`` in ONE launch (inference, forward only): the vendored DeepGEMM MegaGate with the
    logits program's segment(s) in TMEM accumulators (each one ascending chain from zero; one segment under the
    schedule policy), folded by the segment tree, and this package's selection (``select.cuh``) as its gate, so ids
    and weights equal ``route``'s bit for bit.

    ``num_split_k`` (a divisor of the segment count: 1) and ``num_mma_ctas`` (1 / 2) restrict the launch plan's ownership (0 = the
    heuristic's choice); ownership never moves a bit. The first call per CUDA stream (and per shape) must run
    before CUDA graph capture on that stream (JIT compile and the stream's barrier workspace)."""
    _check(x, dtype=torch.bfloat16, ndim=2, name="x")
    _check(router_w, dtype=torch.bfloat16, ndim=2, name="router_w", device=x.device)
    if torch.is_grad_enabled() and (x.requires_grad or router_w.requires_grad):
        raise ValueError("router_sm100 is forward-only")
    x = x.contiguous()
    router_w = router_w.contiguous()
    T, H = x.shape
    E = router_w.size(0)
    if router_w.size(1) != H:
        raise ValueError("x and router_w disagree on the hidden size")
    if not mega_serves(E, H):
        raise ValueError(f"route_mega serves H % 64 == 0 and E <= 512 with E % 4 == 0, got E = {E}, H = {H}")
    if num_split_k not in (0, 1) or num_mma_ctas not in (0, 1, 2):
        raise ValueError("num_split_k must be 0 or 1 (a split-K slice owns whole segments; there is one) and "
                         "num_mma_ctas 0, 1 or 2")
    replayable = mask is None     # a caller's token mask is not replayable from shapes alone (prebuilt row sweep)
    bias, mask = _args(routing, E, top_k, n_group, topk_group, expert_bias, num_shared, mask, T, x.device)
    ks = top_k + num_shared
    idx = torch.empty((T, ks), dtype=torch.int32, device=x.device)
    w = torch.empty((T, ks), dtype=torch.float32, device=x.device)
    if T:
        from isoexec.ops.mm.deepgemm_sm100 import jit_warm
        from isoexec.ops.mm.deepgemm_sm100 import ops as dgo
        if replayable:
            jit_warm.note("gate", e=int(E), h=int(H), routing=routing, top_k=int(top_k), n_group=int(n_group),
                          topk_group=int(topk_group), routed_scale=float(routed_scale), num_shared=int(num_shared),
                          num_split_k=int(num_split_k), num_mma_ctas=int(num_mma_ctas),
                          bias=None if expert_bias is None else str(expert_bias.dtype).split(".")[-1])
        dgo._launch(lambda g: g.bf16_mega_gate_isoexec(
            x, router_w, bias, mask, idx, w, _MODE[routing], int(top_k), int(num_shared), int(n_group),
            int(topk_group), float(routed_scale), int(num_split_k), int(num_mma_ctas)))
    return idx, w


__all__ = ["GROUPED", "LOGITS_REDUCER_BLOCK", "LOGITS_REDUCER_SHORTEST", "RENORM_SCHEDULE", "SQRTSOFTPLUS",
           "declared_segments", "mega_gate_segments", "mega_serves", "route", "route_fused", "route_mega",
           "route_select_only", "route_select_tree"]


#: the token-segment length of dW's fixed segment tree (moe_fp8_sm100/SPEC.md §8.3's SEG_T): a
#: deployment-fixed constant published per (op, shape) -- here a function of E only, never of T.
#: Measured (README "Backward"): 512 wins at E = 64 (shorter chains, more CTAs), 1024 at E >= 256.
#: The backward runs only in the trainer and is not scheduled by the policy (design 7.4: the policy covers the
#: forward pass). dW's sum over the tokens that selected an expert keeps this kernel's own fixed order:
#: segments of exactly ``seg_t_for(E)`` tokens counted from token 0, the last one short, merged by a tree.
SEG_T = 1024


def seg_t_for(num_experts: int) -> int:
    return 512 if num_experts <= 128 else 1024


def backward_segment_tokens(num_tokens: int, num_experts: int) -> int:
    """Tokens per segment of dW's tree: ``seg_t_for(E)``, whatever the number of tokens."""
    return seg_t_for(num_experts)


def route_backward(dp: torch.Tensor, x: torch.Tensor, router_w: torch.Tensor, logits: torch.Tensor,
                   *, routing: str, top_k: int, n_group: int = 1, topk_group: int = 1,
                   expert_bias: torch.Tensor | None = None, routed_scale: float = 1.0,
                   num_shared: int = 0, mask: torch.Tensor | None = None,
                   seg_t: int | None = None):
    """The router backward from ``dp [T, k]`` (fp32, the gradient of the emitted routed weights in
    slot order = ascending expert id; shared-as-routed slots carry no gradient and are not passed).

    ``logits`` are the fp32 logits the forward selected on (``route_fused(..., return_logits=True)``
    or ``route_select_only``'s input): the backward recomputes the selection from them with the
    forward's own selection core, so nothing else is saved. Returns ``(dx bf16 [T, H], dw fp32 [E, H],
    dl fp32 [T, k], idx int32 [T, k+S])``. No gradient reaches the bias or flows through selection.
    """
    _check(dp, dtype=torch.float32, ndim=2, name="dp")
    _check(logits, dtype=torch.float32, ndim=2, name="logits")
    T, E = logits.shape
    seg_t = backward_segment_tokens(T, E) if seg_t is None else seg_t
    if dp.shape != (T, top_k):
        raise ValueError("dp must be fp32 [T, top_k] in slot order")
    if x.shape[0] != T or router_w.shape != (E, x.shape[1]):
        raise ValueError("x [T, H] and router_w [E, H] must match the logits")
    if seg_t <= 0 or seg_t % 16 or seg_t > 1024:
        raise ValueError("seg_t must be a positive multiple of 16, at most 1024")
    if T:   # the forward's geometry checks, once (refuses what the program cannot serve)
        route_select_only(logits[:1], routing=routing, top_k=top_k, n_group=n_group,
                          topk_group=topk_group, expert_bias=expert_bias,
                          routed_scale=routed_scale, num_shared=num_shared,
                          mask=None if mask is None else mask[:1])
    bias, mask = _args(routing, E, top_k, n_group, topk_group, expert_bias, num_shared, mask, T,
                       logits.device)
    ext = extension()
    dl = torch.empty((T, top_k), dtype=torch.float32, device=logits.device)
    idx = torch.empty((T, top_k + num_shared), dtype=torch.int32, device=logits.device)
    ext.bwd_select(logits.contiguous(), bias, mask, _MODE[routing], top_k, num_shared, n_group,
                   topk_group, float(routed_scale), dp.contiguous(), dl, idx)
    x = x.contiguous()
    w = router_w.contiguous()
    H = x.shape[1]
    dx = torch.empty((T, H), dtype=torch.bfloat16, device=x.device)
    ext.dx(idx, dl, w, dx)
    nseg = max(1, -(-T // seg_t))
    part = torch.empty((nseg, E, H), dtype=torch.float32, device=x.device)
    cnt = torch.empty((nseg, E), dtype=torch.int32, device=x.device)
    dw = torch.empty((E, H), dtype=torch.float32, device=x.device)
    ext.dw(idx, dl, x, int(seg_t), part, cnt, dw)
    return dx, dw, dl, idx


__all__ += ["SEG_T", "backward_segment_tokens", "route_backward", "seg_t_for"]
