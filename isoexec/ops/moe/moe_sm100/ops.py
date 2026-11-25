"""SM100 forward API: the same stage seams ``moe_sm90/ops.py`` exposes, on Blackwell."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

from isoexec.ops import schedule_policy

from ._extension import extension

# ------------------------------------------------------------------------- schedule ---
# Every reduction of this operator and the rule of ``isoexec/ops/schedule_policy.py`` that orders it.
# The operator publishes no constant (policy_validation/sparse_design/ONEPAGE.md, Inputs: "Tensor-core
# sums (GEMM, the MoE router, ...) publish nothing and are one segment per shard").
#
#   route_score      one chain over the hidden axis per (token, expert)        rule 3, G = 1
#   fc1, fc1_glu     one chain over the hidden axis per output                 rule 3, G = 1
#   fc2              ``leaves`` = d_max shards of the intermediate axis, one    rule 3, G = 1 per shard,
#                    segment each, balanced tree over the shards               tree over the d_max shards
#   route_mechanics  the selected-set sum: one segment per selected expert,    rule 2
#                    chain in ascending expert id from +0
#   combine          the same chain over a token's selected experts' rows      rule 2

#: The tensor-core reducer's exact block: one ``tcgen05.mma`` consumes 16 K terms (hardware; the same
#: reducer as ``ops/mm/mm_sm100/segments.py``). ``SEGMENT_MULTIPLE`` below is the kernel's K granularity
#: (a segment is padded with zero terms up to it), not a schedule input.
REDUCER_BLOCK = 16
#: The reducer's published shortest segment: none, so rule 3 gives one segment per shard.
REDUCER_SHORTEST = None
#: Rule 2, for the selected-set sum and the combine (``csrc/route.cu`` ``selected_sum`` / ``combine_kernel``).
SELECTED = schedule_policy.selected()


@functools.lru_cache(maxsize=None)
def projection_schedule(K: int, d_max: int = 1) -> schedule_policy.Segments:
    """Rule 3 for one shard of a projection's K reduction. ``d_max`` is 1 for fc1 (the hidden axis is
    not sharded) and the declared leaf count for fc2 (``Placement.leaves`` / ``ISOEXEC_PIK_LEAVES``, the
    largest expert-tensor-parallel degree); a shard shorter than whole blocks is zero-padded capacity."""
    if d_max < 1 or d_max & (d_max - 1) or K < 1 or K % d_max:
        raise ValueError(f"K = {K} is not {d_max} equal shards (d_max must be a power of two)")
    return schedule_policy.reduce(K // d_max, b=REDUCER_BLOCK, L=REDUCER_SHORTEST)


@functools.lru_cache(maxsize=None)
def fc2_leaves(K: int, d_max: int) -> int:
    """Leaves of fc2's fixed tree: ``d_max`` shards of the policy's G segments. Cached per (K, d_max): the policy
    is evaluated once per shape, never on the per-call path."""
    return d_max * projection_schedule(K, d_max).G

#: The row-block granularity of the tile plan. Tiling in M is not numerical -- it is the
#: cta_group::1 UMMA's M -- but the plan and the kernel must agree on it.
BLOCK_M = 128

#: Legal leaf counts (declared d_max of the intermediate axis), as on SM90.
LEAVES = (1, 2, 4, 8)

#: Every K segment must be a multiple of this: one 128-byte TMA swizzle atom of 16-bit
#: operands. SM90 admits multiples of 8; the segment sizes the served models use (I/8 for
#: I in {512, 1536}) satisfy both, so this is a legality bound, not an arithmetic change.
SEGMENT_MULTIPLE = 64


def _check(t, *, dtype=None, device=None, ndim=None, contiguous=True, name="tensor"):
    if not isinstance(t, torch.Tensor) or not t.is_cuda:
        raise ValueError(f"moe_sm100 requires CUDA tensors ({name})")
    if torch.cuda.get_device_capability(t.device) != (10, 0):
        raise ValueError("moe_sm100 requires an SM100 GPU")
    if dtype is not None and t.dtype not in (dtype if isinstance(dtype, tuple) else (dtype,)):
        raise ValueError(f"{name}: expected dtype {dtype}, got {t.dtype}")
    if device is not None and t.device != device:
        raise ValueError("all operands must be on the same device")
    if ndim is not None and t.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim} dimensions, got {t.ndim}")
    if contiguous and not t.is_contiguous():
        raise ValueError(f"{name}: expected a contiguous tensor")
    if torch.is_grad_enabled() and t.requires_grad:
        raise ValueError("moe_sm100 is forward-only; use torch.no_grad() for inference")


@dataclass(frozen=True)
class GroupedPlan:
    """Device-resident expert tiles. Graph-capturable: no value is read on the host."""

    offsets: torch.Tensor
    tiles: torch.Tensor
    total: torch.Tensor
    rows: int

    @property
    def experts(self) -> int:
        return self.offsets.numel() - 1


@dataclass(frozen=True)
class RoutePlan:
    grouped: GroupedPlan
    indices: torch.Tensor
    inverse: torch.Tensor
    restore: torch.Tensor
    #: ``[T, k]`` rows in canonical slot order; what ``combine`` folds.
    combine_rows: torch.Tensor


def _tile_bound(rows: int, experts: int) -> int:
    return min(rows, (rows + BLOCK_M - 1) // BLOCK_M + experts - 1) if rows else 1


# --------------------------------------------------------------------------- router ---


def route_score(x: torch.Tensor, router_w: torch.Tensor, *, out=None) -> torch.Tensor:
    """fp32 router logits ``[T, E]``: one ascending chain over the hidden axis per pair."""
    _check(x, dtype=(torch.bfloat16, torch.float16), ndim=2, contiguous=False, name="x")
    _check(router_w, dtype=x.dtype, device=x.device, ndim=2, contiguous=False, name="router_w")
    if x.size(1) != router_w.size(1) or x.stride(1) != 1 or router_w.stride(1) != 1:
        raise ValueError("router weight must share the hidden axis and be row-contiguous")
    if out is None:
        out = torch.empty((x.size(0), router_w.size(0)), dtype=torch.float32, device=x.device)
    else:
        _check(out, dtype=torch.float32, device=x.device, ndim=2, name="out")
        if out.shape != (x.size(0), router_w.size(0)):
            raise ValueError("incorrect logits shape")
    extension().score(x, router_w, out)
    return out


def route_mechanics(
    logits: torch.Tensor,
    top_k: int,
    *,
    expert_bias: torch.Tensor | None = None,
    routed_scale: float = 1.0,
):
    """``(indices int32 [T,k], probs fp32 [T,k])`` in selection order."""
    _check(logits, dtype=torch.float32, ndim=2, name="logits")
    E = logits.size(1)
    if not 1 <= top_k <= min(32, E) or top_k & (top_k - 1):
        raise ValueError(f"top_k must be a power of two in [1, min(32, E)], got {top_k}")
    if expert_bias is not None:
        _check(expert_bias, dtype=torch.float32, device=logits.device, ndim=1, name="expert_bias")
        if expert_bias.numel() != E:
            raise ValueError("expert correction bias must be fp32 [num_experts]")
    elif routed_scale != 1.0:
        raise ValueError("softmax_renorm does not carry a routed scale")
    indices = torch.empty((logits.size(0), top_k), dtype=torch.int32, device=logits.device)
    probs = torch.empty((logits.size(0), top_k), dtype=torch.float32, device=logits.device)
    extension().mechanics(logits, expert_bias, indices, probs, float(routed_scale))
    return indices, probs


# ------------------------------------------------------------------------- dispatch ---


def make_plan(offsets: torch.Tensor, rows: int) -> GroupedPlan:
    """Build expert tiles from CUDA int32 ``[E+1]`` exclusive offsets."""
    _check(offsets, dtype=torch.int32, ndim=1, name="offsets")
    if not isinstance(rows, int) or not 2 <= offsets.numel() < 2**31 - 256 or not 0 <= rows < 2**31 - 256:
        raise ValueError("require int32 expert offsets and 0 <= rows < 2**31-256")
    experts = offsets.numel() - 1
    bound = _tile_bound(rows, experts)
    tiles = torch.empty((bound, 3), device=offsets.device, dtype=torch.int32)
    total = torch.empty((), device=offsets.device, dtype=torch.int32)
    extension().tiles(offsets, tiles, total, rows, BLOCK_M)
    return GroupedPlan(offsets, tiles, total, rows)


def route(indices: torch.Tensor, num_experts: int) -> RoutePlan:
    """CUDA int32 ``[T, topk]`` expert ids -> padding-free expert groups with exact ranks."""
    _check(indices, dtype=torch.int32, ndim=2, name="indices")
    if (
        not isinstance(num_experts, int)
        or not 0 < num_experts < 2**31 - 256
        or not 1 <= indices.size(1) <= 32
        or indices.numel() >= 2**31 - 256
    ):
        raise ValueError("require int32 E > 0, topk in [1,32] and fewer than 2**31-256 routes")
    device, routes = indices.device, indices.numel()
    blocks = max(1, (routes + 1023) // 1024)
    counts = torch.empty(num_experts, device=device, dtype=torch.int32)
    offsets = torch.empty(num_experts + 1, device=device, dtype=torch.int32)
    block_hist = torch.empty((blocks, num_experts), device=device, dtype=torch.int32)
    inverse = torch.empty(routes, device=device, dtype=torch.int32)
    restore = torch.empty_like(indices)
    combine_rows = torch.empty_like(indices)
    tiles = torch.empty((_tile_bound(routes, num_experts), 3), device=device, dtype=torch.int32)
    total = torch.empty((), device=device, dtype=torch.int32)
    extension().plan(
        indices, counts, offsets, block_hist, inverse, restore, combine_rows, tiles, total, BLOCK_M
    )
    return RoutePlan(GroupedPlan(offsets, tiles, total, routes), indices, inverse, restore, combine_rows)


def dispatch(x: torch.Tensor, probs: torch.Tensor, plan: RoutePlan):
    """Gather token activations and route probabilities into expert-major rows."""
    if not isinstance(plan, RoutePlan):
        raise TypeError("expected a RoutePlan from route")
    _check(x, dtype=(torch.bfloat16, torch.float16), ndim=2, name="x")
    _check(probs, dtype=torch.float32, device=x.device, ndim=2, name="probs")
    if probs.shape != plan.indices.shape or x.size(0) != probs.size(0):
        raise ValueError("incompatible token/probability shapes")
    if x.size(1) % 8 or x.data_ptr() % 16:
        raise ValueError("dispatch needs H divisible by 8 and 16-byte aligned activations")
    out = torch.empty((plan.grouped.rows, x.size(1)), device=x.device, dtype=x.dtype)
    op = torch.empty(plan.grouped.rows, device=x.device, dtype=torch.float32)
    extension().gather(x, probs, plan.inverse, out, op, probs.size(1))
    return out, op


# -------------------------------------------------------------------------- experts ---


def _check_gemm(x, w, plan, *, glu, k_multiple=SEGMENT_MULTIPLE):
    _check(x, dtype=(torch.bfloat16, torch.float16), ndim=2, contiguous=False, name="x")
    _check(w, dtype=x.dtype, device=x.device, ndim=3, contiguous=False, name="weight")
    if not isinstance(plan, GroupedPlan):
        raise TypeError("expected a GroupedPlan from make_plan or route")
    if x.stride(1) != 1 or w.stride(2) != 1:
        raise ValueError("the reduced axis must be contiguous")
    if any(s <= 0 or s % 8 for s in (x.stride(0), w.stride(1), w.stride(0))):
        raise ValueError("row and expert strides must be positive multiples of 8")
    if x.data_ptr() % 16 or w.data_ptr() % 16:
        raise ValueError("operands must be 16-byte aligned")
    if x.size(1) != w.size(2) or w.size(0) != plan.experts or x.size(0) != plan.rows:
        raise ValueError("incompatible grouped GEMM dimensions")
    if x.size(1) % k_multiple:
        raise ValueError(f"K must be a multiple of {k_multiple}")
    if glu and w.size(1) % 2:
        raise ValueError("gate_up must have an even output width")
    for t in (plan.offsets, plan.tiles, plan.total):
        _check(t, dtype=torch.int32, device=x.device, name="plan")
    if plan.tiles.shape != (_tile_bound(plan.rows, plan.experts), 3) or plan.total.numel() != 1:
        raise ValueError("malformed grouped plan")


def fc1(x, weight, plan: GroupedPlan, *, out=None):
    """Grouped ``x @ weight[e].T`` -> operand dtype ``[R, N]``, one K chain, rounded once."""
    _check_gemm(x, weight, plan, glu=False)
    n = weight.size(1)
    if out is None:
        out = torch.empty((x.size(0), n), dtype=x.dtype, device=x.device)
    else:
        _check(out, dtype=x.dtype, device=x.device, ndim=2, name="out")
        if out.shape != (x.size(0), n):
            raise ValueError("incorrect output shape")
    extension().fc1(x, weight, plan.tiles, plan.total, out)
    return out


def activation(projection, probs=None, *, out=None):
    """The FC1 fused epilogue on its own, over an already-rounded ``[R, 2I]`` projection."""
    _check(projection, dtype=(torch.bfloat16, torch.float16), ndim=2, name="projection")
    if projection.size(1) % 2:
        raise ValueError("the projection must hold gate and up side by side")
    n = projection.size(1) // 2
    if probs is None:
        probs = torch.empty(0, dtype=torch.float32, device=projection.device)
    else:
        _check(probs, dtype=torch.float32, device=projection.device, ndim=1, name="probs")
        if probs.numel() != projection.size(0):
            raise ValueError("probs must have one value per row")
    if out is None:
        out = torch.empty((projection.size(0), n), dtype=projection.dtype, device=projection.device)
    else:
        _check(out, dtype=projection.dtype, device=projection.device, ndim=2, name="out")
    extension().activation(projection, probs, out)
    return out


def fc1_glu(x, gate_up, plan: GroupedPlan, probs=None, *, out=None):
    """Grouped FC1 with the separately rounded SiLU, up product and probability product."""
    _check_gemm(x, gate_up, plan, glu=True)
    n = gate_up.size(1) // 2
    if probs is None:
        probs = torch.empty(0, dtype=torch.float32, device=x.device)
    else:
        _check(probs, dtype=torch.float32, device=x.device, ndim=1, name="probs")
        if probs.numel() != x.size(0):
            raise ValueError("probs must have one value per grouped row")
    if out is None:
        out = torch.empty((x.size(0), n), dtype=x.dtype, device=x.device)
    else:
        _check(out, dtype=x.dtype, device=x.device, ndim=2, name="out")
        if out.shape != (x.size(0), n):
            raise ValueError("incorrect output shape")
    extension().fc1_glu(x, gate_up, plan.tiles, plan.total, probs, out)
    return out


def fc2(x, down, plan: GroupedPlan, *, leaves=8, out=None):
    """Grouped FC2: L fixed K segments, each rounded, merged by the balanced tree -> fp32."""
    _check_gemm(x, down, plan, glu=False, k_multiple=8)
    if leaves not in LEAVES:
        raise ValueError(f"leaves must be one of {LEAVES}")
    if x.size(1) % (8 * leaves):
        raise ValueError("each logical K segment must be a positive multiple of 8")
    if fc2_leaves(x.size(1), leaves) != leaves:
        raise ValueError("the schedule policy gives one segment per shard; leaves must be the declared d_max")
    segment = x.size(1) // leaves
    # One leaf: the kernel reads K's tail with zero-filled TMA boxes (the bits of an explicit zero pad), so
    # nothing is copied per call. Several leaves: each declared leaf boundary stays a whole-atom boundary.
    padding = -segment % SEGMENT_MULTIPLE if leaves > 1 else 0
    if padding:
        # Preserve the declared leaf boundaries. Padding the entire contraction
        # would move values between leaves and change their BF16 roundings.
        x = torch.nn.functional.pad(x.reshape(x.size(0), leaves, segment), (0, padding)).flatten(1)
        down = torch.nn.functional.pad(down.reshape(*down.shape[:2], leaves, segment), (0, padding)).flatten(2)
    n = down.size(1)
    if out is None:
        out = torch.empty((x.size(0), n), dtype=torch.float32, device=x.device)
    else:
        _check(out, dtype=torch.float32, device=x.device, ndim=2, name="out")
        if out.shape != (x.size(0), n):
            raise ValueError("incorrect output shape")
    extension().fc2(x, down, plan.tiles, plan.total, out, leaves)
    return out


def experts(x, gate_up, down, plan: GroupedPlan, probs=None, *, leaves=8):
    """Two grouped kernels: fused FC1/activation/probability, then the FC2 segment tree."""
    return fc2(fc1_glu(x, gate_up, plan, probs), down, plan, leaves=leaves)


# -------------------------------------------------------------------------- combine ---


def combine(rows, plan: RoutePlan, *, dtype=torch.bfloat16, out=None):
    """Fold the k slots in canonical slot order with one fp32 chain and one rounding."""
    if not isinstance(plan, RoutePlan):
        raise TypeError("expected a RoutePlan from route")
    _check(rows, dtype=torch.float32, ndim=2, name="rows")
    if rows.size(0) != plan.grouped.rows:
        raise ValueError("incorrect row count")
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("unsupported combine output dtype")
    _check(plan.combine_rows, dtype=torch.int32, device=rows.device, ndim=2, name="combine_rows")
    if out is None:
        out = torch.empty((plan.indices.size(0), rows.size(1)), dtype=dtype, device=rows.device)
    else:
        _check(out, dtype=dtype, device=rows.device, ndim=2, name="out")
    extension().combine(rows, plan.combine_rows, out)
    return out


def shared_expert(x, gate, up, down, expert_gate=None):
    """The always-on expert: one fused gate/up projection, the GLU epilogue, then down.

    Every GEMM names its provider: the projections are ``mm_sm100`` (bf16; a shape it declines is
    refused) and the N = 1 gate the vendored Triton GEMM. ``torch.mm`` would reach whichever
    ``aten::mm`` registration the host process installed last.
    """
    import torch.nn.functional as F

    from isoexec.ops.mm.mm_sm100.ops import mm_sm100
    from isoexec.ops.mm.trainer import mm_vendored

    _check(x, dtype=torch.bfloat16, ndim=2, name="x")
    for name, w in (("gate", gate), ("up", up), ("down", down)):
        _check(w, dtype=x.dtype, device=x.device, ndim=2, contiguous=False, name=name)
    x = x.contiguous()
    inter = torch.cat((mm_sm100(x, gate.contiguous().t()), mm_sm100(x, up.contiguous().t())), dim=-1)
    x_glu, x_linear = torch.chunk(inter, 2, dim=-1)
    h = F.silu(x_glu) * (x_linear + 0.0)
    y = mm_sm100(h.contiguous(), down.contiguous().t()).float()
    if expert_gate is not None:
        _check(expert_gate, dtype=x.dtype, device=x.device, ndim=2, contiguous=False, name="expert_gate")
        y = y * torch.sigmoid(mm_vendored(x, expert_gate)).float()
    return y


# ------------------------------------------------------------------------ whole op ---


def moe(x, gate_up, down, indices, probs, *, leaves=8, out_dtype=None):
    """Dispatch, expert MLP and canonical-order combine from already-selected routing."""
    plan = route(indices, gate_up.size(0))
    grouped_x, grouped_p = dispatch(x, probs, plan)
    rows = experts(grouped_x, gate_up, down, plan.grouped, grouped_p, leaves=leaves)
    return combine(rows, plan, dtype=out_dtype or x.dtype)


def routed_forward(x, router_w, gate_up, down, *, top_k, leaves=8, expert_bias=None, routed_scale=1.0):
    """The whole routed forward: route, dispatch, FC1, FC2, combine. The measured scope."""
    logits = route_score(x, router_w)
    indices, probs = route_mechanics(
        logits, top_k, expert_bias=expert_bias, routed_scale=routed_scale
    )
    return moe(x, gate_up, down, indices, probs, leaves=leaves)


__all__ = [
    "BLOCK_M",
    "LEAVES",
    "SEGMENT_MULTIPLE",
    "GroupedPlan",
    "RoutePlan",
    "activation",
    "combine",
    "dispatch",
    "experts",
    "fc1",
    "fc2_leaves",
    "projection_schedule",
    "fc1_glu",
    "fc2",
    "make_plan",
    "moe",
    "route",
    "route_mechanics",
    "route_score",
    "routed_forward",
    "shared_expert",
]
