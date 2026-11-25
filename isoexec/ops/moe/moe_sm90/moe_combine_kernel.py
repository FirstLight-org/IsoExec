"""Fused MoE combine: unpermute + fixed-order top-k sum in ONE Triton kernel.

``_fixed_order_combine`` seeds ``out[t]`` with a copy of the token's lowest-expert permuted row and adds
the remaining ``k-1`` rows in ascending permuted-row order, accumulating in ``permuted.dtype``. This kernel
walks the same rows in the same order in registers and writes ``[T, H]`` once; nothing arithmetic changes.
With pik-fc2 on, the accumulation is fp32 with a single bf16 round afterwards; with it off the accumulation
is bf16 and rounds after every add, which ``ROUND_BF16`` reproduces, so the kernel is a drop-in for both.
``probs`` is None in production (the router probs are folded into the expert epilogue); the weighted branch
exists only for a generic Megatron dispatcher.

INSTALLATION IS ENGINE-ONLY BY INSTANCE MARK. ``unpermute`` is a module-level function, so the only binding
megatron offers is a process-global rebind, and under ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` the engine and
the trainer share the process -- handing the trainer this bare ``no_grad`` kernel would sever the MoE
backward while the forward-only IsoExec gate stayed green. Since a free function has no ``self`` to mark,
the mark goes on the DISPATCHER and a scope wrapper on its combine entry points publishes it for the
duration of the call. Everything unmarked keeps ``_deterministic_unpermute``.

Rounding schedules recorded on this module's ops (moved verbatim from ``ImplSpec.rounding.documentary``, which
M6.4 deleted; the registry keeps only the pins a host must set):

``moe.combine:pik_leaf_tree``
    Fixed-order top-k combine: sum each token's k rows in ascending global expert id, in fp32, with no atomics
    and no cross-token reduction (it replaced CUDA scatter_add_'s atomicAdd, which is run-to-run
    nondeterministic and batch-variant at top-k>1). Both sides; the fused Triton combine
    (moe_combine_kernel.fused_fixed_order_combine) is a BITWISE-EQUAL accelerated variant, reached through
    moe_combine_backward.fused_combine, whose MoECombine autograd.Function wraps it -- a raw Triton call has no
    grad_fn, so it is never bound directly. (Two citations here outlived their mechanisms:
    `mark_engine_fused_combine`, the bridge-era instance mark deleted with the mcore monkeypatch stack, and the
    ENGINE-ONLY autofuse site of the same name, deleted in v4 M1. Neither leaves the kernel one-sided: the same
    autograd-wrapped path serves both sides.) ISOEXEC_MOE_COMBINE_SORT swaps the stable argsort for a
    counting sort, torch.equal-gated. THE ETP-INVARIANCE THIS ID IS NAMED FOR is a property of the PAIR: the
    leaf-tree fc2 (moe.expert_mlp, which owns `leaves` / `leaf_dtype` / `tree` after the schema-3 split) makes a
    rank at ETP=C compose its local subtree with pik's cross-rank tree into the SAME G-leaf tree for every C,
    and this region contributes the fold order that is invariant to which rank produced a row. Proven offline
    pik_moe_fc2_{leaftree,topk_leaftree}_test.py (maxabsdiff 0.0). WARNS on ETP>1 alltoall (reduce_scatter is
    not tree-invariant -- not an IsoExec target layout).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# Element block along hidden. H is a reduction-free axis here (each hidden element is summed independently
# over the k rows), so this is a pure occupancy knob and ANY value gives bitwise the same result. Fixed
# rather than autotuned so the engine and the trainer cannot pick different configs.
_BLOCK_H = int(os.environ.get("ISOEXEC_MOE_COMBINE_BLOCK_H", "512"))
_NUM_WARPS = int(os.environ.get("ISOEXEC_MOE_COMBINE_WARPS", "4"))


@triton.jit
def _combine_topk_kernel(
    permuted_ptr,  # [P, H]
    rows_ptr,  # [T, K] int32, rows[t, j] = permuted-row index of token t's j-th expert
    probs_ptr,  # [P] fp32 or dummy
    out_ptr,  # [T, H]
    p_stride0,
    p_stride1,
    o_stride0,
    o_stride1,
    H,
    K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_PROBS: tl.constexpr,
    ROUND_BF16: tl.constexpr,
):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    # static_range so `j` is a python int: the j==0 "copy, don't add" special case (the
    # index_select that seeds `out` in _fixed_order_combine) is resolved at compile time, and the
    # k adds are emitted in ascending-j order with no reassociation.
    for j in tl.static_range(K):
        r = tl.load(rows_ptr + t * K + j).to(tl.int64)
        x = tl.load(permuted_ptr + r * p_stride0 + offs * p_stride1, mask=mask, other=0.0).to(tl.float32)
        if USE_PROBS:
            # torch promotes bf16_rows * fp32_probs to fp32 BEFORE the sum; fp32 rows stay fp32.
            x = x * tl.load(probs_ptr + r).to(tl.float32)
        if j == 0:
            acc = x  # seed = a pure copy of the lowest-expert row (no add, so no rounding)
        else:
            acc = acc + x
            if ROUND_BF16:
                # bf16 accumulation: torch's bf16 add computes in fp32 and rounds ONCE per add.
                # Reproduce that round here so an 8-term bf16 chain matches term for term.
                acc = acc.to(tl.bfloat16).to(tl.float32)

    tl.store(out_ptr + t * o_stride0 + offs * o_stride1, acc.to(out_ptr.dtype.element_ty), mask=mask)


def legal_fold_dtype(acc_out_dtype: torch.dtype, out_dtype: torch.dtype | None) -> torch.dtype:
    """Resolve the kernel's STORE dtype, refusing anything that would move a rounding boundary.

    The accumulator dtype (and therefore ``ROUND_BF16``) is decided by ``acc_out_dtype`` and is NEVER
    keyed on this. Only two cases are admissible: ``out_dtype`` absent or equal to ``acc_out_dtype``, so
    the store is unchanged; or an fp32 accumulator with a bf16 ``out_dtype``, which moves the caller's own
    trailing ``.to(torch.bfloat16)`` into the store -- still exactly one round, now in a register. Anything
    else would change the number or order of roundings, so it raises.
    """
    if out_dtype is None or out_dtype == acc_out_dtype:
        return acc_out_dtype
    if acc_out_dtype == torch.float32 and out_dtype == torch.bfloat16:
        return torch.bfloat16
    raise RuntimeError(
        "[isoexec] fused MoE combine: refusing out_dtype="
        f"{out_dtype} against accumulate dtype {acc_out_dtype}. The only fold that preserves the "
        "rounding schedule is fp32-accumulate -> bf16 store; anything else changes the number of "
        "roundings and must be classified as a COMPOSITION EVENT, not a fold."
    )


def build_combine_rows(
    sorted_indices: torch.Tensor, num_tokens: int, *, rows: torch.Tensor | None = None
) -> torch.Tensor | None:
    """``argsort(sorted_indices, stable=True).view(T, k)`` as int32, or None if the layout is not
    "every token routes to exactly k experts" (token dropping / capacity padding).

    Identical to what ``_fixed_order_combine`` computes; kept as its own function so a caller can
    cache it. int32 is a pure index-width change (P <= T*topk is far below 2^31 at any shape this
    stack runs) and does not touch the arithmetic.
    """
    n = int(sorted_indices.numel())
    if num_tokens == 0 or n == 0 or n % num_tokens != 0:
        return None
    k = n // num_tokens
    from .moe_fused_permute import validate_combine_rows

    rows = validate_combine_rows(rows, sorted_indices, num_tokens)
    if rows is not None:
        return rows
    # Counting sort instead of the small-P radix sort. Same permutation, emitted directly as [T, k]
    # int32 so the fused path skips the view/cast too.
    from .moe_combine_rows_kernel import stable_combine_rows

    rows = stable_combine_rows(sorted_indices, num_tokens, dtype=torch.int32)
    if rows is not None:
        return rows
    return torch.argsort(sorted_indices, stable=True).view(num_tokens, k).to(torch.int32).contiguous()


def fused_fixed_order_combine(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape,
    *,
    permuted_probs: torch.Tensor | None = None,
    rows: torch.Tensor | None = None,
    validate: bool = False,
    out_dtype: torch.dtype | None = None,
):
    """Drop-in for ``moe_batch_invariant._fixed_order_combine``, fused into one Triton kernel.

    Returns ``[T, H]`` in ``permuted_tokens.dtype`` (or fp32 when ``permuted_probs`` is given and
    promotes), or ``None`` when the layout is unsupported -- so the caller can fall back exactly
    as the production code does.

    ``permuted_probs`` is the ALREADY-PERMUTED ``[P]`` prob vector (what
    ``probs.T.masked_select(routing_map.T)`` produces); production passes None.
    """
    num_tokens = int(restore_shape[0])
    if rows is None:
        rows = build_combine_rows(sorted_indices, num_tokens)
        if rows is None:
            return None
    k = rows.shape[1]

    if validate:
        expected = torch.arange(num_tokens, device=sorted_indices.device).repeat_interleave(k)
        got = sorted_indices[rows.view(-1).long()].to(expected.dtype)
        if not torch.equal(got, expected):
            raise RuntimeError(
                "[isoexec] fused MoE combine: tokens do not each route to exactly topk experts "
                f"(num_tokens={num_tokens}, permuted_rows={int(sorted_indices.numel())})."
            )

    H = permuted_tokens.shape[-1]
    in_dtype = permuted_tokens.dtype
    use_probs = permuted_probs is not None
    # Match torch's promotion rule for the generic weighted compatibility branch.
    acc_out_dtype = torch.promote_types(in_dtype, permuted_probs.dtype) if use_probs else in_dtype
    # bf16 accumulation rounds after every add; fp32 accumulation does not round at all.
    # KEYED ON acc_out_dtype AND NOT ON THE STORE DTYPE. Folding a trailing bf16 cast into the store
    # must not turn an fp32 sum into a per-term-rounded bf16 sum -- those are different functions.
    round_bf16 = acc_out_dtype == torch.bfloat16
    store_dtype = legal_fold_dtype(acc_out_dtype, out_dtype)

    out = torch.empty(num_tokens, H, dtype=store_dtype, device=permuted_tokens.device)
    grid = (num_tokens, triton.cdiv(H, _BLOCK_H))
    _combine_topk_kernel[grid](
        permuted_tokens,
        rows,
        permuted_probs if use_probs else permuted_tokens,  # dummy ptr when unused
        out,
        permuted_tokens.stride(0),
        permuted_tokens.stride(1),
        out.stride(0),
        out.stride(1),
        H,
        K=k,
        BLOCK_H=_BLOCK_H,
        USE_PROBS=use_probs,
        ROUND_BF16=round_bf16,
        num_warps=_NUM_WARPS,
        # NON-NEGOTIABLE. Without it LLVM contracts `acc + x * prob` into a single FMA -- one rounding
        # step instead of two -- and the weighted combine drifts from torch's. The unweighted production
        # path has no multiply, but the flag stays on so no future edit can reintroduce a contraction.
        enable_fp_fusion=False,
    )
    return out
