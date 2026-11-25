"""Shared MoE combine references and deterministic matmul installation.

The unified hosts use the stable ascending-expert combine reference, scoped
matmul provider installation and differentiable PIK tree reduction defined here.
Owner combine shares the bf16-wire predicate and its losslessness receipt.
Foreign Megatron layer-spec and dispatcher patching are not part of this module.
"""

from __future__ import annotations

import os

import torch


# The bridge-era spec machinery that lived here -- ``force_isoexec_moe_config`` (pinning a foreign
# builder onto this recipe), ``make_isoexec_local_layer_spec`` and their helpers -- was DELETED
# earlier: the unified model declares the recipe as data (an ``Mlp`` row) and the staged executor runs
# it, so there is no foreign provider left to force.
#
# THE DETERMINISTIC INSTALLS WENT THE SAME WAY (kernel-ownership.md §5.2, 2026-09-04).
# ``enable_moe_deterministic_ops`` / ``revert_moe_deterministic_ops`` rebound
# ``moe_utils.unpermute`` and ``topk_routing_with_score_function``;
# ``lift_moe_tp_sp_training_veto`` wrapped the unified MoE layer's forward;
# ``mark_engine_dispatchers_nogather``, ``install_moe_pik_fc2_combine`` and
# ``install_moe_pik_fc2_alltoall_combine`` rebound the two mcore token dispatchers. All six patched
# classes the unified model never builds, and all six had zero callers.
#
# WHAT REMAINS IS THE LIBRARY, and it is what every host actually binds: ``_fixed_order_combine``
# (the ascending-expert reference the fused combine and the leaf tree are gated against),
# ``_install_moe_matmul_invariance`` (LIVE, called unguarded from ``runtimes/trainer_installs.py``),
# ``_PikTreeAllReduce`` (LIVE from ``runtimes/parallel.py``), and the bf16-wire predicate.

# How many distinct (num_tokens, topk) shapes ``_fixed_order_combine`` structurally validates before
# trusting the row layout. Each validation costs one device->host sync.
_VALIDATE_FIRST_N_CALLS = 8
_validated_calls = 0


def _fixed_order_combine(permuted_tokens, sorted_indices, restore_shape, *, rows=None):
    """Sum each token's k expert rows in ascending-expert order. No atomics, no cross-token reduce.

    ``permute`` lays rows out expert-major, so for a fixed token its k rows appear at ascending row
    positions in ascending expert order. A *stable* argsort of ``sorted_indices`` therefore groups
    each token's rows together, ascending expert within the group. Gathering column j across all
    tokens and adding the k columns in order j=0..k-1 gives every token the same summation order
    regardless of how many other tokens are in the batch -- which is the bitwise invariant.

    Returns ``None`` when the layout is not the plain "every token routes to exactly k experts"
    case (e.g. token dropping), so the caller can fall back.
    """
    global _validated_calls

    num_tokens = int(restore_shape[0])
    n = int(sorted_indices.numel())
    if num_tokens == 0 or n == 0 or n % num_tokens != 0:
        return None
    k = n // num_tokens

    # The fused permute producer can emit this exact inverse while it already gathers routed rows,
    # removing every post-FC2 metadata launch. Malformed published metadata raises rather than
    # silently falling back to the sort and hiding an inert arm.
    from .moe_fused_permute import validate_combine_rows

    rows = validate_combine_rows(rows, sorted_indices, num_tokens)
    if rows is None:
        # Exact counting-sort replacement, with the stable argsort as fallback.
        from .moe_combine_rows_kernel import stable_argsort_order

        rows = stable_argsort_order(sorted_indices, num_tokens).view(num_tokens, k)
    if _validated_calls < _VALIDATE_FIRST_N_CALLS:
        _validated_calls += 1
        expected = torch.arange(num_tokens, device=sorted_indices.device).repeat_interleave(k)
        if not torch.equal(sorted_indices[rows.view(-1)].to(expected.dtype), expected):
            raise RuntimeError(
                "[isoexec] MoE combine: tokens do not each route to exactly topk experts "
                f"(num_tokens={num_tokens}, permuted_rows={n}). Token dropping / capacity padding "
                "must be disabled for bitwise IsoExec."
            )

    out = permuted_tokens.index_select(0, rows[:, 0].contiguous())
    for j in range(1, k):
        out = out + permuted_tokens.index_select(0, rows[:, j].contiguous())
    return out


_matmul_invariance_lib = None


def _install_moe_matmul_invariance() -> None:
    """Install our vendored Triton persistent-matmul overrides on the PRIMITIVE GEMM ops (mm/addmm).

    On SM90+ ``enable_batch_invariant_mode`` only pins the cuBLAS workspace config, which gives
    run-to-run determinism but not batch invariance: cuBLAS still selects a different kernel/tiling
    for M=1 than for M=512. The MoE router's fp32 gating mm is measurably not row-invariant, and
    expert GEMMs run at per-expert token counts. Dense models never reach this module.

    Only the primitives are overridden. ``aten::matmul`` and ``aten::linear`` are
    CompositeImplicitAutograd: giving them a CUDA kernel makes them autograd leaves, and torch then
    wants ``aten::matmul_backward`` / ``aten::linear_backward``, which exist only for
    Meta/NestedTensor, so the training backward dies with NotImplementedError.

    Overriding mm/addmm instead loses nothing: matmul/linear decompose into mm/addmm/bmm (bmm is
    already overridden by ``enable_batch_invariant_mode`` on all CUDA platforms), so every GEMM
    still lands on a Triton batch-invariant kernel, and both runtimes decompose identically ->
    engine(no_grad) and trainer(grad) forwards stay bitwise-equal. Verified: fp32 router-GEMM row
    is M-invariant across batch sizes; F.linear/matmul backward work at 2D and 3D, with and
    without bias.
    """
    global _matmul_invariance_lib
    if _matmul_invariance_lib is not None:
        return
    from ...mm import mm_batch_invariant as bi

    if torch.cuda.get_device_capability()[0] == 8:
        return  # vLLM's enable_batch_invariant_mode already overrides matmuls on SM80.

    # The override must stay GLOBAL. Restricting it to the fp32 router and leaving bf16 GEMMs on
    # cuBLAS passes every offline gate but breaks IsoExec live, and is not faster.
    lib = torch.library.Library("aten", "IMPL")

    # Forward-only scoping (ISOEXEC_MM_FWD_ONLY=1, default off). Every FORWARD GEMM on both
    # runtimes still runs the batch-invariant kernel; only the trainer's dgrad/wgrad, which the
    # IsoExec gate does not constrain, fall through to cuBLAS. install() is fail-closed and returns
    # False (leaving the unscoped override below) on a flag-off, a failed self-check, or a hook that
    # will not install.
    from ...mm.mm_fwd_scope import install as _install_mm_fwd_scope

    if not _install_mm_fwd_scope(lib):
        lib.impl("aten::mm", bi.mm_batch_invariant, "CUDA")
        lib.impl("aten::addmm", bi.addmm_batch_invariant, "CUDA")
    _matmul_invariance_lib = lib
    print(
        "[ISOEXEC-MOE] Triton batch-invariant matmul overrides installed GLOBALLY (mm/addmm only; "
        "matmul/linear decompose onto them and must stay autograd-differentiable). Yes, this is "
        "2.6-4x slower than cuBLAS per GEMM -- restricting it to the fp32 router was TRIED and it "
        "BROKE IsoExec (6.9e-07 -> 8.8e-03, live 35B A/B) for ZERO speedup. See the comment above "
        "before you 'optimize' it again.",
        flush=True,
    )


# ``softmax_batch_invariant`` USED TO LIVE HERE, as a forwarder into
# ``vllm.model_executor.layers.batch_invariant``. It is now VENDORED, in
# ``ops/mm/mm_batch_invariant.py`` beside the other eight symbols copied out of that same upstream
# module (kernel-ownership.md section 2a). The forwarder was the last runtime import on the
# reference router's path, so removing it is what lets the reference trainer be built in a process
# with no vLLM installed -- asserted by ``ops/mm/tests/test_softmax_vendor_no_vllm_cpu.py``.


# bf16-wire / fp32-root fast path (kill switch ISOEXEC_MOE_PIK_BF16_WIRE=0). When world == G
# each rank's combine partial is one leaf, already rounded through bf16 by the leaf-tree fc2, so bf16
# on the wire is lossless and halves the combine's bytes. The ROOT must stay fp32: the downstream
# unpermute top-k sum takes its dtype from its input, so a bf16 root would be sum-of-rounded rather
# than fp32-sum-then-one-round. The fp32 root is bit-identical to the fp32-wire root by construction.
# Guarded by world==G only (an internal fp32 node at ETP<G is not bf16-representable, so the trainer
# stays fp32-wire) plus a first-call losslessness self-check with permanent fallback.
_bf16_wire = {"checked": False, "ok": False}


def _pik_bf16_wire_enabled() -> bool:
    return os.environ.get("ISOEXEC_MOE_PIK_BF16_WIRE", "1") == "1"


class _PikTreeAllReduce(torch.autograd.Function):
    """Differentiable pik fixed-tree all-reduce over the ETP (tp_ep) group.

    Forward sums the per-rank fc2 leaf-subtree partials with pik's balanced tree, so the result is
    bitwise identical for any ETP dividing G. Backward passes the gradient through, matching
    Megatron's ``reduce_from_tensor_model_parallel_region`` convention.
    """

    @staticmethod
    def forward(ctx, x, group):
        import torch.distributed as dist

        from ....ops.collectives.pik_tp_invariant import ensure_pik, get_plan

        ensure_pik()
        from pik.allreduce import (  # type: ignore
            p2p_available,
            sym_partial,
            tree_all_reduce,
        )

        x = x.contiguous()
        if (
            _pik_bf16_wire_enabled()
            and x.dtype == torch.float32
            and dist.get_world_size(group) == get_plan().num_leaves
        ):
            if not _bf16_wire["checked"]:
                _bf16_wire["checked"] = True
                _bf16_wire["ok"] = bool((x.to(torch.bfloat16).to(torch.float32) == x).all())
                print(
                    f"[ISOEXEC-MOE] pik combine bf16-wire/fp32-root self-check "
                    f"{'PASS (lossless leaf; shot-1 wire bytes halved)' if _bf16_wire['ok'] else 'FAIL -> fp32 wire permanently'}",
                    flush=True,
                )
            if _bf16_wire["ok"]:
                # bf16 leaf on the wire, fp32 root out. The bf16 leaf is staged straight into
                # peer-visible memory in one fp32->bf16 pass, so `_tree_all_reduce_p2p` recognises its
                # own staging buffer by data_ptr and skips the copy-in. `staged.copy_(x)` is RNE
                # fp32->bf16, bit-identical to `x.to(bf16)`, and is capture-safe (the pool is warmed
                # before capture).
                if p2p_available(group):
                    staged = sym_partial(x.shape, x.device, group, dtype=torch.bfloat16, out_dtype=torch.float32)
                    staged.copy_(x)
                    return tree_all_reduce(staged, group=group, root_dtype=torch.float32)
                return tree_all_reduce(x.to(torch.bfloat16), group=group, root_dtype=torch.float32)
        return tree_all_reduce(x, group=group)

    @staticmethod
    def backward(ctx, g):
        return g, None
