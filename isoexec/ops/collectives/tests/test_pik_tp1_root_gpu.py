"""TP1 row-parallel projection on the real SM90 kernels (1 GPU): the 1-GPU engine/trainer path.

At TP1 the whole leaf tree is local and ``pik/linear.row_parallel_linear`` stores the bf16 root directly
(``ti_gemm(..., out=bf16 root)``) with the plan's G = 8 leaves on the one rank. Before the fix the sm90
segment-tree producer, which writes an fp32 node for m > 1, raised inside schedule admission and every
1-GPU engine failed to build (fix_cpr_async.md). Claims, at Qwen3.8-27B / Qwen3.5-2B projection shapes:

* the TP1 linear runs, at decode and prefill M;
* its bf16 root equals the fp32 root (``ti_gemm`` into fp32) rounded once, bit for bit;
* every schedule (sm90 producer, fused tree, each split) stores the same bf16 root, bit for bit;
* the same holds for bf16 leaves.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires an SM90 GPU")

#: (label, K, N): attention o_proj and MLP fc2 at TP1.
SHAPES = [("27b-o", 6144, 5120), ("27b-fc2", 17408, 5120), ("2b-o", 2048, 2048), ("2b-fc2", 6144, 2048)]
MS = (1, 7, 64, 257, 4096)


@pytest.fixture(scope="module", autouse=True)
def _pik():
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    ensure_pik()


def _operands(m, k, n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, k, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    return x, w


@pytest.mark.parametrize("leaf_dtype", [None, torch.bfloat16], ids=["fp32-leaves", "bf16-leaves"])
@pytest.mark.parametrize("label,k,n", SHAPES)
def test_tp1_root_is_the_rounded_fp32_root_for_every_schedule(label, k, n, leaf_dtype):
    from isoexec.ops.collectives.pik.gemm import _sched_candidates, ti_gemm
    from isoexec.ops.collectives.pik.linear import row_parallel_linear
    from isoexec.ops.collectives.pik.plan import ReductionPlan

    plan = ReductionPlan(num_leaves=8, leaf_dtype=leaf_dtype)
    m_leaves, leaf_k = plan.leaves_per_rank(1), plan.leaf_k(k)
    for M in MS:
        x, w = _operands(M, k, n, seed=M)
        y = row_parallel_linear(x, w, None, plan=plan, tp_size=1, tp_rank=0, k_full=k, group=None,
                                out_dtype=torch.bfloat16)
        assert y.dtype == torch.bfloat16 and tuple(y.shape) == (M, n)
        node = ti_gemm(x, w, plan=plan, tp_size=1, k_full=k)  # the fp32 root
        assert node.dtype == torch.float32
        assert bitwise_equal(y, node.to(torch.bfloat16)), f"{label} M={M}: fused root != rounded fp32 root"
        for sched in _sched_candidates(m_leaves, leaf_k):
            root = torch.empty((M, n), device="cuda", dtype=torch.bfloat16)
            ti_gemm(x, w, plan=plan, tp_size=1, k_full=k, schedule=sched, out=root)
            assert bitwise_equal(root, y), f"{label} M={M}: schedule {sched} stores different bits"
