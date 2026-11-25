"""At tp=1 the row-parallel tree's root is rounded where it is stored, bit-identical to the fp32 root + cast.

At tp=1 ``row_parallel_linear`` rounds the tree root at the tree's own store (the tree-reduce kernel's
``out_bf16`` store, the fused tree's store into a bf16 output, or the split-1 copy) instead of writing
an fp32 root and casting it with a separate ``.to(bfloat16)``. Each is one fp32 -> bf16
round-to-nearest-even of the same fp32 value.

Checked per schedule (every candidate, forced) and through ``row_parallel_linear`` against the fp32
root + cast, at decode and prefill row counts and o_proj / down_proj shapes, as integer views.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

SHAPES = ((4096, 4096), (12288, 4096))  # (K, N): o_proj (attention and GDN), down_proj
MS = (1, 7, 128, 256, 1000)


def _pik():
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    return ensure_pik()


def _operands(m, k, n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, k, device="cuda", generator=g)).bfloat16()
    w = (torch.randn(n, k, device="cuda", generator=g) * k ** -0.5).bfloat16()
    return x, w


def _bits(t):
    return t.contiguous().view(torch.int16)


@pytest.mark.parametrize("leaf_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("m", MS)
def test_every_schedule_rounds_the_root_at_its_store(m, k, n, leaf_dtype):
    pik = _pik()
    from pik.gemm import _sched_candidates, ti_gemm  # type: ignore

    plan = pik.ReductionPlan(num_leaves=8, leaf_dtype=leaf_dtype)
    x, w = _operands(m, k, n, seed=m + k)
    leaf_k = plan.leaf_k(k)
    for sched in _sched_candidates(8, leaf_k):
        ref = ti_gemm(x, w, plan=plan, schedule=sched)  # fp32 root
        assert ref.dtype == torch.float32
        root = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        got = ti_gemm(x, w, plan=plan, schedule=sched, out=root)
        assert got.data_ptr() == root.data_ptr()
        assert torch.equal(_bits(got), _bits(ref.to(torch.bfloat16))), sched


@pytest.mark.parametrize("leaf_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("m", MS)
def test_row_parallel_linear_is_the_fp32_root_cast(m, k, n, leaf_dtype):
    pik = _pik()
    from pik.gemm import ti_gemm  # type: ignore
    from pik.linear import row_parallel_linear  # type: ignore

    plan = pik.ReductionPlan(num_leaves=8, leaf_dtype=leaf_dtype)
    x, w = _operands(m, k, n, seed=3 * m + n)
    got = row_parallel_linear(x, w, None, plan=plan, tp_size=1, tp_rank=0, k_full=k, out_dtype=torch.bfloat16)
    ref = ti_gemm(x, w, plan=plan, tp_size=1, tp_rank=0, k_full=k)
    assert got.dtype == torch.bfloat16 and got.shape == (m, n)
    assert torch.equal(_bits(got), _bits(ref.to(torch.bfloat16)))
