"""Row-strided gate/up views (the packed fc1 output) give the packed silu_mul's bits, with no copies.

The engine's MLP packs gate|up into one GEMM and hands ``mlp.act`` two adjacent column views of its
output; the row-strided kernel reads them in place. Compared as integer views against the contiguous
call and the eager Megatron statement, forward and backward, at decode and prefill row counts.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.mlp.eager import silu_mul_megatron
from isoexec.ops.mlp.mlp_sm100.ops import silu_mul_fused
from isoexec.ops.mlp.mlp_sm100.tests.test_silu_mul_biteq_gpu import POPULATIONS, _population

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="requires SM100"
)

BF16 = torch.bfloat16


def _bits(t):
    return t.contiguous().view(torch.int16)


def _packed(rows, width, pop, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    g = _population((rows, width), pop, "cuda", gen)
    u = _population((rows, width), "large" if pop == "normal" else pop, "cuda", gen)
    return torch.cat([g, u], dim=1)  # [rows, 2 * width]: what the packed fc1 GEMM writes


@pytest.mark.parametrize("pop", POPULATIONS)
@pytest.mark.parametrize("width", [12288, 1536, 136])
@pytest.mark.parametrize("rows", [1, 7, 128, 1000])
def test_strided_views_match_packed_and_eager(rows, width, pop):
    w = _packed(rows, width, pop, seed=rows * 31 + width)
    g, u = w[:, :width], w[:, width:]
    if rows > 1:
        assert not g.is_contiguous() and g.stride(0) == 2 * width
    with torch.no_grad():
        got = silu_mul_fused(g, u)
        packed = silu_mul_fused(g.contiguous(), u.contiguous())
        eager = silu_mul_megatron(g, u)
    assert got.is_contiguous() and got.shape == (rows, width)
    assert torch.equal(_bits(got), _bits(packed))
    assert torch.equal(_bits(got), _bits(eager))


@pytest.mark.parametrize("rows", [1, 7, 256])
def test_strided_backward_matches_packed(rows):
    width = 12288
    w0 = _packed(rows, width, "normal", seed=rows)
    dy = torch.randn(rows, width, device="cuda").to(BF16)

    def run(strided):
        w = w0.detach().clone().requires_grad_(True)
        g, u = w[:, :width], w[:, width:]
        if not strided:
            g, u = g.contiguous(), u.contiguous()
        silu_mul_fused(g, u).backward(dy)
        return w.grad

    assert torch.equal(_bits(run(True)), _bits(run(False)))

