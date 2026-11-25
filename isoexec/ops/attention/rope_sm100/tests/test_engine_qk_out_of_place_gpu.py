"""The engine's q/k norm + RoPE on column windows of the packed projection: out of place, same bits.

When q is a column window of the packed q|k|v|gate GEMM row, the bound ``norms.qk`` callable writes
fresh packed q/k instead of writing over the projection (the attention core reads q packed, so an
in-place q would cost it a separate copy kernel). Compared against the in-place form on the same
operands, as integer views, at decode and prefill token counts and a 16 q / 4 kv heads of 256 geometry
with partial rotary 64, with and without interleaved M-RoPE sections.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="SM100 GPU required")

DEV, BF = "cuda", torch.bfloat16
HQ, HK, D, ROT = 16, 4, 256, 64


def _bits(t):
    return t.contiguous().view(torch.int16)


def _packed_row(T, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = (HQ + 2 * HK + HQ) * D  # q | k | v | gate
    return (torch.randn(T, w, device=DEV, generator=g) * 2.0).to(BF)


def _qk(row):
    T = row.shape[0]
    q = row[:, :HQ * D].view(T, HQ, D)
    k = row[:, HQ * D:(HQ + HK) * D].view(T, HK, D)
    return q, k


@pytest.mark.parametrize("sections", [None, (11, 11, 10)])
@pytest.mark.parametrize("T", [1, 7, 128, 1000])
def test_strided_q_out_of_place_equals_in_place(T, sections):
    from isoexec.ops.attention.rope_sm100 import integration
    from isoexec.ops.attention.rope_sm100.ops import qk_norm_rope as op

    fn = integration.qk_norm_rope()
    g = torch.Generator(device=DEV).manual_seed(T)
    wq = (torch.randn(D, device=DEV, generator=g) * 0.1).to(BF)
    wk = (torch.randn(D, device=DEV, generator=g) * 0.1).to(BF)
    pos = torch.randint(0, 9000, (T,), device=DEV, generator=g)
    rotary = SimpleNamespace(style="neox", theta=1e7, rotary_dim=ROT)
    spec = SimpleNamespace(mrope_section=sections, mrope_interleaved=sections is not None)
    seq = SimpleNamespace(positions=pos)

    row = _packed_row(T, seed=T + 1)
    untouched = row.clone()
    with torch.no_grad():
        q, k = _qk(row)
        yq, yk = fn(q, k, wq, wk, 1e-6, seq=seq, rotary=rotary, spec=spec)
        if T > 1:
            assert yq.is_contiguous() and yq.data_ptr() != q.data_ptr()
            assert torch.equal(_bits(row), _bits(untouched))  # the projection is not written
        # the previous engine path: in place over the projection's own columns
        ref_row = untouched.clone()
        rq, rk = _qk(ref_row)
        tab = integration._TableSet().get(q.device, 1e7, ROT, sections, sections is not None)
        rq, rk = op(rq, rk, wq, wk, pos, tab, 1e-6, zero_centered=True, inplace=True)
    assert torch.equal(_bits(yq), _bits(rq))
    assert torch.equal(_bits(yk), _bits(rk))

