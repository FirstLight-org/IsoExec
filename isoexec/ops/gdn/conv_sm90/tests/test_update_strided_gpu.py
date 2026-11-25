"""conv_sm90 decode ``update`` reading a row-strided x (the q|k|v window of the in_proj GEMM output).

The strided read replaces ``torch.cat([q, k, v])`` on the engine's decode path. Claim: for every lane
(live or padding) the output rows and the carried state are byte-identical to the contiguous in-place
form fed with the cat's copy, and the source GEMM buffer is never written.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires H100/SM90"
)

BF = torch.bfloat16


def _case(n, kd, vd, extra, lines=None, seed=0):
    lines = lines or n + 17
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = 2 * kd + vd + extra  # in_proj row: q | k | v | z, beta, alpha
    gemm = (torch.randn(n, W, generator=g, device="cuda") * 2).to(BF)
    D = 2 * kd + vd
    w = (torch.randn(D, 4, generator=g, device="cuda") * 0.5).to(BF)
    state = (torch.randn(lines, D, 3, generator=g, device="cuda")).to(BF)
    idx = torch.randperm(lines - 1, generator=torch.Generator().manual_seed(seed))[:n].to(torch.int32) + 1
    idx = idx.cuda()
    if n > 2:
        idx[1] = 0  # null block: padding lane
        idx[n - 1] = -1  # negative: padding lane
    return gemm, D, w, state, idx


@pytest.mark.parametrize("n", (1, 2, 5, 64, 256))
@pytest.mark.parametrize("geom", [(512, 1536, 1560), (256, 768, 780), (128, 512, 520)])
def test_strided_matches_contiguous_cat(n, geom):
    from isoexec.ops.gdn.conv_sm90.integration import causal_conv1d_update

    kd, vd, extra = geom
    gemm, D, w, state, idx = _case(n, kd, vd, extra, seed=n)
    view = gemm[:, :D]
    assert view.stride(0) == gemm.shape[1]  # a column window, even at n == 1
    gemm_before = gemm.clone()

    cat = torch.cat([gemm[:, :kd], gemm[:, kd:2 * kd], gemm[:, 2 * kd:D]], dim=-1)
    s_ref = state.clone()
    y_ref = causal_conv1d_update(cat, s_ref, w, None, "silu", conv_state_indices=idx)

    s_got = state.clone()
    y_got = causal_conv1d_update(view, s_got, w, None, "silu", conv_state_indices=idx)

    assert y_got.is_contiguous() and y_got.data_ptr() != gemm.data_ptr()
    assert bitwise_equal(y_got, y_ref)  # live lanes: the conv; padding lanes: their input bytes
    assert bitwise_equal(s_got, s_ref)
    assert bitwise_equal(gemm, gemm_before)  # the GEMM output is never written


def test_strided_update_captures_and_replays():
    from isoexec.ops.gdn.conv_sm90.integration import causal_conv1d_update

    gemm, D, w, state, idx = _case(128, 512, 1536, 1560, seed=11)
    view = gemm[:, :D]
    s_ref = state.clone()
    y_ref = causal_conv1d_update(view.contiguous(), s_ref, w, None, "silu", conv_state_indices=idx)
    s_got = state.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        causal_conv1d_update(view, state.clone(), w, None, "silu", conv_state_indices=idx)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = causal_conv1d_update(view, s_got, w, None, "silu", conv_state_indices=idx)
    graph.replay()
    torch.cuda.synchronize()
    assert bitwise_equal(y, y_ref) and bitwise_equal(s_got, s_ref)
