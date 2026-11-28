"""Layout of the fused decode operands (``engine.fuse_decode_operands``), on CPU tensors (no GPU, no build).

The decode GEMM's output row pitch is the fused weight's row count; ``mm_sm100/general`` reads the (fa x, ga x)
column window of that output in place and needs the pitch to be 16 bytes. 3 H 128 + 256 + H is a multiple of 8 only
when the local head count H is (e.g. 32 heads at TP 8 is H = 4), so the fused weight carries zero rows up to the next
multiple of 8. A head count that is a multiple of 8 gets no padding: its fused weight is the same tensor as before.
"""

import copy

import pytest
import torch

from isoexec.ops.kda.glm_sm100 import engine as E

HIDDEN = 256


def _shard(H):
    g = torch.Generator().manual_seed(H)
    r = lambda *s: torch.randn(*s, generator=g).to(torch.bfloat16)
    HK = H * 128
    return E.KdaShard(rank=0, world=1, H=H, hidden=HIDDEN, w_qkv=r(3 * HK, HIDDEN), conv_w=torch.zeros(3 * HK, 4),
                      w_fa=r(128, HIDDEN), w_fb=r(HK, 128), w_b=r(H, HIDDEN), w_ga=r(128, HIDDEN), w_gb=r(HK, 128),
                      w_o=r(HIDDEN, HK), dt_bias=torch.zeros(HK), A_log=torch.zeros(H), norm_w=r(128), eps=1e-6,
                      lower_bound=-5.0)


@pytest.mark.parametrize("H", [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24])
def test_fused_rows_are_a_multiple_of_eight_and_the_views_are_the_weights_cpu(H):
    base = _shard(H)
    fused = E.fuse_decode_operands(copy.copy(base))
    rows = 3 * H * 128 + 256 + H
    pad = -rows % 8
    assert fused.w_in.shape == (rows + pad, HIDDEN) and fused.w_in.shape[0] % 8 == 0
    assert (fused.w_in.shape[0] * 2) % 16 == 0                     # the GEMM output's row pitch in bytes
    for name in ("w_qkv", "w_fa", "w_ga", "w_b"):
        view, ref = getattr(fused, name), getattr(base, name)
        assert view.shape == ref.shape and torch.equal(view.view(torch.int16), ref.view(torch.int16)), name
        assert view.untyped_storage().data_ptr() == fused.w_in.untyped_storage().data_ptr(), name
    assert torch.equal(fused.w_in[rows:].view(torch.int16), torch.zeros(pad, HIDDEN, dtype=torch.int16))
    if H % 8 == 0:   # no padding: the fused weight is the plain concatenation, bit for bit
        plain = torch.cat([base.w_qkv, base.w_fa, base.w_ga, base.w_b], 0)
        assert torch.equal(fused.w_in.view(torch.int16), plain.view(torch.int16))


def test_a_prebuilt_fused_weight_of_the_old_unpadded_shape_is_refused_cpu():
    base = _shard(4)
    old = torch.cat([base.w_qkv, base.w_fa, base.w_ga, base.w_b], 0).contiguous()
    with pytest.raises(ValueError, match="multiple of 8"):
        E.fuse_decode_operands(copy.copy(base), old)
    E.fuse_decode_operands(copy.copy(base), E.fused_input_weight(base.w_qkv, base.w_fa, base.w_ga, base.w_b))
