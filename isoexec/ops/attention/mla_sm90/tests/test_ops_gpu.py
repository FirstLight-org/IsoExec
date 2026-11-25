"""ops.attention.mla_sm90_attention wiring: projection bits, composition bits, gradient presence, T=1 and ragged packs."""

import pytest
import torch

from isoexec.ops._testing import bitwise_equal

from isoexec.ops.attention.mla_sm90.tests._harness import SCALE, cuda_ok

pytestmark = pytest.mark.skipif(not cuda_ok(), reason="requires an SM90 (H100) device")


def _kernels():
    try:
        from isoexec.ops.attention import gqa_sm90
        from isoexec.ops.attention.mla_sm90.fwd import mla_sm90_fwd

        if not gqa_sm90.available():
            raise RuntimeError("the gqa_sm90 FA3 build (MLA backward) is unavailable")
    except Exception as exc:
        pytest.skip(f"mla_sm90 kernels not importable: {exc}")
    return mla_sm90_fwd


def _inputs(lens, heads=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = sum(lens)

    def r(*s, sc=1.0):
        return (torch.randn(*s, generator=g) * sc).to("cuda", torch.bfloat16)

    return [r(T, heads, 192), r(T, heads, 64), r(T, 512), r(T, 64), r(heads, 192, 512, sc=512 ** -.5),
            r(heads, 256, 512, sc=512 ** -.5)]


def test_projections_match_absorbed_primitive_bits():
    from isoexec.ops.attention.mla_absorbed import absorb_query, unabsorb_output
    from isoexec.ops.attention.mla_sm90.ops import _absorb_fwd, _unabsorb_fwd

    qn, _, _, _, wk, wv = _inputs([300, 17])
    ql = _absorb_fwd(qn, wk)
    assert bitwise_equal(ql, absorb_query(qn, wk))
    assert bitwise_equal(_unabsorb_fwd(ql, wv), unabsorb_output(ql, wv))


@pytest.mark.parametrize("lens", [[1], [700, 3, 1100], [64, 65]])
def test_op_forward_is_the_documented_composition(lens):
    _kernels()
    from isoexec.ops.attention.mla_sm90.ops import kernel_fwd
    from isoexec.ops.attention.mla_absorbed import absorb_query, pack_latent, unabsorb_output
    from isoexec.ops.attention.mla_sm90.ops import mla_sm90_attention

    ts = _inputs(lens)
    y = mla_sm90_attention(*ts, seq_lens=lens, scale=SCALE, seg_tokens=512)
    qn, qp, c, kp, wk, wv = ts
    cache, table = pack_latent(c, kp, lens, 64)
    L = torch.tensor(lens, device="cuda", dtype=torch.int32)
    cu = torch.cat((L.new_zeros(1), L.cumsum(0, dtype=torch.int32)))
    out, _ = kernel_fwd(torch.cat((absorb_query(qn, wk), qp), -1).contiguous(), cache.unsqueeze(2), table, cu, L,
                        SCALE, 512, max(lens))
    assert bitwise_equal(y, unabsorb_output(out, wv))


def test_op_signature_has_no_roles():
    import inspect

    from isoexec.ops.attention.mla_sm90.ops import mla_sm90_attention

    assert "prompt_lens" not in inspect.signature(mla_sm90_attention).parameters


def test_op_gradients_exist_and_are_finite():
    _kernels()
    from isoexec.ops.attention.mla_sm90.ops import mla_sm90_attention

    lens = [600, 5]
    leaves = [t.requires_grad_(True) for t in _inputs(lens, seed=3)]
    y = mla_sm90_attention(*leaves, seq_lens=lens, scale=SCALE, seg_tokens=256)
    y.float().square().mean().backward()
    for t in leaves:
        assert t.grad is not None and t.grad.dtype == t.dtype
        assert torch.isfinite(t.grad).all() and t.grad.count_nonzero()
