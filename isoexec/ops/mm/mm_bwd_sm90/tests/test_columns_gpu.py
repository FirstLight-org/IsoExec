"""split_columns' gradient assembly in one launch is the eager ``torch.add(g, 0, out=...)`` loop, bit for bit."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires H100/SM90"
)

BF16 = torch.bfloat16


def _eager(grads):
    buf = torch.empty((grads[0].shape[0], sum(g.shape[1] for g in grads)), dtype=BF16, device="cuda")
    off = 0
    for g in grads:
        torch.add(g, 0, out=buf[:, off:off + g.shape[1]])
        off += g.shape[1]
    return buf


def _values(shape, gen):
    v = torch.randn(shape, device="cuda", generator=gen)
    flat = v.view(-1)
    flat[::5] = -0.0
    flat[1::7] = float("nan")
    flat[2::11] = float("-inf")
    flat[3::13] = 1e-40
    return v.to(BF16)


# GDN in_proj at TP1/2/4/8 of Qwen3.8-27B: q|k|v|z|beta|alpha widths, plus attention q|k|v|gate.
WIDTHS = ((2048, 2048, 6144, 6144, 48, 48), (1024, 1024, 3072, 3072, 24, 24), (512, 512, 1536, 1536, 12, 12),
          (256, 256, 768, 768, 6, 6), (1536, 256, 256, 1536), (7,), (3, 9, 1, 17))


@pytest.mark.parametrize("widths", WIDTHS)
@pytest.mark.parametrize("t", (1, 5, 1000, 5800))
def test_contiguous_blocks(widths, t):
    from isoexec.ops.mm.mm_bwd_sm90 import ops

    gen = torch.Generator(device="cuda").manual_seed(t + len(widths))
    grads = [_values((t, w), gen) for w in widths]
    got = ops.assemble_columns(grads)
    assert got.is_contiguous() and bitwise_equal(got, _eager(grads))


def test_strided_column_views():
    """q/k/v grads as column views of one conv-backward buffer, z contiguous, beta/alpha odd views."""
    from isoexec.ops.mm.mm_bwd_sm90 import ops

    gen = torch.Generator(device="cuda").manual_seed(3)
    t = 777
    qkv = _values((t, 2560), gen)
    ba = _values((t, 25), gen)
    grads = [qkv[:, :512], qkv[:, 512:1024], qkv[:, 1024:], _values((t, 1536), gen), ba[:, 1:13], ba[:, 13:]]
    assert ops.assemble_supported(grads)
    assert bitwise_equal(ops.assemble_columns(grads), _eager(grads))


def test_split_columns_backward_uses_assembler_with_same_bits():
    from isoexec.core import linear
    from isoexec.ops.mm.wgrad import install_column_assembler

    gen = torch.Generator(device="cuda").manual_seed(9)
    weights = tuple(torch.empty(w, 8, device="cuda") for w in (512, 512, 1536, 1536, 12, 12))
    out0 = _values((300, 4120), gen)
    grads = [_values((300, w.shape[0]), gen) for w in weights]

    def run():
        out = out0.clone().requires_grad_(True)
        parts = linear.split_columns(out, weights)
        torch.autograd.backward(parts, grads)
        return out.grad

    prev = linear._COLUMN_ASSEMBLER
    try:
        linear.set_column_assembler(None)
        want = run()
        assert install_column_assembler()
        assert linear._COLUMN_ASSEMBLER is not None
        assert bitwise_equal(run(), want)
    finally:
        linear.set_column_assembler(prev)
