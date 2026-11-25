"""The TF32-fn projection prototype (``split_proj(pieces=2)`` on ``tf32_fn(fn)``): the two pieces are the exact TF32
weight, so it equals the three-piece projection of the same TF32 weight bit for bit; and it keeps the projection's
relations: rows independent of the batch (subset / permutation), the CTA-group count, and repeat."""
import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.residual.mhc_sm100.ops import split_proj, split_tail, tf32_fn  # noqa: E402

H = 5120


def _bits(t):
    return t.contiguous().view(torch.int32)


def _inputs(T, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(T, 4, H, generator=g, device="cuda") * 2).bfloat16()
    fn = torch.randn(24, 4 * H, generator=g, device="cuda") * 0.01
    return x, fn


def test_tf32_fn_is_two_exact_pieces():
    _, fn = _inputs(1, 0)
    t = tf32_fn(fn)
    assert torch.equal(_bits(t) & 0x1FFF, torch.zeros_like(_bits(t)))
    assert (t - fn).abs().max() <= fn.abs().max() * 2.0 ** -11


@pytest.mark.parametrize("T", [1, 7, 64, 300, 2048])
def test_two_pieces_equal_three_pieces_of_the_tf32_weight(T):
    x, fn = _inputs(T, T)
    t = tf32_fn(fn)
    assert torch.equal(_bits(split_proj(x, t, pieces=2)), _bits(split_proj(x, t, pieces=3)))


def test_batch_groups_and_repeat_invariance():
    x, fn = _inputs(513, 3)
    t = tf32_fn(fn)
    full = split_proj(x, t, pieces=2)
    perm = torch.randperm(513, device="cuda")
    assert torch.equal(_bits(split_proj(x[perm], t, pieces=2)), _bits(full[:, perm]))
    assert torch.equal(_bits(split_proj(x[100:117], t, pieces=2)), _bits(full[:, 100:117]))
    for groups in (1, 3, 16):
        assert torch.equal(_bits(split_proj(x, t, pieces=2, groups=groups)), _bits(full))
    assert torch.equal(_bits(split_proj(x, t, pieces=2)), _bits(full))
    scale = torch.tensor([0.5, 0.7, 0.9], device="cuda")
    base = torch.randn(24, device="cuda") * 0.1
    a = split_tail(full, scale, base, rms_eps=1e-20, eps=1e-6, iters=20)
    b = split_tail(full[:, perm].contiguous(), scale, base, rms_eps=1e-20, eps=1e-6, iters=20)
    assert all(torch.equal(_bits(u[perm]), _bits(v)) for u, v in zip(a, b))
