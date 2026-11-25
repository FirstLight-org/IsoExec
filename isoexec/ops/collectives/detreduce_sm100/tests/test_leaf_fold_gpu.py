"""``leaf.leaf_fold``: the BF16-leaf fold equals its torch oracle bit for bit (P = 1..16, signed zeros, large
magnitudes, cancellation)."""
import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.collectives.detreduce_sm100.leaf import leaf_fold, reference  # noqa: E402


def _same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("P", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("n", [8, 5120, 128 * 5120])
def test_leaf_fold_is_the_oracle(P, n):
    g = torch.Generator(device="cuda").manual_seed(P * 7 + n)
    for scale in (1.0, 1e4, 1e-3):
        planes = (torch.randn(P, n, generator=g, device="cuda") * scale).bfloat16()
        planes[:, :8] = torch.tensor([0.0, -0.0, 1e30, -1e30, 3.0, -3.0, 65504.0, -0.0]).bfloat16().to("cuda")
        assert _same(leaf_fold(planes), reference(planes))

