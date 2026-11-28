"""CPU contract of the Engram package: importing it builds nothing; the oracles run on CPU."""

import subprocess
import sys

import torch

from isoexec.ops.embed.engram_sm100 import reference as R


def test_importing_builds_nothing():
    code = ("import isoexec.ops.embed.engram_sm100.ops;"
            "from isoexec.ops.embed.engram_sm100 import _extension;"
            "assert _extension.extension.cache_info().currsize == 0")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_float64_oracle_is_differentiable_on_cpu():
    g = torch.Generator().manual_seed(3)
    x, key = (torch.randn(2, 5, 4, 16, generator=g) for _ in range(2))
    value = torch.randn(2, 5, 16, generator=g)
    qw, kw = torch.randn(16, generator=g), torch.randn(16, generator=g)
    x.requires_grad_(True)
    out, gate = R.engram_residual_f64(x, key, value, qw, kw, 1e-6)
    assert gate.shape == (2, 5, 4)
    out.sum().backward()
    assert out.dtype == torch.float64 and x.grad is not None and torch.isfinite(x.grad).all()


def test_hash_reciprocals_of_inference_constants():
    """An engine builds its hash constants under inference_mode: inference tensors have no version counter, and the
    reciprocal cache must still serve them (exact floor((2**64 - 1) / p) as int64 bits)."""
    import torch

    from isoexec.ops.embed.engram_sm100 import ops
    with torch.inference_mode():
        primes = torch.tensor([[3, 1000003], [7, 2147483647]], dtype=torch.int64)
        got = ops._hash_magic(primes)
    want = [((1 << 64) - 1) // p for p in (3, 1000003, 7, 2147483647)]
    assert [v & ((1 << 64) - 1) for v in got.flatten().tolist()] == want
    assert ops._hash_magic(primes) is got
