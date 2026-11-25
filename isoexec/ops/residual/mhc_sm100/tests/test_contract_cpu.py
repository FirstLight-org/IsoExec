"""CPU contract: spec tables equal the kernel constants, the package declares no op, the reference transcription
matches the eager oracle on CPU, import builds nothing."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from isoexec.ops.residual.mhc_sm100 import reference as R
from isoexec.ops.residual.mhc_sm100 import spec

CSRC = Path(__file__).resolve().parents[1] / "csrc"


def _const(path, name):
    return int(re.search(rf"constexpr int {name} = (\d+);", (CSRC / path).read_text()).group(1))


def test_tables_agree_with_kernel_constants_cpu():
    assert _const("split_fwd.cu", "kSlices") == spec.NUM_SLICES == 32
    assert _const("split_bwd.cu", "kSegRows") == spec.SEG_ROWS == 512
    assert set(spec.HIDDEN) == {4096, 5120} and all(h % (spec.NUM_SLICES * 16) == 0 for h in spec.HIDDEN)
    assert "H == 4096 || H == 5120" in (CSRC / "split_bwd.cu").read_text()


def test_this_package_declares_no_op_cpu():
    assert not hasattr(spec, "IMPLS")


def test_ops_import_builds_nothing_cpu():
    import subprocess
    import sys

    code = ("import isoexec.ops.residual.mhc_sm100.ops;"
            "from isoexec.ops.residual.mhc_sm100 import _extension;"
            "assert _extension.extension.cache_info().currsize == 0")
    subprocess.run([sys.executable, "-c", code], check=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})


def test_float32_reference_matches_eager_oracle_cpu():
    torch.manual_seed(0)
    n, H = 4, 256
    streams = (torch.randn(5, n, H) * 2).bfloat16()
    fn = (torch.randn(24, n * H) * 0.05).bfloat16()
    base, scale = torch.randn(24) * 0.3, torch.tensor([0.7, 0.4, 0.9])
    post, comb, col = R.eager_split(streams, fn, base, scale, rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20)
    _, rpost, rcomb, rcol, *_ = R.split_reference(streams, fn, scale, base, rms_eps=1e-5, hc_eps=1e-6, iters=20,
                                                  dtype=torch.float64)
    torch.testing.assert_close(post.double(), rpost, atol=2e-6, rtol=0)
    torch.testing.assert_close(comb.double(), rcomb, atol=2e-6, rtol=0)
    torch.testing.assert_close(col.double(), rcol, atol=2e-2, rtol=2 ** -7)


def test_merge_declared_vs_eager_reference_cpu():
    torch.manual_seed(1)
    x = (torch.randn(6, 4, 64) * 2).bfloat16()
    out = torch.randn(6, 64).bfloat16()
    post = torch.rand(6, 4) * 2
    comb = torch.softmax(torch.randn(6, 16), -1).view(6, 4, 4)
    eager = R.eager_merge(x, out, post, comb)
    declared = R.merge_declared(x, out, post, comb)
    mag = (post.double().unsqueeze(-1) * out.double().unsqueeze(-2)).abs() + (comb.double().transpose(-1, -2).abs() @ x.double().abs())
    assert ((eager.double() - declared.double()).abs() <= mag * 2 ** -7).all()
