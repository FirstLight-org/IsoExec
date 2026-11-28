"""RMS norm with ``weight=None`` is refused on the host, before any launch.

The RMS kernels have no no-weight instance: ``launch_rms_fwd`` takes modes 1..3 and ``launch_rms_bwd`` modes
1..2, and every one reads the weight row. The public ``rms_norm`` / ``add_rms_norm`` already refused None;
the raw bindings (``raw_rms_fwd`` / ``raw_rms_bwd``) did not, and passed a null weight pointer to the kernel
(CUDA_ERROR_ILLEGAL_ADDRESS). The guard is now the first statement of ``rms_forward`` / ``rms_backward``.

These tests never launch a kernel with a null weight: each first calls with ZERO rows, where the host skips
the launch whatever the guard does (so a missing guard returns instead of faulting), and goes on to the
non-empty call only once the zero-row call has raised, i.e. once the guard is known to run before the launch.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or live_device_arch() != "sm100", reason="runs on an SM100 device only"
)

N = 1536
REFUSED = "requires a weight"


@pytest.fixture(scope="module")
def ext():
    from isoexec.ops.norms.norms_sm100._extension import extension

    return extension()


def _x(m):
    return torch.randn(m, N, device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("residual", [False, True])
def test_raw_forward_refuses_none(ext, zero_centered, residual):
    for m in (0, 4):  # m = 0 first: no launch either way
        x = _x(m)
        r = _x(m) if residual else None
        with pytest.raises(RuntimeError, match=REFUSED):
            ext.raw_rms_fwd(x, r, None, 1e-6, zero_centered)
        torch.cuda.synchronize()


@pytest.mark.parametrize("zero_centered", [False, True])
def test_raw_backward_refuses_none(ext, zero_centered):
    for m in (0, 4):
        x, dy = _x(m), _x(m)
        rstd = torch.ones(m, device="cuda", dtype=torch.float32)
        with pytest.raises(RuntimeError, match=REFUSED):
            ext.raw_rms_bwd(dy, x, rstd, None, None, zero_centered, 0)
        torch.cuda.synchronize()


def test_public_entry_points_refuse_none():
    from isoexec.ops.norms.norms_sm100 import ops

    for m in (0, 4):
        x, r = _x(m), _x(m)
        with pytest.raises(RuntimeError, match=REFUSED):
            ops.rms_norm(x, None, 1e-6, False)
        with pytest.raises(RuntimeError, match=REFUSED):
            ops.add_rms_norm(x, r, None, 1e-6, False)
        with pytest.raises(ValueError, match=REFUSED):
            ops.add_rms_norm_fused(x, r, None, 1e-6, False)
        torch.cuda.synchronize()

