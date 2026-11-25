"""Engine decode reading the in_proj GEMM output in place: the q|k|v window (no cat) and strided alpha/beta
(compacted inside the q/k/v split launch, not by two ATen copies) give the same output and the same carried
conv/ssm state, bit for bit, as the contiguous inputs the cat path produced. The live GEMM columns
(z, beta, alpha) are never written; a single-row window may be convolved in place (its q|k|v columns are
dead after the conv)."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_recurrent_sm90.tests.test_engine_parity_gpu import _env  # noqa: F401  (autouse env)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the sm90 GDN family is sm_90a only",
)

DEV, D, HK, HV, W = "cuda", 128, 4, 12, 4
KD, VD = HK * D, HV * D
WIDTH = 2 * KD + 2 * VD + 2 * HV  # q | k | v | z | beta | alpha, as the in_proj GEMM writes them


def _pool(blocks, seed):
    from isoexec.ops.gdn.gdn_recurrent_sm90.engine_state import build_recurrent_sm90_gdn

    g = torch.Generator(device=DEV).manual_seed(seed)
    dim = 2 * KD + VD
    w = (torch.randn(dim, W, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    ssm = (torch.randn(blocks, HV, D, D, generator=g, device=DEV) * 0.1)
    conv = torch.randn(blocks, W - 1, dim, generator=g, device=DEV).to(torch.bfloat16).transpose(-1, -2)
    A_log = (torch.randn(HV, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    dt_bias = torch.randn(HV, generator=g, device=DEV).to(torch.bfloat16)
    return build_recurrent_sm90_gdn(
        max_num_seqs=blocks, chunk_size=64, ssm_state=ssm, conv_state=conv, conv_weight=w, conv_bias=None,
        A_log=A_log, dt_bias=dt_bias, num_k_heads=HK, head_k_dim=D, num_v_heads=HV, head_v_dim=D,
        activation="silu", dtype=torch.bfloat16, device=DEV,
    )


def _clone_pool(blocks, seed):
    a, b = _pool(blocks, seed), _pool(blocks, seed)
    assert bitwise_equal(a.ssm_state, b.ssm_state) and bitwise_equal(a.conv_state, b.conv_state)
    return a, b


@pytest.fixture(params=["conv_sm90", "vendored"])
def conv_provider(request):
    """Production installs the conv_sm90 drop-in (``install_arch_impls``); the vendored Triton update is
    the fallback. Both must agree with the contiguous path; only conv_sm90 promises not to write x."""
    from isoexec.ops.gdn.conv_sm90 import integration

    if request.param == "conv_sm90":
        integration.install()
        yield request.param
        integration.uninstall()
    else:
        integration.uninstall()
        yield request.param


@pytest.mark.parametrize("n", (1, 3, 64, 200))
def test_strided_decode_equals_contiguous_decode(n, conv_provider):
    from isoexec.runtimes.vllm import gdn_engine_patch as P
    from isoexec.runtimes.vllm._gdn import _adjacent_qkv

    blocks = n + 8
    ref, got = _clone_pool(blocks, seed=n)
    assert P.__name__ and got._native_kernels
    g = torch.Generator(device=DEV).manual_seed(100 + n)
    slots = (torch.randperm(blocks - 1, generator=torch.Generator().manual_seed(n))[:n] + 1).to(torch.int32).to(DEV)
    for step in range(3):
        y = (torch.randn(n, WIDTH, generator=g, device=DEV) * 2).to(torch.bfloat16)  # the GEMM output
        q = y[:, :KD].unflatten(-1, (HK, D))
        k = y[:, KD:2 * KD].unflatten(-1, (HK, D))
        v = y[:, 2 * KD:2 * KD + VD].unflatten(-1, (HV, D))
        a = y[:, 2 * KD + 2 * VD:2 * KD + 2 * VD + HV]
        b = y[:, 2 * KD + 2 * VD + HV:]
        before = y.clone()
        with torch.no_grad():
            mixed = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
            want = ref.decode(slots, mixed, a.contiguous(), b.contiguous())  # the previous engine path
            window = _adjacent_qkv(q, k, v)
            assert window is not None and window.data_ptr() == y.data_ptr()
            out = got.decode(slots, window, a, b)  # gdn_layer_core with defer_ab: strided a/b
        torch.cuda.synchronize()
        assert bitwise_equal(out, want), f"step {step} output"
        assert bitwise_equal(got.ssm_state, ref.ssm_state), f"step {step} ssm state"
        assert bitwise_equal(got.conv_state, ref.conv_state), f"step {step} conv state"
        live = 2 * KD + VD  # z, beta, alpha: still read after the conv
        assert bitwise_equal(y[:, live:], before[:, live:]), "the live GEMM columns are never written"
        if conv_provider == "conv_sm90" and n > 1:
            assert bitwise_equal(y, before), "conv_sm90 reads a multi-row window without writing it"
