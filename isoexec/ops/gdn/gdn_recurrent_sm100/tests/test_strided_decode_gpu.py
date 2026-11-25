"""Engine decode reading the in_proj GEMM output in place: the q|k|v window (no cat) and strided alpha/beta
(compacted inside the q/k/v split launch, not by two ATen copies) give the same output and the same carried
conv/ssm state, bit for bit, as the contiguous inputs the cat path produced. The live GEMM columns
(z, beta, alpha) are never written; a single-row window may be convolved in place (its q|k|v columns are
dead after the conv). conv_sm100 is installed by the autouse environment, as production installs it."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_recurrent_sm100.tests.test_engine_parity_gpu import _env, heads  # noqa: F401  (autouse env, geometry)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="gdn_recurrent_sm100 is proven on sm_100 only",
)

DEV, D, W = "cuda", 128, 4


@pytest.fixture(autouse=True)
def _geometry(heads, monkeypatch):
    """The parity module's ``heads`` geometry; q | k | v | z | beta | alpha, as the in_proj GEMM writes them."""
    hk, hv = heads
    for name, value in (("HK", hk), ("HV", hv), ("KD", hk * D), ("VD", hv * D),
                        ("WIDTH", 2 * hk * D + 2 * hv * D + 2 * hv)):
        monkeypatch.setattr(__import__(__name__, fromlist=[name]), name, value, raising=False)


def _pool(blocks, seed):
    from isoexec.ops.gdn.gdn_recurrent_sm100.engine_state import build_recurrent_sm100_gdn

    g = torch.Generator(device=DEV).manual_seed(seed)
    dim = 2 * KD + VD
    w = (torch.randn(dim, W, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    ssm = (torch.randn(blocks, HV, D, D, generator=g, device=DEV) * 0.1)
    conv = torch.randn(blocks, W - 1, dim, generator=g, device=DEV).to(torch.bfloat16).transpose(-1, -2)
    A_log = (torch.randn(HV, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    dt_bias = torch.randn(HV, generator=g, device=DEV).to(torch.bfloat16)
    return build_recurrent_sm100_gdn(
        max_num_seqs=blocks, chunk_size=64, ssm_state=ssm, conv_state=conv, conv_weight=w, conv_bias=None,
        A_log=A_log, dt_bias=dt_bias, num_k_heads=HK, head_k_dim=D, num_v_heads=HV, head_v_dim=D,
        activation="silu", dtype=torch.bfloat16, device=DEV,
    )


def _clone_pool(blocks, seed):
    a, b = _pool(blocks, seed), _pool(blocks, seed)
    assert bitwise_equal(a.ssm_state, b.ssm_state) and bitwise_equal(a.conv_state, b.conv_state)
    return a, b


@pytest.mark.parametrize("n", (1, 3, 64, 200))
def test_strided_decode_equals_contiguous_decode(n):
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
        if n > 1:
            assert bitwise_equal(y, before), "conv_sm100 reads a multi-row window without writing it"
