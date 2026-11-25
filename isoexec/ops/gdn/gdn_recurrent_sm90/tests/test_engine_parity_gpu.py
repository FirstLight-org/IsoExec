"""The SM90 native recurrent GDN core: engine prefill/decode on vLLM-layout mamba pages == trainer forward.

Identity convolution isolates the GDN core and state wiring; the conv equivalence is its own claim.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the sm90 GDN family is sm_90a only",
)

DEV = "cuda"
D = 128
HK, HV = 4, 12  # one TP4 rank of Qwen3.8-27B's GDN (16 k-heads / 48 v-heads over TP4)
W = 4


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from isoexec.ops import _dispatch

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    monkeypatch.setenv("ISOEXEC_GDN_KERNEL", "recurrent")
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_KERNELS", "1")
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_STATE", "1")
    monkeypatch.setenv("ISOEXEC_GDN_CHUNKED_PREFILL", "1")
    monkeypatch.setenv("ISOEXEC_GDN_CPR_MIN_PAGES", "0")
    monkeypatch.delenv("ISOEXEC_GDN_CPR_APC", raising=False)


def _inputs(lens, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    T = sum(lens)
    r = lambda *s: torch.randn(*s, generator=g, device=DEV, dtype=torch.float32).to(torch.bfloat16)  # noqa: E731
    q, k, v = r(T, HK, D), r(T, HK, D), r(T, HV, D)
    a, b = r(T, HV), r(T, HV)
    A_log = (torch.randn(HV, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    dt_bias = torch.randn(HV, generator=g, device=DEV).to(torch.bfloat16)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=DEV)
    return q, k, v, a, b, A_log, dt_bias, cu


def _packed(q, k, v):
    T = q.shape[0]
    return torch.cat([q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)], dim=-1).contiguous()


def _pool(blocks, dim, A_log, dt_bias):
    from isoexec.ops.gdn.gdn_recurrent_sm90.engine_state import build_recurrent_sm90_gdn

    w = torch.zeros(dim, W, device=DEV, dtype=torch.bfloat16)
    w[:, W - 1] = 1.0
    ssm = torch.zeros(blocks, HV, D, D, device=DEV, dtype=torch.float32)  # vLLM MambaSpec (HV, V, K) fp32
    conv = torch.zeros(blocks, W - 1, dim, device=DEV, dtype=torch.bfloat16).transpose(-1, -2)  # (D, W-1) view
    return build_recurrent_sm90_gdn(
        max_num_seqs=blocks, chunk_size=64, ssm_state=ssm, conv_state=conv, conv_weight=w, conv_bias=None,
        A_log=A_log, dt_bias=dt_bias, num_k_heads=HK, head_k_dim=D, num_v_heads=HV, head_v_dim=D,
        activation=None, dtype=torch.bfloat16, device=DEV,
    )


def _trainer(d):
    from isoexec.ops.gdn.gdn_recurrent_sm90.integration import gdn_recurrent_sm90_core

    q, k, v, a, b, A_log, dt_bias, cu = d
    with torch.no_grad():
        return gdn_recurrent_sm90_core(q, k, v, a, b, A_log, dt_bias, cu)


@pytest.mark.parametrize("split", [None, [37, 20]])
def test_engine_prefill_then_decode_equals_trainer_bitwise(split):
    lens, n_decode = [130, 45], 9
    d = _inputs(lens, seed=3)
    q, k, v, a, b, A_log, dt_bias, _ = d
    want = _trainer(d)
    x = _packed(q, k, v)
    pool = _pool(16, x.shape[1], A_log, dt_bias)
    slots_cpu = [5, 11]
    slots = torch.tensor(slots_cpu, dtype=torch.int32, device=DEV)
    starts = [0, lens[0]]
    out = torch.full_like(want, float("nan"))
    bounds = [[0, lens[i] - n_decode] for i in range(2)] if split is None else \
        [[0, split[i], lens[i] - n_decode] for i in range(2)]
    for call in range(len(bounds[0]) - 1):
        spans = [(starts[i] + bounds[i][call], starts[i] + bounds[i][call + 1]) for i in range(2)]
        qsl = [0]
        for lo, hi in spans:
            qsl.append(qsl[-1] + hi - lo)
        cat = lambda t: torch.cat([t[lo:hi] for lo, hi in spans]).contiguous()  # noqa: E731
        cu = torch.tensor(qsl, dtype=torch.int32, device=DEV)
        with torch.no_grad():
            o = pool.prefill(slots, slots_cpu, cat(x), cat(a), cat(b), qsl,
                             has_initial_state=[call > 0] * 2, prefill_query_start_loc=cu)
        for i, (lo, hi) in enumerate(spans):
            out[lo:hi] = o[qsl[i]:qsl[i + 1]]
    for step in range(n_decode):
        t = torch.tensor([starts[i] + lens[i] - n_decode + step for i in range(2)], device=DEV)
        with torch.no_grad():
            o = pool.decode(slots, x[t].clone(), a[t].contiguous(), b[t].contiguous())
        out[t] = o
    pre = torch.cat([torch.arange(starts[i], starts[i] + lens[i] - n_decode) for i in range(2)]).to(DEV)
    dec = torch.cat([torch.arange(starts[i] + lens[i] - n_decode, starts[i] + lens[i]) for i in range(2)]).to(DEV)
    assert bitwise_equal(out[pre], want[pre]), "prefill tokens"
    assert bitwise_equal(out[dec], want[dec]), "decode tokens"
