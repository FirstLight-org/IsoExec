"""The engine state machine (``engine_state.CprSm90GDN``, ``build_state_pool``): a prompt prefilled in two
chunks through the pool and then decoded token by token equals one stateless prefill over the same tokens,
bitwise -- outputs and the record the pool is left with. The conv is the identity (width-4 weight
[0, 0, 0, 1], no activation) so the claim is about this family's core and pool, not the conv pair.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_cpr_sm90 import build_state_pool, ops, refuse_incompatible_flags
from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import SM90, state_equal

pytestmark = SM90
DEV, D, HK, HV = "cuda", 128, 2, 4


@pytest.fixture(autouse=True)
def _engine_env(monkeypatch):
    monkeypatch.setenv("ISOEXEC_GDN_KERNEL", "cpr")
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_KERNELS", "1")
    monkeypatch.setenv("ISOEXEC_GDN_CHUNKED_PREFILL", "1")
    monkeypatch.delenv("ISOEXEC_GDN_NATIVE_STATE", raising=False)
    monkeypatch.delenv("ISOEXEC_GDN_CPR_APC", raising=False)
    monkeypatch.delenv("ISOEXEC_GDN_CPR_SLEEP", raising=False)


def _pool(capacity, A_log, dt_bias):
    dim = 2 * HK * D + HV * D
    w = torch.zeros(dim, 4, device=DEV, dtype=torch.bfloat16)
    w[:, 3] = 1.0   # identity conv: y = x exactly
    return build_state_pool(max_num_seqs=capacity, chunk_size=64, conv_weight=w, conv_bias=None, A_log=A_log,
                            dt_bias=dt_bias, num_k_heads=HK, head_k_dim=D, num_v_heads=HV, head_v_dim=D,
                            activation=None, dtype=torch.bfloat16, device=DEV)


def test_refuses_incompatible_flags(monkeypatch):
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_STATE", "1")
    with pytest.raises(RuntimeError, match="gdn_cpr_sm90"):
        refuse_incompatible_flags()


def test_chunked_prefill_then_decode_through_the_pool_equals_one_prefill():
    g = torch.Generator(device=DEV).manual_seed(3)
    lens, split, n_decode = [130, 45], [37, 20], 5
    T = sum(lens)
    r = lambda *s: torch.randn(*s, generator=g, device=DEV).to(torch.bfloat16)
    q, k, v, a, b = r(T, HK, D), r(T, HK, D), r(T, HV, D), r(T, HV), r(T, HV)
    A_log = torch.log(torch.empty(HV, device=DEV).uniform_(1, 16, generator=g))
    dt_bias = torch.randn(HV, device=DEV, generator=g) * 0.5
    cu = torch.tensor([0, lens[0], T], device=DEV, dtype=torch.int32)
    ref_state = ops.CprState.allocate(3, HV, DEV)
    want = ops.prefill(q, k, v, a, b, A_log, dt_bias, cu, state=ref_state,
                       slot_in=torch.tensor([1, 2], device=DEV, dtype=torch.int32))

    pool = _pool(4, A_log, dt_bias)
    x = torch.cat([q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)], dim=-1).contiguous()
    slots_cpu = [3, 1]
    slots = torch.tensor(slots_cpu, device=DEV, dtype=torch.int32)
    got = torch.empty_like(want)
    starts, taken = [0, lens[0]], [0, 0]
    for call in range(2):   # two prefill calls, the second resuming from the pool
        spans = []
        for i in range(2):
            lo = starts[i] + taken[i]
            hi = starts[i] + (split[i] if call == 0 else lens[i] - n_decode)
            spans.append((lo, hi))
            taken[i] = hi - starts[i]
        qsl = [0]
        for lo, hi in spans:
            qsl.append(qsl[-1] + hi - lo)
        idx = torch.cat([torch.arange(lo, hi) for lo, hi in spans]).to(DEV)
        o = pool.prefill(slots, slots_cpu, x[idx], a[idx], b[idx], qsl,
                         has_initial_state=[call == 1] * 2)
        got[idx] = o
    for step in range(n_decode):   # then one token at a time
        idx = torch.tensor([starts[i] + lens[i] - n_decode + step for i in range(2)], device=DEV)
        got[idx] = pool.decode(slots, x[idx].clone(), a[idx], b[idx])
    torch.cuda.synchronize()
    assert bitwise_equal(got, want)
    rows = [pool._slot2row[s] for s in slots_cpu]
    mine = ops.CprState(pool.ssm_state[torch.tensor(rows, device=DEV)].contiguous())
    ref = ops.CprState(ref_state.data[1:3].contiguous())
    assert state_equal(mine, ref, [0, 1])


def test_boundary_checkpoint_adopted_on_another_row_continues_like_the_row_that_scanned_it():
    """Prefix caching's record contract: at a multiple of 64 the boundary bytes, the conv window and ``pos`` are
    the whole state. Request A prefills 192 tokens; its boundary at 128 is captured (``capture_boundaries``)
    after a 128-token chunk. Request B, adopting that checkpoint on a fresh row (``install_boundaries``) whose
    other record bytes are garbage, prefills A's tokens 128.. and decodes; its outputs equal one stateless
    prefill over the same tokens, and its record equals A's."""
    from isoexec.ops.gdn.gdn_cpr_sm90.engine_state import capture_boundaries, install_boundaries

    g = torch.Generator(device=DEV).manual_seed(5)
    T, cut, n_decode = 200, 128, 6
    r = lambda *s: torch.randn(*s, generator=g, device=DEV).to(torch.bfloat16)
    q, k, v, a, b = r(T, HK, D), r(T, HK, D), r(T, HV, D), r(T, HV), r(T, HV)
    A_log = torch.log(torch.empty(HV, device=DEV).uniform_(1, 16, generator=g))
    dt_bias = torch.randn(HV, device=DEV, generator=g) * 0.5
    want = ops.prefill(q, k, v, a, b, A_log, dt_bias, torch.tensor([0, T], device=DEV, dtype=torch.int32))
    x = torch.cat([q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)], dim=-1).contiguous()

    pool = _pool(4, A_log, dt_bias)
    layers = [pool]
    # A: one chunk of 128 tokens into slot 7, then the checkpoint.
    pool.prefill(torch.tensor([7], device=DEV, dtype=torch.int32), [7], x[:cut], a[:cut], b[:cut], [0, cut],
                 has_initial_state=[False])
    row_a = pool._slot2row[7]
    (ckpt,) = capture_boundaries(layers, [row_a], [cut])
    # B: a fresh row filled with garbage, then the checkpoint installed on it.
    (row_b,) = pool._assign_many([9])
    pool.ssm_state[row_b].random_(0, 255, generator=None)
    install_boundaries(layers, [row_b], [cut], [ckpt])
    got = torch.empty_like(want)
    got[:cut] = want[:cut]
    hi = T - n_decode
    got[cut:hi] = pool.prefill(torch.tensor([9], device=DEV, dtype=torch.int32), [9], x[cut:hi], a[cut:hi],
                               b[cut:hi], [0, hi - cut], has_initial_state=[True])
    for t in range(hi, T):
        got[t:t + 1] = pool.decode(torch.tensor([9], device=DEV, dtype=torch.int32), x[t:t + 1].clone(),
                                   a[t:t + 1], b[t:t + 1])
    torch.cuda.synchronize()
    assert bitwise_equal(got, want)
    # A continues the same tokens on its own row: B's record is A's, byte for byte where it is meaningful.
    pool.prefill(torch.tensor([7], device=DEV, dtype=torch.int32), [7], x[cut:hi], a[cut:hi], b[cut:hi],
                 [0, hi - cut], has_initial_state=[True])
    for t in range(hi, T):
        pool.decode(torch.tensor([7], device=DEV, dtype=torch.int32), x[t:t + 1].clone(), a[t:t + 1], b[t:t + 1])
    torch.cuda.synchronize()
    both = torch.tensor([row_a, row_b], device=DEV)
    recs = ops.CprState(pool.ssm_state[both].contiguous())
    assert state_equal(ops.CprState(recs.data[1:2].contiguous()), ops.CprState(recs.data[0:1].contiguous()), [0])
    assert torch.equal(pool.conv_state[row_a], pool.conv_state[row_b])
