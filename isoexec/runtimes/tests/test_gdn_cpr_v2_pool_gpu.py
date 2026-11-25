"""The CPR pool under Model Runner V2's engine contract, on the live arch's CPR package (sm90 or sm100).

* Padding lanes: warmup and CUDA-graph capture dummies prefill every lane on vLLM's null block. A lane on
  the pool's ``null_slot`` resolves to the null row, claims, stamps and evicts nothing, and writes no
  record; real lanes beside it keep the bits they have alone.
* Stack claim: ``assign_prefill_rows`` claims a step's prefill rows in every pool before the forward; each
  layer's prefill then reads the shared plan. Outputs, the state carried into the next chunk, and the
  rows equal those of each layer claiming its own rows.

Random weights; no checkpoint.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from isoexec.ops._testing import bitwise_equal

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((9, 0), (10, 0)),
    reason="requires an SM90 or SM100 GPU",
)

HK, HV, D, NULL = 2, 4, 128, 0


@pytest.fixture
def cpr(monkeypatch):
    from isoexec.ops import _dispatch, _resolve
    from isoexec.ops.gdn import CPR_LAYERS

    monkeypatch.setenv("ISOEXEC_GDN_KERNEL", "cpr")
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_KERNELS", "1")
    monkeypatch.setenv("ISOEXEC_GDN_CHUNKED_PREFILL", "1")
    for flag in ("NATIVE_STATE", "CPR_APC", "CPR_SLEEP"):
        monkeypatch.delenv(f"ISOEXEC_GDN_{flag}", raising=False)
    _resolve._registry.cache_clear()
    assert _dispatch.selected("gdn").startswith("gdn.gdn_cpr_sm")
    conv = importlib.import_module(f"isoexec.ops.{_dispatch.selected('conv')}.integration")
    installed = bool(conv._INSTALLED)
    conv.install()
    count = len(CPR_LAYERS)
    try:
        yield
    finally:
        del CPR_LAYERS[count:]
        if not installed:
            conv.uninstall()
        _resolve._registry.cache_clear()


def _weights(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn((2 * HK + HV) * D, 4, generator=g, device="cuda") * 0.4).bfloat16()
    return w, torch.full((HV,), -4.0, device="cuda"), torch.full((HV,), -1.0, device="cuda")


def _pool(weights, capacity=6):
    from isoexec.ops.gdn import build_state_pool

    w, A, dt = weights
    pool = build_state_pool(max_num_seqs=capacity, chunk_size=64, conv_weight=w, conv_bias=None, A_log=A,
                            dt_bias=dt, num_k_heads=HK, head_k_dim=D, num_v_heads=HV, head_v_dim=D,
                            activation="silu", dtype=torch.bfloat16, device="cuda")
    pool.null_slot = NULL
    pool.note_slot_positions(None)  # requests are lifecycle-owned, as under the engine driver
    return pool


def _tokens(lengths, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lengths)

    def rand(*shape):
        return (torch.randn(*shape, generator=g, device="cuda") * 0.4).bfloat16()

    return rand(T, (2 * HK + HV) * D), rand(T, HV), rand(T, HV)


def _cu(lengths):
    return [0, *torch.tensor(lengths).cumsum(0).tolist()]


def _prefill(pool, slots, lengths, inputs, has):
    x, a, b = inputs
    dev_slots = torch.tensor(slots, device="cuda", dtype=torch.int32)
    return pool.prefill(dev_slots, slots, x, a, b, _cu(lengths), has_initial_state=has)


def _decode(pool, slots, seed):
    x, a, b = _tokens([len(slots)], seed)
    return pool.decode(torch.tensor(slots, device="cuda", dtype=torch.int32), x, a, b)


def test_padding_lanes_take_the_null_row_and_leave_real_lanes_alone(cpr):
    weights = _weights(3)
    padded, plain = _pool(weights), _pool(weights)
    lengths = [70, 33, 129, 5]
    x, a, b = _tokens(lengths, seed=11)
    cu = _cu(lengths)
    real = [0, 2]  # lanes 1 and 3 are padding, on the null block
    got = _prefill(padded, [7, NULL, 9, NULL], lengths, (x, a, b), [False] * 4)
    idx = torch.cat([torch.arange(cu[i], cu[i + 1]) for i in real]).cuda()
    want = _prefill(plain, [7, 9], [lengths[i] for i in real], (x[idx], a[idx], b[idx]), [False, False])
    assert bitwise_equal(got[idx], want)
    assert set(padded._slot2row) == {7, 9} and len(padded._free) == padded.capacity - 2
    # The real rows carried the same state: the next decode step agrees too, and padding decodes inertly.
    assert bitwise_equal(_decode(padded, [7, 9], seed=5), _decode(plain, [7, 9], seed=5))
    assert set(padded._slot2row) == {7, 9}


def test_a_stack_claim_gives_every_layer_its_own_claims_bits(cpr):
    from isoexec.ops.gdn import assign_prefill_rows

    layers = [_weights(s) for s in (21, 22)]
    shared = [_pool(w) for w in layers]
    own = [_pool(w) for w in layers]
    # Step 1: three fresh prompts; step 2: two continue (chunked prefill) beside a fresh one.
    steps = [([4, 8, 15], [96, 41, 130], [False] * 3), ([8, 16, 4], [64, 77, 23], [True, False, True])]
    for n, (slots, lengths, has) in enumerate(steps):
        inputs = _tokens(lengths, seed=100 + n)
        assign_prefill_rows(shared, slots)
        for s, o in zip(shared, own):
            assert s._prefill_plan is not None
            got = _prefill(s, slots, lengths, inputs, has)
            want = _prefill(o, slots, lengths, inputs, has)
            assert bitwise_equal(got, want)
            assert s._slot2row == o._slot2row
            assert bitwise_equal(s.last_used, o.last_used)
    assign_prefill_rows(shared, ())  # a step without prefill retires the last plan
    assert all(s._prefill_plan is None for s in shared)
    for s, o in zip(shared, own):
        assert bitwise_equal(_decode(s, [4, 8, 15, 16], seed=9), _decode(o, [4, 8, 15, 16], seed=9))
