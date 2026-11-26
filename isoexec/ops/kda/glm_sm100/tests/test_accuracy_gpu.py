"""Accuracy against float64 oracles: the contract's token recurrence (reference.recurrence, the equation of
GLMKimiDeltaRule(backend="reference")), the pinned upstream chunk oracle and float64 autograd for the
gradients (including through the returned final state). Tolerances are for bf16 operands with an fp32
state; measured values are recorded in README.md."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.kda.glm_sm100 import ops, reference as R
from isoexec.ops.kda.glm_sm100.tests._fixtures import assert_close, make_inputs, require_sm100

FWD_L2, FWD_MAX = 1e-2, 2.5e-2
BWD_L2, BWD_MAX = 2e-2, 5e-2

SHAPES = {
    "one_token": [1],
    "partial_chunk": [37],
    "one_chunk": [64],
    "chunk_plus_one": [65],
    "four_chunks_and_a_tail": [270],
    "packed_mixed": [130, 7, 64, 1, 200],
    "empty_sequence_inside": [50, 0, 70],
}


@pytest.fixture(autouse=True)
def _gpu():
    require_sm100()


@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("with_state", [True, False])
def test_forward_matches_float64_recurrence(name, with_state):
    q, k, v, g, b, cu, h0 = make_inputs(SHAPES[name], state=with_state)
    o, ht = ops.forward(q, k, v, g, b, cu, h0, output_final_state=True)
    ro, rh = R.recurrence(q, k, v, g, b, cu, h0)
    if o.numel():
        assert_close(o, ro, FWD_L2, FWD_MAX, "output")
    assert_close(ht, rh, FWD_L2, FWD_MAX, "final state")
    assert o.dtype == torch.bfloat16 and ht.dtype == torch.float32


@pytest.mark.parametrize("regime", ["weak_decay", "strong_decay", "bf16_gate_fp32_beta", "unbounded_decay"])
def test_forward_gate_regimes(regime):
    kw = dict(weak_decay=dict(gate_lo=-0.05), strong_decay=dict(gate_lo=-5.0),
              bf16_gate_fp32_beta=dict(g_dtype=torch.bfloat16, beta_dtype=torch.float32),
              unbounded_decay=dict(gate_lo=-60.0))[regime]
    q, k, v, g, b, cu, h0 = make_inputs([300], **kw)
    o, ht = ops.forward(q, k, v, g, b, cu, h0, output_final_state=True)
    ro, rh = R.recurrence(q, k, v, g, b, cu, h0)
    assert torch.isfinite(o.float()).all() and torch.isfinite(ht).all()
    assert_close(o, ro, FWD_L2, FWD_MAX, "output")
    assert_close(ht, rh, FWD_L2, FWD_MAX, "final state")


def test_chunk_algebra_matches_recurrence_in_float64():
    """The recipe's WY / per-channel-decay chunk algebra, evaluated without roundings, is the recurrence."""
    q, k, v, g, b, cu, h0 = make_inputs([150, 64])
    ro, rh = R.recurrence(q, k, v, g, b, cu, h0)
    co, ch = R.chunked(q, k, v, g, b, cu, h0)
    assert_close(co, ro, 1e-12, 1e-12, "chunk algebra output")
    assert_close(ch, rh, 1e-12, 1e-12, "chunk algebra state")


def test_forward_matches_the_pinned_upstream_oracle():
    from isoexec.ops.kda.glm_sm100.tests.upstream_kda import chunk_kimi_delta_attention

    B, T = 2, 96
    q, k, v, g, b, cu, h0 = make_inputs([T] * B, H=3)
    o, ht = ops.forward(q, k, v, g, b, cu, h0, output_final_state=True)
    to4 = lambda t: t.reshape(B, T, *t.shape[1:]).double()
    uo, uh = chunk_kimi_delta_attention(to4(q), to4(k), to4(v), to4(g), to4(b), initial_state=h0.double(),
                                        output_final_state=True, use_qk_l2norm_in_kernel=True)
    assert_close(o.reshape(B, T, 3, 128), uo, FWD_L2, FWD_MAX, "output vs upstream oracle")
    assert_close(ht, uh, FWD_L2, FWD_MAX, "state vs upstream oracle")


@pytest.mark.parametrize("name", ["one_token", "partial_chunk", "chunk_plus_one", "packed_mixed", "empty_sequence_inside"])
@pytest.mark.parametrize("with_state", [True, False])
def test_backward_matches_float64_autograd(name, with_state):
    lens = SHAPES[name]
    q, k, v, g, b, cu, h0 = make_inputs(lens, H=2, state=with_state)
    gen = torch.Generator(device="cuda").manual_seed(7)
    do = torch.randn(q.shape, device="cuda", generator=gen).bfloat16()
    dht = torch.randn(len(lens), 2, 128, 128, device="cuda", generator=gen) * 0.1
    grads = ops.backward(q, k, v, g, b, cu, h0, do, dht)
    ins = [t.detach().double().requires_grad_(True) for t in (q, k, v, g, b)]
    hin = h0.detach().double().requires_grad_(True) if h0 is not None else None
    ro, rh = R.recurrence(*ins, cu, hin)
    wrt = ins + ([hin] if hin is not None else [])
    ref = torch.autograd.grad((ro, rh), wrt, (do.double(), dht.double()), allow_unused=True)
    names = ("dq", "dk", "dv", "dg", "dbeta", "dh0")
    for n, got, want in zip(names, grads, ref):
        if want is None or want.numel() == 0:
            continue
        # dg is exactly zero when no decayed state reaches an output (one token from the zero state); it
        # is then a cancellation of fp32 terms of dk's scale, which floors its denominator.
        assert_close(got, want, BWD_L2, BWD_MAX, n, scale=ref[1] if n == "dg" else None)


def test_backward_through_the_final_state_only():
    """Gradients that reach the inputs only through the returned final state (dO = 0)."""
    q, k, v, g, b, cu, h0 = make_inputs([100], H=2)
    do = torch.zeros(q.shape, device="cuda", dtype=torch.bfloat16)
    dht = torch.randn(1, 2, 128, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5)) * 0.1
    grads = ops.backward(q, k, v, g, b, cu, h0, do, dht)
    ins = [t.detach().double().requires_grad_(True) for t in (q, k, v, g, b, h0)]
    _, rh = R.recurrence(*ins[:5], cu, ins[5])
    ref = torch.autograd.grad(rh, ins, dht.double(), allow_unused=True)
    for n, got, want in zip(("dq", "dk", "dv", "dg", "dbeta", "dh0"), grads, ref):
        if n == "dq":
            assert want is None and got.abs().max().item() == 0.0, "q does not reach the state"
            continue
        assert_close(got, want, BWD_L2, BWD_MAX, n)


@pytest.mark.parametrize("g_dtype,beta_dtype", [(torch.bfloat16, torch.float32), (torch.float32, torch.float32)])
def test_backward_input_dtypes(g_dtype, beta_dtype):
    q, k, v, g, b, cu, h0 = make_inputs([90, 40], H=2, g_dtype=g_dtype, beta_dtype=beta_dtype)
    gen = torch.Generator(device="cuda").manual_seed(3)
    do = torch.randn(q.shape, device="cuda", generator=gen).bfloat16()
    grads = ops.backward(q, k, v, g, b, cu, h0, do, None)
    assert grads[3].dtype == g_dtype and grads[4].dtype == beta_dtype
    ins = [t.detach().double().requires_grad_(True) for t in (q, k, v, g, b, h0)]
    ro, _ = R.recurrence(*ins[:5], cu, ins[5])
    ref = torch.autograd.grad(ro, ins, do.double())
    for n, got, want in zip(("dq", "dk", "dv", "dg", "dbeta", "dh0"), grads, ref):
        assert_close(got, want, BWD_L2, BWD_MAX, n)
