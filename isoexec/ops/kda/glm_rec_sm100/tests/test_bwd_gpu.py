"""Trainer backward of sm100_kda_rec_v1: float64 gradients, run-to-run determinism, batch independence, SASS."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops.kda.glm_rec_sm100 import ops
from isoexec.ops.kda.tests._fixtures import bits_equal, cu_of, make_inputs, rel_l2, require_sm100


@pytest.fixture(autouse=True)
def _gpu():
    require_sm100()


def _f64_grads(q, k, v, g, b, cu, h0, do, dht):
    from isoexec.ops.kda.glm_sm100 import reference as R
    leaves = [t.detach().double().requires_grad_(True) for t in (q, k, v, g, b)]
    h = None if h0 is None else h0.detach().double().transpose(-1, -2).contiguous().requires_grad_(True)
    o, ht = R.recurrence(*leaves, cu, h)
    loss = (o * do.double()).sum()
    if dht is not None:
        loss = loss + (ht * dht.double().transpose(-1, -2)).sum()
    ins = leaves + ([h] if h is not None else [])
    grads = torch.autograd.grad(loss, ins, allow_unused=True)
    return grads


@pytest.mark.parametrize("lens,state,with_dht", [([130, 64, 1, 200, 65], True, True), ([300], False, False), ([7, 63], True, False)])
def test_backward_against_float64_gradients(lens, state, with_dht):
    H = 3
    q, k, v, g, b, cu, h0 = make_inputs(lens, H=H, seed=21, state=state)
    gen = torch.Generator(device="cuda").manual_seed(33)
    do = torch.randn(sum(lens), H, 128, device="cuda", generator=gen).bfloat16()
    dht = torch.randn(len(lens), H, 128, 128, device="cuda", generator=gen) * 0.05 if with_dht else None
    o, ht, ckpt, base = ops.forward_train(q, k, v, g, b, cu, h0)
    o_ref, ht_ref = ops.forward(q, k, v, g, b, cu, h0)
    bits_equal(o, o_ref, "forward_train o == forward o")
    bits_equal(ht, ht_ref, "forward_train state == forward state")
    dq, dk, dv, dg, db, dh0 = ops.backward(q, k, v, g, b, cu, ckpt, base, do, dht, need_dh0=state)
    want = _f64_grads(q, k, v, g, b, cu, h0, do, dht)
    for name, got, ref in (("dq", dq, want[0]), ("dk", dk, want[1]), ("dv", dv, want[2]), ("dg", dg, want[3]), ("dbeta", db, want[4])):
        assert rel_l2(got, ref) < 2e-2, (name, rel_l2(got, ref))
    if state:
        assert rel_l2(dh0, want[5].transpose(-1, -2)) < 2e-2, rel_l2(dh0, want[5].transpose(-1, -2))


def test_backward_is_run_to_run_bitwise_and_independent_of_the_batch_for_each_sequence_alone():
    lens = [130, 70]
    H = 2
    q, k, v, g, b, cu, h0 = make_inputs(lens, H=H, seed=5)
    do = torch.randn(sum(lens), H, 128, device="cuda").bfloat16()
    o, ht, ckpt, base = ops.forward_train(q, k, v, g, b, cu, h0)
    r1 = ops.backward(q, k, v, g, b, cu, ckpt, base, do, None, need_dh0=True)
    r2 = ops.backward(q, k, v, g, b, cu, ckpt, base, do, None, need_dh0=True, grid=1)   # different CTA schedule
    for a, c in zip(r1, r2):
        bits_equal(a, c, "run to run / grid")
    # sequence 1 alone == sequence 1 inside the batch
    s = slice(130, 200)
    o1, ht1, ck1, b1 = ops.forward_train(q[s].contiguous(), k[s].contiguous(), v[s].contiguous(), g[s].contiguous(), b[s].contiguous(), cu_of([70]), h0[1:2])
    r3 = ops.backward(q[s].contiguous(), k[s].contiguous(), v[s].contiguous(), g[s].contiguous(), b[s].contiguous(), cu_of([70]), ck1, b1, do[s].contiguous(), None, need_dh0=True)
    for a, c in zip(r1[:5], r3[:5]):
        bits_equal(a[s], c, "alone vs batch")
    bits_equal(r1[5][1:2], r3[5], "dh0 alone vs batch")


def test_autograd_core_matches_explicit_backward():
    lens = [100, 33]
    q, k, v, g, b, cu, h0 = make_inputs(lens, H=2, seed=6)
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v, g, b)]
    o, ht = ops.core(*leaves, cu, h0)
    do = torch.randn_like(o)
    (o * do).sum().backward() if False else torch.autograd.backward([o], [do])
    _, _, ckpt, base = ops.forward_train(q, k, v, g, b, cu, h0)
    want = ops.backward(q, k, v, g, b, cu, ckpt, base, do, None, need_dh0=False)
    for got, ref in zip([t.grad for t in leaves], want[:5]):
        bits_equal(got, ref, "autograd")
