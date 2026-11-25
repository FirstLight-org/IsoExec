"""Contiguous context parallelism at CP = 2, 3, 4, 8, simulated rank by rank in one process.

The CP wrappers run with :class:`MailboxTransport` over the plain-torch reference cores, whose per-row
(attention) and per-token (GDN) arithmetic does not depend on how rows are grouped into calls, just
like the production kernels' contract. Claims:

* forward: CP=k outputs equal CP=1 outputs bit for bit (attention, conv + GDN core);
* backward: CP=k gradients match CP=1 to fp32 rounding and are bitwise deterministic run to run;
* recompute: a record pass plus a replay pass (no forward messages) gives the plain-mode gradients.
"""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F

from isoexec.ops.attention.reference import ref_attention_core_qk
from isoexec.ops.collectives.cp_attention import make_cp_attention_core
from isoexec.ops.collectives.cp_contiguous import ContiguousPlan, register_plan
from isoexec.ops.collectives.cp_transport import MailboxTransport, cp_region
from isoexec.ops.gdn.cp import make_cp_conv, make_cp_gdn_core
from isoexec.ops.gdn.reference import ref_gdn_core_stateful

HQ, HK, D = 4, 2, 16
GK, GV, DK, DV, WIDTH = 2, 4, 8, 8, 4


def _cu(cp: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    lengths = [cp * rng.randint(1, 6) for _ in range(rng.randint(2, 5))]
    lengths[rng.randrange(len(lengths))] = cp * rng.randint(6, 12)  # spans several ranks
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return cu


def tap_conv(x, w, cu_seqlens):
    """Causal depthwise conv + SiLU with one fp32 multiply and add per tap in ascending tap order: an
    output's arithmetic depends only on its own taps, never on the row count of the call."""
    width, rows = w.shape[-1], x.shape[0]
    cu = cu_seqlens.tolist()
    starts = torch.zeros(rows, dtype=torch.long)
    for a, b in zip(cu, cu[1:]):
        starts[a:b] = a
    pos = torch.arange(rows)
    acc = torch.zeros(rows, x.shape[1], dtype=torch.float32)
    for tap in range(width):
        src = pos - (width - 1 - tap)
        valid = (src >= starts).unsqueeze(1)
        acc = acc + (x[src.clamp(min=0)].float() * valid) * w[:, 0, tap].float()
    # SiLU one row at a time: CPU picks vector/remainder paths by element count.
    return torch.stack([F.silu(row) for row in acc]).to(x.dtype) if rows else acc.to(x.dtype)


class World:
    """``cp`` simulated ranks of one microbatch, sharing a mailbox."""

    def __init__(self, cu, cp):
        self.cp, self.box = cp, {}
        self.plans = [ContiguousPlan.build(cu, cp, rank) for rank in range(cp)]
        self.cus = []
        for plan in self.plans:
            local = torch.tensor(plan.local_cu, dtype=torch.int32)
            register_plan(local, plan)
            self.cus.append(local)

    def transport(self, rank):
        t = MailboxTransport(self.box, self.cp, rank)
        return lambda: t

    def rows(self, rank):
        plan = self.plans[rank]
        return slice(plan.lo, plan.lo + plan.local)


def _attention(world: World, q, k, v, dout, *, mode=None):
    cores = [make_cp_attention_core(ref_attention_core_qk, transport_factory=world.transport(r), cp=world.cp,
                                    rank=r, site="layer.attn") for r in range(world.cp)]
    leaves = [[x[world.rows(r)].detach().clone().requires_grad_(True) for x in (q, k, v)] for r in range(world.cp)]

    def forward(r):
        ql, kl, vl = leaves[r]
        return cores[r](ql, kl, vl, world.cus[r], world.plans[r].max_q, D ** -0.5)

    if mode == "recompute":
        with torch.no_grad(), cp_region("record"):
            first = [forward(r) for r in range(world.cp)]
        with cp_region("replay"):
            outs = [forward(r) for r in reversed(range(world.cp))][::-1]  # no forward messages: any order
        assert all(torch.equal(a, b) for a, b in zip(first, outs))
    else:
        outs = [forward(r) for r in range(world.cp)]
    for r in reversed(range(world.cp)):
        torch.autograd.backward(outs[r], dout[world.rows(r)])
    grads = [torch.cat([leaves[r][i].grad for r in range(world.cp)]) for i in range(3)]
    return torch.cat([o.detach() for o in outs]), grads


def _gdn(world: World, x, a, b, w, A_log, dt_bias, dout, *, mode=None):
    convs = [make_cp_conv(tap_conv, transport_factory=world.transport(r), cp=world.cp, rank=r, site="layer.gdn.conv")
             for r in range(world.cp)]
    cores = [make_cp_gdn_core(ref_gdn_core_stateful, transport_factory=world.transport(r), cp=world.cp, rank=r,
                              site="layer.gdn.core") for r in range(world.cp)]
    leaves = [[t[world.rows(r)].detach().clone().requires_grad_(True) for t in (x, a, b)] for r in range(world.cp)]
    params = [w.detach().clone().requires_grad_(True), A_log.detach().clone().requires_grad_(True),
              dt_bias.detach().clone().requires_grad_(True)]
    kd = GK * DK

    def forward(r):
        xl, al, bl = leaves[r]
        y = convs[r](xl, params[0], world.cus[r])
        q = y[:, :kd].unflatten(-1, (GK, DK))
        k = y[:, kd:2 * kd].unflatten(-1, (GK, DK))
        v = y[:, 2 * kd:].unflatten(-1, (GV, DV))
        return cores[r](q, k, v, al, bl, params[1], params[2], world.cus[r])

    if mode == "recompute":
        with torch.no_grad(), cp_region("record"):
            first = [forward(r) for r in range(world.cp)]
        with cp_region("replay"):
            outs = [forward(r) for r in reversed(range(world.cp))][::-1]
        assert all(torch.equal(a_, b_) for a_, b_ in zip(first, outs))
    else:
        outs = [forward(r) for r in range(world.cp)]
    for r in reversed(range(world.cp)):
        torch.autograd.backward(outs[r], dout[world.rows(r)])
    grads = [torch.cat([leaves[r][i].grad for r in range(world.cp)]) for i in range(3)] + [p.grad for p in params]
    return torch.cat([o.detach() for o in outs]), grads


def _attention_inputs(total, seed):
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(total, h, D, generator=g) for h in (HQ, HK, HK))
    return q, k, v, torch.randn(total, HQ, D, generator=g)


def _gdn_inputs(total, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(total, 2 * GK * DK + GV * DV, generator=g)
    a, b = torch.randn(total, GV, generator=g), torch.randn(total, GV, generator=g)
    w = torch.randn(x.shape[1], 1, WIDTH, generator=g) * 0.5
    A_log, dt_bias = torch.randn(GV, generator=g) * 0.3, torch.randn(GV, generator=g) * 0.3
    return x, a, b, w, A_log, dt_bias, torch.randn(total, GV, DV, generator=g)


def _close(actual, expected):
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("cp", [2, 3, 4, 8])
def test_attention_forward_bitwise_and_backward_sound(cp):
    cu = _cu(cp, seed=cp)
    q, k, v, dout = _attention_inputs(cu[-1], seed=cp)
    ref_out, ref_grads = _attention(World(cu, 1), q, k, v, dout)
    out, grads = _attention(World(cu, cp), q, k, v, dout)
    assert torch.equal(out, ref_out)
    _close(grads, ref_grads)
    out2, grads2 = _attention(World(cu, cp), q, k, v, dout)
    assert torch.equal(out2, out) and all(torch.equal(x, y) for x, y in zip(grads2, grads))


@pytest.mark.parametrize("cp", [2, 3, 4, 8])
def test_gdn_conv_and_state_forward_bitwise_and_backward_sound(cp):
    cu = _cu(cp, seed=100 + cp)
    inputs = _gdn_inputs(cu[-1], seed=cp)
    ref_out, ref_grads = _gdn(World(cu, 1), *inputs)
    out, grads = _gdn(World(cu, cp), *inputs)
    assert torch.equal(out, ref_out)
    _close(grads, ref_grads)
    out2, grads2 = _gdn(World(cu, cp), *inputs)
    assert torch.equal(out2, out) and all(torch.equal(x, y) for x, y in zip(grads2, grads))


def test_short_pieces_pass_halo_and_state_through_a_rank():
    """cp=8 over one 16-row sequence: every rank holds 2 rows (< w-1 = 3), so conv tails pass through."""
    cu, cp = [0, 16], 8
    inputs = _gdn_inputs(cu[-1], seed=7)
    ref_out, ref_grads = _gdn(World(cu, 1), *inputs)
    out, grads = _gdn(World(cu, cp), *inputs)
    assert torch.equal(out, ref_out)
    _close(grads, ref_grads)
    q, k, v, dout = _attention_inputs(cu[-1], seed=7)
    ref_out, _ = _attention(World(cu, 1), q, k, v, dout)
    out, _ = _attention(World(cu, cp), q, k, v, dout)
    assert torch.equal(out, ref_out)


@pytest.mark.parametrize("cp", [2, 4])
def test_record_replay_recompute_matches_plain_mode(cp):
    cu = _cu(cp, seed=50 + cp)
    q, k, v, dout = _attention_inputs(cu[-1], seed=cp)
    out, grads = _attention(World(cu, cp), q, k, v, dout)
    out_r, grads_r = _attention(World(cu, cp), q, k, v, dout, mode="recompute")
    assert torch.equal(out, out_r) and all(torch.equal(x, y) for x, y in zip(grads, grads_r))
    inputs = _gdn_inputs(cu[-1], seed=cp)
    out, grads = _gdn(World(cu, cp), *inputs)
    out_r, grads_r = _gdn(World(cu, cp), *inputs, mode="recompute")
    assert torch.equal(out, out_r) and all(torch.equal(x, y) for x, y in zip(grads, grads_r))


def test_missing_plan_refuses():
    from isoexec.ops.collectives.cp_contiguous import CpPlanError

    core = make_cp_attention_core(ref_attention_core_qk, transport_factory=lambda: None, cp=2, rank=0, site="s")
    q = torch.zeros(4, HQ, D)
    with pytest.raises(CpPlanError, match="no contiguous CP plan"):
        core(q, q[:, :HK], q[:, :HK], torch.tensor([0, 4], dtype=torch.int32), 4, 1.0)
