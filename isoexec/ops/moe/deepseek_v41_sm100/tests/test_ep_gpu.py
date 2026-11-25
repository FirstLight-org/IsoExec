"""Deterministic EP of the DeepSeek-V4.1 MoE, simulated ranks on one GPU (``ep_fixed_order.SimWorld``):
the BF16 actor's output, dx and every expert gradient are bitwise the single-rank recipe on the concatenated batch at EP = 1, 2, 4, 8 under contiguous and
scrambled expert maps (shared expert at home or owned by one rank), ragged and empty shards."""
from __future__ import annotations

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

from isoexec.ops.moe import ep_fixed_order as epf  # noqa: E402
from isoexec.ops.moe.deepseek_v41_sm100 import ep, ops, training  # noqa: E402
from isoexec.ops.moe.deepseek_v41_sm100.tests._reference import rel  # noqa: E402

DEV = "cuda"
E, K, H, I, SCALE, L = 256, 6, 512, 256, 1.5, 10.0
SHARDS = {1: [150], 2: [90, 60], 4: [37, 0, 64, 49], 8: [20, 17, 0, 31, 9, 25, 30, 18]}
KW = dict(top_k=K, routed_scale=SCALE, limit=L)


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def _maps(ep_n):
    yield "contiguous_home", epf.ExpertMap.contiguous(E, ep_n, shared="home")
    yield "permuted_owned", epf.ExpertMap.permuted(E, ep_n, seed=ep_n, shared=ep_n - 1)


@pytest.mark.parametrize("ep_n", [1, 2, 4, 8])
def test_bf16_actor_ep_is_ep1(ep_n):
    g = _gen(7 + ep_n)
    gu = (torch.randn(E + 1, 2 * I, H, generator=g, device=DEV) * 0.05).to(torch.bfloat16)
    dn = (torch.randn(E + 1, H, I, generator=g, device=DEV) * 0.05).to(torch.bfloat16)
    rw = (torch.randn(E, H, generator=g, device=DEV) * 0.05).to(torch.bfloat16)
    rb = torch.randn(E, generator=g, device=DEV) * 0.01
    counts = SHARDS[ep_n]
    X = torch.randn(sum(counts), H, generator=g, device=DEV).to(torch.bfloat16)
    dY = torch.randn(sum(counts), H, generator=g, device=DEV).to(torch.bfloat16)
    xr, gur, dnr, rwr = (t.clone().requires_grad_() for t in (X, gu, dn, rw))
    yr = training.moe_train(xr, rwr, rb, gur, dnr, **KW)
    yr.backward(dY)
    assert torch.equal(yr.detach(), ops.moe_forward(X, rw, rb, gu, dn, **KW))
    xs, dys = list(X.split(counts)), list(dY.split(counts))
    for name, emap in _maps(ep_n):
        def rank(comm, emap=emap):
            ids = emap.local_experts(comm.rank)
            sel = torch.tensor(ids, device=DEV)
            gl = gu.index_select(0, sel).contiguous().requires_grad_()
            dl = dn.index_select(0, sel).contiguous().requires_grad_()
            x = xs[comm.rank].clone().requires_grad_()
            r = rw.clone().requires_grad_()
            y = ep.ep_moe_train(x, r, rb, gl, dl, emap=emap, comm=comm, **KW)
            y.backward(dys[comm.rank])
            roll = ep.ep_moe_forward(xs[comm.rank], rw, rb, gl.detach(), dl.detach(), emap=emap, comm=comm, **KW)
            return ids, y.detach(), x.grad, gl.grad, dl.grad, r.grad, roll
        out = epf.run_sim(ep_n, rank)
        assert torch.equal(torch.cat([o[1] for o in out]), yr.detach()), name
        assert torch.equal(torch.cat([o[6] for o in out]), yr.detach()), name           # rollout == training
        assert torch.equal(torch.cat([o[2] for o in out]), xr.grad), name
        for ids, _, _, ggu, gdn, _, _ in out:
            keep = [i for i, e in enumerate(ids) if e < E or not emap.shared_home]
            sel = torch.tensor([ids[i] for i in keep], device=DEV)
            li = torch.tensor(keep, device=DEV)
            assert torch.equal(ggu.index_select(0, li), gur.grad.index_select(0, sel)), name
            assert torch.equal(gdn.index_select(0, li), dnr.grad.index_select(0, sel)), name
        for o in out:
            assert torch.equal(o[5], out[0][5]), name
            assert rel(o[5].float(), rwr.grad.float()) < 1e-2, name
            if emap.shared_home:
                assert torch.equal(o[3][-1], out[0][3][-1]), name
                assert torch.equal(o[4][-1], out[0][4][-1]), name
                assert rel(o[3][-1].float(), gur.grad[-1].float()) < 1e-2, name
                assert rel(o[4][-1].float(), dnr.grad[-1].float()) < 1e-2, name


@pytest.mark.parametrize("ep_n", [2, 4])
def test_per_stack_row_counts(ep_n):
    """Model-lane R7: one forward runs an encoder MoE on T rows and a decoder MoE on R <= T gathered rows (ragged per
    rank) through the same EP group; both layers' outputs, dx and expert gradients are the single-rank recipe's bits."""
    g = _gen(31 + ep_n)
    gu = (torch.randn(E + 1, 2 * I, H, generator=g, device=DEV) * 0.05).to(torch.bfloat16)
    dn = (torch.randn(E + 1, H, I, generator=g, device=DEV) * 0.05).to(torch.bfloat16)
    rw = (torch.randn(E, H, generator=g, device=DEV) * 0.05).to(torch.bfloat16)
    rb = torch.randn(E, generator=g, device=DEV) * 0.01
    T = [40, 9, 0, 33][:ep_n]
    keep = [[i for i in range(t) if i % 3 != 1] for t in T]          # the decoder's rows of each rank
    X = torch.randn(sum(T), H, generator=g, device=DEV).to(torch.bfloat16)
    dY = torch.randn(sum(len(k) for k in keep), H, generator=g, device=DEV).to(torch.bfloat16)
    sel_all = torch.tensor([sum(T[:r]) + i for r, k in enumerate(keep) for i in k], device=DEV, dtype=torch.long)

    xr, gur, dnr, rwr = (t.clone().requires_grad_() for t in (X, gu, dn, rw))
    a = training.moe_train(xr, rwr, rb, gur, dnr, **KW)
    yr = training.moe_train(a.index_select(0, sel_all), rwr, rb, gur, dnr, **KW)
    yr.backward(dY)
    xs = list(X.split(T))
    offs = [sum(len(k) for k in keep[:r]) for r in range(ep_n)]
    emap = epf.ExpertMap.contiguous(E, ep_n, shared="home")

    def rank(comm):
        ids = emap.local_experts(comm.rank)
        sel = torch.tensor(ids, device=DEV)
        gl = gu.index_select(0, sel).contiguous().requires_grad_()
        dl = dn.index_select(0, sel).contiguous().requires_grad_()
        x = xs[comm.rank].clone().requires_grad_()
        r = rw.clone().requires_grad_()
        a = ep.ep_moe_train(x, r, rb, gl, dl, emap=emap, comm=comm, **KW)
        loc = torch.tensor(keep[comm.rank], device=DEV, dtype=torch.long)
        y = ep.ep_moe_train(a.index_select(0, loc), r, rb, gl, dl, emap=emap, comm=comm, **KW)
        n = len(keep[comm.rank])
        y.backward(dY[offs[comm.rank]:offs[comm.rank] + n])
        return ids, y.detach(), x.grad, gl.grad, dl.grad
    out = epf.run_sim(ep_n, rank)
    assert torch.equal(torch.cat([o[1] for o in out]), yr.detach())
    assert torch.equal(torch.cat([o[2] for o in out]), xr.grad)
    for ids, _, _, ggu, gdn in out:
        own = [i for i, e in enumerate(ids) if e < E]
        s = torch.tensor([ids[i] for i in own], device=DEV)
        li = torch.tensor(own, device=DEV)
        assert torch.equal(ggu.index_select(0, li), gur.grad.index_select(0, s))
        assert torch.equal(gdn.index_select(0, li), dnr.grad.index_select(0, s))
