"""Token-disjoint EP training over real NCCL ranks (``DistComm``; the cross-host realization, here on 2 GPUs): ragged
per-rank token counts, an encoder layer on T rows then a decoder layer on R gathered rows (model-lane R7 / R8); outputs,
dx and owned experts' gradients are the single-rank recipe's bits on the concatenated batch, the replicated router and
shared-expert gradients are identical on every rank. Skipped without a two-GPU lease."""
from __future__ import annotations

import os

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="needs a leased SM100 GPU")

E, K, H, I = 256, 6, 512, 256
KW = dict(top_k=K, routed_scale=1.5, limit=10.0)
T = [37, 64]


def _data():
    g = torch.Generator(device="cuda").manual_seed(5)
    gu = (torch.randn(E + 1, 2 * I, H, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    dn = (torch.randn(E + 1, H, I, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    rw = (torch.randn(E, H, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    rb = torch.randn(E, generator=g, device="cuda") * 0.01
    X = torch.randn(sum(T), H, generator=g, device="cuda").to(torch.bfloat16)
    keep = [[i for i in range(t) if i % 4 != 2] for t in T]
    dY = torch.randn(sum(len(k) for k in keep), H, generator=g, device="cuda").to(torch.bfloat16)
    return gu, dn, rw, rb, X, keep, dY


def _rank(rank, world, port, out):
    import torch.distributed as dist
    from isoexec.ops.moe import ep_fixed_order as epf
    from isoexec.ops.moe.deepseek_v41_sm100 import ep, training
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    gu, dn, rw, rb, X, keep, dY = _data()
    sel_all = torch.tensor([sum(T[:r]) + i for r, k in enumerate(keep) for i in k], device="cuda", dtype=torch.long)
    # the single-rank recipe on the concatenated batch (every rank computes it)
    xr, gur, dnr, rwr = (t.clone().requires_grad_() for t in (X, gu, dn, rw))
    a = training.moe_train(xr, rwr, rb, gur, dnr, **KW)
    yr = training.moe_train(a.index_select(0, sel_all), rwr, rb, gur, dnr, **KW)
    yr.backward(dY)
    emap = epf.ExpertMap.contiguous(E, world, shared="home")
    comm = epf.DistComm()
    ids = emap.local_experts(rank)
    sel = torch.tensor(ids, device="cuda")
    gl = gu.index_select(0, sel).contiguous().requires_grad_()
    dl = dn.index_select(0, sel).contiguous().requires_grad_()
    x = X.split(T)[rank].clone().requires_grad_()
    r = rw.clone().requires_grad_()
    a = ep.ep_moe_train(x, r, rb, gl, dl, emap=emap, comm=comm, **KW)
    loc = torch.tensor(keep[rank], device="cuda", dtype=torch.long)
    y = ep.ep_moe_train(a.index_select(0, loc), r, rb, gl, dl, emap=emap, comm=comm, **KW)
    o = sum(len(k) for k in keep[:rank])
    y.backward(dY[o:o + len(keep[rank])])
    ok = {}
    o2 = sum(len(k) for k in keep[:rank])
    ok["y"] = torch.equal(y.detach(), yr.detach()[o2:o2 + len(keep[rank])])
    ok["dx"] = torch.equal(x.grad, xr.grad[sum(T[:rank]):sum(T[:rank + 1])])
    own = [i for i, e in enumerate(ids) if e < E]
    s = torch.tensor([ids[i] for i in own], device="cuda")
    li = torch.tensor(own, device="cuda")
    ok["d_gate_up"] = torch.equal(gl.grad.index_select(0, li), gur.grad.index_select(0, s))
    ok["d_down"] = torch.equal(dl.grad.index_select(0, li), dnr.grad.index_select(0, s))
    # replicated gradients: the same bits on every rank (fixed rank-order sum)
    for name, t in (("router", r.grad), ("shared_gate_up", gl.grad[-1]), ("shared_down", dl.grad[-1])):
        parts = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(parts, t.contiguous())
        ok[name + "_same_on_ranks"] = all(torch.equal(p, parts[0]) for p in parts)
    torch.save(ok, f"{out}.{rank}")
    dist.barrier()
    dist.destroy_process_group()


def test_token_disjoint_ep_training_over_nccl(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires a two-GPU lease")
    import torch.multiprocessing as mp
    port = 29800 + os.getpid() % 90
    mp.spawn(_rank, args=(2, port, str(tmp_path / "ok")), nprocs=2, join=True)
    for r in range(2):
        res = torch.load(tmp_path / f"ok.{r}")
        assert all(res.values()), (r, res)
