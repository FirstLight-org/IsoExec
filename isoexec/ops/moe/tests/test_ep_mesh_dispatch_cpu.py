"""Expert-parallel dispatch over Megatron's own meshes, on CPU (gloo), at 2 to 16 ranks.

The meshes come from ``megatron.core.parallel_state.RankGenerator``, the class the trainer's
``initialize_model_parallel`` builds its groups with (dense ``tp-cp-ep-dp-pp``; experts ETP=1, so
an EP group is a run of consecutive ranks and the expert-DP group strides over them). Each TP group
is one data shard (one (CP, DP) coordinate) with its own tokens. The cases cover

* EP a multiple of TP, EP equal to TP, and EP not a multiple of TP (TP4/EP2, TP2/EP3, TP8/EP4);
* expert data parallelism (EDP = TP*CP*DP/EP) of 1, 2, 4 and 8;
* CP ranks as further token shards inside one EP group;
* the trainer dispatch with and without SP, and the engine's static-capacity dispatch.

The expert is a per-row stand-in (``rows * scale[e]`` then the route weight), so the claim checked is
the dispatch's: every token's k terms come back and fold in ascending global expert id, in fp32, with
one narrowing, so the bits equal the EP=1 program. Backward: input, route-weight and expert-scale
gradients (summed over expert-DP replicas, as Megatron's expert DDP group does) match the EP=1
autograd reference, and two backward passes give identical bits. The GPU batteries
(``ops/moe/tests/ep_matrix_gpu.py``) run the same meshes on the real sm90 expert kernels.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H, K = 8, 3


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def mesh(world: int, tp: int, cp: int, ep: int) -> dict[str, list[list[int]]]:
    """Megatron's dense and expert groups for this world (PP=1, ETP=1)."""
    from megatron.core.parallel_state import RankGenerator

    dp = world // (tp * cp)
    assert tp * cp * dp == world and world % ep == 0
    dense = RankGenerator(tp=tp, ep=1, dp=dp, pp=1, cp=cp, order="tp-cp-ep-dp-pp")
    expert = RankGenerator(tp=1, ep=ep, dp=world // ep, pp=1, cp=1, order="tp-cp-ep-dp-pp")
    return {"tp": dense.get_ranks("tp"), "cp": dense.get_ranks("cp"), "dp": dense.get_ranks("dp"),
            "ep": expert.get_ranks("ep"), "edp": expert.get_ranks("dp")}


def _shard_tokens(shard: int) -> int:
    return 5 + 3 * shard  # unequal shards: the engine's static capacity must cover the largest peer


def _shard_inputs(shard: int, experts: int):
    g = torch.Generator().manual_seed(1000 + shard)
    n = _shard_tokens(shard)
    x = torch.randn(n, H, generator=g).to(torch.bfloat16)
    logits = torch.randn(n, experts, generator=g)
    vals, idx = logits.topk(K, dim=1)  # sorted by score, not by expert id
    probs = torch.softmax(vals, dim=1)
    gy = torch.randn(n, H, generator=g)
    return x, idx.to(torch.int32), probs, gy


def _scales(experts: int) -> torch.Tensor:
    return torch.linspace(0.5, 1.5, experts, dtype=torch.float32)


def _inner(lo: int, scale: torch.Tensor):
    """Per-row expert over local experts ``lo..``: ``(row * scale[e]).bf16 * prob.bf16`` in fp32 rows,
    permuted by expert like the real kernels (``extra`` is the permutation)."""
    from isoexec.ops.moe.carriers import ExpertOutputs

    def f(rows, routing, gate_up, down, **kw):
        experts = routing.indices[:, 0].long()
        perm = torch.argsort(experts, stable=True)
        e = experts[perm]
        term = (rows[perm].float() * scale.index_select(0, e).unsqueeze(1)).to(torch.bfloat16)
        out = (term * routing.probs[perm].to(torch.bfloat16)).float()
        return ExpertOutputs(rows=out, layout="permuted", routing=routing, extra=perm)

    return f


def _combine(routed, shared):
    rows = routed.rows.index_select(0, routed.combine_rows.long().flatten()).view(*routed.combine_rows.shape, -1)
    total = rows[:, 0]
    for j in range(1, rows.shape[1]):
        total = total + rows[:, j]
    return total


def reference(x, idx, probs, scale):
    """The EP=1 program: terms in ascending global expert id, one fp32 chain, autograd-capable."""
    order = idx.long().argsort(dim=1)
    e = idx.long().gather(1, order)
    p = probs.gather(1, order)
    total = None
    for j in range(K):
        term = (x.float() * scale.index_select(0, e[:, j]).unsqueeze(1)).to(torch.bfloat16)
        term = (term * p[:, j : j + 1].to(torch.bfloat16)).float()
        total = term if total is None else total + term
    return total


def _worker(rank, world, tp, cp, ep, sp, experts, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from isoexec.ops.moe import expert_parallel as ops
    from isoexec.ops.moe.carriers import RoutingResult
    from isoexec.schema import Axis, Placement

    groups = mesh(world, tp, cp, ep)
    handles = {}
    for kind in ("tp", "ep"):
        for ranks in groups[kind]:
            handle = dist.new_group(ranks)
            if rank in ranks:
                handles[kind] = (handle, ranks.index(rank), len(ranks))
    shard = next(i for i, ranks in enumerate(groups["tp"]) if rank in ranks)
    tp_group, tp_rank, _ = handles["tp"]
    ep_group, ep_rank, _ = handles["ep"]
    e_local = experts // ep
    lo = ep_rank * e_local
    scale_global = _scales(experts)
    one = Axis(1, 0, None)

    def placement(side, use_sp):
        return Placement(tp=Axis(tp, tp_rank, tp_group if tp > 1 else None), ep=Axis(ep, ep_rank, ep_group),
                         dp=one, pp=one, etp=one, cp=one, side=side, leaves=1, sp=use_sp)

    x, idx, probs, gy = _shard_inputs(shard, experts)
    T = x.shape[0]
    gate_up = torch.zeros(e_local, 2, H)
    down = torch.zeros(e_local, H, 1)
    record = {"shard": shard, "tp_rank": tp_rank, "ep_rank": ep_rank, "steps": []}
    use_sp = sp and tp > 1
    if use_sp:
        # SP keeps whole TP token slices: pad the shard to a multiple of TP with zero-weight rows.
        pad = (-T) % tp
        if pad:
            x = torch.cat([x, torch.zeros(pad, H, dtype=x.dtype)])
            idx = torch.cat([idx, idx[:1].expand(pad, K)])
            probs = torch.cat([probs, torch.zeros(pad, K)])
            gy = torch.cat([gy, torch.zeros(pad, H)])
            T = x.shape[0]
    t0, t1 = ops.token_shard(T, tp, tp_rank)
    record["slice"] = (t0, t1) if use_sp else (0, T)
    for _ in range(2):
        scale = scale_global[lo : lo + e_local].clone().requires_grad_(True)
        xr = x.clone().requires_grad_(True)
        pr = probs.clone().requires_grad_(True)
        fn = ops.expert_dispatch(_inner(lo, scale), ep=ep, ep_rank=ep_rank, ep_group=ep_group, combine=_combine,
                                 placement=placement("trainer", use_sp))
        out = fn(xr, RoutingResult(idx, pr, None), gate_up, down)
        assert out.layout == ("combined_local" if use_sp else "combined"), out.layout
        y = out.rows
        g = gy[t0:t1] if use_sp else gy
        (y.float() * g).sum().backward()
        record["steps"].append({"y": y.detach().view(torch.int32).clone(), "dx": xr.grad.clone(),
                                "dp": pr.grad.clone(),
                                # an owner no route reached has no gradient: Megatron's main_grad stays zero
                                "dscale": torch.zeros_like(scale) if scale.grad is None else scale.grad.clone()})
    # Expert-DP replicas see disjoint tokens; Megatron's expert DDP group sums them. Every rank that
    # holds expert e contributes to the global gradient of e.
    full = torch.zeros(experts)
    full[lo : lo + e_local] = record["steps"][0]["dscale"]
    dist.all_reduce(full)
    record["dscale_global"] = full
    if not use_sp:
        ops.set_engine_token_bound(lambda n: max(_shard_tokens(s) for s in range(len(groups["tp"]))))
        try:
            with torch.no_grad():
                eng = ops.engine_expert_dispatch(_inner(lo, scale_global[lo : lo + e_local]), ep=ep, ep_rank=ep_rank,
                                                 ep_group=ep_group, combine=_combine,
                                                 placement=placement("engine", False))(
                    x, RoutingResult(idx, probs, None), gate_up, down)
        finally:
            ops.set_engine_token_bound(None)
        record["engine"] = eng.rows.view(torch.int32).clone()
    results[rank] = record
    dist.barrier()
    dist.destroy_process_group()


# (world, tp, cp, ep): EP | TP*CP*DP always (Megatron's own constraint), nothing else assumed.
CASES = [
    (2, 1, 1, 2),   # TP1 x DP2, EP2: EDP 1
    (2, 2, 1, 2),   # TP2, EP2 = TP
    (4, 2, 1, 4),   # TP2 x DP2, EP4: EP a multiple of TP
    (4, 2, 1, 2),   # TP2 x DP2, EP2: EDP 2
    (4, 4, 1, 2),   # TP4, EP2: EP not a multiple of TP (EP divides TP)
    (4, 1, 1, 2),   # TP1 x DP4, EP2: EDP 2
    (4, 1, 2, 2),   # CP2 x DP2, EP2: the EP group spans CP ranks
    (4, 2, 2, 4),   # TP2 x CP2, EP4
    (6, 2, 1, 3),   # TP2 x DP3, EP3: EP and TP coprime; an EP group holds parts of two TP groups
    (8, 1, 1, 2),   # TP1 x DP8, EP2: EDP 4
    (8, 8, 1, 4),   # TP8, EP4: EP not a multiple of TP
    (8, 2, 2, 8),   # TP2 x CP2 x DP2, EP8
    (8, 4, 1, 8),   # TP4 x DP2, EP8
    (16, 1, 1, 2),  # TP1 x DP16, EP2: EDP 8
    (16, 2, 1, 2),  # TP2 x DP8, EP2: EDP 8 with TP
]


@pytest.mark.parametrize("sp", [False, True], ids=["nosp", "sp"])
@pytest.mark.parametrize(("world", "tp", "cp", "ep"), CASES, ids=[f"w{w}_tp{t}_cp{c}_ep{e}" for w, t, c, e in CASES])
def test_dispatch_is_ep1_bitwise_and_backward_matches(world, tp, cp, ep, sp):
    if sp and tp == 1:
        pytest.skip("SP needs TP > 1")
    experts = 24 if world == 6 else 16
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_worker, args=(world, tp, cp, ep, sp, experts, _free_port(), results), nprocs=world, join=True)
    scale = _scales(experts).requires_grad_(True)
    dscale_ref = torch.zeros(experts)
    for rank in range(world):
        r = results[rank]
        x, idx, probs, gy = _shard_inputs(r["shard"], experts)
        xr, pr = x.clone().requires_grad_(True), probs.clone().requires_grad_(True)
        want = reference(xr, idx, pr, scale)
        t0, t1 = r["slice"]
        T = x.shape[0]
        first, again = r["steps"]
        # forward: EP=1 bits, both passes
        got = first["y"].view(torch.float32)
        assert torch.equal(got[: max(0, min(t1, T) - t0)].view(torch.int32),
                           want.detach()[t0:min(t1, T)].view(torch.int32)), f"rank {rank} forward differs from EP1"
        assert torch.equal(first["y"], again["y"])
        if "engine" in r:
            assert torch.equal(r["engine"], want.detach().view(torch.int32)), f"rank {rank} engine differs from EP1"
        # backward: deterministic, and the EP=1 gradient
        for key in ("dx", "dp", "dscale"):
            assert torch.equal(first[key], again[key]), f"rank {rank} {key} not deterministic"
        (want.float() * gy).sum().backward()
        assert torch.allclose(first["dx"][:T].float(), xr.grad.float(), rtol=1e-2, atol=1e-3), rank
        assert torch.allclose(first["dp"][:T], pr.grad, rtol=1e-4, atol=1e-5), rank
        if r["tp_rank"] == 0:
            dscale_ref += scale.grad  # one TP rank per shard: its shard's contribution
        scale.grad = None
    for rank in range(world):
        assert torch.allclose(results[rank]["dscale_global"], dscale_ref, rtol=1e-4, atol=1e-4), rank
