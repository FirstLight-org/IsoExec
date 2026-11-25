"""Expert-parallel invariance battery on SM90: the Qwen3.5-MoE layer over Megatron and vLLM meshes.

Claim (``moe.combine.korder_fp32``; paper: "each expert's complete output is returned to the token's
rank, which adds the outputs in ascending expert identity"): at any EP degree, any TP / CP / DP around
it, SP on or off, trainer or engine dispatch, every token's MoE output has the bits of the EP=1 TP=1
trainer program on the same tokens. Backward is not EP-invariant by contract; it must be correct
(close to the EP=1 autograd gradient once expert-DP replicas are summed, as Megatron's expert DDP
group does) and deterministic run to run.

Geometry: one Qwen3.5-35B-A3B MoE layer (H 2048, I 512, gated shared expert 512, softmax-renorm
routing, weights applied after fc2) with 16 experts top 4, so EP 2/4/8 divide the experts.

Meshes are Megatron's own (``RankGenerator``, dense ``tp-cp-ep-dp-pp``, experts ETP=1) for the
trainer, and vLLM's for the engine (TP groups of consecutive ranks, EP = DP x TP). Each TP group is one
data shard with its own tokens (a (CP, DP) coordinate for the trainer, a DP engine for vLLM); the MoE
is per token, so CP ranks are token shards here (the CP x EP model composition runs on ``opt/cp``).

Modes (each writes ``<out>/<name>/rank<r>.pt``; ``compare`` judges them and prints one line per layout):

* ``ref``      -- 1 process: the EP=1 TP=1 trainer layer (and the engine-side EP=1 layer) on every
                  shard's tokens, forward (grad and no-grad), backward, routed-only rows and grads.
* ``layout``   -- ``torchrun --nproc-per-node N ep_matrix_gpu.py layout NAME OUT``: the full
                  ``nn.MoE`` (shared expert, PIK row-parallel, SP seams) over NCCL, one GPU per rank.
* ``sim``      -- ``python ep_matrix_gpu.py sim NAME OUT``: N processes on ONE GPU (8 or 16 ranks),
                  collectives over gloo with CUDA payloads staged through host memory (bytes only).
                  Runs the EP region itself -- routing, the dispatch, the sm90 expert kernels and the
                  fixed-order combine -- with no shared expert (whose TP reduction is PIK's, covered by
                  the TP batteries), and checks it against the reference's routed rows.
* ``compare``  -- ``python ep_matrix_gpu.py compare OUT``.
* ``suite``    -- ``python ep_matrix_gpu.py suite OUT [--sims] [--layouts]``: reference, the simulations
                  (ranks spread over the visible GPUs) and every layout that fits the visible GPUs, then
                  compare; exit 0 iff all that ran passed. Runs at 1, 4 and 8 GPUs.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import zlib
from pathlib import Path

import torch

FIXTURE = Path(__file__).resolve().parents[3] / "models/qwen3_5_moe/tests/fixtures/qwen3_5_35b_a3b.json"
SITE = "layers.0.mlp"
EXPERTS, TOP_K = 16, 4
MAX_SHARDS = 16
PREFIXES = (1, 2, 4, 8, 16)
STEPS = 2

#: name -> (world, tp, cp, ep, sp, side). Trainer layouts first, then vLLM engine layouts.
LAYOUTS: dict[str, tuple[int, int, int, int, bool, str]] = {
    # 2 GPUs
    "t_tp1_dp2_ep2": (2, 1, 1, 2, False, "trainer"),
    "t_tp2_ep2": (2, 2, 1, 2, False, "trainer"),
    "t_tp2_ep2_sp": (2, 2, 1, 2, True, "trainer"),
    "t_cp2_ep2": (2, 1, 2, 2, False, "trainer"),
    "e_tp2_ep2": (2, 2, 1, 2, False, "engine"),
    "e_dp2_ep2": (2, 1, 1, 2, False, "engine"),
    # 4 GPUs
    "t_tp1_dp4_ep4": (4, 1, 1, 4, False, "trainer"),
    "t_tp1_dp4_ep2": (4, 1, 1, 2, False, "trainer"),      # EDP 2
    "t_tp2_dp2_ep4": (4, 2, 1, 4, False, "trainer"),
    "t_tp2_dp2_ep2": (4, 2, 1, 2, False, "trainer"),      # EDP 2
    "t_tp2_dp2_ep4_sp": (4, 2, 1, 4, True, "trainer"),
    "t_tp4_ep2": (4, 4, 1, 2, False, "trainer"),          # EP not a multiple of TP
    "t_tp4_ep2_sp": (4, 4, 1, 2, True, "trainer"),
    "t_tp4_ep4": (4, 4, 1, 4, False, "trainer"),
    "t_cp2_dp2_ep2": (4, 1, 2, 2, False, "trainer"),      # CP x EP, EDP 2
    "t_tp2_cp2_ep4": (4, 2, 2, 4, False, "trainer"),      # CP x EP x TP
    "t_tp2_cp2_ep2_sp": (4, 2, 2, 2, True, "trainer"),    # CP x EP x TP+SP, EDP 2
    "e_tp4_ep4": (4, 4, 1, 4, False, "engine"),
    "e_tp2_dp2_ep4": (4, 2, 1, 4, False, "engine"),
    # 8 GPUs (one node)
    "t_tp1_dp8_ep8": (8, 1, 1, 8, False, "trainer"),      # EP8
    "t_tp1_dp8_ep2": (8, 1, 1, 2, False, "trainer"),      # EDP 4
    "t_tp2_dp4_ep8_sp": (8, 2, 1, 8, True, "trainer"),
    "t_tp2_dp4_ep2": (8, 2, 1, 2, False, "trainer"),      # EDP 4
    "t_tp4_dp2_ep2_sp": (8, 4, 1, 2, True, "trainer"),    # EP not a multiple of TP, EDP 4
    "t_tp8_ep4": (8, 8, 1, 4, False, "trainer"),          # EP not a multiple of TP
    "t_tp8_ep8_sp": (8, 8, 1, 8, True, "trainer"),
    "t_tp2_cp2_dp2_ep8": (8, 2, 2, 8, False, "trainer"),  # CP x EP8
    "t_tp1_cp2_dp4_ep2": (8, 1, 2, 2, False, "trainer"),  # CP x EP2, EDP 4
    "e_tp8_ep8": (8, 8, 1, 8, False, "engine"),
    "e_tp2_dp4_ep8": (8, 2, 1, 8, False, "engine"),
    "e_tp4_dp2_ep8": (8, 4, 1, 8, False, "engine"),
    # 16 GPUs (two nodes): EDP 8 needs EP >= 2 over 16 ranks. The suite reports these as needing 16 GPUs;
    # run them with a multi-node torchrun of ``layout``, or as the spread simulation below.
    "t_tp1_dp16_ep2": (16, 1, 1, 2, False, "trainer"),    # EDP 8
    "t_tp2_dp8_ep2": (16, 2, 1, 2, False, "trainer"),     # EDP 8 with TP
}

#: Single-GPU simulations (gloo transport): the EP region at 8 and 16 ranks.
SIMS: dict[str, tuple[int, int, int, int, bool, str]] = {
    "s_tp1_dp8_ep8": (8, 1, 1, 8, False, "trainer"),
    "s_tp1_dp8_ep2": (8, 1, 1, 2, False, "trainer"),      # EDP 4
    "s_tp2_dp4_ep8": (8, 2, 1, 8, False, "trainer"),
    "s_tp2_dp4_ep2": (8, 2, 1, 2, False, "trainer"),      # EDP 4
    "s_tp4_dp2_ep2_sp": (8, 4, 1, 2, True, "trainer"),    # EP not a multiple of TP, EDP 4
    "s_tp8_ep4": (8, 8, 1, 4, False, "trainer"),          # EP not a multiple of TP
    "s_tp8_ep4_sp": (8, 8, 1, 4, True, "trainer"),
    "s_tp8_ep8": (8, 8, 1, 8, False, "trainer"),
    "s_tp2_cp2_dp2_ep8": (8, 2, 2, 2 * 4, False, "trainer"),
    "s_tp1_cp4_dp2_ep4": (8, 1, 4, 4, False, "trainer"),  # CP x EP, EDP 2
    "s_tp1_dp16_ep2": (16, 1, 1, 2, False, "trainer"),    # EDP 8
    "s_tp2_dp8_ep2": (16, 2, 1, 2, False, "trainer"),     # EDP 8
    "s_tp4_dp4_ep8_sp": (16, 4, 1, 8, True, "trainer"),   # EDP 2
    "se_tp8_ep8": (8, 8, 1, 8, False, "engine"),
    "se_tp2_dp4_ep8": (8, 2, 1, 8, False, "engine"),
    "se_tp4_dp2_ep8": (8, 4, 1, 8, False, "engine"),
}


# ----------------------------------------------------------------------------------------- shared
def shard_tokens(shard: int) -> int:
    """Unequal, multiple-of-8 token counts (SP needs whole TP slices up to TP 8), crossing the
    small-plan (<= 256 routes) and the grouped paths."""
    return 8 * (6 + (shard * 47) % 110)


def seeded(name: str, shape, scale: float, device) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(zlib.crc32(name.encode()))
    return torch.randn(shape, generator=g, device=device, dtype=torch.float32) * scale


def scale_of(name: str) -> float:
    return 0.05 if name.startswith("router") else 0.02


def shard_inputs(shard: int, H: int, device):
    T = shard_tokens(shard)
    return (seeded(f"x{shard}", (T, H), 1.0, device).bfloat16(), seeded(f"gy{shard}", (T, H), 1.0, device),
            seeded(f"gs{shard}", (T, EXPERTS), 1.0, device))


def text_config() -> dict:
    raw = json.loads(FIXTURE.read_text())
    text = dict(raw["text_config"])
    text.update(num_experts=EXPERTS, num_experts_per_tok=TOP_K)
    return text


def install(side: str, *, transport: bool = True) -> None:
    from isoexec.compiler.source import ModelSource
    from isoexec.ops.collectives.pik_tp_invariant import select_engine_transport, select_trainer_transport
    from isoexec.runtimes.trainer_installs import install_trainer_kernels

    raw = json.loads(FIXTURE.read_text())
    install_trainer_kernels("qwen3_5_moe_ep_battery", source=ModelSource("isoexec.models.qwen3_5_moe"),
                            raw_config=raw)
    if not transport:
        return  # the simulated EP region binds no PIK reduction
    if side == "engine":
        select_engine_transport("ENGINE")
    else:
        select_trainer_transport("TRAINER")


def build(placement, device) -> torch.nn.Module:
    """nn.MoE placed at ``placement``, every held tensor filled from its GLOBAL seeded value."""
    from isoexec import nn as inn
    from isoexec.runtimes.parallel import colwise_ranges, validate_component_placement

    text = text_config()
    module = inn.MoE(hidden=text["hidden_size"], num_experts=EXPERTS, top_k=TOP_K,
                     intermediate=text["moe_intermediate_size"],
                     shared_intermediate=text["shared_expert_intermediate_size"],
                     routing="softmax_renorm", weight_after_fc2=True, device="meta")
    inn.place(module, placement, prefix=SITE)
    validate_component_placement(module, placement)
    tp, rank = placement.tp.size, placement.tp.rank
    shared_axes = module.shared.tensor_axes()
    for name, p in list(module.named_parameters()):
        full = seeded(name, p.shape, scale_of(name), device)  # meta shapes are global until allocation
        if name.startswith("shared.") and tp > 1:
            axis = shared_axes[name.split(".", 1)[1]]
            if axis.kind in ("col", "row"):
                d = 0 if axis.kind == "col" else 1
                (lo, hi), = colwise_ranges(full.shape[d], tp, rank)
                full = full.narrow(d, lo, hi - lo)
        parent, leaf = module.get_submodule(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1]
        setattr(parent, leaf, torch.nn.Parameter(full.to(p.dtype).contiguous()))
    return module


def expert_tensors(lo: int, hi: int, device):
    """``gate_up[E, 2I, H]`` and ``down[E, H, I]`` for global experts ``lo..hi`` from the same seeds."""
    text = text_config()
    H, I = text["hidden_size"], text["moe_intermediate_size"]
    gu = torch.stack([torch.cat([seeded(f"experts.{e}.gate", (I, H), 0.02, device),
                                 seeded(f"experts.{e}.up", (I, H), 0.02, device)]) for e in range(lo, hi)])
    dn = torch.stack([seeded(f"experts.{e}.down", (H, I), 0.02, device) for e in range(lo, hi)])
    return gu.bfloat16().contiguous(), dn.bfloat16().contiguous()


def trainer_mesh(world: int, tp: int, cp: int, ep: int):
    from megatron.core.parallel_state import RankGenerator

    dp = world // (tp * cp)
    assert tp * cp * dp == world and world % ep == 0 and EXPERTS % ep == 0
    dense = RankGenerator(tp=tp, ep=1, dp=dp, pp=1, cp=cp, order="tp-cp-ep-dp-pp")
    expert = RankGenerator(tp=1, ep=ep, dp=world // ep, pp=1, cp=1, order="tp-cp-ep-dp-pp")
    return {"tp": dense.get_ranks("tp"), "ep": expert.get_ranks("ep"), "edp": expert.get_ranks("dp")}


def engine_mesh(world: int, tp: int, ep: int):
    """vLLM: TP groups of consecutive ranks, one EP group over every DP x TP rank."""
    assert ep == world, "vLLM's EP group is DP x TP"
    return {"tp": [list(range(s, s + tp)) for s in range(0, world, tp)], "ep": [list(range(world))],
            "edp": [[r] for r in range(world)]}


def groups_for(rank: int, spec, new_group):
    world, tp, cp, ep, _sp, side = spec
    mesh = engine_mesh(world, tp, ep) if side == "engine" else trainer_mesh(world, tp, cp, ep)
    mine = {}
    for kind in ("tp", "ep", "edp"):
        for ranks in mesh[kind]:
            handle = new_group(ranks) if len(ranks) > 1 else None
            if rank in ranks:
                mine[kind] = (handle, ranks.index(rank), len(ranks), ranks)
    shard = next(i for i, ranks in enumerate(mesh["tp"]) if rank in ranks)
    return mine, shard, len(mesh["tp"])


def placement_for(mine, side: str, sp: bool):
    from isoexec.ops.collectives.pik_tp_invariant import _leaf_dtype, _num_leaves
    from isoexec.schema import Axis, Placement

    one = Axis(1, 0, None)
    tp_h, tp_r, tp_n, _ = mine["tp"]
    ep_h, ep_r, ep_n, _ = mine["ep"]
    tp = Axis(tp_n, tp_r, tp_h) if tp_n > 1 else one
    return Placement(tp=tp, ep=Axis(ep_n, ep_r, ep_h) if ep_n > 1 else one, dp=one, pp=one,
                     etp=tp if ep_n == 1 else one, cp=one, side=side, leaves=_num_leaves(),
                     leaf_dtype="bf16" if _leaf_dtype() == torch.bfloat16 else "fp32", sp=sp)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view({torch.float32: torch.int32, torch.bfloat16: torch.int16}[t.dtype])


# ---------------------------------------------------------------------------------------- reference
def run_reference(out: Path) -> None:
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    from isoexec.runtimes.parallel import trivial_placement

    record: dict = {"shards": {}, "prefix": {}}
    for side in ("trainer", "engine"):
        install(side)
        module = build(trivial_placement(side=side), dev)
        H = module.hidden
        if side == "engine":
            with torch.no_grad():
                for s in range(MAX_SHARDS):
                    x, _, _ = shard_inputs(s, H, dev)
                    record["shards"][s]["engine"] = module(x.clone(), None).cpu()
            continue
        grads: dict[str, torch.Tensor] = {}
        routed_grads: dict[str, torch.Tensor] = {}
        for s in range(MAX_SHARDS):
            x, gy, gs = shard_inputs(s, H, dev)
            entry: dict = {}
            with torch.no_grad():
                entry["no_grad"] = module(x.clone(), None).cpu()
            module.zero_grad(set_to_none=True)
            xr = x.clone().requires_grad_(True)
            y, scores = module(xr, None, return_aux=True)
            (y.float() * gy).sum().add((scores * gs).sum()).backward()
            entry["out"], entry["dx"] = y.detach().cpu(), xr.grad.cpu()
            for n, p in module.named_parameters():
                if p.grad is not None and (n == "router.weight" or n.startswith("experts.")):
                    grads[n] = grads.get(n, 0) + p.grad.float().cpu()
            # The EP region alone: routing, experts, fixed-order combine; no shared expert.
            gate_up, down = module.expert_weight_views()
            gate_up = gate_up.detach().clone().requires_grad_(True)
            down = down.detach().clone().requires_grad_(True)
            xr = x.clone().requires_grad_(True)
            routing = module.slots.route_select(module.slots.route_score(xr, module.router.weight), module._spec, False)
            rows = module.slots.combine(module.slots.expert_mlp(xr, routing, gate_up, down, weight_after_fc2=True), None)
            (rows.float() * gy).sum().backward()
            entry["routed"], entry["routed_dx"] = rows.detach().cpu(), xr.grad.cpu()
            for n, t in (("gate_up", gate_up), ("down", down)):
                routed_grads[n] = routed_grads.get(n, 0) + t.grad.float().cpu()
            record["shards"][s] = entry
            if s + 1 in PREFIXES:
                # A layout with n shards runs shards 0..n-1: keep the gradient sums over each prefix.
                record["prefix"][s + 1] = {"grads": dict(grads), "routed": dict(routed_grads)}
        record["resolved"] = {k: v.impl for k, v in __import__("isoexec").nn.resolved_ops(module)[SITE].items()
                              if hasattr(v, "impl")}
        del module
    (out / "ref").mkdir(parents=True, exist_ok=True)
    torch.save(record, out / "ref" / "rank0.pt")
    print(f"[EP-MATRIX] ref written ({MAX_SHARDS} shards)", flush=True)


# ------------------------------------------------------------------------------------- NCCL layout
def run_layout(name: str, out: Path) -> None:
    import torch.distributed as dist

    spec = LAYOUTS[name]
    world, sp, side = spec[0], spec[4], spec[5]
    rank = int(os.environ["RANK"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    assert int(os.environ["WORLD_SIZE"]) == world, f"{name} needs {world} ranks"
    dev = torch.device("cuda", local)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    install(side)
    mine, shard, shards = groups_for(rank, spec, dist.new_group)
    placement = placement_for(mine, side, sp)
    if side == "engine":
        from isoexec.ops.moe import expert_parallel as ops

        bound = max(shard_tokens(s) for s in range(shards))
        ops.set_engine_token_bound(lambda n: max(n, bound))
    module = build(placement, dev)
    H = module.hidden
    x, gy, gs = shard_inputs(shard, H, dev)
    T = x.shape[0]
    tp_h, tp_r, tp_n, _ = mine["tp"]
    lo, hi = (T * tp_r // tp_n, T * (tp_r + 1) // tp_n) if sp else (0, T)

    def gather_tp(t):
        if not sp or tp_n == 1:
            return t
        parts = [torch.empty_like(t) for _ in range(tp_n)]
        dist.all_gather(parts, t.contiguous(), group=tp_h)
        return torch.cat(parts)

    record: dict = {"spec": spec, "shard": shard, "tp_rank": tp_r, "ep_rank": mine["ep"][1],
                    "resolved": {k: v.impl for k, v in __import__("isoexec").nn.resolved_ops(module)[SITE].items()
                                 if hasattr(v, "impl")},
                    "held": [int(k) for k in module.experts._modules], "steps": []}
    with torch.no_grad():
        record["no_grad"] = gather_tp(module(x[lo:hi].clone(), None)).cpu()
    if side == "trainer":
        for _ in range(STEPS):
            module.zero_grad(set_to_none=True)
            xr = x[lo:hi].clone().requires_grad_(True)
            y, scores = module(xr, None, return_aux=True)
            (y.float() * gy[lo:hi]).sum().add((scores * gs[lo:hi]).sum()).backward()
            grads = {n: p.grad.detach().cpu() for n, p in module.named_parameters() if p.grad is not None}
            record["steps"].append({"out": gather_tp(y.detach()).cpu(), "dx": gather_tp(xr.grad).cpu(), "grads": grads})
        # Global expert gradients: every holder adds its rows (expert-DP replicas see disjoint tokens).
        g = record["steps"][0]["grads"]
        for role in ("gate", "up", "down"):
            shape = getattr(next(iter(module.experts._modules.values())), role).shape
            full = torch.zeros((EXPERTS, *shape), device=dev)
            for e in record["held"]:
                key = f"experts.{e}.{role}"
                if key in g:
                    full[e] = g[key].to(dev).float()
            dist.all_reduce(full)  # sum over all ranks (ascending rank order is NCCL's; fp32)
            record[f"experts_{role}"] = full.cpu()
        router = g["router.weight"].to(dev).float() if tp_r == 0 else torch.zeros_like(g["router.weight"], device=dev).float()
        dist.all_reduce(router)  # one TP rank per shard: the sum over shards
        record["router_sum"] = router.cpu()
    path = out / name
    path.mkdir(parents=True, exist_ok=True)
    torch.save(record, path / f"rank{rank}.pt")
    dist.barrier()
    dist.destroy_process_group()
    print(f"[EP-MATRIX] {name} rank {rank} written", flush=True)


# ---------------------------------------------------------------------------------- 1-GPU simulation
def _staged_collectives():
    """Patch the three collectives the EP region calls so CUDA payloads ride gloo through host memory.
    Bytes only: the arithmetic stays in the same kernels on the same device."""
    import torch.distributed as dist

    a2a, agt, ar = dist.all_to_all_single, dist.all_gather_into_tensor, dist.all_reduce

    def all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
        o = torch.empty_like(output, device="cpu")
        a2a(o, input.detach().cpu(), output_split_sizes, input_split_sizes, group=group)
        output.copy_(o)

    def all_gather_into_tensor(output, input, group=None, async_op=False):
        o = torch.empty_like(output, device="cpu")
        agt(o, input.detach().cpu(), group=group)
        output.copy_(o)

    def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        t = tensor.detach().cpu()
        ar(t, op=op, group=group)
        tensor.copy_(t)

    dist.all_to_all_single, dist.all_gather_into_tensor, dist.all_reduce = all_to_all_single, all_gather_into_tensor, all_reduce


def _sim_child(rank: int, name: str, out: str, port: int, devices: int = 1) -> None:
    import torch.distributed as dist

    spec = SIMS[name]
    world, sp, side = spec[0], spec[4], spec[5]
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dev = torch.device("cuda", rank % devices)  # spread: several ranks per GPU, gloo between them
    torch.cuda.set_device(dev)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _staged_collectives()
    install("trainer", transport=False)
    from isoexec.ops.moe import expert_parallel as ops
    from isoexec.ops.moe.eager import combine_fixed_order, route_mechanics_megatron, route_score_fp32
    from isoexec.ops.moe.moe_sm90 import integration
    from isoexec.schema import Mlp

    mine, shard, shards = groups_for(rank, spec, dist.new_group)
    placement = placement_for(mine, side, sp)
    ep_h, ep_r, ep_n, _ = mine["ep"]
    _, tp_r, tp_n, _ = mine["tp"]
    text = text_config()
    H = text["hidden_size"]
    spec_mlp = Mlp(intermediate_size=text["moe_intermediate_size"], num_experts=EXPERTS, top_k=TOP_K,
                   shared_intermediate=0, routing="softmax_renorm", routed_scale=1.0, shared_expert_gate=False)
    router = seeded("router.weight", (EXPERTS, H), scale_of("router.weight"), dev)
    e_local = EXPERTS // ep_n
    gate_up0, down0 = expert_tensors(ep_r * e_local, (ep_r + 1) * e_local, dev)
    mechanics = route_mechanics_megatron
    combine = combine_fixed_order
    if side == "engine":
        mechanics = integration.engine_route_mechanics(route_mechanics_megatron)
        combine = integration.engine_combine(combine_fixed_order)
        bound = max(shard_tokens(s) for s in range(shards))
        ops.set_engine_token_bound(lambda n: max(n, bound))
    dispatch = ops.engine_expert_dispatch if side == "engine" else ops.expert_dispatch
    fn = dispatch(integration.expert_mlp, ep=ep_n, ep_rank=ep_r, ep_group=ep_h, combine=combine, placement=placement)
    x, gy, _ = shard_inputs(shard, H, dev)
    T = x.shape[0]
    lo, hi = (T * tp_r // tp_n, T * (tp_r + 1) // tp_n) if sp else (0, T)
    record: dict = {"spec": spec, "shard": shard, "tp_rank": tp_r, "ep_rank": ep_r, "slice": (lo, hi),
                    "experts": (ep_r * e_local, (ep_r + 1) * e_local), "steps": []}

    def forward(xr, gate_up, down):
        routing = mechanics(route_score_fp32(xr, router), spec_mlp, False)
        return fn(xr, routing, gate_up, down, weight_after_fc2=True).rows

    with torch.no_grad():
        record["no_grad"] = forward(x.clone(), gate_up0, down0).cpu()
    if side == "trainer":
        for _ in range(STEPS):
            xr = x.clone().requires_grad_(True)
            gu = gate_up0.clone().requires_grad_(True)
            dn = down0.clone().requires_grad_(True)
            rows = forward(xr, gu, dn)
            (rows.float() * gy[lo:hi]).sum().backward()
            zero = lambda t: torch.zeros_like(t) if t.grad is None else t.grad  # an owner no route reached has no gradient
            record["steps"].append({"out": rows.detach().cpu(), "dx": xr.grad.cpu(),
                                    "gate_up": zero(gu).cpu(), "down": zero(dn).cpu()})
        for role, held in (("gate_up", record["steps"][0]["gate_up"]), ("down", record["steps"][0]["down"])):
            full = torch.zeros((EXPERTS, *held.shape[1:]), device=dev)
            a, b = record["experts"]
            full[a:b] = held.to(dev).float()
            dist.all_reduce(full)
            record[f"global_{role}"] = full.cpu()
    path = Path(out) / name
    path.mkdir(parents=True, exist_ok=True)
    torch.save(record, path / f"rank{rank}.pt")
    dist.barrier()
    dist.destroy_process_group()


def run_sim(name: str, out: Path, devices: int = 1) -> None:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    torch.multiprocessing.spawn(_sim_child, args=(name, str(out), port, devices), nprocs=SIMS[name][0], join=True)
    print(f"[EP-MATRIX] {name} (sim) written", flush=True)


# ------------------------------------------------------------------------------------------ compare
def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def compare(out: Path) -> int:
    ref = torch.load(out / "ref" / "rank0.pt", weights_only=False)
    shards = ref["shards"]
    failures = 0
    # EP=1: engine == trainer on every shard (the trainer==engine base case).
    eq = all(torch.equal(bits(shards[s]["engine"]), bits(shards[s]["out"])) for s in shards)
    eq = eq and all(torch.equal(bits(shards[s]["no_grad"]), bits(shards[s]["out"])) for s in shards)
    print(f"[EP-MATRIX] ref EP1: engine==trainer and no_grad==grad on {len(shards)} shards: {'PASS' if eq else 'FAIL'}")
    failures += not eq
    for name in [*LAYOUTS, *SIMS]:
        path = out / name
        if not path.is_dir():
            continue
        spec = (LAYOUTS | SIMS)[name]
        world, tp, cp, ep, sp, side = spec
        files = sorted(path.glob("rank*.pt"))
        if len(files) != world:
            print(f"[EP-MATRIX] {name}: FAIL incomplete ({len(files)}/{world} ranks)")
            failures += 1
            continue
        ranks = [torch.load(f, weights_only=False) for f in files]
        sim = name in SIMS
        problems: list[str] = []
        fwd_key, dx_key = ("routed", "routed_dx") if sim else ("out", "dx")
        for r, rec in enumerate(ranks):
            want = shards[rec["shard"]][fwd_key]
            if sim and sp:
                a, b = rec["slice"]
                want = want[a:b]
            outs = [rec["no_grad"], *(s["out"] for s in rec["steps"])]
            for i, got in enumerate(outs):
                if got.shape != want.shape or not torch.equal(bits(got), bits(want)):
                    n = (bits(got) != bits(want)).sum().item() if got.shape == want.shape else "shape"
                    problems.append(f"rank{r} fwd{i}: {n} elements differ from EP1")
            if rec["steps"]:
                first, *rest = rec["steps"]
                for again in rest:
                    for key in first:
                        a, b = first[key], again[key]
                        if isinstance(a, dict):
                            bad = [k for k in a if not torch.equal(bits(a[k]), bits(b[k]))]
                            if bad:
                                problems.append(f"rank{r} {key} nondeterministic: {bad[:3]}")
                        elif not torch.equal(bits(a), bits(b)):
                            problems.append(f"rank{r} {key} nondeterministic")
                rel = _rel(first["dx"], shards[rec["shard"]][dx_key])
                if not rel < 2e-2:
                    problems.append(f"rank{r} dx rel err {rel:.2e}")
        if side == "trainer":
            if sim:
                for role in ("gate_up", "down"):
                    rel = _rel(ranks[0][f"global_{role}"], ref["prefix"][_shards(ranks)]["routed"][role])
                    if not rel < 2e-2:
                        problems.append(f"expert {role} grad rel err {rel:.2e}")
            else:
                for role in ("gate", "up", "down"):
                    g = ref["prefix"][_shards(ranks)]["grads"]
                    want = torch.stack([g[f"experts.{e}.{role}"] for e in range(EXPERTS)])
                    rel = _rel(ranks[0][f"experts_{role}"], want)
                    if not rel < 2e-2:
                        problems.append(f"expert {role} grad rel err {rel:.2e}")
                rel = _rel(ranks[0]["router_sum"], ref["prefix"][_shards(ranks)]["grads"]["router.weight"])
                if not rel < 2e-2:
                    problems.append(f"router grad rel err {rel:.2e}")
        status = "PASS" if not problems else "FAIL"
        failures += bool(problems)
        detail = "" if not problems else " :: " + "; ".join(problems[:6])
        print(f"[EP-MATRIX] {name} world={world} tp={tp} cp={cp} ep={ep} sp={sp} side={side}"
              f"{' sim' if sim else ''}: {status}{detail}", flush=True)
    return failures


def _shards(ranks) -> int:
    used = sorted({r["shard"] for r in ranks})
    assert used == list(range(len(used))), used
    return len(used)


def run_suite(out: Path, *, sims: bool, layouts: bool) -> int:
    """Reference, then simulations (spread over the visible GPUs) and/or every NCCL layout that fits the
    visible GPUs (one ``torchrun`` each), then ``compare``. Exit 0 iff everything that ran passed; a
    layout larger than the node is listed, not run. This is the entry point for a multi-layout sweep."""
    import subprocess

    out.mkdir(parents=True, exist_ok=True)
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d] or \
        [str(i) for i in range(torch.cuda.device_count())]
    me = [sys.executable, str(Path(__file__).resolve())]
    env0 = dict(os.environ, CUDA_VISIBLE_DEVICES=visible[0])
    failures = 0
    if not (out / "ref" / "rank0.pt").exists():
        failures += subprocess.run([*me, "ref", str(out)], env=env0, check=False).returncode != 0
    if sims:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(visible))
        for name in SIMS:
            rc = subprocess.run([*me, "sim", name, str(out), str(len(visible))], env=env, check=False).returncode
            print(f"[EP-MATRIX] sim {name} rc={rc}", flush=True)
            failures += rc != 0
    if layouts:
        for name, spec in LAYOUTS.items():
            world = spec[0]
            if world > len(visible):
                print(f"[EP-MATRIX] {name}: needs {world} GPUs, {len(visible)} visible; not run", flush=True)
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(visible[:world]))
            rc = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                                 f"--nproc-per-node={world}", str(Path(__file__).resolve()), "layout", name, str(out)],
                                env=env, check=False).returncode
            print(f"[EP-MATRIX] layout {name} rc={rc}", flush=True)
            failures += rc != 0
    failures += compare(out)
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    mode = argv[0]
    if mode == "ref":
        run_reference(Path(argv[1]))
    elif mode == "layout":
        try:
            run_layout(argv[1], Path(argv[2]))
        except Exception:  # noqa: BLE001 (report and fail this layout; the session runs the rest)
            traceback.print_exc()
            return 1
    elif mode == "sim":
        run_sim(argv[1], Path(argv[2]), int(argv[3]) if len(argv) > 3 else 1)
    elif mode == "suite":
        # suite OUT [--sims] [--layouts]   (neither flag: both)
        flags = set(argv[2:])
        both = not flags & {"--sims", "--layouts"}
        return run_suite(Path(argv[1]), sims=both or "--sims" in flags, layouts=both or "--layouts" in flags)
    elif mode == "compare":
        return 1 if compare(Path(argv[1])) else 0
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
