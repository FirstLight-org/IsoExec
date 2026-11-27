"""``moe.combine.korder_fp32`` on SM100: the Qwen3.5-35B-A3B MoE layer at EP 1, 2 and 4 over NCCL.

The class's claim is EP invariance: a token's k expert terms are folded over ascending GLOBAL expert
id in one fp32 chain, so an EP split cannot reorder the sum. This battery runs the trainer's
``nn.MoE`` (``sm100_tcgen05_after_fc2_v1`` experts, ``sm100_after_fc2_combine_v1`` fold, the same
native combine kernel ``sm100_gather_korder_v1`` launches) at the released geometry -- H 2048,
256 experts, top 8, I 512, a gated 512-wide shared expert -- with identical weights and tokens at

* EP1  -- one process, the class's program;
* EP2  -- TP2 / EP2 / ETP1 / SP, two NCCL processes (the trainer's layout family);
* EP4  -- TP4 / EP4 / ETP1 / SP, four NCCL processes (the production trainer layout);
* ETP4 -- TP4 / EP1 / ETP4 / SP, four processes (the EP1 layout of the same mesh),

and asserts

1. forward bits are equal across every layout (outputs gathered over SP);
2. the EP1 output is the class's program, recomputed here: the expert rows folded in a python fp32
   chain over ascending global expert id, plus the shared expert, narrowed once;
3. backward is deterministic per layout (three steps, identical bits for dx and every gradient),
   with torch's deterministic algorithms OFF as in the production trainer (Megatron
   ``deterministic_mode`` false; ``ISOEXEC_EP_KORDER_DETERMINISTIC_MODE=1`` turns them on).
   Gradient EP invariance is not the contract and is only reported.

Run: ``CUDA_VISIBLE_DEVICES=<4 B200s> python -m pytest -q -s <this file>``.
"""

from __future__ import annotations

import json
import os
import socket
import zlib
from pathlib import Path

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or live_device_arch() != "sm100", reason="SM100 (B200) GPU required"
)

_RAW = Path(__file__).resolve().parents[4] / "models/qwen3_5_moe/tests/fixtures/qwen3_5_35b_a3b.json"
SITE = "layers.0.mlp"
T = 4096
STEPS = 3
#: name -> (world, ep, etp)
LAYOUTS = {"ep1": (1, 1, 1), "ep2": (2, 2, 1), "ep4": (4, 4, 1), "etp4": (4, 1, 4)}
#: The impls this battery proves members of ``moe.combine.korder_fp32``.
MEMBERS = ("sm100_after_fc2_combine_v1", "sm100_gather_korder_v1")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view({torch.float32: torch.int32, torch.bfloat16: torch.int16}[t.dtype])


def _seeded(name: str, shape, scale: float, device) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(zlib.crc32(name.encode()))
    return torch.randn(shape, generator=g, device=device, dtype=torch.float32) * scale


def _scale(name: str) -> float:
    if name.startswith("router"):
        return 0.05
    return 0.02


def _fold_reference(rows: torch.Tensor, combine_rows: torch.Tensor) -> torch.Tensor:
    """The class's program in torch: one fp32 chain over the canonical slots, no tensor core."""
    order = combine_rows.long()
    acc = rows.index_select(0, order[:, 0]).clone()
    for j in range(1, order.shape[1]):
        acc = acc + rows.index_select(0, order[:, j])
    return acc


def _child(rank: int, world: int, ep: int, etp: int, port: int, out: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    import torch.distributed as dist

    from isoexec import nn as inn
    from isoexec.compiler.source import ModelSource
    from isoexec.ops.collectives.pik_tp_invariant import _leaf_dtype, _num_leaves, select_trainer_transport
    from isoexec.runtimes.parallel import colwise_ranges, validate_component_placement
    from isoexec.runtimes.trainer_installs import install_trainer_kernels
    from isoexec.schema import Axis, Placement

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    raw = json.loads(_RAW.read_text())
    # The production trainer runs with Megatron ``deterministic_mode`` off, so torch's
    # deterministic algorithms are OFF: every adjoint must be fixed-order on its own.
    deterministic = os.environ.get("ISOEXEC_EP_KORDER_DETERMINISTIC_MODE", "0") == "1"
    install_trainer_kernels("qwen3_5_35b_a3b", source=ModelSource("isoexec.models.qwen3_5_moe"), raw_config=raw,
                            deterministic_attention_backward=deterministic)
    select_trainer_transport("TRAINER")  # pik's tree on NCCL, as the production trainer installs it
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    text = raw["text_config"]
    H = text["hidden_size"]

    one = Axis(1, 0, None)
    tp_group = dist.new_group(list(range(world))) if world > 1 else None
    ep_group = dist.new_group(list(range(world))) if ep > 1 else None
    tp = Axis(world, rank, tp_group) if world > 1 else one
    placement = Placement(
        tp=tp, ep=Axis(ep, rank, ep_group) if ep > 1 else one, dp=one, pp=one,
        etp=tp if etp > 1 else one, cp=one, side="trainer", leaves=_num_leaves(),
        leaf_dtype="bf16" if _leaf_dtype() == torch.bfloat16 else "fp32", sp=world > 1,
    )
    module = inn.MoE(hidden=H, num_experts=text["num_experts"], top_k=text["num_experts_per_tok"],
                     intermediate=text["moe_intermediate_size"],
                     shared_intermediate=text["shared_expert_intermediate_size"],
                     routing="softmax_renorm", weight_after_fc2=True, device="meta")
    inn.place(module, placement, prefix=SITE)
    validate_component_placement(module, placement)  # the construction check the trainer runs
    resolved = {k: v.impl for k, v in inn.resolved_ops(module)[SITE].items() if hasattr(v, "impl")}

    # Materialize this rank's tensors from the GLOBAL values: experts by id, shared by TP slice.
    shared_axes = module.shared.tensor_axes()
    for name, p in list(module.named_parameters()):
        full = _seeded(name, p.shape, _scale(name), dev)
        axis = None
        if name.startswith("shared.") and world > 1:
            axis = shared_axes[name.split(".", 1)[1]]
        elif name.startswith("experts.") and etp > 1:
            # ETP splits an expert by role: gate/up output rows, the down contraction.
            axis = module.get_submodule(name.rsplit(".", 1)[0]).tensor_plan_metadata()[name.rsplit(".", 1)[1]].expert_axis
        if axis is not None:
            if axis.kind in ("col", "row"):
                d = 0 if axis.kind == "col" else 1
                (lo, hi), = colwise_ranges(full.shape[d], world, rank)
                full = full.narrow(d, lo, hi - lo)
        parent, leaf = module.get_submodule(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1]
        setattr(parent, leaf, torch.nn.Parameter(full.to(p.dtype).contiguous()))

    x_full = _seeded("x", (T, H), 1.0, dev).bfloat16()
    gy_full = _seeded("gy", (T, H), 1.0, dev)
    gs_full = _seeded("gs", (T, text["num_experts"]), 1.0, dev)
    lo, hi = (T * rank // world, T * (rank + 1) // world) if world > 1 else (0, T)

    def gather(t: torch.Tensor) -> torch.Tensor:
        if world == 1:
            return t
        parts = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(parts, t.contiguous())
        return torch.cat(parts)

    record: dict = {"resolved": resolved, "steps": []}
    with torch.no_grad():
        record["no_grad_out"] = gather(module(x_full[lo:hi].clone(), None)).cpu()
    for _ in range(STEPS):
        module.zero_grad(set_to_none=True)
        x = x_full[lo:hi].clone().requires_grad_(True)
        y, scores = module(x, None, return_aux=True)
        (y.float() * gy_full[lo:hi]).sum().add((scores * gs_full[lo:hi]).sum()).backward()
        grads = {n: p.grad.detach().cpu() for n, p in module.named_parameters() if p.grad is not None}
        record["steps"].append({"out": gather(y.detach()).cpu(), "dx": gather(x.grad).cpu(), "grads": grads})

    if world == 1:
        # The program: rows from the bound expert stage, folded here, plus the shared expert.
        with torch.no_grad():
            xx = x_full.clone()
            routing = module.slots.route_select(module.slots.route_score(xx, module.router.weight), module._spec, False)
            gate_up, down = module.expert_weight_views()
            routed = module.slots.expert_mlp(xx, routing, gate_up, down, weight_after_fc2=True)
            # Canonical slot j of token t holds that token's j-th smallest GLOBAL expert id.
            from isoexec.ops.moe.moe_sm100 import ops as sm100_ops

            plan = sm100_ops.route(routing.indices.contiguous().to(torch.int32), gate_up.size(0))
            assert torch.equal(plan.combine_rows, routed.combine_rows)
            expert_of_row = routing.indices.long().flatten()[plan.inverse.long()]
            record["slots_ascending"] = bool(torch.equal(
                expert_of_row[routed.combine_rows.long()], routing.indices.long().sort(dim=1).values))
            s = module.shared
            shared = module.slots.shared_expert(xx, s.gate, s.up, s.down, s.expert_gate)
            # nn.MoE returns the fold's own dtype here (fp32; the graph build narrows it at the
            # boundary), so the program is compared unnarrowed and narrowed.
            program = _fold_reference(routed.rows, routed.combine_rows) + shared
            kernel = module.slots.combine(routed, shared)
            record["program"] = program.cpu()
            record["kernel_fold"] = kernel.cpu()
            # The forward-only member launches the same kernel on the same rows.
            from isoexec.ops.moe.sm100 import combine as gather_korder

            record["gather_korder"] = gather_korder(routed, shared).cpu()
    torch.save(record, os.path.join(out, f"{rank}.pt"))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(name: str, root: Path) -> list[dict]:
    world, ep, etp = LAYOUTS[name]
    if torch.cuda.device_count() < world:
        pytest.skip(f"{name} needs {world} visible GPUs")
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    torch.multiprocessing.spawn(_child, args=(world, ep, etp, _free_port(), str(out)), nprocs=world, join=True)
    return [torch.load(out / f"{r}.pt", weights_only=False) for r in range(world)]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("ep_korder")
    return {name: _run(name, root) for name in LAYOUTS}


def test_layouts_bind_the_class_members(runs):
    for name, ranks in runs.items():
        for r in ranks:
            assert r["resolved"]["combine"] in MEMBERS, (name, r["resolved"])
            assert r["resolved"]["expert_mlp"] == "sm100_tcgen05_after_fc2_v1", (name, r["resolved"])


def test_ep1_is_the_class_program(runs):
    (ref,) = runs["ep1"]
    assert ref["slots_ascending"], "combine_rows does not walk ascending global expert id"
    assert torch.equal(_bits(ref["kernel_fold"]), _bits(ref["program"]))
    assert torch.equal(_bits(ref["gather_korder"]), _bits(ref["program"]))
    assert torch.equal(_bits(ref["steps"][0]["out"]), _bits(ref["program"]))
    assert torch.equal(_bits(ref["no_grad_out"]), _bits(ref["program"]))
    narrowed = ref["program"].to(torch.bfloat16)
    assert torch.equal(_bits(ref["steps"][0]["out"].to(torch.bfloat16)), _bits(narrowed))


@pytest.mark.parametrize("name", [n for n in LAYOUTS if n != "ep1"])
def test_forward_bits_are_layout_invariant(runs, name):
    want = runs["ep1"][0]["steps"][0]["out"]
    for r in runs[name]:
        for got in (r["no_grad_out"], *(s["out"] for s in r["steps"])):
            diff = (_bits(got) != _bits(want)).sum().item()
            assert diff == 0, f"{name}: {diff} of {want.numel()} output elements differ from EP1"


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_backward_is_deterministic_per_layout(runs, name):
    for rank, r in enumerate(runs[name]):
        first, *rest = r["steps"]
        assert torch.isfinite(first["dx"].float()).all()
        assert first["grads"]["router.weight"].abs().sum() > 0
        for again in rest:
            assert torch.equal(_bits(first["dx"]), _bits(again["dx"])), (name, rank)
            assert set(first["grads"]) == set(again["grads"])
            for n, g in first["grads"].items():
                assert torch.equal(_bits(g), _bits(again["grads"][n])), (name, rank, n)


def test_report_gradient_layout_differences(runs):
    """Informational: gradient EP invariance is not the contract."""
    ref = runs["ep1"][0]["steps"][0]
    for name in LAYOUTS:
        if name == "ep1":
            continue
        dx = runs[name][0]["steps"][0]["dx"]
        n = (_bits(dx) != _bits(ref["dx"])).sum().item()
        experts = {}
        for r in runs[name]:
            for k, g in r["steps"][0]["grads"].items():
                # ETP holds role slices of each expert; only whole experts compare elementwise.
                if k.startswith("experts.") and g.shape == ref["grads"][k].shape:
                    experts[k] = (_bits(g) != _bits(ref["grads"][k])).sum().item()
        print(f"[ep_korder] {name}: dx elements differing from EP1 {n}/{dx.numel()}; "
              f"expert grads differing {sum(1 for v in experts.values() if v)}/{len(experts)}")
