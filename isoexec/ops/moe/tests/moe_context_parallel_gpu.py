"""The trainer's ``nn.MoE`` under a CP=2 cut, with and without EP=2, on one SM90 or SM100 GPU (two Gloo ranks).

Each rank holds Megatron's zigzag rows of a packed batch (chunks ``r`` and ``2*cp-1-r`` of every sequence)
and runs the released Qwen3.5-35B-A3B mixture -- H 2048, 256 experts, top 8, I 512, a gated 512-wide
shared expert, the live arch's after-fc2 experts and korder fold -- at

* CP2 EP1 -- every expert on both ranks, each rank routes and folds its own rows;
* CP2 EP2 -- the CP pair is the EP group: each rank owns 128 experts and serves both ranks' routes,

against CP1 EP1 computed in the same process on the whole batch. Claims, per rank and layout:

1. the output rows and the router scores equal the CP1 rows bitwise;
2. dx and the score-gradient path give every local row the CP1 input-gradient bits;
3. backward is deterministic (two steps, identical bits for dx and every parameter gradient).

Expert, router and shared weight gradients are sums over a different token split and are only reported.
Gloo moves the bytes (``_gloo_cuda``); what this harness cannot show is NCCL itself. Run directly.
"""

from __future__ import annotations

import json
import tempfile
import zlib
from pathlib import Path

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() not in ("sm90", "sm100"),
                                reason="requires an SM90 or SM100 GPU")

_RAW = Path(__file__).resolve().parents[3] / "models/qwen3_5_moe/tests/fixtures/qwen3_5_35b_a3b.json"
SITE = "layers.0.mlp"
CU = [0, 2400, 3600, 4096]
STEPS = 2


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().view(-1).view(torch.uint8)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _seeded(name: str, shape, scale: float, device) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(zlib.crc32(name.encode()))
    return torch.randn(shape, generator=g, device=device, dtype=torch.float32) * scale


def _module(placement, text, dev):
    from isoexec import nn as inn
    from isoexec.runtimes.parallel import validate_component_placement

    module = inn.MoE(hidden=text["hidden_size"], num_experts=text["num_experts"], top_k=text["num_experts_per_tok"],
                     intermediate=text["moe_intermediate_size"],
                     shared_intermediate=text["shared_expert_intermediate_size"],
                     routing="softmax_renorm", weight_after_fc2=True, device="meta")
    inn.place(module, placement, prefix=SITE)
    validate_component_placement(module, placement)
    # Every rank materializes its held tensors from the same global values (experts keep global ids).
    for name, p in list(module.named_parameters()):
        full = _seeded(name, p.shape, 0.05 if name.startswith("router") else 0.02, dev)
        parent, leaf = module.get_submodule(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1]
        setattr(parent, leaf, torch.nn.Parameter(full.to(p.dtype).contiguous()))
    return module


def _steps(module, x_full, gy, gs):
    """STEPS forward/backward passes; returns the last pass and whether every pass had the same bits."""
    runs = []
    for _ in range(STEPS):
        module.zero_grad(set_to_none=True)
        x = x_full.clone().requires_grad_(True)
        y, scores = module(x, None, return_aux=True)
        (y.float() * gy).sum().add((scores * gs).sum()).backward()
        runs.append({"out": y.detach(), "scores": scores.detach(), "dx": x.grad,
                     "grads": {n: p.grad.detach().clone() for n, p in module.named_parameters() if p.grad is not None}})
    first, last = runs[0], runs[-1]
    deterministic = (_same(first["dx"], last["dx"]) and _same(first["out"], last["out"])
                     and all(_same(first["grads"][n], g) for n, g in last["grads"].items()))
    return last, deterministic


def _body(rank, world, out, init_method):
    from isoexec.ops.collectives.cp_layout import local_to_global
    from isoexec.ops.collectives.pik_tp_invariant import _leaf_dtype, _num_leaves
    from isoexec.ops.collectives.tests._gloo_cuda import init_world
    from isoexec.compiler.source import ModelSource
    from isoexec.runtimes.trainer_installs import install_trainer_kernels
    from isoexec.schema import Axis, Placement

    group = init_world(rank, world, init_method)
    raw = json.loads(_RAW.read_text())
    install_trainer_kernels("qwen3_5_35b_a3b", source=ModelSource("isoexec.models.qwen3_5_moe"), raw_config=raw,
                            deterministic_attention_backward=False)
    text, dev = raw["text_config"], torch.device("cuda")
    one = Axis(1, 0, None)
    leaves = dict(leaves=_num_leaves(), leaf_dtype="bf16" if _leaf_dtype() == torch.bfloat16 else "fp32")

    T, H, E = CU[-1], text["hidden_size"], text["num_experts"]
    x_full = _seeded("x", (T, H), 1.0, dev).bfloat16()
    gy_full, gs_full = _seeded("gy", (T, H), 1.0, dev), _seeded("gs", (T, E), 1.0, dev)
    mine = torch.tensor(local_to_global(CU, world, rank), device=dev)

    ref, ref_det = _steps(_module(Placement(tp=one, ep=one, dp=one, pp=one, etp=one, cp=one, side="trainer",
                                            **leaves), text, dev), x_full, gy_full, gs_full)
    res = {"cp1_deterministic": ref_det}
    for ep in (1, 2):
        placement = Placement(tp=one, ep=Axis(ep, rank, group) if ep > 1 else one, dp=one, pp=one, etp=one,
                              cp=Axis(world, rank, group), side="trainer", **leaves)
        got, det = _steps(_module(placement, text, dev), x_full[mine], gy_full[mine], gs_full[mine])
        tag = f"cp2_ep{ep}"
        res[f"{tag}_out"] = _same(got["out"], ref["out"][mine])
        res[f"{tag}_scores"] = _same(got["scores"], ref["scores"][mine])
        res[f"{tag}_dx"] = _same(got["dx"], ref["dx"][mine])
        res[f"{tag}_deterministic"] = det
        # Weight gradients: token-split sums, reported only (the CP pair's sum is the CP1 gradient's value).
        report = {}
        for name, g in got["grads"].items():
            total = g.float().clone()
            if ep == 1 or not name.startswith("experts."):
                torch.distributed.all_reduce(total, group=group)
            want = ref["grads"][name].float()
            report[name.split(".")[0]] = max(report.get(name.split(".")[0], 0.0),
                                             float((total - want).abs().max() / want.abs().max().clamp_min(1e-30)))
        res[f"{tag}_grad_rel_maxdiff"] = report
    out[rank] = res


def run() -> dict:
    from isoexec.ops.collectives.tests._gloo_harness import run_world

    with tempfile.TemporaryDirectory() as tmp:
        return run_world(_body, world=2, rendezvous=f"{tmp}/rdzv", timeout_s=3600)


@pytest.fixture(scope="module")
def results():
    return run()


CLAIMS = ("cp1_deterministic",) + tuple(f"cp2_ep{ep}_{c}" for ep in (1, 2)
                                        for c in ("out", "scores", "dx", "deterministic"))


@pytest.mark.parametrize("claim", CLAIMS)
def test_cp2_moe_rows_equal_cp1(results, claim):
    from isoexec.ops.collectives.tests._gloo_harness import assert_claim

    assert_claim(results, claim, 2)


if __name__ == "__main__":
    results = run()
    for rank, res in sorted(results.items()):
        print(rank, res)
    failed = [c for c in CLAIMS for r in results.values() if not r[c]]
    raise SystemExit(1 if failed else 0)
