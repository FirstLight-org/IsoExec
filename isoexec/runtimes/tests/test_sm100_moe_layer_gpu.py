"""The SM100 MoE layer at Qwen3.5-35B-A3B shapes: engine == trainer forward bits, batch invariance,
and a deterministic trainer backward through every stage including the router.

Random weights at the released geometry; one B200.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or live_device_arch() != "sm100",
                                reason="SM100 (B200) GPU required")

_RAW = Path(__file__).resolve().parents[2] / "models/qwen3_5_moe/tests/fixtures/qwen3_5_35b_a3b.json"
_T = 512


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view({torch.float32: torch.int32, torch.bfloat16: torch.int16}[t.dtype])


@pytest.fixture(scope="module")
def layers():
    from isoexec import nn
    from isoexec.compiler.source import ModelSource
    from isoexec.runtimes.parallel import trivial_placement
    from isoexec.runtimes.trainer_installs import install_trainer_kernels

    raw = json.loads(_RAW.read_text())
    install_trainer_kernels("qwen3_5_35b_a3b", source=ModelSource("isoexec.models.qwen3_5_moe"), raw_config=raw)
    text = raw["text_config"]

    def build(side):
        module = nn.MoE(hidden=text["hidden_size"], num_experts=text["num_experts"],
                        top_k=text["num_experts_per_tok"], intermediate=text["moe_intermediate_size"],
                        shared_intermediate=text["shared_expert_intermediate_size"],
                        routing="softmax_renorm", weight_after_fc2=True, device="cuda")
        nn.place(module, trivial_placement(side=side), prefix="layers.0.mlp")
        return module

    trainer, engine = build("trainer"), build("engine")
    generator = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for parameter in trainer.parameters():
            parameter.copy_((torch.randn(parameter.shape, generator=generator, device="cuda") * 0.02).to(parameter.dtype))
    engine.load_state_dict(trainer.state_dict())
    x = torch.randn(_T, text["hidden_size"], generator=generator, device="cuda").bfloat16()
    return trainer, engine, x


def test_bindings_are_the_sm100_recipes(layers):
    from isoexec import nn

    trainer, engine, _ = layers
    for module in (trainer, engine):
        bound = nn.resolved_ops(module)["layers.0.mlp"]
        assert {name: bound[name].impl for name in ("route_score", "route_select", "expert_mlp", "combine",
                                                    "shared_expert")} == {
            "route_score": "sm100_router_gemm_fp32_v1", "route_select": "sm100_fused_topk_v2",
            "expert_mlp": "sm100_tcgen05_after_fc2_v1", "combine": "sm100_after_fc2_combine_v2",
            "shared_expert": "sm100_after_fc2_shared_v2"}


def test_engine_forward_equals_trainer_forward(layers):
    trainer, engine, x = layers
    with torch.no_grad():
        rollout = engine(x, None)
        scored = trainer(x, None)
    trained = trainer(x.clone().requires_grad_(True), None)
    assert torch.equal(_bits(rollout), _bits(scored))
    assert torch.equal(_bits(rollout), _bits(trained.detach()))


def test_engine_rows_do_not_depend_on_the_batch(layers):
    _, engine, x = layers
    with torch.no_grad():
        full = engine(x, None)
        for rows in (1, 37, 300):
            assert torch.equal(_bits(engine(x[:rows].clone(), None)), _bits(full[:rows])), rows


def test_trainer_backward_is_deterministic_and_reaches_the_router(layers):
    trainer, _, x = layers

    def step():
        trainer.zero_grad(set_to_none=True)
        inputs = x.clone().requires_grad_(True)
        out, scores = trainer(inputs, None, return_aux=True)
        assert scores.shape == (_T, trainer.num_experts) and scores.requires_grad
        (out.float().square().mean() + scores.square().mean()).backward()
        grads = {name: p.grad.clone() for name, p in trainer.named_parameters() if p.grad is not None}
        return inputs.grad.clone(), grads

    dx, grads = step()
    assert set(grads) == {name for name, p in trainer.named_parameters() if p.requires_grad}
    assert all(torch.isfinite(g).all() for g in (dx, *grads.values()))
    assert grads["router.weight"].abs().sum() > 0
    dx2, grads2 = step()
    assert torch.equal(_bits(dx), _bits(dx2))
    assert all(torch.equal(grads[name], grads2[name]) for name in grads)


def test_router_and_shared_expert_reach_no_host_installed_gemm(layers, monkeypatch):
    """The router and shared-expert GEMMs name their providers. A host may wrap ``matmul_persistent``
    with its own provider and another may not, and ``torch.mm`` reaches whichever
    ``aten::mm`` registration a host installed last; with both made to refuse, neither side's bits
    move, so the two processes run the same GEMMs whatever they installed."""
    from isoexec.ops.mm import mm_batch_invariant as V

    trainer, engine, x = layers
    with torch.no_grad():
        want = {"engine": engine(x, None), "trainer": trainer(x, None)}

    def host_installed(*args, **kwargs):
        raise AssertionError("an SM100 MoE GEMM reached a host-installed provider")

    stock = getattr(V, "_isoexec_stock_matmul_persistent", V.matmul_persistent)
    monkeypatch.setattr(V, "_isoexec_stock_matmul_persistent", stock, raising=False)
    monkeypatch.setattr(V, "matmul_persistent", host_installed)
    monkeypatch.setattr(torch, "mm", host_installed)
    with torch.no_grad():
        for side, module in (("engine", engine), ("trainer", trainer)):
            assert torch.equal(_bits(module(x, None)), _bits(want[side])), side
    assert torch.equal(_bits(want["engine"]), _bits(want["trainer"]))


def test_sorted_topk_adjoint_matches_the_eager_softmax_over_the_selected():
    from isoexec.ops.moe import routing
    from isoexec.ops.moe.sm100 import mechanics
    from isoexec.schema import Mlp

    spec = Mlp(intermediate_size=512, num_experts=256, top_k=8, routing="softmax_renorm")
    generator = torch.Generator(device="cuda").manual_seed(1)
    logits = torch.randn(_T, 256, generator=generator, device="cuda")
    upstream = torch.randn(_T, 8, generator=generator, device="cuda")
    native_in = logits.clone().requires_grad_(True)
    native = mechanics("softmax_renorm")(native_in, spec, False)
    eager_in = logits.clone().requires_grad_(True)
    eager = routing.softmax_renorm(eager_in, spec, want_scores_full=False)
    assert torch.equal(native.indices, eager.indices)
    torch.testing.assert_close(native.probs, eager.probs, rtol=1e-6, atol=1e-7)
    (native.probs * upstream).sum().backward()
    (eager.probs * upstream).sum().backward()
    torch.testing.assert_close(native_in.grad, eager_in.grad, rtol=1e-5, atol=1e-7)


def test_dispatch_gather_adjoint_matches_index_select_without_atomics():
    from isoexec.ops.moe.moe_sm100 import ops
    from isoexec.ops.moe.moe_sm100.integration import _Gather

    generator = torch.Generator(device="cuda").manual_seed(2)
    T, E, k, H = 300, 256, 8, 2048
    indices = torch.stack([torch.randperm(E, generator=generator, device="cuda")[:k] for _ in range(T)]).int()
    plan = ops.route(indices, E)
    inverse = plan.inverse.long()
    tokens = inverse.div(k, rounding_mode="floor")
    x = torch.randn(T, H, generator=generator, device="cuda").bfloat16().requires_grad_(True)
    probs = torch.rand(T, k, generator=generator, device="cuda").requires_grad_(True)
    gx, gp = _Gather.apply(x, probs, inverse, tokens, plan.restore)
    assert torch.equal(gx, x.detach().index_select(0, tokens))
    assert torch.equal(gp, probs.detach().flatten().index_select(0, inverse))
    ux = torch.randn(gx.shape, generator=generator, device="cuda").bfloat16()
    up = torch.randn(gp.shape, generator=generator, device="cuda")
    dx, dp = torch.autograd.grad((gx.float() * ux.float()).sum() + (gp * up).sum(), (x, probs))
    exact = x.detach().float().requires_grad_(True)
    rx, rp = torch.autograd.grad((exact.index_select(0, tokens) * ux.float()).sum()
                                 + (probs.flatten().index_select(0, inverse) * up).sum(), (exact, probs))
    # One fp32 sum per token rounded once to bf16: within half a bf16 ulp of the exact sum.
    torch.testing.assert_close(dx.float(), rx, rtol=2.0**-8, atol=1e-30)
    assert torch.equal(dp, rp)
    again = torch.autograd.grad((_Gather.apply(x, probs, inverse, tokens, plan.restore)[0].float() * ux.float()).sum(), x)[0]
    assert torch.equal(_bits(dx), _bits(again))
