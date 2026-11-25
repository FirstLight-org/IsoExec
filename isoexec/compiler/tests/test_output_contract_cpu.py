"""Logical output metadata and the explicit MoE output rounding boundary."""

from collections.abc import Callable

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor

import isoexec.nn as inn
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.text import text_inputs
from isoexec.ops.moe.reference import (
    ref_combine, ref_expert_mlp, ref_route_mechanics, ref_route_score, ref_shared_expert,
)
from isoexec.ops.table import OpTable
from isoexec.runtimes.parallel import trivial_placement
from isoexec.schema import SequenceContext, make_sequence_context


def _sequence() -> SequenceContext:
    return make_sequence_context(torch.arange(4), torch.tensor([0, 4], dtype=torch.int32), max_q=4)


def _fill(module: nn.Module) -> None:
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)


def _moe() -> inn.MoE:
    return inn.MoE(hidden=16, num_experts=4, top_k=2, intermediate=8,
                   shared_intermediate=8, device="cpu")


def _gradients(module: nn.Module, forward: Callable[[], torch.Tensor]):
    module.zero_grad(set_to_none=True)
    output = forward()
    output.float().square().sum().backward()
    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        gradients[name] = parameter.grad.clone()
    return output.detach(), gradients


def _equal(left: torch.Tensor, right: torch.Tensor) -> None:
    assert left.dtype == right.dtype
    assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_moe_rounds_after_complete_combine_and_preserves_unprepared_output(input_dtype: torch.dtype):
    original, prepared = _moe(), _moe()
    _fill(original)
    prepared.load_state_dict(original.state_dict())
    before = dict(original.facts())
    prepared.prepare_output_contract()
    assert original.facts() == before and prepared.facts() != before
    x = torch.randn(4, 16, generator=torch.Generator().manual_seed(29), dtype=input_dtype)
    x.requires_grad_()
    seq = _sequence()
    assert prepared.abstract_output(x, seq).dtype == input_dtype
    for module in (original, prepared):
        inn.place(module, trivial_placement(side="trainer"), prefix="mixture", op_table=OpTable(mode="reference"))
    assert original(x, seq).dtype == torch.float32
    expected, expected_grads = _gradients(original, lambda: original(x, seq).to(input_dtype))
    assert x.grad is not None
    expected_input_grad = x.grad.clone()
    x.grad = None
    actual, actual_grads = _gradients(prepared, lambda: prepared(x, seq))
    _equal(actual, expected)
    assert x.grad is not None
    _equal(x.grad, expected_input_grad)
    for name in expected_grads:
        _equal(actual_grads[name], expected_grads[name])

    # Premature BF16 rounding of routed/shared terms is not the input-dtype contract.
    routing = ref_route_mechanics(ref_route_score(x, original.router.weight), original._spec)
    routed = ref_expert_mlp(x, routing, *original.expert_weight_views())
    shared = original.shared
    assert shared is not None
    shared_output = ref_shared_expert(x, shared.gate, shared.up, shared.down, shared.expert_gate)
    early = (ref_combine(routed, None).to(torch.bfloat16) + shared_output.to(torch.bfloat16)).to(input_dtype)
    assert not torch.equal(actual, early)


@pytest.mark.parametrize("kind", ["embedding", "head", "norm", "mlp", "moe", "block"])
def test_abstract_forward_matches_real_metadata_without_placing_original(kind: str):
    factories: dict[str, Callable[[], inn.Component]] = {
        "embedding": lambda: inn.Embedding(vocab=32, hidden=16, device="cpu"),
        "head": lambda: inn.LMHead(vocab=32, padded_vocab=40, hidden=16, device="cpu"),
        "norm": lambda: inn.RMSNorm(hidden=16, device="cpu"),
        "mlp": lambda: inn.MLP(hidden=16, intermediate=8, device="cpu"),
        "moe": _moe,
        "block": lambda: inn.DecoderBlock(hidden=16, device="cpu",
            mixer=inn.MLP(hidden=16, intermediate=8, device="cpu"), mlp=_moe()),
    }
    module = factories[kind]()
    for child in module.modules():
        if isinstance(child, inn.Component):
            child.prepare_output_contract()
    _fill(module)
    x = (torch.arange(4) if kind == "embedding" else torch.ones(4, 16, dtype=torch.bfloat16))
    args = (x, _sequence()) if kind in ("mlp", "moe", "block") else (x,)
    parameters = tuple(module.parameters())
    fake = module.abstract_output(*args)
    assert isinstance(fake, FakeTensor)
    assert all(child.placement is None for child in module.modules() if isinstance(child, inn.Component))
    assert all(left is right for left, right in zip(parameters, module.parameters(), strict=True))
    inn.place(module, trivial_placement(side="trainer"), prefix="fixture", op_table=OpTable(mode="reference"))
    actual = module(*args)
    assert (fake.shape, fake.dtype, fake.device, fake.stride()) == (
        actual.shape, actual.dtype, actual.device, actual.stride(),
    )


def test_abstract_mla_rejects_incompatible_operands_before_runtime():
    module = inn.LatentAttention(hidden=16, heads=4, q_lora_rank=8, kv_lora_rank=8,
        qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4, rope_theta=10000, device="cpu")
    with pytest.raises(TypeError, match="matching latent and up-projection dtypes"):
        module.abstract_output(torch.ones(4, 16, dtype=torch.float32), _sequence())
    assert module.placement is None


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("routing", ["softmax_renorm", "sigmoid_noaux_bias_select"])
@pytest.mark.parametrize("shared_intermediate,top_k", [(8, 2), (0, 1), (8, 4)],
                         ids=["shared", "no_shared_top1", "all_experts"])
def test_export_preserves_authored_operations_after_moe_and_real_op_table(
    input_dtype: torch.dtype, routing: str, shared_intermediate: int, top_k: int,
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = inn.Embedding(vocab=32, hidden=16, device="cpu")
            self.layers = nn.ModuleList([inn.MoE(
                hidden=16, num_experts=4, top_k=top_k, intermediate=8,
                shared_intermediate=shared_intermediate, routing=routing, device="cpu",
            )])
            self.head = inn.LMHead(vocab=32, padded_vocab=40, hidden=16, device="cpu")

        def forward(self, input_ids, seq):
            x = self.embed(input_ids).to(input_dtype)
            for layer in self.layers:
                x = layer(x, seq)
            x += 0.125 if x.dtype == torch.bfloat16 else 0.25
            return self.head(x)

    model = Model()
    _fill(model)
    examples = text_inputs(model, device="cpu", max_tokens=8)
    result = capture(model, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes)
    assert [layer.path for layer in result.layers] == ["layers.0"]
    assert result.program.get_submodule("layers.0") is model.layers[0]
    assert all("isoexec_capture" not in str(node.target) for node in result.program.graph.nodes)
    table = OpTable(mode="reference")
    inn.place(model, trivial_placement(side="trainer"), op_table=table)
    assert model.layers[0]._op_table is table
    for tokens in (1, 4):
        ids = torch.arange(tokens)
        cu = torch.tensor([0, tokens], dtype=torch.int32)
        seq = make_sequence_context(torch.arange(tokens), cu, max_q=tokens, max_kv=tokens,
                                    input_ids=ids, cu_seqlens_cpu=cu.clone())
        assert model.layers[0](model.embed(ids).to(input_dtype), seq).dtype == input_dtype
        expected, expected_grads = _gradients(model, lambda: model(ids, seq))
        actual, actual_grads = _gradients(model, lambda: result.program(ids, seq))
        _equal(actual, expected)
        for name in expected_grads:
            _equal(actual_grads[name], expected_grads[name])
