"""Binding selection and side effects, without executing model operations."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import distributed as dist
from torch import fx, nn

import isoexec.nn as inn
from isoexec.core.bindings import BoundOp
from isoexec.compiler.realization.graph import clone_graph_module
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.partition import define_stage
from isoexec.compiler.tests.test_capture_cpu import Model, example_inputs
from isoexec.nn._seams import no_movement
from isoexec.ops import _dispatch
from isoexec.ops.table import OpResolutionError, OpTable
from isoexec.runtimes.megatron import _binding
from isoexec.runtimes.megatron._binding import bind_trainer_ops


def _must_not_execute(*args, **kwargs):
    raise AssertionError("Binding executed an operation")


@pytest.fixture
def tp_group():
    return dist.ProcessGroup(0, 1)


def _graph_module(component, *input_names):
    root = nn.Module()
    root.layer = component
    graph = fx.Graph()
    inputs = tuple(graph.placeholder(name) for name in input_names)
    graph.output(graph.call_module("layer", inputs))
    return fx.GraphModule(root, graph)


def test_binding_uses_declared_facts_without_placement_materialization_or_execution(monkeypatch, tp_group):
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    model = Model().eval()
    model.units[0].mix = inn.Attention(hidden=16, heads=4, kv_heads=2, head_dim=4,
                                     rope_theta=10000, eps=2e-6)
    model.units[1].mix = inn.GDN(hidden=16, key_heads=2, value_heads=4,
                               key_head_dim=4, value_head_dim=4, conv_kernel=4, eps=3e-6)
    captured = capture(model, example_inputs())
    source = define_stage(captured, ((0, 2),), pp_rank=0).module
    target = clone_graph_module(source)
    graph_before = str(target.graph)
    parameters = dict(target.named_parameters(remove_duplicate=False))
    reference, requests = OpTable(mode="reference"), {}
    row_parallel_requests = {}
    factory = _binding.bind_row_parallel

    def bind(op, *, site, **kwargs):
        requests[site] = (op, kwargs)
        return replace(reference.bind(op, site=site, **kwargs), fn=_must_not_execute)

    def row_parallel(*, tp_group, site, sp):
        bound = replace(factory(tp_group=tp_group, site=site, sp=sp), fn=_must_not_execute)
        row_parallel_requests[site] = (tp_group, bound)
        return bound

    monkeypatch.setattr(_binding, "bind_row_parallel", row_parallel)
    bind_trainer_ops(target, tp_group=tp_group, op_table=SimpleNamespace(bind=bind))

    assert requests["units.0.norm.norm"] == (
        "norms.rms.pre_attn", {"inputs": {"eps": 1e-6}, "convention": "rms_zero_centered", "facts": {"width": 16}},
    )
    assert requests["units.0.ffn.act"] == ("mlp.act", {"facts": {"activation": "silu_mul"}})
    assert requests["units.0.mix.core"] == (
        "attention.core.gqa", {"inputs": {"scale": 0.5, "head_dim": 4}},
    )
    assert requests["units.1.mix.out_norm"] == (
        "norms.gated_out", {"inputs": {"eps": 3e-6}, "convention": "gated_rms_zero_centered",
                            "facts": {"width": 4, "activation": "silu"}},
    )
    assert requests["readout.head"] == (
        "logprobs.head", {"inputs": {"tp": 1, "boundary_out_dtype": "fp32"}},
    )
    assert row_parallel_requests.keys() == {
        "units.0.ffn.row_parallel", "units.0.mix.row_parallel",
        "units.1.ffn.row_parallel", "units.1.mix.row_parallel",
    }
    for site, (group, bound) in row_parallel_requests.items():
        module = target.get_submodule(site.removesuffix(".row_parallel"))
        assert group is tp_group
        assert module.slots.resolved()["row_parallel"] is bound
        assert bound.impl == "pik_tree" and bound.fn is _must_not_execute
    assert str(source.graph) == str(target.graph) == graph_before
    assert target.get_parameter("lexicon.weight") is target.get_parameter("readout.weight")
    for name, parameter in parameters.items():
        assert target.get_parameter(name) is parameter and parameter.is_meta
        assert parameter.shape == source.get_parameter(name).shape
        assert parameter.dtype == source.get_parameter(name).dtype
    for site, module in target.named_modules():
        if isinstance(module, inn.Component):
            assert module.placement is None
            assert module.training is source.get_submodule(site).training
            assert not source.get_submodule(site).slots.resolved()
            assert not any(unbound.required for unbound in module.slots.unfilled().values())
            if "tp_copy" in module.slots:
                bound = module.slots.resolved()["tp_copy"]
                assert bound.impl == "none" and bound.fn is no_movement
                assert dict(bound.inputs) == {"tp": 1, "side": "trainer", "sp": False}
    attention = target.get_submodule("units.0.mix")
    kv_gather, cp_gather = (attention.slots.resolved()[slot] for slot in ("kv_gather", "cp_gather"))
    assert kv_gather.impl == cp_gather.impl == "none"
    assert kv_gather.fn is cp_gather.fn is no_movement
    assert dict(kv_gather.inputs) == {"tp": 1, "side": "trainer", "kv_heads": 2}
    assert dict(cp_gather.inputs) == {"cp": 1}
    gdn = target.get_submodule("units.1.mix")
    assert gdn.slots.resolved()["conv"].op == "conv.short"
    assert "conv.short" not in gdn.slots.resolved()["core"].ops


def test_production_norm_binding_selects_declared_provider_on_meta(monkeypatch, tp_group):
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    target = _graph_module(inn.RMSNorm(hidden=256, eps=2e-6, convention="rms", role="final"), "hidden")
    bind_trainer_ops(target, tp_group=tp_group)

    bound = target.layer.slots.resolved()["norm"]
    assert bound.op == "norms.rms.final"
    assert bound.provider == "ops.norms.norms_sm90" and bound.version is not None
    assert dict(bound.inputs) == {"eps": 2e-6, "convention": "rms"}
    assert target.layer.weight.is_meta and target.layer.placement is None


def test_fused_qk_rope_keeps_both_operations_and_marks_separate_norm_optional(monkeypatch, tp_group):
    attention = inn.Attention(hidden=16, heads=4, kv_heads=2, head_dim=4, rope_theta=10000)
    target = _graph_module(attention, "hidden", "seq")
    reference = OpTable(mode="reference")

    def bind(op, *, site, **kwargs):
        assert op != "rope.apply", "The fused q/k norm already owns rotation"
        if op == "norms.qk":
            return BoundOp(op, "test.fused_qk_rope", None, "test", fn=_must_not_execute,
                           owns={"norms.qk": "test.fused_qk_rope", "rope.apply": "test.fused_qk_rope"})
        return reference.bind(op, site=site, **kwargs)

    monkeypatch.setattr(_binding, "bind_row_parallel", lambda **kwargs: BoundOp(
        "collectives.row_parallel", "test.row_parallel", None, "test", fn=_must_not_execute,
    ))
    bind_trainer_ops(target, tp_group=tp_group, op_table=SimpleNamespace(bind=bind))
    assert not attention.slots.unfilled()["qk_norm"].required
    assert set(attention.slots.resolved()["qk_rope"].ops) == {"norms.qk", "rope.apply"}


def test_missing_gdn_operation_ownership_refuses_binding_without_changing_source(tp_group):
    source = _graph_module(inn.GDN(hidden=16, key_heads=2, value_heads=4,
                                   key_head_dim=4, value_head_dim=4, conv_kernel=4), "hidden", "seq")
    target = clone_graph_module(source)
    reference = OpTable(mode="reference")

    def bind(op, *, site, **kwargs):
        if op == "gdn.core":
            return BoundOp(op, "test.missing_gating", None, "test", fn=_must_not_execute,
                           owns={"gdn.l2norm": "test.missing_gating"})
        return reference.bind(op, site=site, **kwargs)

    with pytest.raises(OpResolutionError, match=r"layer\.core.*gdn\.gating"):
        bind_trainer_ops(target, tp_group=tp_group, op_table=SimpleNamespace(bind=bind))
    assert not source.layer.slots.resolved()


def test_unsupported_norm_width_refuses_at_its_module_site(monkeypatch, tp_group):
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    target = _graph_module(inn.RMSNorm(hidden=17), "hidden")
    with pytest.raises(OpResolutionError, match=r"layer\.norm.*17"):
        bind_trainer_ops(target, tp_group=tp_group)


def test_tp2_group_binds_sharded_head_operations(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    target = _graph_module(inn.LMHead(vocab=32, hidden=16), "hidden")
    group = dist.ProcessGroup(0, 2)
    bind_trainer_ops(target, tp_group=group)
    assert target.layer.tp_group is group
    assert target.layer.slots.resolved()["head"].inputs["tp"] == 2
    assert target.layer.slots.resolved()["tp_copy"].impl == "identity_fwd_tree_sum_bwd"


@pytest.mark.parametrize("rank", [0, 1])
def test_native_head_masks_local_padding_and_trims_gathered_logits(rank):
    head = inn.LMHead(vocab=3, padded_vocab=8, hidden=4, device="cpu")
    head.tp_group = dist.ProcessGroup(rank, 2)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.tensor([[1.0, 2.0, 3.0, -float("inf")]]) if rank == 0 else torch.full((1, 4), -float("inf"))
    assert torch.equal(head.trim_to_vocab(logits), expected)
    gathered = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    assert torch.equal(head.trim_to_vocab(gathered), torch.tensor([[0.0, 1.0, 2.0]]))


def test_tp2_gdn_norm_keeps_the_bound_kernel_callable(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    target = _graph_module(inn.GDN(hidden=8, key_heads=2, value_heads=4,
                                  key_head_dim=4, value_head_dim=4, conv_kernel=4), "hidden", "seq")
    reference = OpTable(mode="reference")

    def bind(op, **kwargs):
        bound = reference.bind(op, **kwargs)
        if op == "norms.gated_out":
            return replace(bound, fn=lambda output, gate, weight, eps: output * weight + gate)
        return bound

    bind_trainer_ops(target, tp_group=dist.ProcessGroup(0, 2), op_table=SimpleNamespace(bind=bind))
    output = torch.ones(1, 2, 4)
    gate = torch.full_like(output, 2)
    weight = torch.arange(4, dtype=torch.float32)
    assert torch.equal(target.layer.slots.out_norm(output, gate, weight, 1e-6), output * weight + gate)


@pytest.mark.parametrize("rank", [0, 1])
def test_ep2_moe_binding_keeps_global_experts_and_refuses_an_unqualified_combine(monkeypatch, rank):
    from isoexec.runtimes.parallel import ShardError, combine_twin_members

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    one = dist.ProcessGroup(0, 1)
    groups = SimpleNamespace(tp=one, ep=dist.ProcessGroup(rank, 2), dp=one, pp=one, cp=one, expt_tp=one)
    placement = _binding.placement_from_groups(groups)

    def moe():
        return _graph_module(inn.MoE(hidden=8, num_experts=4, top_k=2, intermediate=6, shared_intermediate=4), "x")

    target = moe()
    _binding.bind_trainer_ops(target, tp_group=one, placement=placement)
    held = sorted({int(name.split(".")[2]) for name, _ in target.named_parameters() if ".experts." in name})
    assert held == [2 * rank, 2 * rank + 1]
    assert target.layer.slots.resolved()["combine"].impl in combine_twin_members()
    # A combine outside moe.combine.korder_fp32 may fold experts in another order at another EP layout.
    with pytest.raises(ShardError, match="korder_fp32"):
        _binding.bind_trainer_ops(moe(), tp_group=one, placement=placement, op_table=OpTable(mode="reference"))



@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_ep2_moe_binding_holds_whole_experts(monkeypatch, rank):
    from isoexec.runtimes.parallel import combine_twin_members

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    one, pair = dist.ProcessGroup(0, 1), dist.ProcessGroup(rank, 2)
    # TP2 EP2 runs with ETP=1: dispatch splits tokens over TP and experts stay whole.
    groups = SimpleNamespace(tp=pair, ep=pair, dp=one, pp=one, cp=one, expt_tp=one)
    placement = _binding.placement_from_groups(groups)
    target = _graph_module(inn.MoE(hidden=8, num_experts=4, top_k=2, intermediate=6, shared_intermediate=4), "x")

    _binding.bind_trainer_ops(target, tp_group=pair, placement=placement)

    assert (placement.tp.size, placement.ep.size, placement.etp.size) == (2, 2, 1)
    held = sorted({int(name.split(".")[2]) for name, _ in target.named_parameters() if ".experts." in name})
    assert held == [2 * rank, 2 * rank + 1]
    assert target.layer.slots.resolved()["combine"].impl in combine_twin_members()
    assert target.layer.slots.resolved()["expert_mlp"].inputs["dispatch_shard"] == 2


def test_placement_bound_components_bind_themselves_on_the_trainer(tp_group):
    from isoexec.runtimes.vllm.tests.test_definition_cpu import _Bound
    from isoexec.schema import Axis, Placement

    target = _graph_module(_Bound(), "x")
    with pytest.raises(OpResolutionError, match="requires a placement"):
        bind_trainer_ops(target, tp_group=tp_group)
    one = Axis(1, 0, None)
    placement = Placement(tp=Axis(1, 0, tp_group), ep=one, dp=one, pp=one, etp=one, cp=one, side="trainer", leaves=1)
    target = _graph_module(_Bound(), "x")
    bind_trainer_ops(target, tp_group=tp_group, placement=placement)
    assert target.layer.placed_with is placement and target.layer.site == "layer"
    assert target.layer.norm.site is None


class _Own(inn.Component):
    """A package's own (not placement-bound) component."""

    def __init__(self):
        super().__init__()
        self.placed_with = None

    def facts(self):
        return {"kind": "own"}

    def _place(self, placement):
        self.placed_with = placement

    def forward(self, x):
        return x


def _two(first, second):
    root = nn.Module()
    root.first, root.second = first, second
    graph = fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_module("second", (graph.call_module("first", (x,)),)))
    return fx.GraphModule(root, graph)


@pytest.mark.parametrize("side", ["trainer", "engine"])
def test_package_components_place_once_beside_a_release_moe(monkeypatch, side):
    """Under a placement a package's own component places itself; the release MoE and its children bind
    exactly as before (nothing placed twice, the same resolved ops)."""
    from isoexec.runtimes.vllm import _binding as engine

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    one = dist.ProcessGroup(0, 1)
    placement = _binding.placement_from_groups(SimpleNamespace(tp=one, ep=one, dp=one, pp=one, cp=one, expt_tp=one))
    if side == "engine":
        placement = replace(placement, side="engine")
    calls = []
    original = inn.Component.apply_placement

    def record(self, *args, **kwargs):
        calls.append(id(self))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(inn.Component, "apply_placement", record)

    def moe():
        return inn.MoE(hidden=8, num_experts=4, top_k=2, intermediate=6, shared_intermediate=4)

    def bind(target):
        if side == "trainer":
            _binding.bind_trainer_ops(target, tp_group=one, placement=placement)
        else:
            engine.bind_body_ops(target, tp_group=one, placement=placement)

    alone = _graph_module(moe(), "x")
    bind(alone)
    calls.clear()
    target = _two(moe(), _Own())
    bind(target)
    assert len(calls) == len(set(calls)) and target.second.placed_with is placement
    want = {k: v.impl for k, v in alone.layer.slots.resolved().items()}
    assert {k: v.impl for k, v in target.first.slots.resolved().items()} == want
    if side == "trainer":      # without a placement the trainer still refuses an unknown kind
        with pytest.raises(Exception, match="trainer binding does not support"):
            _binding.bind_trainer_ops(_graph_module(_Own(), "x"), tp_group=one)
