"""VllmModel construction preserves source state, names and local ties."""

import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import distributed as dist
from torch import fx, nn

import isoexec.nn as inn
from isoexec.core.bindings import BoundOp
from isoexec.compiler.realization.graph import clone_graph_module
from isoexec.compiler.realization.storage import allocate_tensors
from isoexec.compiler.realization.tests import test_weights_cpu
from isoexec.compiler.partition import StageDefinition
from isoexec.compiler.source import ModelSource
from isoexec.nn._seams import no_movement
from isoexec.ops import _dispatch
from isoexec.ops.logprobs.eager import head_bf16
from isoexec.ops.table import OpTable
from isoexec.runtimes.engine_ops import lm_head_slice_sampled_rows
from isoexec.runtimes.vllm import _binding
from isoexec.runtimes.vllm._binding import bind_body_ops, bind_head_ops
from isoexec.runtimes.vllm._definition import _split_body_and_head
from isoexec.runtimes.vllm.model import VllmModel
from isoexec.runtimes.vllm.tests._gdn_fixtures import install_gdn


@pytest.fixture(autouse=True)
def production_bindings(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")


@pytest.fixture
def tp_group(monkeypatch):
    group = dist.ProcessGroup(0, 1)
    distributed = ModuleType("vllm.distributed")
    distributed.get_tp_group = lambda: SimpleNamespace(device_group=group)
    distributed.get_pp_group = lambda: SimpleNamespace(device_group=dist.ProcessGroup(0, 1))
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    # vLLM's default stage layout without VLLM_PP_LAYER_PARTITION.
    utils = ModuleType("vllm.distributed.utils")
    utils.get_pp_indices = lambda layers, rank, size: (layers * rank // size, layers * (rank + 1) // size)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    return group


@pytest.fixture
def hf_config():
    return test_weights_cpu.qwen_config()


@pytest.fixture
def vllm_config(monkeypatch, tp_group):
    reference, production = OpTable(mode="reference"), OpTable()

    def bind(op, **kwargs):
        table = production if op == "logprobs.head" else reference
        return table.bind(op, **kwargs)

    factory = _binding.bind_row_parallel

    def row_parallel(*, tp_group: dist.ProcessGroup, site):
        assert tp_group is group
        return factory(tp_group=tp_group, site=site)

    group = tp_group
    monkeypatch.setattr(_binding, "OpTable", lambda: SimpleNamespace(mode="reference", bind=bind))
    monkeypatch.setattr(_binding, "bind_row_parallel", row_parallel)

    def allocate_on_cpu(module, *, device):
        assert device == torch.device("cuda", 3)
        allocate_tensors(module, device=torch.device("cpu"))

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr("isoexec.runtimes.vllm.model.allocate_tensors", allocate_on_cpu)
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16, enforce_eager=True, max_model_len=128),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype="auto", enable_prefix_caching=False,
                                     mamba_cache_dtype="auto", mamba_ssm_cache_dtype="float32"),
        scheduler_config=SimpleNamespace(max_num_seqs=4, enable_chunked_prefill=False),
        compilation_config=SimpleNamespace(static_forward_context={}),
        quant_config=None,
        speculative_config=None,
    )

    class NativeAttention(nn.Module):
        def __init__(self, *, num_heads, head_size, scale, num_kv_heads, cache_config,
                     quant_config, prefix, attn_backend):
            super().__init__()
            self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
            self.head_size = self.head_size_v = head_size
            self.impl = SimpleNamespace(scale=scale)
            self.cache_config, self.quant_config = cache_config, quant_config
            self.layer_name, self.attn_backend = prefix, attn_backend
            self.kv_cache = torch.empty(0)
            self.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
            config.compilation_config.static_forward_context[prefix] = self

    class PyTorchVarlenAttentionBackend:
        forward_includes_kv_cache_update = False

    attention = ModuleType("vllm.model_executor.layers.attention")
    attention.Attention = NativeAttention
    backend = ModuleType("isoexec.ops.attention.varlen_backend")
    backend.PyTorchVarlenAttentionBackend = PyTorchVarlenAttentionBackend
    monkeypatch.setitem(sys.modules, attention.__name__, attention)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    install_gdn(monkeypatch)
    return config


def _module(*, tied=True, auxiliary=False, keyword=False):
    root = nn.Module()
    root.embed = inn.Embedding(vocab=8, hidden=4)
    layer = nn.Module()
    layer.attn = inn.Attention(hidden=4, heads=2, kv_heads=1, head_dim=16, rope_theta=10000,
                               rotary_fraction=0.5, mrope_section=(1, 1, 2), mrope_interleaved=True, eps=2e-6)
    layer.gdn = inn.GDN(hidden=4, key_heads=1, value_heads=2, key_head_dim=4, value_head_dim=4,
                        conv_kernel=4, eps=3e-6)
    layer.mlp = inn.MLP(hidden=4, intermediate=8)
    root.layers = nn.ModuleDict({"12": layer})
    root.norm = inn.RMSNorm(hidden=4, eps=1e-6)
    root.head = inn.LMHead(vocab=8, hidden=4, tied_to=root.embed.weight if tied else None)
    root.embed.weight.requires_grad_(False)
    root.register_buffer("offset", torch.empty(4, dtype=torch.bfloat16, device="meta"), persistent=False)
    root.register_buffer("scale", torch.tensor(0.5, dtype=torch.float32))
    graph = fx.Graph()
    ids, seq = graph.placeholder("input_ids"), graph.placeholder("seq")
    hidden = graph.call_module("embed", (ids,))
    hidden = graph.call_module("layers.12.attn", (hidden, seq))
    hidden = graph.call_module("layers.12.gdn", (hidden, seq))
    hidden = graph.call_module("layers.12.mlp", (hidden, seq))
    hidden = graph.call_function(torch.add, (hidden, graph.get_attr("offset")))
    hidden = graph.call_function(torch.mul, (hidden, graph.get_attr("scale")))
    hidden = graph.call_module("norm", (hidden,))
    aux = graph.call_function(torch.mean, (hidden,)) if auxiliary else None
    if keyword:
        logits = graph.call_module("head", kwargs={"hidden": hidden})
    else:
        logits = graph.call_module("head", (hidden,))
    hidden.meta["val"] = torch.empty(5, 4, dtype=torch.bfloat16, device="meta")
    logits.meta["val"] = torch.empty(5, 8, dtype=torch.float32, device="meta")
    if aux is not None:
        aux.meta["val"] = torch.empty((), dtype=torch.bfloat16, device="meta")
        output = graph.output((logits, {"aux": aux}))
        output.meta["val"] = (logits.meta["val"], {"aux": aux.meta["val"]})
    else:
        output = graph.output(logits)
        output.meta["val"] = logits.meta["val"]
    source = fx.GraphModule(root, graph).eval()
    source.register_buffer("offset", source.get_buffer("offset"), persistent=False)
    return source


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_constructor_preserves_source_objects_names_and_embedding_head_ties(tied, auxiliary, vllm_config, hf_config):
    source = _module(tied=tied, auxiliary=auxiliary)
    source_graph = str(source.graph)
    source_head = source.get_submodule("head")
    parameters = dict(source.named_parameters(remove_duplicate=False))
    offset = source.get_buffer("offset")
    scale = source.get_buffer("scale")
    source_output = next(node for node in source.graph.nodes if node.op == "output")
    source_output_value = source_output.meta["val"]

    model = VllmModel(StageDefinition(source, None),
                      source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                      vllm_config=vllm_config, prefix="model")
    body, head = model.body, model.head

    assert str(source.graph) == source_graph and source.get_submodule("head") is source_head
    assert source.get_buffer("offset") is offset and offset.is_meta
    assert source.get_buffer("scale") is scale and scale.item() == 0.5
    assert not source_head.slots.resolved()
    assert body is not source and head is not source_head
    assert set(head.slots.resolved()) == {"head", "lm_head_slice", "tp_copy"}
    assert model.training is body.training is head.training is False
    assert not hasattr(body, "head")
    body_output = next(node for node in body.graph.nodes if node.op == "output")
    assert isinstance(body_output.args[0], fx.Node)
    assert body_output.args[0].op == "call_module" and body_output.args[0].target == "norm"
    assert body_output.meta["val"].shape == (5, 4)
    assert source_output.meta["val"] is source_output_value
    assert all(node.target is not torch.mean for node in body.graph.nodes)
    assert (head.weight is body.get_parameter("embed.weight")) is tied

    assert set(dict(model.named_parameters(remove_duplicate=False))) == {
        name if name == "head.weight" else f"body.{name}" for name in parameters
    }
    for name, original in parameters.items():
        assert source.get_parameter(name) is original and original.is_meta
        copied = model.get_parameter(name if name == "head.weight" else f"body.{name}")
        assert copied is not original and copied.device == torch.device("cpu")
        assert (copied.shape, copied.dtype, copied.requires_grad) == (
            original.shape, original.dtype, original.requires_grad,
        )
    assert body.get_buffer("offset") is not offset
    assert body.get_buffer("offset").shape == offset.shape and body.get_buffer("offset").dtype == offset.dtype
    assert all(buffer.device == torch.device("cpu") for buffer in model.buffers())
    assert body.get_buffer("scale") is not scale
    assert body.get_buffer("scale").dtype == torch.float32 and torch.equal(body.get_buffer("scale"), scale)
    assert "offset" not in source.state_dict() and "body.offset" not in model.state_dict()
    assert "scale" in source.state_dict() and "body.scale" in model.state_dict()
    for path, module in body.named_modules():
        if isinstance(module, (inn.Embedding, inn.MLP, inn.Attention)):
            assert not module.slots.unfilled()
        elif isinstance(module, inn.RMSNorm):  # add_norm: filled only where the fusion pass rewrote a join
            assert set(module.slots.unfilled()) <= {"add_norm"}
            assert not any(slot.required for slot in module.slots.unfilled().values())
        elif isinstance(module, inn.GDN):
            assert set(module.slots.unfilled()) == {"conv"}
            assert not module.slots.unfilled()["conv"].required
        if isinstance(module, inn.Component):
            assert not source.get_submodule(path).slots.resolved()


def test_body_binding_uses_component_facts_and_preserves_registered_state(monkeypatch, tp_group):
    source = _module()
    source_graph = str(source.graph)
    body, head = _split_body_and_head(clone_graph_module(source))
    graph = str(body.graph)
    parameters = dict(body.named_parameters(remove_duplicate=False))
    buffers = dict(body.named_buffers())
    reference, requests = OpTable(mode="reference"), {}
    factory, row_bindings = _binding.bind_row_parallel, {}

    def must_not_execute(*args, **kwargs):
        raise AssertionError("Binding executed an operation")

    def bind(op, *, site, **kwargs):
        requests[site] = (op, kwargs)
        return replace(reference.bind(op, site=site, **kwargs), fn=must_not_execute)

    def row_parallel(*, tp_group, site):
        bound = replace(factory(tp_group=tp_group, site=site), fn=must_not_execute)
        row_bindings[site] = (tp_group, bound)
        return bound

    monkeypatch.setattr(_binding, "OpTable", lambda: SimpleNamespace(mode="reference", bind=bind))
    monkeypatch.setattr(_binding, "bind_row_parallel", row_parallel)
    bind_body_ops(body, tp_group=tp_group)

    assert requests == {
        "body.embed.lookup": ("embed.lookup", {"inputs": {"tp": 1}}),
        "body.layers.12.attn.mm": ("mm", {}),
        "body.layers.12.attn.qk_norm": ("norms.qk", {
            "inputs": {"eps": 2e-6}, "convention": "rms_zero_centered",
        }),
        "body.layers.12.attn.qk_rope": ("rope.apply", {
            "inputs": {"style": "neox", "rotary_dim": 8, "theta": 10000.0},
            "facts": {"style": "neox", "rotary_dim": 8, "theta": 10000.0,
                      "mrope_section": (1, 1, 2), "mrope_interleaved": True, "qk_norm": "rms_zero_centered"},
        }),
        "body.layers.12.attn.gate": ("attention.gate", {
            "inputs": {"gate": "sigmoid"}, "facts": {"gate": "sigmoid"},
        }),
        "body.layers.12.gdn.mm": ("mm", {}),
        "body.layers.12.gdn.out_norm": ("norms.gated_out", {
            "inputs": {"eps": 3e-6}, "convention": "gated_rms_zero_centered",
            "facts": {"width": 4, "activation": "silu"},
        }),
        "body.layers.12.mlp.mm": ("mm", {}),
        "body.layers.12.mlp.act": ("mlp.act", {"facts": {"activation": "silu_mul"}}),
        "body.norm.norm": ("norms.rms.pre_attn", {
            "inputs": {"eps": 1e-6}, "convention": "rms_zero_centered", "facts": {"width": 4},
        }),
    }
    assert set(row_bindings) == {
        "body.layers.12.attn.row_parallel", "body.layers.12.gdn.row_parallel", "body.layers.12.mlp.row_parallel",
    }
    for site, (group, bound) in row_bindings.items():
        module = body.get_submodule(site.removeprefix("body.").removesuffix(".row_parallel"))
        assert group is tp_group
        assert module.slots.resolved()["row_parallel"] is bound
        assert bound.impl == "pik_tree" and bound.fn is must_not_execute
        tp_copy = module.slots.resolved()["tp_copy"]
        assert tp_copy.fn is no_movement
        assert tp_copy.inputs == {"tp": 1, "side": "engine", "sp": False}
    attention = body.get_submodule("layers.12.attn")
    kv_gather, cp_gather = (attention.slots.resolved()[slot] for slot in ("kv_gather", "cp_gather"))
    assert kv_gather.fn is cp_gather.fn is no_movement
    assert kv_gather.inputs == {"tp": 1, "side": "engine", "kv_heads": 1}
    assert cp_gather.inputs == {"cp": 1}
    assert set(attention.slots.resolved()["qk_rope"].ops) == {"norms.qk", "rope.apply"}
    assert str(source.graph) == source_graph and str(body.graph) == graph
    assert head.weight is body.get_parameter("embed.weight")
    assert set(dict(body.named_parameters(remove_duplicate=False))) == set(parameters)
    for name, parameter in parameters.items():
        assert body.get_parameter(name) is parameter and parameter.is_meta
    for name, buffer in buffers.items():
        assert body.get_buffer(name) is buffer
    for path, module in body.named_modules():
        if isinstance(module, (inn.Embedding, inn.MLP)):
            assert not module.slots.unfilled()
        elif isinstance(module, inn.RMSNorm):  # add_norm: filled only where the fusion pass rewrote a join
            assert set(module.slots.unfilled()) <= {"add_norm"}
            assert not any(slot.required for slot in module.slots.unfilled().values())
        elif isinstance(module, inn.Attention):
            assert set(module.slots.unfilled()) == {"core"}
            assert module.slots.unfilled()["core"].required
        elif isinstance(module, inn.GDN):
            assert set(module.slots.unfilled()) == {"core", "conv"}
            assert all(slot.required for slot in module.slots.unfilled().values())
        if isinstance(module, inn.Component):
            assert module.placement is None
            assert not source.get_submodule(path).slots.resolved()


def test_attention_without_norm_or_gate_keeps_optional_slots_unbound(vllm_config, hf_config):
    source = _module()
    source.get_submodule("layers.12").attn = inn.Attention(
        hidden=4, heads=2, kv_heads=1, head_dim=4, rope_theta=10000, qk_norm=None, out_gate=None,
    )

    model = VllmModel(StageDefinition(source, None),
                      source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                      vllm_config=vllm_config)

    attention = model.body.get_submodule("layers.12.attn")
    assert not any(slot.required for slot in attention.slots.unfilled().values())
    assert not attention.slots.unfilled()["qk_norm"].required
    assert not attention.slots.unfilled()["gate"].required
    assert attention.slots.resolved()["qk_rope"].ops == ("rope.apply",)


def test_fused_qk_rope_preserves_ownership_without_a_separate_rotation(vllm_config, hf_config, monkeypatch):
    source = _module()
    reference = OpTable(mode="reference")

    def bind(op, *, site, **kwargs):
        assert op != "rope.apply", "The fused q/k norm already owns rotation"
        if op == "norms.qk":
            return BoundOp(op, "test.fused_qk_rope", None, "test", fn=lambda *args, **kwargs: None,
                           owns={"norms.qk": "test.fused_qk_rope", "rope.apply": "test.fused_qk_rope"})
        return reference.bind(op, site=site, **kwargs)

    monkeypatch.setattr(_binding, "OpTable", lambda: SimpleNamespace(mode="reference", bind=bind))
    model = VllmModel(StageDefinition(source, None),
                      source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                      vllm_config=vllm_config)

    attention = model.body.get_submodule("layers.12.attn")
    assert not attention.slots.unfilled()["qk_norm"].required
    assert attention.slots.resolved()["core"].impl == "sm90_fa3_segsplit_v1_paged"
    assert set(attention.slots.resolved()["qk_rope"].ops) == {"norms.qk", "rope.apply"}
    assert not source.get_submodule("layers.12.attn").slots.resolved()


def test_tp_engine_takes_the_pik_transport_decision_at_construction(vllm_config, hf_config, monkeypatch):
    from isoexec.ops.collectives import pik_tp_invariant

    # An architecture policy refusing P2P is observable at construction (none ships one today).
    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    monkeypatch.setitem(pik_tp_invariant.NCCL_ONLY_ARCHS, "sm100", "arch policy: sm100 (test)")
    monkeypatch.setenv("ISOEXEC_PIK_P2P", "1")
    monkeypatch.setattr(sys.modules["vllm.distributed"], "get_tp_group",
                        lambda: SimpleNamespace(device_group=dist.ProcessGroup(0, 2)))
    monkeypatch.setattr(vllm_config.parallel_config, "tensor_parallel_size", 2)

    with pytest.raises(RuntimeError, match="ISOEXEC_PIK_P2P=1 .*arch policy: sm100"):
        VllmModel(StageDefinition(_module(), None),
                  source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                  vllm_config=vllm_config)


def test_constructor_refuses_mismatched_tp_config_and_group(vllm_config, hf_config, monkeypatch):
    source = _module()
    monkeypatch.setattr(sys.modules["vllm.distributed"], "get_tp_group",
                        lambda: SimpleNamespace(device_group=dist.ProcessGroup(0, 2)))

    with pytest.raises(ValueError, match="matching TP config and group size"):
        VllmModel(StageDefinition(source, None),
                  source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                  vllm_config=vllm_config)

    for module in source.modules():
        if isinstance(module, inn.Component):
            assert not module.slots.resolved()


def test_head_binding_preserves_registered_state_and_selects_native_rows(tp_group):
    module = _module()
    graph = str(module.graph)
    head = module.head
    weight = head.weight

    bind_head_ops(head, tp_group=tp_group)

    assert str(module.graph) == graph
    assert module.head is head
    assert head.weight is weight is module.embed.weight
    assert weight.is_meta
    bindings = head.slots.resolved()
    assert set(bindings) == {"head", "lm_head_slice", "tp_copy"}
    assert bindings["head"].op == "logprobs.head"
    assert bindings["head"].impl == "eager"
    assert bindings["head"].fn is head_bf16
    assert bindings["head"].inputs == {"tp": 1, "boundary_out_dtype": "fp32"}
    assert bindings["lm_head_slice"].op == "logprobs.lm_head_slice"
    assert bindings["lm_head_slice"].impl == "sampled_rows"
    assert bindings["lm_head_slice"].fn is lm_head_slice_sampled_rows
    assert bindings["tp_copy"].op == "collectives.tp_copy"
    assert bindings["tp_copy"].impl == "none"
    assert bindings["tp_copy"].fn is no_movement
    assert bindings["tp_copy"].inputs == {"tp": 1, "side": "engine", "sp": False}


def test_constructor_accepts_a_keyword_hidden_input(vllm_config, hf_config):
    source = _module(keyword=True, auxiliary=True)

    model = VllmModel(StageDefinition(source, None),
                      source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                      vllm_config=vllm_config)

    body_output = next(node for node in model.body.graph.nodes if node.op == "output")
    assert isinstance(body_output.args[0], fx.Node)
    assert body_output.args[0].target == "norm"
    assert model.head.weight is model.body.get_parameter("embed.weight")


@pytest.mark.parametrize("section,field,value", [
    ("parallel_config", "tensor_parallel_size", 2),
    ("parallel_config", "pipeline_parallel_size", 2),
    ("model_config", "dtype", torch.float32),
    ("compilation_config", "cudagraph_mode", SimpleNamespace(name="PIECEWISE")),
    ("compilation_config", "mode", 3),
    (None, "quant_config", object()),
])
def test_constructor_refuses_unimplemented_configs_without_changing_source(
    vllm_config, hf_config, section, field, value,
):
    source = _module()
    source_graph = str(source.graph)
    parameters = dict(source.named_parameters(remove_duplicate=False))
    offset = source.get_buffer("offset")
    setattr(vllm_config if section is None else getattr(vllm_config, section), field, value)

    with pytest.raises(ValueError, match="VllmModel construction"):
        VllmModel(StageDefinition(source, None),
                  source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                  vllm_config=vllm_config)

    assert str(source.graph) == source_graph
    assert source.get_buffer("offset") is offset
    for name, parameter in parameters.items():
        assert source.get_parameter(name) is parameter


def test_constructor_requires_meta_parameters(vllm_config, hf_config):
    source = _module().to_empty(device="cpu")

    with pytest.raises(ValueError, match="requires meta parameters"):
        VllmModel(StageDefinition(source, None),
                  source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                  vllm_config=vllm_config)


@pytest.mark.parametrize("breakable,v2", [("0", True), ("1", False)])
def test_constructor_refuses_piecewise_without_breakable_v2_graphs(vllm_config, hf_config, monkeypatch,
                                                                   breakable, v2):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", breakable == "1", raising=False)
    vllm_config.compilation_config.cudagraph_mode = SimpleNamespace(name="FULL_AND_PIECEWISE")
    vllm_config.use_v2_model_runner = v2

    with pytest.raises(ValueError, match="FULL_AND_PIECEWISE requires"):
        VllmModel(StageDefinition(_module(), None),
                  source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                  vllm_config=vllm_config)


def test_constructor_refuses_a_cudagraph_mode_vllm_rewrote(vllm_config, hf_config, monkeypatch):
    """vLLM turns FULL_AND_PIECEWISE into FULL_DECODE_ONLY, with one log line, in a process that lacks
    breakable graphs; the model must refuse the engine rather than run another mode silently."""
    monkeypatch.setattr(vllm_config, "additional_config",
                        {"isoexec": {"cudagraph_mode": "FULL_AND_PIECEWISE"}}, raising=False)
    vllm_config.compilation_config.cudagraph_mode = SimpleNamespace(name="FULL_DECODE_ONLY")

    with pytest.raises(ValueError, match="FULL_AND_PIECEWISE was requested but vLLM built the engine with "
                                         "FULL_DECODE_ONLY.*VLLM_USE_BREAKABLE_CUDAGRAPH=1"):
        VllmModel(StageDefinition(_module(), None),
                  source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                  vllm_config=vllm_config)


def test_constructor_accepts_decode_only_cuda_graphs(vllm_config, hf_config):
    vllm_config.compilation_config.cudagraph_mode = SimpleNamespace(name="FULL_DECODE_ONLY")

    model = VllmModel(StageDefinition(_module(), None),
                      source=ModelSource("isoexec.models.qwen3_5"), hf_config=hf_config,
                      vllm_config=vllm_config)

    assert model.head is not None


class _Bound(inn.Component):
    """A placement-bound kind that binds its own child components."""

    PLACEMENT_BOUND = True

    def __init__(self) -> None:
        super().__init__()
        self.norm = inn.RMSNorm(hidden=8, eps=1e-6, device="meta")
        self.placed_with, self.engine_state = None, None

    def facts(self):
        return {"kind": "bound"}

    def _place(self, placement):
        self.placed_with = placement

    def bind_engine_state(self, *, vllm_config, layer_prefix):
        self.engine_state = (vllm_config, layer_prefix)


def test_placement_bound_components_bind_themselves_and_attach_engine_state(tp_group):
    body = nn.Module()
    body.layer = _Bound()
    assert _binding.has_placement_bound(body)
    placement = _binding.placement_from_groups(tp_group=tp_group, pp_group=dist.ProcessGroup(0, 1))
    _binding.bind_body_ops(body, tp_group=tp_group, placement=placement)
    assert body.layer.placed_with is placement and body.layer.site == "body.layer"
    # The child belongs to the placement-bound owner, so the runtime does not bind it.
    assert body.layer.norm.site is None
    with pytest.raises(Exception, match="requires a placement"):
        _binding.bind_body_ops(_with(_Bound()), tp_group=tp_group)
    config = object()
    _binding.bind_engine_states(body, vllm_config=config, prefix="model")
    assert body.layer.engine_state == (config, "model.body.layer")


def _with(child):
    body = nn.Module()
    body.layer = child
    return body


def test_engine_expert_parallel_holds_whole_experts_and_follows_the_tp_ranks(monkeypatch, tp_group):
    ep = dist.ProcessGroup(0, 1)
    monkeypatch.setattr(sys.modules["vllm.distributed"], "get_ep_group",
                        lambda: SimpleNamespace(device_group=ep), raising=False)
    assert _binding.engine_expert_group(tp_group) is ep
    monkeypatch.setattr(sys.modules["vllm.distributed"], "get_ep_group",
                        lambda: SimpleNamespace(device_group=dist.ProcessGroup(0, 2)))
    with pytest.raises(ValueError, match="order the TP ranks"):
        _binding.engine_expert_group(tp_group)
    monkeypatch.delattr(sys.modules["vllm.distributed"], "get_ep_group")
    with pytest.raises(ValueError, match="expert-parallel group"):
        _binding.engine_expert_group(tp_group)


def test_decode_graphs_refuse_a_host_sized_expert_exchange():
    moe = inn.MoE(hidden=8, num_experts=2, top_k=1, intermediate=4, routing="softmax_renorm", device="meta")
    host_sized = BoundOp("moe.expert_mlp", "impl", 1, "test", {}, lambda *a, **k: None)
    moe.slots.expert_mlp = host_sized
    body = _with(moe)
    _binding.refuse_uncapturable_expert_parallel(body, "NONE")
    with pytest.raises(ValueError, match="CUDA-graph capturable"):
        _binding.refuse_uncapturable_expert_parallel(body, "FULL_DECODE_ONLY")

    def device_sized(*a, **k):
        return None

    device_sized.graph_capturable = True
    moe.slots.expert_mlp = BoundOp("moe.expert_mlp", "impl", 1, "test", {}, device_sized)
    _binding.refuse_uncapturable_expert_parallel(body, "FULL_DECODE_ONLY")
