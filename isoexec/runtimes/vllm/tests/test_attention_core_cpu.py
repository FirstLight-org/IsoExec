"""Native Attention registration and core routing without GPU execution."""

import math
import sys

import pytest
import torch
from torch import fx, nn

import isoexec.nn as inn
from isoexec.compiler.realization.graph import clone_graph_module
from isoexec.compiler.realization.storage import allocate_tensors
from isoexec.runtimes.vllm._binding import bind_attention_cores, bind_body_ops
from isoexec.runtimes.vllm.bindings import PagedAttentionBinding
from isoexec.runtimes.vllm.tests import test_definition_cpu

production_bindings = test_definition_cpu.production_bindings
tp_group = test_definition_cpu.tp_group
vllm_config = test_definition_cpu.vllm_config


def _body():
    root = nn.Module()
    root.layers = nn.ModuleDict()
    graph = fx.Graph()
    hidden, seq = graph.placeholder("hidden"), graph.placeholder("seq")
    for index, heads, kv_heads, head_dim in (("3", 4, 2, 4), ("7", 2, 1, 8)):
        layer = nn.Module()
        layer.mixer = inn.Attention(hidden=8, heads=heads, kv_heads=kv_heads, head_dim=head_dim, rope_theta=10000)
        root.layers[index] = layer
        hidden = graph.call_module(f"layers.{index}.mixer", (hidden, seq))
    graph.output(hidden)
    return fx.GraphModule(root, graph).eval()


@pytest.mark.parametrize("prefix", ["", "model"])
def test_native_attention_registration_preserves_graph_weights_and_layer_identity(vllm_config, tp_group, prefix):
    source = _body()
    body = clone_graph_module(source)
    bind_body_ops(body, tp_group=tp_group)
    allocate_tensors(body, device=torch.device("cpu"))
    graph = str(body.graph)
    parameters = dict(body.named_parameters())
    context = vllm_config.compilation_config.static_forward_context
    existing = nn.Module()
    context["body.layers.0.mixer.state"] = existing

    bind_attention_cores(body, vllm_config=vllm_config, prefix=prefix)

    names = {"body.layers.0.mixer.state"}
    for index, heads, kv_heads, head_dim in (("3", 4, 2, 4), ("7", 2, 1, 8)):
        module = body.get_submodule(f"layers.{index}.mixer")
        layer = module.get_submodule("core")
        name = f"{prefix}.body.layers.{index}.mixer.core" if prefix else f"body.layers.{index}.mixer.core"
        names.add(name)
        assert context[name] is layer
        assert layer.layer_name == name
        assert (layer.num_heads, layer.num_kv_heads, layer.head_size) == (heads, kv_heads, head_dim)
        assert layer.impl.scale == 1 / math.sqrt(head_dim)
        assert layer.cache_config is vllm_config.cache_config and layer.quant_config is None
        backend = sys.modules["isoexec.ops.attention.varlen_backend"].PyTorchVarlenAttentionBackend
        assert layer.attn_backend is backend
        bound = module.slots.resolved()["core"]
        assert bound.op == "attention.core.gqa" and bound.impl == "sm90_fa3_segsplit_v1_paged"
        assert bound.provider == "runtimes.vllm" and bound.version is not None
        assert bound.inputs == {} and not module.slots.unfilled()
        assert body.get_buffer(f"layers.{index}.mixer.core._k_scale") is layer._k_scale
    assert set(context) == names and context["body.layers.0.mixer.state"] is existing
    assert str(body.graph) == str(source.graph) == graph
    assert dict(body.named_parameters()).keys() == parameters.keys()
    for name, parameter in parameters.items():
        assert body.get_parameter(name) is parameter
    assert all(parameter.is_meta for parameter in source.parameters())
    for index in ("3", "7"):
        assert "core" not in source.get_submodule(f"layers.{index}.mixer")._modules


def test_cores_route_to_the_registered_layers_current_cache(vllm_config, tp_group, monkeypatch):
    body = _body()
    bind_body_ops(body, tp_group=tp_group)
    allocate_tensors(body, device=torch.device("cpu"))
    bind_attention_cores(body, vllm_config=vllm_config)
    calls = []

    def run(layer, q, k, v, *, out):
        calls.append((layer, layer.kv_cache, out))

    monkeypatch.setattr(PagedAttentionBinding, "run", staticmethod(run))
    for index, heads, kv_heads, head_dim in (("3", 4, 2, 4), ("7", 2, 1, 8)):
        module = body.get_submodule(f"layers.{index}.mixer")
        layer = module.core
        cache = torch.empty(2, 1, 16, kv_heads, head_dim, device="meta")
        layer.kv_cache = cache
        q = torch.empty(3, heads, head_dim, device="meta")
        k = v = torch.empty(3, kv_heads, head_dim, device="meta")

        output = module.slots.core(q, k, v, None, None, module.scale)

        assert calls[-1][0] is layer and calls[-1][1] is cache and calls[-1][2] is output
        assert output.shape == (3, heads, head_dim)
