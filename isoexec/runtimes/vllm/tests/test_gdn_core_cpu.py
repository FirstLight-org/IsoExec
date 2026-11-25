"""GDN pool registration and native core routing without GPU execution."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import fx, nn

import isoexec.nn as inn
from isoexec.compiler.realization.graph import clone_graph_module
from isoexec.compiler.realization.storage import allocate_tensors
from isoexec.ops import gdn
from isoexec.ops.table import OpTable
from isoexec.runtimes.vllm import _binding, gdn_engine_patch
from isoexec.runtimes.vllm._binding import bind_body_ops, bind_gdn_cores
from isoexec.runtimes.vllm.tests import _gdn_fixtures, test_definition_cpu

production_bindings = test_definition_cpu.production_bindings


@pytest.fixture(autouse=True)
def _cpr_selected(monkeypatch):
    """These tests cover the private-pool provider (gdn_cpr_sm90). sm90's default is the native recurrent
    core, whose state lives in vLLM's mamba pages and has no pool, so select CPR for the module."""
    from isoexec.ops import _dispatch

    row = _dispatch.FAMILIES["gdn"]
    monkeypatch.setitem(_dispatch.FAMILIES, "gdn", replace(row, archs={**row.archs, "sm90": "gdn.gdn_cpr_sm90"}))
tp_group = test_definition_cpu.tp_group
vllm_config = test_definition_cpu.vllm_config


def _body():
    root = nn.Module()
    root.layers = nn.ModuleDict()
    graph = fx.Graph()
    hidden, seq = graph.placeholder("hidden"), graph.placeholder("seq")
    for index, key_heads, value_heads, key_dim, value_dim in (("3", 2, 4, 4, 8), ("7", 1, 2, 8, 4)):
        layer = nn.Module()
        layer.mixer = inn.GDN(hidden=8, key_heads=key_heads, value_heads=value_heads,
                              key_head_dim=key_dim, value_head_dim=value_dim, conv_kernel=4)
        root.layers[index] = layer
        hidden = graph.call_module(f"layers.{index}.mixer", (hidden, seq))
    graph.output(hidden)
    return fx.GraphModule(root, graph).eval()


@pytest.fixture
def body(vllm_config, tp_group):
    source = _body()
    target = clone_graph_module(source)
    bind_body_ops(target, tp_group=tp_group)
    allocate_tensors(target, device=torch.device("cpu"))
    return source, target


@pytest.mark.parametrize("prefix", ["", "model"])
def test_gdn_registration_preserves_weights_and_publishes_current_pool_buffers(body, vllm_config, prefix, monkeypatch):
    source, target = body
    graph = str(target.graph)
    parameters = dict(target.named_parameters(remove_duplicate=False))
    context = vllm_config.compilation_config.static_forward_context
    existing = nn.Module()
    context["body.layers.0.mixer.core"] = existing

    bind_gdn_cores(target, vllm_config=vllm_config, prefix=prefix)

    names = {"body.layers.0.mixer.core"}
    for index, conv_shape, ssm_shape in (("3", (3, 48), (4, 8, 4)), ("7", (3, 24), (2, 4, 8))):
        module = target.get_submodule(f"layers.{index}.mixer")
        layer = module.core
        pool = layer._pool
        name = f"{prefix}.body.layers.{index}.mixer.core" if prefix else f"body.layers.{index}.mixer.core"
        names.add(name)
        assert isinstance(layer, _gdn_fixtures.MambaBase)
        assert context[name] is layer and layer.prefix == name
        assert layer.mamba_type is _gdn_fixtures.MambaAttentionBackendEnum.GDN_ATTN
        assert layer.get_state_shape() == (conv_shape, ssm_shape)
        assert layer.get_state_dtype() == (torch.bfloat16, torch.float32)
        assert pool.capacity == 4 and pool.chunk_size == 64
        assert pool.conv_weight.data_ptr() == module.conv_w.data_ptr()
        assert pool.A_log is module.A_log and pool.dt_bias is module.dt_bias
        for buffer_name in ("ssm_state", "conv_state", "slot2row"):
            path = f"layers.{index}.mixer.core.{buffer_name}"
            assert target.get_buffer(path) is getattr(pool, buffer_name)
            assert path not in target.state_dict()
        assert pool.ssm_state.dtype == torch.uint8
        assert pool.conv_state.dtype == torch.bfloat16 and pool.slot2row.dtype == torch.long
        expanded = torch.zeros(17, dtype=torch.long)
        pool._publish("slot2row", expanded)
        assert target.get_buffer(f"layers.{index}.mixer.core.slot2row") is expanded
        bound = module.slots.resolved()["core"]
        assert bound.op == "gdn.core"
        assert set(bound.owns) == {"gdn.core", "gdn.l2norm", "gdn.gating", "conv.short"}
        assert set(module.slots.unfilled()) == {"conv"} and not module.slots.unfilled()["conv"].required
    assert set(context) == names and context["body.layers.0.mixer.core"] is existing
    assert str(source.graph) == str(target.graph) == graph
    assert dict(target.named_parameters(remove_duplicate=False)).keys() == parameters.keys()
    for name, parameter in parameters.items():
        assert target.get_parameter(name) is parameter
    assert all(parameter.is_meta for parameter in source.parameters())
    for index in ("3", "7"):
        assert not source.get_submodule(f"layers.{index}.mixer").slots.resolved()
    monkeypatch.setitem(gdn.__dict__, "gdn_cpr_min_pages", lambda: True)
    assert target.get_submodule("layers.3.mixer.core").get_state_shape() == ((1, 8), (1, 1, 4))


@pytest.mark.parametrize("actual_tokens", [2, 3])
def test_gdn_core_routes_native_metadata_and_current_weights_to_the_same_pool(body, vllm_config, monkeypatch, actual_tokens):
    _, target = body
    bind_gdn_cores(target, vllm_config=vllm_config)
    metadata = {f"body.layers.{index}.mixer.core": object() for index in ("3", "7")}
    calls = []

    def run(pool, md, mixed, alpha, beta):
        output = torch.empty(actual_tokens, pool.num_v_heads, pool.head_v_dim, dtype=mixed.dtype, device=mixed.device)
        calls.append((pool, md, mixed, alpha, beta, output))
        return output

    monkeypatch.setattr(gdn_engine_patch, "gdn_metadata", metadata.__getitem__)
    monkeypatch.setattr(gdn_engine_patch, "gdn_layer_core", run)
    for index, key_heads, value_heads, key_dim, value_dim in (("3", 2, 4, 4, 8), ("7", 1, 2, 8, 4)):
        module = target.get_submodule(f"layers.{index}.mixer")
        layer, weight = module.core, module.conv_w
        with torch.no_grad():
            weight.fill_(0.25)
        q = k = torch.empty(3, key_heads, key_dim, dtype=torch.bfloat16, device="meta")
        v = torch.empty(3, value_heads, value_dim, dtype=torch.bfloat16, device="meta")
        alpha = beta = torch.empty(3, value_heads, dtype=torch.bfloat16, device="meta")
        A_log, dt_bias = torch.ones_like(module.A_log), torch.zeros_like(module.dt_bias)

        output = module.slots.core(q, k, v, alpha, beta, A_log, dt_bias, None, cu_seqlens_cpu=None)

        pool, md, mixed, seen_alpha, seen_beta, kernel_output = calls[-1]
        assert pool is layer._pool and md is metadata[layer.prefix]
        assert mixed.shape == (3, 2 * key_heads * key_dim + value_heads * value_dim)
        assert seen_alpha is alpha and seen_beta is beta
        assert pool.A_log is A_log and pool.dt_bias is dt_bias
        assert pool.conv_weight.data_ptr() == weight.data_ptr()
        assert torch.equal(pool.conv_weight, torch.full_like(pool.conv_weight, 0.25))
        assert output.shape == v.shape and output.dtype == v.dtype
        if actual_tokens == 3:
            assert output is kernel_output
        assert module.conv_w is weight


def test_gdn_metadata_free_profile_returns_the_value_shape_without_running_the_pool(body, vllm_config, monkeypatch):
    _, target = body
    bind_gdn_cores(target, vllm_config=vllm_config)

    def must_not_execute(*args, **kwargs):
        raise AssertionError("Metadata-free profiling executed the GDN pool")

    monkeypatch.setattr(gdn_engine_patch, "gdn_layer_core", must_not_execute)
    q = k = torch.empty(3, 2, 4, dtype=torch.bfloat16, device="meta")
    v = torch.empty(3, 4, 8, dtype=torch.bfloat16, device="meta")
    alpha = beta = torch.empty(3, 4, dtype=torch.bfloat16, device="meta")
    module = target.get_submodule("layers.3.mixer")

    output = module.slots.core(q, k, v, alpha, beta, module.A_log, module.dt_bias, None)

    assert output.shape == v.shape and output.dtype == v.dtype and output.device == v.device


def test_gdn_core_refuses_missing_fused_operation_ownership(body, vllm_config, monkeypatch):
    _, target = body
    reference = OpTable(mode="reference")

    def bind(op, **kwargs):
        bound = reference.bind(op, **kwargs)
        return replace(bound, owns={}) if op == "gdn.core" else bound

    monkeypatch.setattr(_binding, "OpTable", lambda: SimpleNamespace(mode="reference", bind=bind))

    with pytest.raises(inn.PlacementError, match="gdn.l2norm"):
        bind_gdn_cores(target, vllm_config=vllm_config)
