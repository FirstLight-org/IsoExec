from types import SimpleNamespace

import pytest
import torch
from torch import distributed as dist
from transformers import PretrainedConfig

from isoexec.compiler.realization import weights as weight_loading
from isoexec.compiler.realization.graph import clone_graph_module
from isoexec.compiler.realization.storage import allocate_tensors
from isoexec.compiler.realization.tests import test_tp_cpu
from isoexec.compiler.realization.tp import shard_parameter_shapes
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.partition import define_stage
from isoexec.compiler.source import ModelSource
from isoexec.compiler.tests.test_capture_cpu import Model, example_inputs
from isoexec.compiler.text import text_inputs
from isoexec.nn.component import Component
from isoexec.ops.table import OpTable
from isoexec.runtimes.megatron import _binding
from isoexec.schema import make_sequence_context

native = pytest.importorskip("isoexec.runtimes.megatron.model", exc_type=ImportError)

tp_source = test_tp_cpu.source

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


def test_native_tp_attributes_describe_current_registered_shards(tp_source):
    target = clone_graph_module(tp_source)
    shard_parameter_shapes(target, tp_size=2, side="trainer")
    allocate_tensors(target, device=torch.device("cpu"))
    parameters = dict(target.named_parameters())
    _binding.set_tensor_parallel_attributes(target)
    assert target.embed.weight is target.head.weight
    for name in ("embed.weight", "attn.q", "gdn.conv_w", "gdn.A_log", "mlp.gate"):
        parameter = target.get_parameter(name)
        assert parameter.tensor_model_parallel and parameter.partition_dim == 0 and parameter.partition_stride == 1
    for name in ("attn.o", "gdn.o", "mlp.down"):
        assert target.get_parameter(name).tensor_model_parallel and target.get_parameter(name).partition_dim == 1
    for name in ("norm.weight", "attn.q_norm", "attn.k_norm", "gdn.out_norm"):
        assert not target.get_parameter(name).tensor_model_parallel and target.get_parameter(name).partition_dim == -1
    assert all(target.get_parameter(name) is parameter for name, parameter in parameters.items())
    assert all(not hasattr(parameter, "tensor_model_parallel") for parameter in tp_source.parameters())


@pytest.fixture(autouse=True)
def reference_bindings(monkeypatch):
    reference = OpTable(mode="reference")
    monkeypatch.setattr(_binding, "OpTable", lambda: reference)

    def row_parallel(*, tp_group, site, sp):
        assert tp_group is dist.group.WORLD and not sp
        return reference.bind("collectives.row_parallel", site=site)

    monkeypatch.setattr(_binding, "bind_row_parallel", row_parallel)
    mapping_module = SimpleNamespace(
        checkpoint_conversion_mapping=lambda config: [],
    )
    monkeypatch.setattr(ModelSource, "weight_module", lambda self: mapping_module)
    monkeypatch.setattr(weight_loading, "ignored_patterns", lambda config: (r"^unused\.",))


@pytest.fixture
def groups(tmp_path):
    dist.init_process_group("gloo", init_method=(tmp_path / "group").as_uri(), rank=0, world_size=1)
    collection = ProcessGroupCollection()
    collection.tp = collection.pp = dist.group.WORLD
    try:
        yield collection
    finally:
        dist.destroy_process_group()


def test_native_constructor_and_loading_keep_registered_state(groups, monkeypatch):
    allocate = native.allocate_tensors
    device_index = torch.cuda.current_device() if torch.cuda.is_available() else 3

    def allocate_on_cpu(graph_module, *, device):
        assert device == torch.device("cuda", device_index)
        allocate(graph_module, device=torch.device("cpu"))

    monkeypatch.setattr(torch.cuda, "current_device", lambda: device_index)
    monkeypatch.setattr(native, "allocate_tensors", allocate_on_cpu)
    model = Model().eval()
    captured = capture(model, example_inputs())
    definition = define_stage(captured, ((0, 2),), pp_rank=0)
    config = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4,
                               bf16=True, params_dtype=torch.bfloat16)
    hf_config = PretrainedConfig()
    target = native.MegatronModel(definition,
                                  source=ModelSource("fixture_model"), hf_config=hf_config, config=config, pg_collection=groups)

    assert isinstance(target, MegatronModule)
    assert target.config is config
    assert target.hf_config is hf_config
    assert tuple(target.named_children()) == (("graph_module", target.graph_module),)
    assert target.graph_module is not definition.module
    assert target.training is target.graph_module.training is False
    assert target.graph_module.get_parameter("lexicon.weight") is target.graph_module.get_parameter("readout.weight")
    assert target.graph_module.get_parameter("lexicon.weight") is not model.lexicon.weight
    assert all(parameter.device == torch.device("cpu") for parameter in target.parameters())
    assert all(parameter.is_meta for parameter in definition.module.parameters())
    for module in target.graph_module.modules():
        if isinstance(module, Component):
            assert not any(unbound.required for unbound in module.slots.unfilled().values())

    weight = target.graph_module.lexicon.weight

    def stream():
        yield "lexicon.weight", torch.full_like(weight, 0.25)
        assert torch.equal(weight, torch.full_like(weight, 0.25))
        yield "finish.weight", torch.zeros_like(target.graph_module.finish.weight)
        yield "unused.weight", torch.ones(1)

    assert target.load_weights(stream()) == {"graph_module.lexicon.weight", "graph_module.finish.weight"}
    assert target.graph_module.lexicon.weight is weight
    assert target.load_weights(()) == set()


def test_pp_stages_use_native_activation_shapes_and_keep_local_objects(groups, monkeypatch):
    allocate = native.allocate_tensors
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(native, "allocate_tensors", lambda module, **kwargs: allocate(module, device=torch.device("cpu")))
    authored = Model().eval()
    examples = text_inputs(authored, device="cpu", max_tokens=8, profile="trainer")
    captured = capture(authored, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes,
                       root_layouts=examples.root_layouts, seq=examples.seq)
    full = {name: torch.full(parameter.shape, 0.02, dtype=parameter.dtype)
            for name, parameter in authored.named_parameters()}
    models = []
    for rank in range(2):
        collection = SimpleNamespace(tp=groups.tp, pp=dist.ProcessGroup(rank, 2))
        config = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4,
                                   pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16,
                                   bf16=True, params_dtype=torch.bfloat16)
        model = native.MegatronModel(define_stage(captured, ((0, 1), (1, 2)), pp_rank=rank),
                                     source=ModelSource("fixture_model"), hf_config=PretrainedConfig(), config=config, pg_collection=collection)
        model.load_weights(full.items())
        models.append(model)
    ids = torch.tensor([[1, 2, 3, 4]])
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                             max_seqlen_q=2, max_seqlen_kv=2)
    models[0].set_input_tensor([None])
    hidden = models[0](ids, None, None, packed_seq_params=packed)
    assert hidden.shape == (4, 1, 16) and hidden.dtype == torch.bfloat16 and hidden._base is None
    received = hidden.detach().requires_grad_()
    models[1].set_input_tensor([received])
    unused_padded_ids = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]])
    output = models[1](unused_padded_ids, None, None, packed_seq_params=packed)
    assert output.shape == (1, 4, 32) and output.dtype == torch.float32
    output.square().sum().backward()
    assert received.grad is not None and torch.isfinite(received.grad).all()
    assert models[0].shared_embedding_or_output_weight() is not models[1].shared_embedding_or_output_weight()
    assert all(parameter.is_meta for parameter in authored.parameters())
    with pytest.raises(ValueError, match="activation"):
        models[1](ids, None, None, packed_seq_params=packed)
    # Each stage accepts only this packed batch's [T, 1, hidden] stream.
    for wrong in (received[:, :, :8], received[:3], received.squeeze(1)):
        models[1].set_input_tensor([wrong.detach()])
        with pytest.raises(ValueError, match=r"\[4, 1, 16\] activation"):
            models[1](unused_padded_ids, None, None, packed_seq_params=packed)
    longer = PackedSeqParams(qkv_format="thd", cu_seqlens_q=torch.tensor([0, 2, 6], dtype=torch.int32),
                             cu_seqlens_kv=torch.tensor([0, 2, 6], dtype=torch.int32), max_seqlen_q=4, max_seqlen_kv=4)
    models[0].set_input_tensor([None])
    hidden = models[0](torch.tensor([[1, 2, 3, 4, 5, 6]]), None, None, packed_seq_params=longer)
    models[1].set_input_tensor([hidden.detach().requires_grad_()])
    assert models[1](unused_padded_ids, None, None, packed_seq_params=longer).shape == (1, 6, 32)


@pytest.mark.parametrize("field,value", [
    ("tensor_model_parallel_size", 2),
    ("pipeline_model_parallel_size", 2),
    ("expert_model_parallel_size", 2),
    ("sequence_parallel", True),
    ("cuda_graph_impl", "local"),
    ("params_dtype", torch.float32),
])
def test_native_constructor_refuses_unimplemented_configs_without_changing_source(groups, field, value):
    model = Model()
    captured = capture(model, example_inputs())
    definition = define_stage(captured, ((0, 2),), pp_rank=0)
    source_weight = model.lexicon.weight
    config = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4,
                               bf16=True, params_dtype=torch.bfloat16)
    setattr(config, field, value)
    with pytest.raises(ValueError, match="MegatronModel construction"):
        native.MegatronModel(definition, source=ModelSource("fixture_model"), hf_config=PretrainedConfig(),
                             config=config, pg_collection=groups)
    assert definition.module.get_parameter("lexicon.weight") is source_weight


@pytest.mark.parametrize("granularity,method,span", [
    ("selective", None, None), ("full", "block", 1), ("full", "uniform", 0),
])
def test_native_constructor_refuses_unsupported_recompute_before_allocation(groups, granularity, method, span):
    config = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4,
                               bf16=True, params_dtype=torch.bfloat16)
    config.recompute_granularity = granularity
    config.recompute_method = method
    config.recompute_num_layers = span
    with pytest.raises(ValueError, match="full uniform recompute"):
        native.MegatronModel(None,
                             source=ModelSource("fixture_model"), hf_config=PretrainedConfig(), config=config, pg_collection=groups)


def test_native_packed_forward_preserves_logits_gradients_ties_and_updates(groups, monkeypatch):
    allocate = native.allocate_tensors
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(native, "allocate_tensors", lambda graph, **kwargs: allocate(graph, device=torch.device("cpu")))
    authored = Model().eval()
    examples = text_inputs(authored, device="cpu", max_tokens=8)
    captured = capture(authored, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes,
                       root_layouts=examples.root_layouts)
    definition = define_stage(captured, ((0, 2),), pp_rank=0)
    config = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4,
                               bf16=True, params_dtype=torch.bfloat16)
    model = native.MegatronModel(definition, source=ModelSource("fixture_model"), hf_config=PretrainedConfig(),
                                config=config, pg_collection=groups)
    generator = torch.Generator().manual_seed(42)
    weights = [(name, torch.randn(parameter.shape, generator=generator) * 0.03)
               for name, parameter in model.graph_module.named_parameters()]
    model.load_weights(iter(weights))
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                             max_seqlen_q=3, max_seqlen_kv=3, total_tokens=5)
    assert model.share_embeddings_and_output_weights
    assert model.shared_embedding_or_output_weight() is model.graph_module.get_parameter("lexicon.weight")
    model.set_input_tensor([None])
    expected_seq = make_sequence_context(torch.tensor([0, 1, 0, 1, 2]), cu, max_q=3, max_kv=3,
                                         input_ids=ids[0], cu_seqlens_cpu=cu.clone())
    expected = model.graph_module(input_ids=expected_seq.input_ids, seq=expected_seq).unsqueeze(0)
    expected.square().sum().backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)

    actual = model(ids, None, None, packed_seq_params=packed)
    actual.square().sum().backward()

    assert actual.shape == (1, 5, 32) and actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.equal(parameter.grad, gradients[name])
    before = actual.detach().clone()
    name = "lexicon.weight"
    model.load_weights(((name, torch.zeros_like(model.graph_module.get_parameter(name))),))
    assert not torch.equal(model(ids, None, None, packed_seq_params=packed), before)
    with pytest.raises(ValueError, match="attention_mask=None"):
        model(ids, None, torch.ones(1, 5), packed_seq_params=packed)
    with pytest.raises(ValueError, match="pipeline activation"):
        model.set_input_tensor(torch.zeros(5, 1, 16))
