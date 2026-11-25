"""Native weight sync preserves registered state across complete chunked updates."""

import importlib
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from isoexec.integrations.skyrl.tests import test_native_models_cpu
from isoexec.compiler.realization.graph import clone_graph_module
from isoexec.compiler.realization.storage import allocate_tensors
from isoexec.compiler.realization.tests import test_weights_cpu
from isoexec.compiler.realization.tp import shard_parameter_shapes
from isoexec.compiler.source import ModelSource
from isoexec.ops.collectives.tp_tree import bind_row_parallel
from isoexec.runtimes.vllm import _binding as vllm_binding
from isoexec.runtimes.vllm.model import VllmModel
from isoexec.runtimes.vllm.tests import test_definition_cpu, test_native_cpu

framework = test_native_models_cpu.framework
worker = test_native_models_cpu.worker
qwen_stage = test_weights_cpu.qwen_stage
production_bindings = test_definition_cpu.production_bindings
tp_group = test_definition_cpu.tp_group
vllm_config = test_definition_cpu.vllm_config
native_registry = test_native_cpu.native_registry


@pytest.mark.parametrize("worker", ["custom_weights"], indirect=True)
@pytest.mark.parametrize("runtime", ["trainer", "vllm"])
def test_native_model_loads_the_selected_packages_checkpoint(
    worker, framework, native_registry, vllm_config, runtime,
):
    from isoexec.runtimes.vllm.registration import NATIVE_ARCH, register_native

    if runtime == "trainer":
        model, = framework.integration.build_module(
            worker, wrap_with_ddp=False, ddp_config=None, lora_config=None, bf16=True,
        )
        body, head = model.graph_module, model.graph_module.head
    else:
        models, _ = native_registry
        kwargs = register_native(worker.source, worker.raw, checkpoint=worker.cfg.policy.model.path)
        vllm_config.additional_config = kwargs["additional_config"]
        vllm_config.device_config = SimpleNamespace(device=torch.device("cpu"))
        vllm_config.scheduler_config.max_num_batched_tokens = 8
        vllm_config.model_config.hf_text_config = worker.source.build_config(worker.raw).get_text_config()
        module, name = models[NATIVE_ARCH].split(":")
        model = getattr(importlib.import_module(module), name)(vllm_config=vllm_config)
        weights = load_file(worker.cfg.policy.model.path + "/model.safetensors")
        loaded = model.load_weights(weights.items())
        assert {model.get_parameter(name) for name in loaded} == set(model.parameters())
        body, head = model.body, model.head

    assert model.source == worker.source
    assert torch.equal(body.embed.weight, torch.full_like(body.embed.weight, 0.25))
    norm = body.get_parameter("layers.0.mixer.out_norm")
    assert torch.equal(norm, torch.full_like(norm, -0.75))
    assert head.weight is body.embed.weight


@dataclass(frozen=True)
class ParamMeta:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]


@pytest.fixture
def pair(framework, worker, qwen_stage, vllm_config, monkeypatch):
    base = ModuleType("vllm.distributed.weight_transfer.base")
    base.ParamMeta, base.WeightSource = ParamMeta, object
    monkeypatch.setitem(sys.modules, base.__name__, base)
    source_module = importlib.import_module("isoexec.integrations.skyrl.weight_source")
    modules = framework.integration.build_module(
        worker, wrap_with_ddp=True, ddp_config=None, lora_config=None, bf16=True,
    )
    source = source_module.IsoExecWeightSource(modules, torch.bfloat16)
    rollout = VllmModel(qwen_stage,
                        source=worker.source, hf_config=worker.source.build_config(worker.raw),
                        vllm_config=vllm_config)
    return modules[0].module, source, rollout


def test_metadata_and_chunks_use_current_parameters_without_hf_conversion(pair):
    policy, source, rollout = pair
    parameters = dict(rollout.named_parameters(remove_duplicate=False))
    pointers = {name: value.data_ptr() for name, value in parameters.items()}
    weights = list(source.iter_full())
    metadata = source.metadata

    assert [(item.name, item.dtype, item.shape) for item in metadata] == [
        (name, tensor.dtype, tuple(tensor.shape)) for name, tensor in weights
    ]
    assert {name for name, _ in weights} == {name for name, _ in rollout.named_parameters()}
    assert "head.weight" not in dict(weights)
    rollout.start_weight_update()
    for start in range(0, len(weights), 3):
        chunk = weights[start:start + 3]
        assert rollout.load_weights(chunk) == {name for name, _ in chunk}
    rollout.finish_weight_update()

    for name, tensor in weights:
        assert torch.equal(rollout.get_parameter(name), tensor)
    assert rollout.body.embed.weight is rollout.head.weight
    assert torch.equal(rollout.body.get_parameter("layers.0.mixer.out_norm"),
                       policy.graph_module.get_parameter("layers.0.mixer.out_norm"))
    for name, parameter in parameters.items():
        assert rollout.get_parameter(name) is parameter and parameter.data_ptr() == pointers[name]

    with torch.no_grad():
        policy.graph_module.embed.weight.fill_(0.75)
    current = dict(source.iter_full())
    assert current["body.embed.weight"].data_ptr() == policy.graph_module.embed.weight.data_ptr()
    assert torch.equal(current["body.embed.weight"], torch.full_like(current["body.embed.weight"], 0.75))
    rollout.start_weight_update()
    rollout.load_weights(source.iter_full())
    rollout.finish_weight_update()
    assert rollout.head.weight is parameters["head.weight"]
    assert torch.equal(rollout.head.weight, current["body.embed.weight"])


def test_finish_rejects_partial_update_and_new_start_can_replace_it(pair):
    _, source, rollout = pair
    weights = list(source.iter_full())
    rollout.start_weight_update()
    rollout.load_weights(weights[:-1])
    with pytest.raises(ValueError, match="missing parameters"):
        rollout.finish_weight_update()
    with pytest.raises(ValueError, match="failed"):
        rollout.load_weights(weights[-1:])
    rollout.start_weight_update()
    rollout.load_weights(source.iter_full())
    rollout.finish_weight_update()


@pytest.mark.parametrize("failure", ["name", "shape", "dtype", "iterator"])
def test_failed_chunk_cannot_be_finished_even_after_prior_full_coverage(pair, failure):
    _, source, rollout = pair
    weights = list(source.iter_full())
    name, tensor = weights[0]
    rollout.start_weight_update()
    rollout.load_weights(weights)

    def invalid():
        if failure == "iterator":
            raise RuntimeError("source iteration failed")
        yield ("unknown.weight" if failure == "name" else name,
               tensor[:1] if failure == "shape" else tensor.float() if failure == "dtype" else tensor)

    with pytest.raises((ValueError, RuntimeError)):
        rollout.load_weights(invalid())
    with pytest.raises(ValueError, match="failed"):
        rollout.finish_weight_update()


def test_tied_values_must_agree_across_chunks(pair):
    _, source, rollout = pair
    incoming = dict(source.iter_full())["body.embed.weight"]
    parameter = rollout.head.weight
    rollout.start_weight_update()
    assert rollout.load_weights([("body.embed.weight", incoming)]) == {"body.embed.weight"}
    assert rollout.load_weights([("head.weight", incoming.clone())]) == {"head.weight"}
    with pytest.raises(ValueError, match="conflicting"):
        rollout.load_weights([("head.weight", incoming + 1)])
    assert rollout.head.weight is rollout.body.embed.weight is parameter
    assert torch.equal(parameter, incoming)


@pytest.fixture
def verify_weight_updates(vllm_config):
    vllm_config.additional_config = {"isoexec": {"verify_weight_updates": True}}


def test_a_verified_update_finishes_only_after_every_parameter_was_compared_with_a_second_copy(
    verify_weight_updates, pair, capsys,
):
    _, source, rollout = pair
    weight_source = importlib.import_module("isoexec.integrations.skyrl.weight_source")
    rollout.start_weight_update()
    rollout.load_weights(source.iter_full())
    with pytest.raises(ValueError, match="never compared with a second copy"):
        rollout.finish_weight_update()
    rollout.start_weight_update()
    rollout.load_weights(weight_source.RepeatedWeightSource(source).iter_full())
    rollout.finish_weight_update()
    assert "[ISOEXEC-WEIGHT-SYNC] PASS" in capsys.readouterr().out


def test_update_calls_require_one_active_operation(pair):
    _, _, rollout = pair
    with pytest.raises(ValueError, match="No weight update"):
        rollout.finish_weight_update()
    rollout.start_weight_update()
    with pytest.raises(ValueError, match="already active"):
        rollout.start_weight_update()


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_receiver_selects_full_tensor_coordinates_and_preserves_registered_state(
    pair, qwen_stage, vllm_config, monkeypatch, rank,
):
    policy, source, _ = pair
    for parameter in policy.graph_module.parameters():
        with torch.no_grad():
            parameter.copy_(torch.arange(parameter.numel()).reshape(parameter.shape))
    weights = list(source.iter_full())
    group = torch.distributed.ProcessGroup(rank, 2)
    monkeypatch.setattr(sys.modules["vllm.distributed"], "get_tp_group",
                        lambda: type("Group", (), {"device_group": group})())
    # The fixture's row-parallel check expects its TP1 group; this receiver binds the TP2 group.
    monkeypatch.setattr(vllm_binding, "bind_row_parallel", bind_row_parallel)
    vllm_config.parallel_config.tensor_parallel_size = 2
    vllm_config.compilation_config.static_forward_context.clear()
    rollout = VllmModel(qwen_stage, source=policy.source, hf_config=policy.hf_config, vllm_config=vllm_config)
    parameters = dict(rollout.named_parameters(remove_duplicate=False))
    pointers = {name: parameter.data_ptr() for name, parameter in parameters.items()}
    full = dict(weights)
    rollout.start_weight_update()
    for start in range(0, len(weights), 3):
        rollout.load_weights(weights[start:start + 3])
    rollout.finish_weight_update()
    expected = {
        "body.embed.weight": full["body.embed.weight"][64 * rank:64 * (rank + 1)],
        "body.layers.0.mlp.down": full["body.layers.0.mlp.down"][:, 16 * rank:16 * (rank + 1)],
        "body.norm.weight": full["body.norm.weight"],
        "body.layers.0.mixer.conv_w": full["body.layers.0.mixer.conv_w"]
            [list(range(8 * rank, 8 * (rank + 1))) + list(range(16 + 8 * rank, 24 + 8 * rank))
             + list(range(32 + 8 * rank, 40 + 8 * rank))],
    }
    for name, tensor in expected.items():
        assert torch.equal(rollout.get_parameter(name), tensor)
    assert rollout.head.weight is rollout.body.embed.weight
    assert all(rollout.get_parameter(name) is parameter and parameter.data_ptr() == pointers[name]
               for name, parameter in parameters.items())
    rollout.start_weight_update()
    with pytest.raises(ValueError, match="full tensor shape"):
        rollout.load_weights([("body.embed.weight", rollout.body.embed.weight.detach().clone())])
    with pytest.raises(ValueError, match="failed"):
        rollout.finish_weight_update()


def test_tp2_export_metadata_needs_no_collectives_and_iteration_reads_current_shards(pair, qwen_stage, monkeypatch):
    from isoexec.integrations.skyrl.weight_source import IsoExecWeightSource

    policy, _, _ = pair
    local = clone_graph_module(qwen_stage.module)
    shard_parameter_shapes(local, tp_size=2, side="trainer")
    allocate_tensors(local, device=torch.device("cpu"))
    policy.graph_module = local
    policy.pg_collection.tp = torch.distributed.ProcessGroup(0, 2)
    source = IsoExecWeightSource([policy], torch.bfloat16)
    metadata = source.metadata
    shapes = dict(qwen_stage.module.named_parameters())
    assert [(item.name, item.dtype, item.shape) for item in metadata] == [
        ("body." + name, parameter.dtype, tuple(parameter.shape)) for name, parameter in shapes.items()
    ]
    called = []

    def gather(parts, tensor, *, group):
        assert group is policy.pg_collection.tp
        called.append(tensor)
        parts[0].copy_(tensor)
        parts[1].copy_(tensor + 1)

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    with torch.no_grad():
        for parameter in local.parameters():
            parameter.fill_(2)
    weights = dict(source.iter_full())
    assert torch.equal(weights["body.embed.weight"], torch.cat((torch.full((64, 16), 2), torch.full((64, 16), 3))))
    assert torch.equal(weights["body.layers.0.mlp.down"],
                       torch.cat((torch.full((16, 16), 2), torch.full((16, 16), 3)), dim=1))
    assert torch.equal(weights["body.norm.weight"], torch.full((16,), 2))
    assert called and "head.weight" not in weights
    with torch.no_grad():
        local.embed.weight.fill_(4)
    assert dict(source.iter_full())["body.embed.weight"][0, 0].item() == 4


@pytest.mark.parametrize("rank", [0, 1])
def test_pp2_receiver_initial_hf_and_chunked_updates_preserve_cross_stage_ties(
    pair, worker, vllm_config, monkeypatch, rank,
):
    from isoexec.compiler.capture.capture import capture
    from isoexec.compiler.partition import define_stage
    from isoexec.compiler.text import text_inputs

    _, source, _ = pair
    raw = worker.raw
    author = worker.source.build_meta(raw)
    inputs = text_inputs(author, device="cpu", max_tokens=8, profile="engine")
    captured = capture(author, inputs.args, inputs.kwargs, dynamic_shapes=inputs.dynamic_shapes,
                       root_layouts=inputs.root_layouts, seq=inputs.seq)
    definition = define_stage(captured, ((0, 1), (1, 2)), pp_rank=rank)
    group = torch.distributed.ProcessGroup(rank, 2)
    monkeypatch.setattr(sys.modules["vllm.distributed"], "get_pp_group",
                        lambda: SimpleNamespace(device_group=group))
    vllm_config.parallel_config.pipeline_parallel_size = 2
    vllm_config.additional_config = {"isoexec": {"model_source": worker.source.to_dict(), "raw_model_config": raw}}
    vllm_config.model_config.get_hidden_size = lambda: 16
    vllm_config.compilation_config.static_forward_context.clear()
    rollout = VllmModel(definition,
                        source=worker.source, hf_config=worker.source.build_config(raw),
                        vllm_config=vllm_config)
    parameters = dict(rollout.named_parameters(remove_duplicate=False))
    hf = "model.language_model.embed_tokens.weight"
    assert rollout.load_weights([(hf, torch.full((128, 16), 0.125))]) == {
        "body.embed.weight" if rank == 0 else "head.weight",
    }
    shared = rollout.body.embed.weight if rank == 0 else rollout.head.weight
    assert torch.equal(shared, torch.full_like(shared, 0.125))
    with pytest.raises(ValueError, match="Unknown HF"):
        rollout.load_weights([("model.missing.weight", torch.ones(1))])
    rollout.start_weight_update()
    weights = list(source.iter_full())
    assert "head.weight" not in dict(weights)
    for start in range(0, len(weights), 3):
        rollout.load_weights(weights[start:start + 3])
    rollout.finish_weight_update()
    assert torch.equal(shared, dict(weights)["body.embed.weight"])
    assert all(rollout.get_parameter(name) is parameter for name, parameter in parameters.items())
    assert rollout._mrope and (rollout.head is None) == (rank == 0)
    sequence = ModuleType("vllm.sequence")

    class IntermediateTensors(dict):
        pass

    sequence.IntermediateTensors = IntermediateTensors
    monkeypatch.setitem(sys.modules, sequence.__name__, sequence)
    context = ModuleType("vllm.forward_context")
    context.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)
    monkeypatch.setitem(sys.modules, context.__name__, context)
    calls = []

    def body(**kwargs):
        calls.append(kwargs)
        return torch.full((4, 16), 0.25, dtype=torch.bfloat16)

    monkeypatch.setattr(rollout.body, "forward", body)
    hidden = torch.ones(4, 16, dtype=torch.bfloat16)
    output = rollout(torch.arange(4) if rank == 0 else None, torch.arange(4),
                     None if rank == 0 else IntermediateTensors(hidden_states=hidden))
    assert calls[0]["seq"].positions.shape == (3, 4)
    if rank == 0:
        assert isinstance(output, IntermediateTensors) and output["hidden_states"].shape == (4, 16)
    else:
        assert calls[0]["hidden_states"] is hidden and calls[0]["seq"].input_ids is None
        assert isinstance(output, torch.Tensor) and output.shape == (4, 16)
    assert rollout.make_empty_intermediate_tensors(5, torch.bfloat16, torch.device("cpu"))["hidden_states"].shape == (5, 16)


def _pp_rank_sync(rank, rendezvous):
    from datetime import timedelta

    from megatron.core.distributed.finalize_model_grads import _allreduce_word_embedding_grads
    from megatron.core.transformer.transformer_config import TransformerConfig
    from torch import distributed as dist
    from torch import fx, nn
    from transformers import PretrainedConfig

    from isoexec.compiler.partition import StageDefinition
    from isoexec.nn import Embedding, LMHead, RMSNorm
    from isoexec.ops.table import OpTable
    from isoexec.runtimes.megatron import _binding
    from isoexec.runtimes.megatron import model as native
    from isoexec.schema import SequenceContext

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        tp_groups = [dist.new_group([index]) for index in range(2)]
        group = SimpleNamespace(tp=tp_groups[rank], pp=dist.group.WORLD)
        root, graph = nn.Module(), fx.Graph()
        ids, seq = graph.placeholder("input_ids"), graph.placeholder("seq")
        ids.meta["isoexec_input_type"], seq.meta["isoexec_input_type"] = torch.Tensor, SequenceContext
        embedding = Embedding(vocab=4, hidden=4)
        if rank == 0:
            root.embed, root.norm_first = embedding, RMSNorm(hidden=4)
            value = graph.call_module("embed", (ids,))
            value = graph.call_module("norm_first", (value,))
            value.meta["val"] = torch.empty(3, 4, dtype=torch.bfloat16, device="meta")
            graph.output((value,))
            shared_name = "embed.weight"
        else:
            hidden = graph.placeholder("hidden")
            hidden.meta["val"] = torch.empty(3, 4, dtype=torch.bfloat16, device="meta")
            root.norm_last = RMSNorm(hidden=4)
            root.head = LMHead(vocab=4, hidden=4, tied_to=embedding.weight)
            value = graph.call_module("norm_last", (hidden,))
            graph.output(graph.call_module("head", (value,)))
            shared_name = "head.weight"
        definition = StageDefinition(fx.GraphModule(root, graph), shared_name)
        allocate = native.allocate_tensors
        native.allocate_tensors = lambda module, **kwargs: allocate(module, device=torch.device("cpu"))
        torch.cuda.current_device = lambda: 0
        reference = OpTable(mode="reference")
        _binding.OpTable = lambda: reference
        config = TransformerConfig(num_layers=2, hidden_size=4, num_attention_heads=1,
                                   pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16,
                                   params_dtype=torch.bfloat16, bf16=True)
        model = native.MegatronModel(definition,
                                     source=ModelSource("isoexec.models.qwen2_moe"), hf_config=PretrainedConfig(), config=config, pg_collection=group)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(rank + 1)
            model.shared_embedding_or_output_weight().copy_(torch.arange(16).reshape(4, 4))
        shared = model.shared_embedding_or_output_weight()
        shared.main_grad = torch.full_like(shared, rank + 1, dtype=torch.float32)
        wrapped = SimpleNamespace(module=model, ddp_config=SimpleNamespace(use_megatron_fsdp=False))
        _allreduce_word_embedding_grads([wrapped], config, embd_group=dist.group.WORLD, pp_group=dist.group.WORLD)
        assert torch.equal(shared.main_grad, torch.full((4, 4), 3.0))
        base = ModuleType("vllm.distributed.weight_transfer.base")
        base.ParamMeta, base.WeightSource = ParamMeta, object
        sys.modules[base.__name__] = base
        from isoexec.integrations.skyrl.weight_source import IsoExecWeightSource

        source = IsoExecWeightSource([model], torch.bfloat16)
        metadata = source.metadata
        assert [item.name for item in metadata] == ["body.embed.weight", "body.norm_first.weight", "body.norm_last.weight"]
        weights = dict(source.iter_full())
        assert torch.equal(weights["body.embed.weight"], torch.arange(16).reshape(4, 4).bfloat16())
        assert torch.equal(weights["body.norm_first.weight"], torch.ones(4))
        assert torch.equal(weights["body.norm_last.weight"], torch.full((4,), 2))
        assert [(item.name, item.dtype, item.shape) for item in metadata] == [
            (name, tensor.dtype, tuple(tensor.shape)) for name, tensor in weights.items()
        ]
        with torch.no_grad():
            shared.fill_(0.25)
        assert torch.equal(dict(source.iter_full())["body.embed.weight"], torch.full((4, 4), 0.25))
    finally:
        dist.destroy_process_group()


def test_real_pp_group_merges_shared_gradients_and_streams_all_stages(tmp_path):
    torch.multiprocessing.spawn(_pp_rank_sync, args=((tmp_path / "pp").as_uri(),), nprocs=2, join=True)


def _ep_rank_sync(rank, rendezvous):
    from datetime import timedelta

    from megatron.core.transformer.transformer_config import TransformerConfig
    from torch import distributed as dist
    from torch import fx, nn
    from transformers import PretrainedConfig

    import isoexec.nn as inn
    from isoexec.compiler.partition import StageDefinition
    from isoexec.ops import _dispatch
    from isoexec.runtimes.megatron import model as native

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        # Production bindings admit the declared architecture; nothing here executes a kernel.
        _dispatch.ARCH = "sm90"
        single = [dist.new_group([index]) for index in range(2)][rank]
        group = SimpleNamespace(tp=single, pp=single, cp=single, expt_tp=single, dp=dist.group.WORLD,
                                ep=dist.group.WORLD)
        root, graph = nn.Module(), fx.Graph()
        root.moe = inn.MoE(hidden=8, num_experts=4, top_k=2, intermediate=6, shared_intermediate=0)
        graph.output(graph.call_module("moe", (graph.placeholder("x"),)))
        allocate = native.allocate_tensors
        native.allocate_tensors = lambda module, **kwargs: allocate(module, device=torch.device("cpu"))
        torch.cuda.current_device = lambda: 0
        config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1, num_moe_experts=4,
                                   moe_router_topk=2, expert_model_parallel_size=2, add_bias_linear=False,
                                   bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16)
        model = native.MegatronModel(StageDefinition(fx.GraphModule(root, graph), None),
                                     source=ModelSource("isoexec.models.qwen2_moe"), hf_config=PretrainedConfig(),
                                     config=config, pg_collection=group)
        held = model.graph_module.moe.held_experts()
        assert [int(name) for name in model.graph_module.moe.experts._modules] == [2 * rank, 2 * rank + 1]
        assert all(parameter.allreduce is False for expert in held for parameter in expert.parameters())
        assert getattr(model.graph_module.moe.router.weight, "allreduce", True)

        def expert_value(index, role, shape, offset=0.0):
            return torch.full(shape, 10 * index + ("gate", "up", "down").index(role) + offset, dtype=torch.bfloat16)

        def fill(offset):
            with torch.no_grad():
                model.graph_module.moe.router.weight.fill_(0.5)
                for name, expert in model.graph_module.moe.experts.named_children():
                    for role in ("gate", "up", "down"):
                        parameter = expert.get_parameter(role)
                        parameter.copy_(expert_value(int(name), role, parameter.shape, offset))

        base = ModuleType("vllm.distributed.weight_transfer.base")
        base.ParamMeta, base.WeightSource = ParamMeta, object
        sys.modules[base.__name__] = base
        from isoexec.integrations.skyrl.weight_source import IsoExecWeightSource

        source = IsoExecWeightSource([model], torch.bfloat16)
        roles = {"gate": (6, 8), "up": (6, 8), "down": (8, 6)}
        expected = [("body.moe.router.weight", torch.float32, (4, 8))] + [
            (f"body.moe.experts.{index}.{role}", torch.bfloat16, shape)
            for role, shape in roles.items() for index in range(4)
        ]
        assert [(item.name, item.dtype, item.shape) for item in source.metadata] == expected
        for offset in (0.0, 0.25):
            fill(offset)
            weights = list(source.iter_full())
            assert [(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in weights] == expected
            for name, tensor in weights[1:]:
                index, role = int(name.split(".")[3]), name.split(".")[4]
                assert torch.equal(tensor, expert_value(index, role, roles[role], offset)), (rank, name)
    finally:
        dist.destroy_process_group()


def test_ep2_source_streams_every_global_expert_from_the_held_stacks(tmp_path):
    torch.multiprocessing.spawn(_ep_rank_sync, args=((tmp_path / "ep").as_uri(),), nprocs=2, join=True)

