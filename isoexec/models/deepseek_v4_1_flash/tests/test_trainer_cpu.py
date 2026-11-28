"""CPU contract of the V4.1 trainer binding: trainable tensors, packed-sequence layout, the gradient
reductions Megatron installs, and the weight stream the engine receives."""
import copy
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch

from isoexec.compiler.capture.capture import capture
from isoexec.compiler.realization.tp import shard_parameter_shapes
from isoexec.models.deepseek_v4_1_flash import training
from isoexec.models.deepseek_v4_1_flash.holders import MoEWeights
from isoexec.models.deepseek_v4_1_flash.tests.test_deepseek_v4_1_flash_cpu import SOURCE, placement, raw
from isoexec.schema import make_sequence_context

_binding = pytest.importorskip("isoexec.runtimes.megatron._binding", exc_type=ImportError)


@pytest.fixture(scope="module")
def debug_model():
    return SOURCE.build_meta(raw("debug_config.json"))


def _placed(debug_model, tp, rank, ep=1):
    model = copy.deepcopy(debug_model)
    p = placement(tp, rank % tp, ep, rank % ep, "trainer")
    shard_parameter_shapes(model, tp_size=tp, side="trainer", etp_size=1 if ep > 1 else tp)
    for i, block in enumerate(model.layers()):
        block.apply_placement(p, site=f"layers.{i}")
    model.decoder.entry.apply_placement(p, site="decoder.entry")
    model.embed.apply_placement(p, site="embed")
    model.norm.apply_placement(p, site="norm")
    return model


@pytest.mark.parametrize("tp,ep", [(1, 1), (2, 1), (2, 2)])
def test_trainer_placement_marks_the_actor_recipe_trainable(debug_model, tp, ep):
    model = _placed(debug_model, tp, 1, ep)
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    assert frozen == {name for name, _ in model.named_parameters()
                      if ".indexer." in name or name.endswith("gate.bias")}
    assert frozen and all(f".attn.indexer." in n or ".ffn.gate.bias" in n for n in frozen)
    assert model.embed.weight.requires_grad and model.norm.weight.requires_grad
    held = model.layers()[0].ffn.held_experts()
    E = model.cfg.n_routed_experts
    assert held == (tuple(range(E // 2, E)) if ep == 2 else tuple(range(E)))
    # the engine keeps its copies read-only
    engine = copy.deepcopy(debug_model)
    engine.layers()[0].apply_placement(placement(tp, 0, ep, 0, "engine"), site="layers.0")
    assert not any(p.requires_grad for p in engine.layers()[0].parameters())


def test_trainer_packed_experts_are_stacked_copies_not_repointed():
    torch.manual_seed(0)
    cfg = SimpleNamespace(n_routed_experts=4, hidden_size=8, moe_intermediate_size=6)
    ffn = MoEWeights(cfg, "cpu")
    for p in ffn.parameters():
        p.data.normal_()
        p.requires_grad_(True)
    block = SimpleNamespace(ffn=ffn, ep=SimpleNamespace(local=(2, 3)), placement=SimpleNamespace(side="trainer"),
                            _packed=None)
    from isoexec.models.deepseek_v4_1_flash.block import V41Layer
    pointers = [p.data_ptr() for p in ffn.parameters()]
    gate_up, down = V41Layer.packed_experts(block)
    assert [p.data_ptr() for p in ffn.parameters()] == pointers       # DDP buffer views stay in place
    experts = [ffn.experts["2"], ffn.experts["3"], ffn.shared_experts]
    assert gate_up.shape == (3, 12, 8) and down.shape == (3, 8, 6)
    for g, e in enumerate(experts):
        assert torch.equal(gate_up[g], torch.cat((e.w1.weight, e.w3.weight)))
        assert torch.equal(down[g], e.w2.weight)
    # the copies name their parameters, so the MoE backward saves those instead of the copies
    from isoexec.ops.moe.deepseek_v41_sm100.packing import EXPERT_PARAMS_ATTR
    assert getattr(gate_up, EXPERT_PARAMS_ATTR) == tuple((e.w1.weight, e.w3.weight) for e in experts)
    assert getattr(down, EXPERT_PARAMS_ATTR) == tuple((e.w2.weight,) for e in experts)
    (gate_up * torch.arange(gate_up.numel()).view(gate_up.shape)).sum().backward()
    assert torch.equal(experts[1].w3.weight.grad, torch.arange(gate_up.numel()).view(gate_up.shape)[1, 6:].float())
    assert ffn.experts["0"].w1.weight.grad is None


@pytest.mark.parametrize("tp,ep", [(1, 1), (1, 2), (2, 2)])
def test_routed_expert_freeze_switch_freezes_routed_experts_only(debug_model, tp, ep, monkeypatch):
    from isoexec.models.deepseek_v4_1_flash import config as v41_config
    recipe = {name for name, p in _placed(debug_model, tp, 1, ep).named_parameters() if not p.requires_grad}
    assert v41_config.program_recorded()["trained_set"] == training.TRAINED_SET_ACTOR == training.trained_set()
    monkeypatch.setenv(training.FREEZE_ROUTED_EXPERTS_ENV, "1")
    model = _placed(debug_model, tp, 1, ep)
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    routed = {name for name, _ in model.named_parameters() if ".ffn.experts." in name}
    assert routed and frozen == recipe | routed and not recipe & routed
    shared = [p for name, p in model.named_parameters() if ".ffn.shared_experts." in name]
    assert shared and all(p.requires_grad for p in shared)
    assert all(p.requires_grad for name, p in model.named_parameters() if ".ffn.gate.weight" in name)
    assert v41_config.program_recorded()["trained_set"] == "actor_routed_experts_frozen"
    monkeypatch.setenv(training.FREEZE_ROUTED_EXPERTS_ENV, "yes")
    with pytest.raises(ValueError, match="ISOEXEC_DS_FREEZE_ROUTED_EXPERTS must be 0 or 1"):
        training.trained_set()


def test_routed_expert_freeze_switch_is_a_classified_trainer_flag():
    from isoexec.core import flags
    flag = flags.get(training.FREEZE_ROUTED_EXPERTS_ENV)
    assert flag.default == "0" and flag.sides == ("trainer",) and flag.disposition == flags.FUNCTION
    assert training.FREEZE_ROUTED_EXPERTS_ENV in flags.NUMERICS_SWITCHES


def test_sequence_rows_pad_unpad_and_gradients_are_gathers():
    bounds = torch.tensor([0, 3, 4, 9], dtype=torch.int32)
    layout = training.SequenceRows.from_boundaries(bounds.tolist(), torch.device("cpu"))
    assert (layout.sequences, layout.longest, layout.tokens, layout.lengths) == (3, 5, 9, (3, 1, 5))
    assert layout.rows.tolist() == [0, 1, 2, 5, 10, 11, 12, 13, 14]
    x = torch.arange(1, 9 * 2 + 1, dtype=torch.float32).view(1, 9, 2).requires_grad_(True)
    padded = layout.pad(x)
    assert padded.shape == (3, 5, 2)
    assert torch.equal(padded[0, :3], x[0, :3]) and torch.equal(padded[1, 0], x[0, 3])
    assert torch.equal(padded[2], x[0, 4:]) and not padded[0, 3:].any() and not padded[1, 1:].any()
    back = layout.unpad(padded * 2)
    assert torch.equal(back, x * 2)
    weight = torch.randn_like(back)
    (back * weight).sum().backward()
    assert torch.equal(x.grad, 2 * weight)
    # padding positions receive no gradient from the packed rows
    y = torch.randn(3, 5, 2, requires_grad=True)
    (layout.unpad(y) * weight).sum().backward()
    assert not y.grad[0, 3:].any() and not y.grad[1, 1:].any() and torch.equal(y.grad[2], weight[0, 4:])
    with pytest.raises(ValueError):
        training.SequenceRows.from_boundaries([0, 3, 3], torch.device("cpu"))


def test_sequence_rows_from_context():
    x = torch.zeros(1, 7, 4)
    one = make_sequence_context(torch.arange(7), torch.tensor([0, 7], dtype=torch.int32), max_q=7,
                                cu_seqlens_cpu=torch.tensor([0, 7], dtype=torch.int32))
    assert training.sequence_rows(None, x) is None and training.sequence_rows(one, x) is None
    cu = torch.tensor([0, 2, 7], dtype=torch.int32)
    two = make_sequence_context(torch.tensor([0, 1, 0, 1, 2, 3, 4]), cu, max_q=5, cu_seqlens_cpu=cu)
    layout = training.sequence_rows(two, x)
    assert layout.lengths == (2, 5) and training.sequence_rows(two, x) is layout
    with pytest.raises(ValueError, match="host sequence boundaries"):
        training.sequence_rows(make_sequence_context(torch.arange(7), cu, max_q=5), x)


def test_trainer_hashes_each_packed_sequence_from_an_empty_history(debug_model, monkeypatch):
    from isoexec.ops.embed.engram_sm100 import ops as engram_ops
    calls = []

    seg = []

    def fake_hash(ids, *args, **kwargs):
        calls.append((ids.clone(), kwargs.get("state")))
        if kwargs.get("seg_start") is not None:
            seg.append(kwargs["seg_start"])
        return ids[..., None, None].expand(*ids.shape, 2, 24).clone()

    monkeypatch.setattr(engram_ops, "ngram_hash", fake_hash)
    embed = copy.deepcopy(debug_model.embed)
    embed.placement = placement(1, 0)
    embed._hash = {"cpu": (None,) * 4 + (0,)}
    monkeypatch.setattr(type(embed), "_hash_tensors", lambda self, device: (None,) * 4 + (0,))
    ids = torch.arange(10, 17).view(1, 7)
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    seq = make_sequence_context(torch.tensor([0, 1, 2, 0, 1, 2, 3]), cu, max_q=4, cu_seqlens_cpu=cu)
    hashes = embed._packed_hashes(ids, seq)
    # one launch over the packed rows, each token's sequence start given (every sequence from an empty history)
    assert [c[0].tolist() for c in calls] == [[[10, 11, 12, 13, 14, 15, 16]]]
    assert all(state is None for _, state in calls)
    assert seg[0].tolist() == [0, 0, 0, 3, 3, 3, 3]
    assert torch.equal(hashes[..., 0, 0], ids)


def _graph(debug_model, tp, rank, ep=1):
    """The captured program with its rows placed as the trainer binder places them (after capture)."""
    graph = capture(copy.deepcopy(debug_model), (torch.zeros(1, 16, dtype=torch.long),)).program
    p = placement(tp, rank % tp, ep, rank % ep, "trainer")
    shard_parameter_shapes(graph, tp_size=tp, side="trainer", etp_size=1 if ep > 1 else tp)
    for name, module in list(graph.named_modules()):
        if getattr(module, "PLACEMENT_BOUND", False):
            module.apply_placement(p, site=name)
    return graph, None


def test_partial_gradient_sums_complete_the_replicated_attention_tensors_in_rank_order(debug_model, monkeypatch):
    from isoexec.ops.collectives import ordered_sum as ordered
    seen = []
    monkeypatch.setattr(ordered, "ordered_sum", lambda grad, group: seen.append(group) or grad + 1)
    group = SimpleNamespace(size=lambda: 2)
    graph, _ = _graph(debug_model, 2, 0)
    hooked = _binding.set_partial_gradient_sums(graph, group)
    expected = sorted(name for name, p in graph.named_parameters() if p.requires_grad and ".attn." in name
                      and not name.endswith(("attn_sink", "wq_b.weight", "wo_a.weight", "wo_b.weight")))
    assert sorted(hooked) == expected
    assert any(".compressor." in n for n in hooked) and not any(".indexer." in n for n in hooked)
    assert all(".attn." in n for n in hooked)
    parameter = graph.get_parameter(hooked[0])
    hook = next(iter(parameter._backward_hooks.values()))
    assert torch.equal(hook(torch.zeros(2)), torch.ones(2)) and seen == [group]
    with pytest.raises(ValueError, match="already installed"):
        _binding.set_partial_gradient_sums(graph, group)
    assert _binding.set_partial_gradient_sums(_graph(debug_model, 1, 0)[0], SimpleNamespace(size=lambda: 1)) == []


def test_expert_tensors_reduce_over_expert_data_parallel_only(debug_model):
    graph, _ = _graph(debug_model, 2, 1, 2)
    _binding.set_expert_parallel_attributes(graph)
    marked = {name for name, p in graph.named_parameters() if getattr(p, "allreduce", True) is False}
    assert marked == {name for name, _ in graph.named_parameters() if ".ffn.experts." in name}
    assert all(int(name.split(".experts.")[1].split(".")[0]) >= 128 for name in marked)


def test_megatron_config_takes_the_bound_rows_dimensions(debug_model):
    from isoexec.runtimes.megatron._config import make_transformer_config
    graph = capture(copy.deepcopy(debug_model), (torch.zeros(1, 16, dtype=torch.long),))
    config = make_transformer_config(graph, tensor_model_parallel_size=2, expert_model_parallel_size=2,
                                     expert_tensor_parallel_size=1)
    cfg = debug_model.cfg
    # pipeline units: the encoder blocks and the decoder as one unit (no cut inside it)
    assert (config.num_layers, config.num_attention_heads, config.kv_channels) == (len(cfg.encoder) + 1,
                                                                                  cfg.num_attention_heads, 512)
    assert (config.num_moe_experts, config.moe_ffn_hidden_size) == (256, 128)


# ------------------------------------------------------------------------- weight transport (CPU fakes)
@dataclass
class ParamMeta:
    name: str
    dtype: torch.dtype
    shape: tuple


def _fake_transfer_base():
    base = ModuleType("vllm.distributed.weight_transfer.base")
    base.ParamMeta, base.WeightSource = ParamMeta, object
    return base


@pytest.fixture
def weight_source(monkeypatch):
    base = _fake_transfer_base()
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.delitem(sys.modules, "isoexec.integrations.skyrl.weight_source", raising=False)
    from isoexec.integrations.skyrl import weight_source
    return weight_source


def _source(module, weight_source, groups):
    src = weight_source.IsoExecWeightSource.__new__(weight_source.IsoExecWeightSource)
    src._module, src._groups, src._tp_group, src._pp_group = module, groups, groups.tp, groups.pp
    src._head_name = "head" if hasattr(module, "head") else None
    return src


def test_stream_names_every_engine_registered_tensor_of_the_debug_model(debug_model, weight_source):
    one = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    source = _source(debug_model, weight_source, SimpleNamespace(tp=one, pp=one, ep=one))
    names = [m.name for m in source.metadata]
    want = [("head." + n[5:]) if n.startswith("head.") else "body." + n for n, _ in debug_model.named_parameters()]
    assert names == want and "body.encoder.blocks.4.engram.wkv.weight" in names
    assert not any(".engram.embed." in n for n in names)        # frozen checkpoint tables are not streamed


def _ep_round_trip(rank, rendezvous, results):
    from datetime import timedelta

    from torch import distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        base = _fake_transfer_base()
        sys.modules[base.__name__] = base
        from isoexec.integrations.skyrl import weight_source
        cfg = SimpleNamespace(n_routed_experts=4, hidden_size=8, moe_intermediate_size=6)

        def filled(module):
            with torch.no_grad():
                for name, p in module.named_parameters():
                    p.copy_(torch.full_like(p, float(sum(map(ord, name)) % 97)))
            return module

        trainer = torch.nn.Module()
        trainer.layers = torch.nn.ModuleList([torch.nn.Module()])
        trainer.layers[0].ffn = filled(MoEWeights(cfg, "cpu"))
        for key in [k for k in trainer.layers[0].ffn.experts._modules if int(k) // 2 != rank]:
            del trainer.layers[0].ffn.experts._modules[key]
        assert _binding.mark_declared_experts(trainer) == 6
        trainer.layers[0].ffn.mark_ep_summed()       # block._place at trainer TP1 x EP > 1 (token-disjoint EP)
        one = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
        source = _source(trainer, weight_source, SimpleNamespace(tp=one, pp=one, ep=dist.group.WORLD))
        names = [m.name for m in source.metadata]
        stream = list(source.iter_full())
        assert [n for n, _ in stream] == names
        # the engine side: one full replica receives every global expert under its registered name
        engine = torch.nn.Module()
        engine.layers = torch.nn.ModuleList([torch.nn.Module()])
        engine.layers[0].ffn = MoEWeights(cfg, "cpu")
        registered = dict(engine.named_parameters())
        with torch.no_grad():
            for name, tensor in stream:
                registered[name.removeprefix("body.")].copy_(tensor)
        reference = filled(MoEWeights(cfg, "cpu"))
        ok = all(torch.equal(p, dict(reference.named_parameters())[n])
                 for n, p in engine.layers[0].ffn.named_parameters())
        results[rank] = (ok, len(names))
    finally:
        dist.destroy_process_group()


def test_ep2_stream_round_trips_every_global_expert_into_an_engine_replica(tmp_path):
    results = torch.multiprocessing.Manager().dict()
    torch.multiprocessing.spawn(_ep_round_trip, args=((tmp_path / "ep").as_uri(), results), nprocs=2, join=True)
    assert dict(results) == {0: (True, 2 + 4 * 3 + 3), 1: (True, 2 + 4 * 3 + 3)}     # router weight, bias


def test_cross_row_state_crosses_rows_in_the_captured_packed_geometry():
    layout = training.SequenceRows.from_boundaries([0, 5, 6, 13], torch.device("cpu"))   # lengths 5, 1, 7
    B, L, N, r = 3, 7, 13, 2
    per = L // r                                               # 3 entries per padded sequence
    kv = torch.arange(B * per * 4, dtype=torch.float32).view(B * per, 4).requires_grad_(True)
    codes = torch.randint(0, 255, (B, 128, 8), dtype=torch.uint8)
    topk = torch.arange(B * L * 2, dtype=torch.int32).view(B * L, 2)
    packed = layout.pack_state(("kv", "index_codes", "topk"), (kv, codes, topk), r)
    assert packed[0].shape == (N // r, 4) and packed[1].shape == (1, 128, 8) and packed[2].shape == (N, 2)
    valid = [0, 1, 2 * per + 0, 2 * per + 1, 2 * per + 2]      # seq 0: 2 entries, seq 1: none, seq 2: 3
    assert torch.equal(packed[0][:5], kv[valid].detach()) and not packed[0][5:].any()
    assert torch.equal(packed[1][0, :5], codes.view(-1, 8)[[0, 1, 256, 257, 258]])
    back = layout.unpack_state({"kv": packed[0], "index_codes": packed[1], "topk": packed[2]}, r)
    assert torch.equal(back["kv"][valid], kv[valid].detach()) and not back["kv"][[2, 3, 4, 5]].any()
    assert back["index_codes"].shape == (B, 128, 8)
    assert torch.equal(back["topk"][layout.rows], topk[layout.rows])
    assert torch.equal(back["topk"][1 * L + 3], topk[1 * L + 0])     # padding repeats the last token row
    back["kv"].sum().backward()
    assert kv.grad[valid].eq(1).all() and not kv.grad[[2, 3, 4, 5]].any()
