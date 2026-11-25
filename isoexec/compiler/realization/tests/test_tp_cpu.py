"""TP-local declarations preserve graph state, aliases, and head granularity."""

from types import SimpleNamespace

import pytest
import torch
from torch import fx, nn
from transformers import PretrainedConfig
from transformers.core_model_loading import WeightRenaming

import isoexec.nn as inn
from isoexec.compiler.realization import weights as weight_loading
from isoexec.compiler.realization.graph import clone_graph_module
from isoexec.compiler.realization.storage import allocate_tensors
from isoexec.compiler.realization.tests import test_weights_cpu
from isoexec.compiler.realization.tp import (
    gather_weight,
    global_shape,
    set_head_output_shape,
    set_local_dims,
    shard_parameter_shapes,
    slice_weight,
)
from isoexec.compiler.realization.weights import load_weights
from isoexec.compiler.capture.export_normalize import _checked_output
from isoexec.compiler.source import ModelSource
from isoexec.schema import col, replicated, row

qwen_stage = test_weights_cpu.qwen_stage


@pytest.mark.parametrize("tp_size", [1, 2])
def test_captured_head_check_uses_local_width_without_changing_source(qwen_stage, tp_size):
    source = qwen_stage.module
    original = str(source.graph)
    target = clone_graph_module(source)
    names = [node.name for node in target.graph.nodes]
    shard_parameter_shapes(target, tp_size=tp_size, side="trainer")
    set_head_output_shape(target)
    guard = next(node for node in target.graph.nodes
                 if node.target is _checked_output and node.args[2] == "head")
    assert guard.args[3][-1] == guard.args[4][-2] == 128 // tp_size
    assert [node.name for node in target.graph.nodes] == names
    assert str(source.graph) == original
    shape = [5, 128 // tp_size]
    output = torch.empty(shape, dtype=torch.float32)
    hidden = torch.empty(5, 16)
    assert _checked_output(output, guard.args[1], guard.args[2], shape, guard.args[4],
                           guard.args[5], (hidden,)) is output
    with pytest.raises(RuntimeError, match="metadata"):
        _checked_output(torch.empty(5, shape[-1] + 1), guard.args[1], guard.args[2], shape,
                        guard.args[4], guard.args[5], (hidden,))


@pytest.fixture
def source():
    root = nn.Module()
    root.embed = inn.Embedding(vocab=16, hidden=8)
    root.norm = inn.RMSNorm(hidden=8, eps=1e-6)
    root.attn = inn.Attention(hidden=8, heads=4, kv_heads=2, head_dim=4, rope_theta=10000)
    root.gdn = inn.GDN(hidden=8, key_heads=2, value_heads=4, key_head_dim=4, value_head_dim=4, conv_kernel=4)
    root.mlp = inn.MLP(hidden=8, intermediate=16)
    root.head = inn.LMHead(vocab=16, hidden=8, tied_to=root.embed.weight)
    root.norm.weight.requires_grad_(False)
    root.gdn.A_log = nn.Parameter(torch.empty(4, dtype=torch.float32, device="meta"))
    graph = fx.Graph()
    ids, seq = graph.placeholder("ids"), graph.placeholder("seq")
    hidden = graph.call_module("embed", (ids,))
    hidden = graph.call_module("norm", (hidden,))
    hidden = graph.call_module("attn", (hidden, seq))
    hidden = graph.call_module("gdn", (hidden, seq))
    hidden = graph.call_module("mlp", (hidden,))
    graph.output(graph.call_module("head", (hidden,)))
    module = fx.GraphModule(root, graph).eval()
    module.register_buffer("positions", torch.arange(4), persistent=False)
    return module


@pytest.mark.parametrize("tp_size", [1, 2])
def test_local_shapes_preserve_source_graph_and_registered_state(source, tp_size):
    target = clone_graph_module(source)
    original = dict(target.named_parameters(remove_duplicate=False))
    global_shapes = {name: tuple(value.shape) for name, value in source.named_parameters(remove_duplicate=False)}
    properties = {name: (value.dtype, value.requires_grad) for name, value in original.items()}
    positions = target.positions
    expected = global_shapes if tp_size == 1 else {
        "embed.weight": (8, 8), "norm.weight": (8,),
        "attn.q": (8, 8), "attn.k": (4, 8), "attn.v": (4, 8), "attn.gate": (8, 8),
        "attn.q_norm": (4,), "attn.k_norm": (4,), "attn.o": (8, 8),
        "gdn.q": (4, 8), "gdn.k": (4, 8), "gdn.v": (8, 8), "gdn.z": (8, 8),
        "gdn.beta": (2, 8), "gdn.alpha": (2, 8), "gdn.conv_w": (16, 1, 4),
        "gdn.A_log": (2,), "gdn.dt_bias": (2,), "gdn.out_norm": (4,), "gdn.o": (8, 8),
        "mlp.gate": (8, 8), "mlp.up": (8, 8), "mlp.down": (8, 8), "head.weight": (8, 8),
    }

    shard_parameter_shapes(target, tp_size=tp_size, side="trainer")
    set_local_dims(target, tp_size=tp_size)

    assert (target.attn.heads_local, target.attn.kv_heads_local) == (4 // tp_size, 2 // tp_size)
    assert (target.gdn.key_heads_local, target.gdn.value_heads_local) == (2 // tp_size, 4 // tp_size)
    assert (target.gdn.key_dim_local, target.gdn.value_dim_local) == (8 // tp_size, 16 // tp_size)
    assert (target.attn.heads, target.attn.kv_heads, target.gdn.key_heads, target.gdn.value_heads) == (4, 2, 2, 4)
    assert {name: tuple(value.shape) for name, value in target.named_parameters(remove_duplicate=False)} == expected
    assert str(target.graph) == str(source.graph)
    assert target.embed.weight is target.head.weight
    assert target.positions is positions and "positions" not in target.state_dict()
    assert not target.training
    for name, parameter in target.named_parameters(remove_duplicate=False):
        assert parameter is original[name] and parameter.is_meta
        assert (parameter.dtype, parameter.requires_grad) == properties[name]
    assert {name: tuple(value.shape) for name, value in source.named_parameters(remove_duplicate=False)} == global_shapes
    assert source.embed.weight is source.head.weight


@pytest.mark.parametrize("etp_size", [2, 1])
def test_experts_split_along_their_etp_axes(etp_size):
    # ETP=TP at EP=1; EP>1 holds whole experts (ETP=1) while the shared expert still splits over TP.
    root = nn.Module()
    root.moe = inn.MoE(hidden=8, num_experts=2, top_k=1, intermediate=6, shared_intermediate=4)
    full = {name: torch.randn(parameter.shape) for name, parameter in root.named_parameters()}

    shard_parameter_shapes(root, tp_size=2, side="trainer", etp_size=etp_size)

    width = 6 // etp_size
    assert {name: tuple(value.shape) for name, value in root.named_parameters()} == {
        "moe.router.weight": (2, 8),
        **{f"moe.experts.{e}.{role}": shape for e in range(2)
           for role, shape in (("gate", (width, 8)), ("up", (width, 8)), ("down", (8, width)))},
        "moe.shared.gate": (2, 8), "moe.shared.up": (2, 8), "moe.shared.down": (8, 2),
        "moe.shared.expert_gate": (1, 8),
    }
    for name, expected in (("moe.experts.1.gate", full["moe.experts.1.gate"][6 - width:]),
                           ("moe.experts.1.down", full["moe.experts.1.down"][:, 6 - width:])):
        local = slice_weight(root, name, full[name], tp_size=2, tp_rank=1, side="trainer", etp_size=etp_size)
        assert torch.equal(local, expected)
        assert global_shape(root, name, local, tp_size=2, side="trainer", etp_size=etp_size) == full[name].shape


@pytest.mark.parametrize("case", ["tie", "heads", "below", "segments", "stride", "strided_segments"])
def test_invalid_declarations_fail_without_changing_parameters(source, monkeypatch, case):
    target = clone_graph_module(source)
    tp_size = 2
    if case == "tie":
        monkeypatch.setattr(target.head, "tensor_axes", lambda: {"weight": row()})
        match = "shared parameter"
    elif case == "heads":
        target.gdn = inn.GDN(hidden=8, key_heads=3, value_heads=6, key_head_dim=4, value_head_dim=4, conv_kernel=4)
        match = "3 units do not divide"
    elif case == "below":
        # Trainers split only attention KV heads below TP; GDN has no partial-head gather.
        tp_size, match = 4, "2 units below tp=4"
    else:
        axes = dict(target.mlp.tensor_axes())
        axes["up"] = {"segments": col(segments=((3, None), (13, None))), "stride": col(stride=16),
                      "strided_segments": col(stride=2, segments=((8, None), (8, None)))}[case]
        monkeypatch.setattr(target.mlp, "tensor_axes", lambda: axes)
        match = "do not support stride" if case == "strided_segments" else "not divisible by tp=2"
    original = {name: (value, tuple(value.shape)) for name, value in target.named_parameters(remove_duplicate=False)}

    with pytest.raises(ValueError, match=match):
        shard_parameter_shapes(target, tp_size=tp_size, side="trainer")

    for name, parameter in target.named_parameters(remove_duplicate=False):
        assert parameter is original[name][0] and tuple(parameter.shape) == original[name][1]


@pytest.mark.parametrize("rank", [0, 1])
def test_qwen_hf_conversion_precedes_rank_selection(qwen_stage, rank):
    target = clone_graph_module(qwen_stage.module)
    shard_parameter_shapes(target, tp_size=2, side="trainer")
    set_local_dims(target, tp_size=2)
    allocate_tensors(target, device=torch.device("cpu"))
    parameters = dict(target.named_parameters(remove_duplicate=False))
    prefix = "model.language_model."
    embedding = torch.arange(128 * 16, dtype=torch.float32).reshape(128, 16)
    qkv = torch.arange(48 * 16, dtype=torch.float32).reshape(48, 16)
    q_gate = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16)
    conv = torch.arange(48 * 4, dtype=torch.float32).reshape(48, 1, 4)
    down = torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32)
    weights = {
        prefix + "embed_tokens.weight": embedding,
        prefix + "layers.0.linear_attn.in_proj_qkv.weight": qkv,
        prefix + "layers.0.linear_attn.conv1d.weight": conv,
        prefix + "layers.1.self_attn.q_proj.weight": q_gate,
        prefix + "layers.0.mlp.down_proj.weight": down,
        prefix + "norm.weight": torch.arange(16, dtype=torch.float32),
    }
    before = {name: value.clone() for name, value in weights.items()}

    loaded = load_weights(target, weights.items(),
                          source=ModelSource("isoexec.models.qwen3_5"), hf_config=test_weights_cpu.qwen_config(), tp_size=2, tp_rank=rank,
                          side="trainer")

    q_rows, gate_rows = (([0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15]) if rank == 0 else
                         ([16, 17, 18, 19, 24, 25, 26, 27], [20, 21, 22, 23, 28, 29, 30, 31]))
    conv_rows = (list(range(8)) + list(range(16, 24)) + list(range(32, 40)) if rank == 0 else
                 list(range(8, 16)) + list(range(24, 32)) + list(range(40, 48)))
    expected = {
        "embed.weight": embedding[64 * rank:64 * (rank + 1)],
        "layers.1.mixer.q": q_gate[q_rows], "layers.1.mixer.gate": q_gate[gate_rows],
        "layers.0.mixer.conv_w": conv[conv_rows],
        "layers.0.mlp.down": down[:, 16 * rank:16 * (rank + 1)],
        "norm.weight": torch.arange(16, dtype=torch.float32),
    }
    for role, offset in (("q", 0), ("k", 16), ("v", 32)):
        expected[f"layers.0.mixer.{role}"] = qkv[offset + 8 * rank:offset + 8 * (rank + 1)]
    assert loaded == expected.keys()
    for name, value in expected.items():
        parameter = target.get_parameter(name)
        assert torch.equal(parameter, value.to(parameter.dtype)), name
    assert target.embed.weight is target.head.weight
    assert all(target.get_parameter(name) is value for name, value in parameters.items())
    assert all(torch.equal(value, before[name]) for name, value in weights.items())
    assert all(parameter.is_meta for parameter in qwen_stage.module.parameters())


@pytest.mark.parametrize("axis,rank,indices", [
    (col(stride=2), 0, [0, 1, 2, 3, 8, 9, 10, 11]),
    (col(stride=2), 1, [4, 5, 6, 7, 12, 13, 14, 15]),
    (col(segments=((4, None), (12, None))), 0, [0, 1, 4, 5, 6, 7, 8, 9]),
    (col(segments=((4, None), (12, None))), 1, [2, 3, 10, 11, 12, 13, 14, 15]),
    (row(), 1, list(range(8, 16))),
    (replicated(), 1, list(range(16))),
])
def test_strided_segmented_row_and_replicated_entries(monkeypatch, axis, rank, indices):
    mapping_module = SimpleNamespace(checkpoint_conversion_mapping=lambda config: [WeightRenaming("^hf.weight$", "weight")])
    monkeypatch.setattr(ModelSource, "weight_module", lambda self: mapping_module)
    monkeypatch.setattr(weight_loading, "ignored_patterns", lambda config: ())
    model = nn.Module()
    shape = (4, 16) if axis.kind == "row" else (16, 4)
    model.weight = nn.Parameter(torch.empty(shape, dtype=torch.bfloat16, device="meta"))
    model.tensor_axes = lambda: {"weight": axis}
    shard_parameter_shapes(model, tp_size=2, side="trainer")
    allocate_tensors(model, device=torch.device("cpu"))
    weight = model.weight
    incoming = torch.arange(64, dtype=torch.float32).reshape(shape)
    before = incoming.clone()

    load_weights(model, [("hf.weight", incoming)],
                 source=ModelSource("isoexec.models.qwen3_5"), hf_config=PretrainedConfig(), tp_size=2, tp_rank=rank,
                 side="trainer")

    expected = incoming[:, indices] if axis.kind == "row" else incoming[indices]
    assert torch.equal(weight, expected.bfloat16()) and model.weight is weight
    assert torch.equal(incoming, before)
    if axis.kind != "replicated":
        with pytest.raises(ValueError, match="global shape"):
            load_weights(model, [("hf.weight", torch.zeros_like(weight))],
                         source=ModelSource("isoexec.models.qwen3_5"), hf_config=PretrainedConfig(),
                         tp_size=2, tp_rank=rank, side="trainer")
        assert torch.equal(weight, expected.bfloat16())


def _geometry():
    """Qwen3.5-0.8B head ratios at small widths: 8 query heads, 2 KV heads, GDN heads divisible by 8."""
    root = nn.Module()
    root.attn = inn.Attention(hidden=8, heads=8, kv_heads=2, head_dim=4, rope_theta=10000)
    root.gdn = inn.GDN(hidden=8, key_heads=8, value_heads=8, key_head_dim=2, value_head_dim=2, conv_kernel=4)
    return root


def _numbered(shape):
    return torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)


def _expected_kv_rows(side, tp_size, rank):
    """Trainers split the 8 KV rows over TP; engines hold one whole 4-row head once TP exceeds 2 heads."""
    if side == "trainer" or tp_size <= 2:
        width = 8 // tp_size
        return list(range(rank * width, (rank + 1) * width))
    head = rank // (tp_size // 2)
    return list(range(4 * head, 4 * head + 4))


@pytest.mark.parametrize("side", ["trainer", "engine"])
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_kv_heads_and_gdn_segments_follow_the_side_policy(side, tp_size):
    source = _geometry()
    target = _geometry()
    shard_parameter_shapes(target, tp_size=tp_size, side=side)
    set_local_dims(target, tp_size=tp_size)
    # Below the KV head granule each rank consumes one whole KV head on both sides.
    assert (target.attn.heads_local, target.attn.kv_heads_local) == (8 // tp_size, max(1, 2 // tp_size))
    assert (target.gdn.key_heads_local, target.gdn.value_heads_local) == (8 // tp_size, 8 // tp_size)
    kv_rows = len(_expected_kv_rows(side, tp_size, 0))
    assert tuple(target.attn.k.shape) == tuple(target.attn.v.shape) == (kv_rows, 8)
    assert tuple(target.attn.q.shape) == (32 // tp_size, 8)
    assert tuple(target.gdn.conv_w.shape) == (48 // tp_size, 1, 4)
    for rank in range(tp_size):
        for name in ("attn.k", "attn.v"):
            full = _numbered(source.get_parameter(name).shape)
            local = slice_weight(target, name, full, tp_size=tp_size, tp_rank=rank, side=side)
            assert torch.equal(local, full[_expected_kv_rows(side, tp_size, rank)])
            assert global_shape(target, name, local, tp_size=tp_size, side=side) == tuple(full.shape)
        conv = _numbered(source.gdn.conv_w.shape)
        width = 16 // tp_size
        rows = [offset + rank * width + i for offset in (0, 16, 32) for i in range(width)]
        local = slice_weight(target, "gdn.conv_w", conv, tp_size=tp_size, tp_rank=rank, side=side)
        assert torch.equal(local, conv[rows])
        assert global_shape(target, "gdn.conv_w", local, tp_size=tp_size, side=side) == tuple(conv.shape)
    assert all(parameter.is_meta for parameter in source.parameters())
    if tp_size == 1:
        assert slice_weight(target, "attn.k", full, tp_size=1, tp_rank=0, side=side) is full


def _gather_rank(rank, rendezvous):
    from datetime import timedelta

    from torch import distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=30))
    try:
        group = dist.group.WORLD
        full = _numbered((16, 4))
        cases = [
            (col(), list(range(4 * rank, 4 * rank + 4))),
            (col(stride=2), [2 * rank, 2 * rank + 1, 8 + 2 * rank, 9 + 2 * rank]),
            (col(segments=((4, None), (12, None))), [rank] + list(range(4 + 3 * rank, 7 + 3 * rank))),
            (row(), None),
            (replicated(), None),
        ]
        for axis, coordinates in cases:
            model = nn.Module()
            model.weight = nn.Parameter(torch.empty(full.shape, device="meta"))
            model.tensor_axes = lambda axis=axis: {"weight": axis}
            local = full[:, rank:rank + 1] if axis.kind == "row" else full if coordinates is None else full[coordinates]
            local = local.clone()
            assert torch.equal(slice_weight(model, "weight", full, tp_size=4, tp_rank=rank, side="trainer"), local)
            before = local.clone()
            gathered = gather_weight(model, "weight", local, tp_group=group, side="trainer")
            assert torch.equal(gathered, full) and torch.equal(local, before)
        for side in ("trainer", "engine"):
            target = _geometry()
            shard_parameter_shapes(target, tp_size=4, side=side)
            kv = _numbered((8, 8))
            local = kv[_expected_kv_rows(side, 4, rank)].clone()
            assert torch.equal(gather_weight(target, "attn.k", local, tp_group=group, side=side), kv)
    finally:
        dist.destroy_process_group()


def test_four_rank_gather_reconstructs_independent_global_coordinates(tmp_path):
    torch.multiprocessing.spawn(_gather_rank, args=((tmp_path / "gloo").as_uri(),), nprocs=4, join=True)
