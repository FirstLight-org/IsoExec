"""CPU contract of the DeepSeek V4.1 Flash package: construction, capture, checkpoint keys, decode, placement.

The released checkpoint's tensor index (names, dtypes, shapes from the safetensors headers; routed experts
collapsed to one entry per layer and projection) is a fixture, as are the released config and the config of
a reduced-depth debug checkpoint (8 released rows, 256 routed experts of 128 channels).
"""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isoexec.compiler.realization.weights import _conversion_mapping, _find_mapping, package_ignored_patterns
from isoexec.compiler.source import ModelSource
from isoexec.core.checkpoint_quant import quantize_checkpoint_weight
from isoexec.models.deepseek_v4_1_flash.config import FORWARD_PROGRAM
from isoexec.models.deepseek_v4_1_flash.weight import DecodeBlocks
from isoexec.nn.inspect import inventory
from isoexec.schema import Axis, Placement

SOURCE = ModelSource("isoexec.models.deepseek_v4_1_flash")
FIXTURES = Path(__file__).parent / "fixtures"
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn, "I8": torch.int8,
          "F8_E8M0": torch.float8_e8m0fnu}


def raw(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(scope="module")
def debug_model():
    return SOURCE.build_meta(raw("debug_config.json"))


def checkpoint_index():
    """``{key: (dtype, shape)}`` of the released checkpoint."""
    data = raw("checkpoint_index.json")
    out = {key: (dtype, tuple(shape)) for key, (dtype, shape) in data["tensors"].items()}
    for template, item in data["experts"].items():
        out.update({template.format(e=e): (item["dtype"], tuple(item["shape"])) for e in range(item["count"])})
    return out


def placement(tp=1, tp_rank=0, ep=1, ep_rank=0, side="trainer"):
    group = object()      # a process-group handle; binding on meta never communicates
    one = Axis(1, 0, None)
    return Placement(tp=Axis(tp, tp_rank, group), ep=Axis(ep, ep_rank, group if ep > 1 else None), dp=one, pp=one,
                     etp=Axis(tp, tp_rank, group) if ep == 1 else one, cp=one, side=side, leaves=8)


@pytest.mark.parametrize("key,value", [("hc_mult", 2), ("scoring_func", "softmax"), ("head_dim", 256)])
def test_unsupported_geometry_refused(key, value):
    config = raw("debug_config.json")
    config["text_config"][key] = value
    with pytest.raises(ValueError):
        SOURCE.build_meta(config)


def _mapped(config, keys):
    renamings, converters = _conversion_mapping(SOURCE, config)
    mapped = {}
    for key in keys:
        names = _find_mapping(key, renamings, converters)[2]
        assert len(names) == 1
        mapped.setdefault(names[0], []).append(key)
    return mapped, package_ignored_patterns(SOURCE, config)


@pytest.mark.parametrize("name", ["config.json", "debug_config.json"])
def test_checkpoint_key_coverage(name, request):
    model = request.getfixturevalue("debug_model") if name == "debug_config.json" else SOURCE.build_meta(raw(name))
    config = SOURCE.build_config(raw(name))
    index = checkpoint_index()
    mapped, ignored = _mapped(config, index)
    registered = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    received = {n: keys for n, keys in mapped.items() if n in registered}
    assert received.keys() == registered.keys()
    assert all(any(re.search(p, n) for p in ignored) for n in mapped.keys() - registered.keys())
    for target, keys in received.items():
        tensor = registered[target]
        stored = {k.rsplit(".", 1)[1]: index[k] for k in keys}
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu) or len(keys) == 1:
            (dtype, shape), = stored.values()
            assert DTYPES[dtype] == tensor.dtype, target
            if not model.cfg.subset or ".ffn.gate." not in target:
                assert shape == tuple(tensor.shape), target
            continue
        # one decoded master from its codes and UE8M0 scales
        assert set(stored) == {"weight", "scale"} and stored["scale"][0] == "F8_E8M0", target
        dtype, shape = stored["weight"]
        n, k = tensor.shape
        assert tensor.dtype == torch.bfloat16
        if dtype == "I8":      # packed E2M1, 1 x 32 blocks
            assert (shape[0] >= n and shape[1] >= k // 2) and ".ffn.experts." in target
        else:                  # E4M3, 32 x 32 blocks
            assert dtype == "F8_E4M3" and shape[0] >= n and shape[1] >= k
        if not model.cfg.subset:
            assert shape == ((n, k // 2) if dtype == "I8" else (n, k))


def _reference_decode(codes, scale, storage):
    """Independent decode: E4M3 / E2M1 code points times 2^(exponent - 127) per block, in float64."""
    if storage == "deepseek_fp4":
        table = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        b = codes.view(torch.uint8).to(torch.int64)
        nib = torch.stack((b & 15, b >> 4), -1).flatten(-2)
        values = torch.tensor([table[c & 7] * (-1.0 if c & 8 else 1.0) for c in nib.flatten().tolist()],
                              dtype=torch.float64).view(nib.shape)
        rows, cols = 1, 32
    else:
        values = codes.to(torch.float64)
        rows, cols = 32, 32
    factor = torch.pow(2.0, scale.view(torch.uint8).to(torch.float64) - 127)
    factor = factor.repeat_interleave(rows, 0).repeat_interleave(cols, 1)
    return (values * factor[:values.shape[0], :values.shape[1]]).to(torch.bfloat16)


@pytest.mark.parametrize("storage,n,k,tn,tk", [
    ("deepseek_fp8", 64, 96, 64, 96), ("deepseek_fp8", 40, 72, 40, 72), ("deepseek_fp4", 6, 128, 6, 128),
    ("deepseek_fp8", 64, 96, 32, 64), ("deepseek_fp4", 6, 128, 4, 64)])
def test_decode_converter_is_exact(storage, n, k, tn, tk):
    g = torch.Generator().manual_seed(n * k + tn)
    codes, scale = quantize_checkpoint_weight(torch.randn(n, k, generator=g) * 40, storage)
    # spread the block exponents over a wide range
    scale = (scale.view(torch.uint8).to(torch.int16) + torch.randint(-9, 9, scale.shape, generator=g)).to(
        torch.uint8).view(torch.float8_e8m0fnu)
    target = torch.empty(tn, tk, dtype=torch.bfloat16, device="meta")
    patterns = [r"\.w\.weight$", r"\.w\.scale$"]

    def convert(reduced):
        return DecodeBlocks(reduced=reduced).convert(
            {patterns[0]: [codes], patterns[1]: [scale]}, patterns, [".w.weight"], full_layer_name="x.w.weight",
            model=SimpleNamespace(get_parameter=lambda name: target))[".w.weight"]

    reduced = (tn, tk) != (n, k)
    if reduced:
        with pytest.raises(ValueError, match="checkpoint shape"):
            convert(False)
    out = convert(reduced)
    ref = _reference_decode(codes, scale, storage)[:tn, :tk]
    assert out.dtype == torch.bfloat16 and torch.equal(out.view(torch.int16), ref.view(torch.int16))
    assert out.abs().max() > 0


def test_placement_binding_on_meta(debug_model):
    import copy
    cfg = debug_model.cfg
    E, H, D, G, R = cfg.n_routed_experts, cfg.num_attention_heads, cfg.head_dim, cfg.o_groups, cfg.o_lora_rank
    from isoexec.compiler.realization.tp import shard_parameter_shapes
    from isoexec.models.deepseek_v4_1_flash.providers import _engine_ep_moe, _engine_local_moe, _ep_moe, _local_moe
    seams = {("trainer", False): _local_moe, ("trainer", True): _ep_moe, ("engine", False): _engine_local_moe,
             ("engine", True): _engine_ep_moe}
    for tp, ep, side in ((1, 1, "trainer"), (2, 1, "trainer"), (2, 2, "trainer"), (1, 1, "engine"), (2, 2, "engine"),
                         (1, 2, "engine")):
        for rank in range(max(tp, ep)):
            model = copy.deepcopy(debug_model)
            p = placement(tp, rank % tp, ep, rank % ep, side)
            for i, block in enumerate(model.layers()):
                block.apply_placement(p, site=f"layers.{i}")
                assert block.moe_forward is seams[side, ep > 1]
                assert block.facts()["program"] == FORWARD_PROGRAM
                assert all(lin.grouped_deepgemm for lin in block.linear.values())
                assert (block.engine_expert_exchange() is block.moe_forward) == (ep > 1)
                assert getattr(block.moe_forward, "graph_capturable", False) == (ep > 1 and side == "engine")
                assert block.tp.size == tp and block.ep.size == ep and block.ep.partitioned == (ep > 1 and tp == 1)
                lo = (rank % ep) * E // ep
                assert block.ffn.held_experts() == tuple(range(lo, lo + E // ep)) == block.ep.local
                assert set(block.linear) == {"attention", "indexer", "engram"}
            model.decoder.entry.apply_placement(p, site="decoder.entry")
            shard_parameter_shapes(model, tp_size=tp, side=side, etp_size=1 if ep > 1 else tp)
            attn = model.layers()[2].attn
            assert tuple(attn.wq_b.weight.shape) == (H * D // tp, cfg.q_lora_rank)
            assert tuple(attn.attn_sink.shape) == (H // tp,)
            assert tuple(attn.wo_a.weight.shape) == (G * R // tp, H * D // G)
            assert tuple(attn.wo_b.weight.shape) == (cfg.hidden_size, G * R // tp)
            assert tuple(attn.indexer.wq_b.weight.shape) == (cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)
            if side == "trainer":
                with pytest.raises(Exception, match="engine-placed"):
                    model.layers()[0].bind_engine_state(vllm_config=None, layer_prefix="model.layers.0")
    model = copy.deepcopy(debug_model)
    with pytest.raises(Exception, match="TP = 3"):
        model.layers()[0].apply_placement(placement(3, 0), site="layers.0")
    # token-disjoint EP training (the recipe's trainer layout, GPU pending): the router and the shared expert are
    # summed over the EP group inside ep_moe_train, so Megatron reduces them as expert tensors, not again over EP
    from isoexec.models.deepseek_v4_1_flash.providers import _ep_moe
    disjoint = copy.deepcopy(debug_model).layers()[0]
    disjoint.apply_placement(placement(1, 0, 2, 1), site="layers.0")
    assert disjoint.ep.partitioned and disjoint.moe_forward is _ep_moe
    marked = {n for n, p in disjoint.named_parameters() if getattr(p, "allreduce", True) is False}
    assert marked == {n for n, _ in disjoint.named_parameters() if n.startswith(("ffn.gate.", "ffn.shared_experts."))}
    # pipeline stages need declared, admissible starts
    one = Axis(1, 0, None)
    staged = Placement(tp=one, ep=one, dp=one, pp=Axis(2, 0, object()), etp=one, cp=one, side="trainer", leaves=8)
    with pytest.raises(Exception, match="PP_STAGES_UNDECLARED"):
        copy.deepcopy(debug_model).layers()[0].apply_placement(staged, site="layers.0")
    staged_model = copy.deepcopy(debug_model)
    staged_model.declare_pipeline_stages((0, staged_model.cfg.boundary))      # the CED cut
    staged_model.layers()[0].apply_placement(staged, site="layers.0")


def _synthetic(dtype, shape, g):
    if dtype in ("BF16", "F32"):
        return torch.randn(shape, generator=g).to(DTYPES[dtype])
    if dtype == "F8_E8M0":
        return torch.randint(118, 136, shape, generator=g, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    codes = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
    if dtype == "F8_E4M3":
        codes = torch.where((codes & 0x7F) == 0x7F, codes ^ 1, codes)        # no NaN codes
    return codes.view(DTYPES[dtype])


def test_loader_decodes_renames_and_skips_on_a_reduced_row(debug_model):
    import copy
    from torch import nn

    from isoexec.compiler.realization.storage import allocate_tensors
    from isoexec.compiler.realization.weights import load_weights

    row, source, ep = 3, 3, 128                       # row 3 reads checkpoint layer 3; two local experts
    target = nn.Module()                              # the CED module path of an encoder row
    target.encoder = nn.Module()
    target.encoder.blocks = nn.ModuleDict({str(row): copy.deepcopy(debug_model.layers()[row])})
    block = target.encoder.blocks[str(row)]
    held_before = {n for n, _ in target.named_parameters()}
    block.narrow_held(placement(1, 0, ep, 0), site=f"encoder.blocks.{row}")
    remote = held_before - {n for n, _ in target.named_parameters()}
    allocate_tensors(target, device=torch.device("cpu"))
    g = torch.Generator().manual_seed(0)
    index = checkpoint_index()
    keys = [k for k in index if k.startswith(f"layers.{source}.") and
            not re.search(r"\.experts\.([2-9]|\d\d+)\.", k)] + [f"layers.{source}.ffn.experts.{e}.w1.{leaf}"
                                                           for e in (17, 300) for leaf in ("weight", "scale")]
    keys += ["layers.13.attn_norm.weight", "mtp.0.attn.wq_a.weight", "vision.blocks.0.attn.wo.bias"]
    weights = {k: _synthetic(*index[k], g) for k in keys}
    config = SOURCE.build_config(raw("debug_config.json"))
    loaded = load_weights(target, list(weights.items()), source=SOURCE, hf_config=config, remote=remote)
    assert loaded == {n for n, _ in target.named_parameters()} | {n for n, _ in target.named_buffers()}
    I, E = debug_model.cfg.moe_intermediate_size, debug_model.cfg.n_routed_experts
    expert = block.ffn.experts["1"].w2.weight
    codes, scale = (weights[f"layers.{source}.ffn.experts.1.w2.{leaf}"] for leaf in ("weight", "scale"))
    want = _reference_decode(codes[:, :I // 2], scale[:, :I // 32], "deepseek_fp4")
    assert torch.equal(expert.view(torch.int16), want.view(torch.int16))
    wq_a = _reference_decode(*(weights[f"layers.{source}.attn.wq_a.{leaf}"] for leaf in ("weight", "scale")),
                             "deepseek_fp8")
    assert torch.equal(block.attn.wq_a.weight.view(torch.int16), wq_a.view(torch.int16))
    assert torch.equal(block.ffn.gate.weight, weights[f"layers.{source}.ffn.gate.weight"][:E])
    with pytest.raises(ValueError, match="sources incomplete"):
        load_weights(target, [(f"layers.{source}.attn.wq_a.weight", weights[f"layers.{source}.attn.wq_a.weight"])],
                     source=SOURCE, hf_config=config)


def test_engine_steps_switch_static_ep_routes_without_preparing(debug_model, monkeypatch):
    """Static EP is prepared once (bind time); each engine forward only switches the routes: graph decode
    buckets static, eager / profiling steps host-counted. No step prepares or communicates."""
    import copy
    from types import SimpleNamespace
    model = copy.deepcopy(debug_model)
    block = model.layers()[0]
    block.apply_placement(placement(2, 0, 2, 0, "engine"), site="layers.0")
    comm = SimpleNamespace(static_routes=False)
    monkeypatch.setattr(block.ep, "_comm", comm)
    block.ep.static_home_rows = 32                       # what bind_engine_state prepared
    monkeypatch.setattr(type(block), "prepare_static_ep",
                        lambda self, rows: pytest.fail("a step re-prepared static EP"))
    from isoexec.models.deepseek_v4_1_flash import engine
    monkeypatch.setattr(engine, "ROUTE_STATS", {"static": {}, "captured": {}, "counted": 0})
    decode = lambda capturing=False: SimpleNamespace(graph=True, tokens=8, capturing=capturing)  # noqa: E731
    for step, static in ((decode(), True), (SimpleNamespace(graph=False, tokens=40, capturing=False), False),
                         (None, False), (decode(True), True)):
        assert block.select_ep_routes(step) is True and comm.static_routes is static
    assert engine.ROUTE_STATS == {"static": {8: 1}, "captured": {8: 1}, "counted": 2}
    block.ep.static_home_rows = None                     # graphs off: never static
    assert block.select_ep_routes(decode()) is False and comm.static_routes is True


def test_source_parses_with_the_package_config_class_whatever_autoconfig_maps(monkeypatch):
    """A runtime may register its own class for ``deepseek_v41`` (vLLM flattens the text tower); the package
    still parses its configuration with its own class."""
    from transformers import PretrainedConfig
    from transformers.models.auto import configuration_auto
    from isoexec.models.deepseek_v4_1_flash.config import DeepseekV41Config

    class Flattened(PretrainedConfig):
        model_type = "deepseek_v41"
    monkeypatch.setitem(configuration_auto.CONFIG_MAPPING._extra_content, "deepseek_v41", Flattened)
    config = SOURCE.build_config(raw("debug_config.json"))
    assert type(config) is DeepseekV41Config and config.isoexec_subset["layer_sources"][4] == 14
    assert SOURCE.build_meta(raw("debug_config.json")).cfg.num_hidden_layers == 8


def test_engine_load_hook_prepares_and_refresh_refuses_moved_parameters(debug_model, monkeypatch):
    """``prepare_engine_execution`` packs the experts, prepares the MegaMoE workspace and records the addresses
    graphs read in place; a weight update that moved one is refused at the commit, an in-place one commits."""
    import copy
    from isoexec.models.deepseek_v4_1_flash import engine
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops as attn_ops
    model = copy.deepcopy(debug_model)
    block = model.layers()[0]
    with pytest.raises(Exception, match="engine-placed"):
        block.prepare_engine_execution()
    block.apply_placement(placement(2, 0, 2, 0, "engine"), site="layers.0")
    calls = []
    monkeypatch.setattr(type(block), "packed_experts", lambda self: calls.append("pack"))
    monkeypatch.setattr(type(block), "prepare_expert_workspace", lambda self: calls.append("workspace"))
    addresses = [(("attn.wq_b.weight", 4096, (8, 8), torch.bfloat16),)]
    monkeypatch.setattr(type(block), "_read_addresses", lambda self: addresses[-1])
    assert block.moved_parameters() == []                  # nothing recorded yet
    block.prepare_engine_execution()
    assert calls == ["pack", "workspace"] and block.moved_parameters() == []
    monkeypatch.setattr(attn_ops, "refresh_derived_weights", lambda module: 5)
    monkeypatch.setattr(engine, "RECEIPTS", {"eager": {}, "captured": {}, "replays": {}, "weights": 0})
    block.engine_state = SimpleNamespace(swa=None, comp=None, index=None, state=None)
    assert block.refresh_derived_weights() == 5 and engine.RECEIPTS["weights"] == 1
    addresses.append((("attn.wq_b.weight", 8192, (8, 8), torch.bfloat16),))
    with pytest.raises(RuntimeError, match=r"moved \['attn.wq_b.weight'\]"):
        block.refresh_derived_weights()
    assert engine.RECEIPTS["weights"] == 1


def test_engine_packed_experts_are_recorded_once_and_follow_every_weight_change(debug_model, monkeypatch):
    """A prepared engine row returns the packed expert aliases it recorded (no per-forward walk): an in-place update
    is read through them, a rebound anchor master re-derives, a moved non-anchor master is refused at the commit
    (which re-records), and the trainer never caches."""
    import copy
    from isoexec.models.deepseek_v4_1_flash import engine
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops as attn_ops

    def expected(block):
        experts = [block.ffn.experts[str(e)] for e in block.ep.local] + [block.ffn.shared_experts]
        gu = torch.stack([p for e in experts for p in (e.w1.weight, e.w3.weight)])
        return gu.view(len(experts), -1, gu.shape[-1]), torch.stack([e.w2.weight for e in experts])

    block = copy.deepcopy(debug_model).layers()[0]
    block.apply_placement(placement(1, 0, 1, 0, "engine"), site="layers.0")
    block.to_empty(device="cpu")
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in block.parameters():
            p.copy_(torch.randn(p.shape, generator=gen).to(p.dtype))
    monkeypatch.setattr(type(block), "pack_shared_input", lambda self: 0)
    monkeypatch.setattr(type(block), "prepare_expert_workspace", lambda self: False)
    walks = []
    real_stack = __import__("isoexec.ops.moe.clamped_swiglu_sm100.packing", fromlist=["Stack"]).Stack.apply
    monkeypatch.setattr(__import__("isoexec.ops.moe.clamped_swiglu_sm100.packing", fromlist=["Stack"]).Stack,
                        "apply", staticmethod(lambda *a: walks.append(1) or real_stack(*a)))
    assert block._packed is None
    block.prepare_engine_execution()
    assert block._packed is not None
    with torch.no_grad():
        walks.clear()
        gu, dn = block.packed_experts()
        assert walks == [] and gu is block._packed[0] and dn is block._packed[1]
        e_gu, e_dn = expected(block)
        assert torch.equal(gu, e_gu) and torch.equal(dn, e_dn)
        middle = block.ffn.experts[str(block.ep.local[len(block.ep.local) // 2])]
        middle.w3.weight.add_(1.0)                       # an in-place update (IPC / NCCL / file push)
        gu, dn = block.packed_experts()
        assert walks == [] and torch.equal(gu, expected(block)[0])
        first = block.ffn.experts[str(block.ep.local[0])].w1.weight
        first.data = first.data.clone() + 2.0            # an anchor rebound outside a commit: re-derived
        gu, dn = block.packed_experts()
        assert walks and torch.equal(gu, expected(block)[0]) and torch.equal(dn, expected(block)[1])
        walks.clear()
        gu2, _ = block.packed_experts()
        assert walks == [] and torch.equal(gu2, expected(block)[0])
    monkeypatch.setattr(attn_ops, "refresh_derived_weights", lambda module: 0)
    monkeypatch.setattr(engine, "RECEIPTS", {"eager": {}, "captured": {}, "replays": {}, "weights": 0})
    block.engine_state = None
    block.prepare_engine_execution()                     # re-record the addresses after the rebinding above
    middle.w2.weight.data = middle.w2.weight.data.clone()  # a moved non-anchor master reaches the commit
    with pytest.raises(RuntimeError, match="moved"):
        block.refresh_derived_weights()
    assert block._packed is None                         # the commit dropped the aliases before refusing
    trainer = copy.deepcopy(debug_model).layers()[0]
    trainer.apply_placement(placement(1, 0, 1, 0, "trainer"), site="layers.0")
    trainer.to_empty(device="cpu")
    trainer.packed_experts()
    assert trainer._packed is None


def test_engine_rows_declare_the_tree_reducer_scope_once_prepared(debug_model, monkeypatch):
    import copy
    from isoexec.models.deepseek_v4_1_flash import block as block_module
    reducer = SimpleNamespace(prepared=False)            # one reducer per TP group (providers.tp_reducer)
    monkeypatch.setattr(block_module, "tp_reducer", lambda group: reducer)
    model = copy.deepcopy(debug_model)
    a, b = model.layers()[0], model.layers()[1]
    for i, block in enumerate((a, b)):
        block.apply_placement(placement(2, 0, 2, 0, "engine"), site=f"layers.{i}")
    assert a.engine_execution_scopes() == []                # transport not prepared (no device)
    reducer.prepared = True
    assert a.engine_execution_scopes() == [reducer] == b.engine_execution_scopes()
    one = copy.deepcopy(debug_model).layers()[0]
    one.apply_placement(placement(1, 0, 1, 0, "engine"), site="layers.0")
    trainer = copy.deepcopy(debug_model).layers()[0]
    trainer.apply_placement(placement(2, 0, 2, 0, "trainer"), site="layers.0")
    assert one.engine_execution_scopes() == [] == trainer.engine_execution_scopes()


def test_program_receipts_of_forwards_captures_and_replays(monkeypatch):
    from isoexec.models.deepseek_v4_1_flash import engine
    monkeypatch.setattr(engine, "RECEIPTS", {"eager": {}, "captured": {}, "replays": {}, "weights": 0})

    def step(tokens, capturing, *notes):
        s = SimpleNamespace(tokens=tokens, capturing=capturing, program={})
        for key, value in notes:
            engine.EngineStep.note(s, key, value)
        return s
    rows = [("reducer", "tree"), ("experts", "mega"), ("routes", "counted"), ("reducer", "tree")]
    engine.record_program(step(37, False, *rows))
    engine.record_program(step(8, True, ("reducer", "tree"), ("experts", "mega"), ("routes", "static")))
    engine.record_program(step(5, False, ("reducer", "tree"), ("reducer", "all_gather")))
    engine.record_replay(8)
    engine.commit_weights()
    engine.record_replay(8)
    engine.record_replay(8)
    assert engine.RECEIPTS == {
        "eager": {"experts=mega,reducer=tree,routes=counted": 1, "reducer=mixed": 1},
        "captured": {"8": {"program": "experts=mega,reducer=tree,routes=static", "weights": 0}},
        "replays": {"8": {"0": 1, "1": 2}}, "weights": 1}


def test_host_engram_tables_take_the_runtime_mapping_or_register_their_own(monkeypatch):
    """The engine state takes the device views of the runtime's node-shared host copy and registers nothing
    itself (fail closed when a shared table has none); a table this process holds alone (the trainer's) is
    registered here."""
    from isoexec.models.deepseek_v4_1_flash import engine_state
    from isoexec.models.deepseek_v4_1_flash.holders import EngramTable
    table = EngramTable(4, 32, "cpu")
    rows = [SimpleNamespace(engram=None), SimpleNamespace(engram=SimpleNamespace(embed=table))]
    monkeypatch.setattr(engine_state, "_rows", lambda model: rows)
    registered = []
    monkeypatch.setattr(engine_state, "register_host_table", lambda t, device: registered.append(t) or (1 / 0))
    with pytest.raises(ZeroDivisionError):
        engine_state.map_host_tables(object(), "cpu")       # a trainer's own copy: registered here
    assert registered == [table]
    table.__dict__["_host_shared"] = {"weight": object()}
    with pytest.raises(RuntimeError, match="no device mapping"):
        engine_state.map_host_tables(object(), "cpu")
    assert registered == [table]
    table.device_view = ("w", "s")                       # mapped already: left as is
    assert engine_state.map_host_tables(object(), "cpu") == 0


def test_a_trainer_host_table_is_registered_and_released_in_pieces(monkeypatch):
    """A trainer's own host Engram table is registered in ``REGISTER_CHUNK`` pieces (one device range per
    buffer) and released piece by piece, explicitly (``release_trainer_state``) or by its finalizer: a
    registration left to the context teardown is dropped in one driver critical section at process exit and
    stalls every other driver client (first-4: ``nvidia-smi`` beyond 10 s after the trainer stage)."""
    import gc
    import sys
    import types

    from isoexec.models.deepseek_v4_1_flash import engine_state
    from isoexec.models.deepseek_v4_1_flash.holders import EngramTable
    from isoexec.models.deepseek_v4_1_flash.training import release_trainer_state
    from isoexec.runtimes.vllm import host_tables
    calls, unregistered = [], []
    rt = types.SimpleNamespace(
        cudaHostRegisterMapped=1, cudaHostRegisterPortable=2, cudaError_t=types.SimpleNamespace(cudaSuccess=0),
        cudaHostRegister=lambda base, n, flags: (calls.append((base, n, flags)), (0,))[1],
        cudaHostGetDevicePointer=lambda base, flags: (0, base + (1 << 44)),
        cudaHostUnregister=lambda base: unregistered.append(base))
    bindings = types.ModuleType("cuda.bindings")
    bindings.runtime = rt
    monkeypatch.setitem(sys.modules, "cuda", types.ModuleType("cuda"))
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", rt)
    monkeypatch.setattr(host_tables, "REGISTER_CHUNK", 8192)
    monkeypatch.setattr(torch, "as_tensor", lambda iface, device: torch.zeros(iface.__cuda_array_interface__["shape"],
                                                                             dtype=torch.uint8))

    def aligned(like):
        n = like.numel() * like.element_size()
        raw = torch.empty(n + 4096, dtype=torch.uint8)
        off = (-raw.data_ptr()) % 4096
        return raw[off:off + n].view(like.dtype).view(like.shape)

    def table():
        t = EngramTable(100, 256, "meta")
        t.weight, t.scale = aligned(torch.empty((100, 256), dtype=torch.float8_e4m3fn)), \
            aligned(torch.empty((100, 8), dtype=torch.float8_e8m0fnu))
        return t
    t = table()
    views = engine_state.register_host_table(t, "cpu")
    w, s = t.weight.data_ptr(), t.scale.data_ptr()
    assert [(b, n) for b, n, _ in calls] == [(w, 8192), (w + 8192, 8192), (w + 16384, 8192), (w + 24576, 1024),
                                             (s, 800)]
    assert max(n for _, n, _ in calls) <= host_tables.REGISTER_CHUNK and all(f == 3 for *_, f in calls)
    assert views[0].shape == t.weight.shape and views[1].shape == t.scale.shape
    t.device_view = views
    rows = [SimpleNamespace(engram=None), SimpleNamespace(engram=SimpleNamespace(embed=t))]
    monkeypatch.setattr(engine_state, "_rows", lambda model: rows)
    assert release_trainer_state(object()) == {"host_tables": 1}
    assert unregistered == [b for b, _, _ in calls] and t.device_view is None
    assert release_trainer_state(object()) == {"host_tables": 0} and len(unregistered) == 5     # idempotent
    # the finalizer releases a table that was never released explicitly
    calls.clear(), unregistered.clear()
    t2 = table()
    engine_state.register_host_table(t2, "cpu")
    del t2
    gc.collect()
    assert unregistered == [b for b, _, _ in calls] and len(calls) == 5
    # a failure part-way drops every piece registered so far
    calls.clear(), unregistered.clear()
    rt.cudaHostGetDevicePointer = lambda base, flags: (0, base + (1 << 12 if len(calls) == 3 else 0))
    with pytest.raises(RuntimeError, match="contiguous"):
        engine_state.register_host_table(table(), "cpu")
    assert unregistered == [b for b, _, _ in calls] and len(calls) == 3


def test_engine_placeholders_persist_per_row_and_padded_token_count(debug_model):
    """Produced-state placeholders of engine forwards are allocated once per row and padded token count (graph
    sizes kept, one eager slot) and satisfy the captured boundary: exact shapes and strides, offset 0, no alias
    of the row's inputs (another row's placeholders)."""
    from isoexec.compiler.capture.export_normalize import _checked_output
    producers = [b for b in debug_model.layers() if b.produces()]
    a, b = producers[0], producers[1]
    cfg = debug_model.cfg
    x = lambda T: torch.empty((1, T, cfg.hc_mult, cfg.hidden_size), dtype=torch.bfloat16)  # noqa: E731
    graph, eager = SimpleNamespace(graph=True), SimpleNamespace(graph=False)
    for block in (a, b):
        block.__dict__.pop("_placeholder_cache", None)
    g8 = a._placeholders(x(8), graph)
    assert a._placeholders(x(8), graph) is g8 and a._placeholders(x(8), eager) is g8
    e40 = a._placeholders(x(40), eager)
    assert a._placeholders(x(40), eager) is e40
    e41 = a._placeholders(x(41), eager)                       # the eager slot moves; graph sizes stay
    assert set(a.__dict__["_placeholder_cache"]) == {((1, 8), x(8).device), ((1, 41), x(41).device)}
    assert a._placeholders(x(40), eager) is not e40 and a._placeholders(x(8), graph) is g8
    want = a.abstract_output(x(41), torch.empty((1, 41, cfg.hc_mult)))[5:]
    other = b._placeholders(x(41), eager)
    is_alias = torch._C._is_alias_of
    for got, ref in zip(a._placeholders(x(41), eager), want, strict=True):
        assert _checked_output(got, ref.dtype, "row", ref.shape, ref.stride(), ref.device, other) is got
        assert got.storage_offset() == 0 and all(not is_alias(got, o) for o in other)
    assert len(e41) == len(a.produces())
    consumer = next(blk for blk in debug_model.layers() if not blk.produces())
    assert consumer._placeholders(x(8), graph) == ()



def test_derived_weight_guard_compares_versions_without_a_walk_and_probes_contents(debug_model, monkeypatch):
    from torch import nn
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops as attn_ops
    block = debug_model.layers()[0]
    assert block.derived_weight_guard() is block.derived_weight_guard()         # one guard per row
    assert block.stale_derived_weights() == []
    rows = [nn.Linear(4, 3), nn.Linear(4, 3)]
    norm = nn.Parameter(torch.ones(3))
    rows[1].register_parameter("norm", norm)
    attn_ops._bf16_weight(rows[0], "w0", rows[0].weight, False)
    attn_ops._cache(rows[1]).get("cat", (rows[1].weight, rows[1].bias),
                                 lambda: torch.cat([rows[1].weight.flatten(), rows[1].bias]).bfloat16())
    attn_ops.f32_param(norm)
    guard = attn_ops.DerivedWeightGuard.combine([attn_ops.DerivedWeightGuard([r], [f"layer {i}"])
                                                 for i, r in enumerate(rows)])
    walks = []
    entries = attn_ops.derived_weight_entries
    monkeypatch.setattr(attn_ops, "derived_weight_entries", lambda m: walks.append(m) or entries(m))
    assert guard.stale() == [] and len(guard) == 4 and walks == []
    with torch.no_grad():
        rows[1].bias.add_(1)                                  # a write that bypassed the refresh
        norm.mul_(2)
    assert guard.stale() == ["layer 1: cat", f"layer 1: f32:{id(norm)}"]
    assert attn_ops.refresh_derived_weights(rows[1]) == 2
    assert guard.stale() == [] and len(walks) == 1            # re-read after the refresh, no tree walk
    with torch.no_grad():
        rows[0].weight.data.add_(1)                           # no version counter records it
    assert guard.stale() == []
    flags = {}
    for _ in range(3):
        label, flag = guard.probe_content()
        flags[label] = int(flag)
    assert flags == {"layer 0: w0": 1, "layer 1: cat": 0, f"layer 1: f32:{id(norm)}": 0}


def test_step_guard_reads_its_packet_a_step_later_and_fails_closed():
    """The per-step packet (per-table max block id, sampled min, cross-rank digests) is read once it arrived,
    never waited for."""
    from isoexec.models.deepseek_v4_1_flash.engine_state import StepGuard

    class Event:
        def __init__(self, done):
            self.done = done

        def query(self):
            return self.done

        def synchronize(self):
            pass

    guard = StepGuard(1, None, limit=100)
    guard._world = 2
    guard._host = torch.tensor([3, 99, 0, 7, 7, 0], dtype=torch.int64)
    guard.pending = (Event(False), 2, True, True)
    assert guard.resolve() is False and guard.pending is not None        # in flight: no wait
    guard.pending = (Event(True), 2, True, True)
    assert guard.resolve() is True and guard.pending is None
    for packet, tables, low, digest, match in [([5, 100], 2, False, False, "outside"),
                                               ([5, 9, -1], 2, True, False, "negative"),
                                               ([5, 9, 7, 8], 2, False, True, "disagree")]:
        guard._host = torch.tensor(packet + [0] * 4, dtype=torch.int64)
        guard.pending = (Event(True), tables, low, digest)
        with pytest.raises(RuntimeError, match=match):
            guard.resolve()
    guard._host = torch.tensor([5, 9, -1, 8], dtype=torch.int64)
    guard.pending = (Event(True), 2, False, False)                        # no min, no digest this step
    assert guard.resolve() is True


def test_step_guard_never_skips_its_tp_collective_on_a_rank_local_in_flight_packet():
    """Defect 12: at TP > 1 a step whose previous packet is still in flight WAITS for it and issues the digest
    all-gather like every other rank (skipping is rank-local timing and mis-pairs the TP collectives); at TP = 1
    the step is skipped without waiting, as before."""
    from isoexec.models.deepseek_v4_1_flash.engine_state import StepGuard

    calls = {"sync": 0}

    class Event:
        def query(self):
            return False                  # still in flight on this rank

        def synchronize(self):
            calls["sync"] += 1

    guard = StepGuard(1, object(), limit=None)
    guard._world, guard._host = 2, torch.zeros(8, dtype=torch.int64)
    guard.pending = (Event(), 1, False, False)
    assert guard.resolve(wait=guard.group is not None and guard.interval > 0) is True and calls["sync"] == 1
    solo = StepGuard(1, None, limit=None)
    solo._host = torch.zeros(8, dtype=torch.int64)
    solo.pending = (Event(), 1, False, False)
    assert solo.resolve(wait=solo.group is not None and solo.interval > 0) is False and calls["sync"] == 1
    import inspect
    src = inspect.getsource(StepGuard.step)
    assert "self.resolve(wait=self.group is not None" in src


def test_cache_page_limit_is_the_largest_cache_page_count():
    from isoexec.models.deepseek_v4_1_flash.engine_state import cache_page_limit
    context = {"a": SimpleNamespace(kv_cache=torch.empty(5, 2)), "b": SimpleNamespace(kv_cache=[torch.empty(9, 3)]),
               "c": SimpleNamespace(kv_cache=torch.tensor([]))}
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=context))
    groups = [[SimpleNamespace(layer_names=["a", "c"])], [SimpleNamespace(layer_names=["b"])]]
    assert cache_page_limit(cfg, groups) == 9 and cache_page_limit(cfg, []) is None


def test_small_fp32_root_gemm_pads_into_a_persistent_buffer_with_zero_tail(monkeypatch):
    """The decode head's padded A: the same values as concatenating zeros, rows a longer call wrote cleared."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops as attn_ops
    x8, x3 = torch.arange(8 * 4, dtype=torch.float32).view(8, 4) + 1, torch.ones(3, 4)
    assert torch.equal(attn_ops._padded_rows(x8, 16), attn_ops._pad_rows(x8, 16))      # CPU: the plain form

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=7))
    monkeypatch.setattr(attn_ops, "_PADDED", {})

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))      # the device path, on CPU

    def padded(x):
        return attn_ops._padded_rows(x, 16)
    first = padded(x8)
    assert torch.equal(first, attn_ops._pad_rows(x8, 16))
    second = padded(x3)
    assert second.data_ptr() == first.data_ptr() and torch.equal(second, attn_ops._pad_rows(x3, 16))


def test_engine_rows_pack_shared_inputs_through_the_attention_ops(monkeypatch):
    """Attention requests M-D2 / M-D3: ``pack_shared_input`` packs ``wq_a | wkv (| indexer weights)`` with
    ``ops.pack_shared_input_rows`` and ``wq_b | indexer wq_b`` with ``ops.pack_qr_rows`` for this row's layer id,
    counting the groups the ops packed; host (or trainer) weights are left alone."""
    from isoexec.models.deepseek_v4_1_flash.block import V41Layer
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops as v41_ops
    calls = []
    monkeypatch.setattr(v41_ops, "pack_shared_input_rows", lambda m, cfg, i: calls.append(("x", i)) or torch.empty(1))
    monkeypatch.setattr(v41_ops, "pack_qr_rows", lambda m, cfg, i: calls.append(("qr", i)) or (None if i != 2 else torch.empty(1)))
    cuda = SimpleNamespace(is_cuda=True)
    row = SimpleNamespace(attn=SimpleNamespace(wq_a=SimpleNamespace(weight=cuda), wq_b=SimpleNamespace(weight=cuda),
                                               wkv=SimpleNamespace(weight=cuda)), cfg=object(), index=2)
    assert V41Layer.pack_shared_input(row) == 2 and calls == [("x", 2), ("qr", 2)]
    row.index, calls[:] = 1, []
    assert V41Layer.pack_shared_input(row) == 1 and calls == [("x", 1), ("qr", 1)]
    host = SimpleNamespace(is_cuda=False)
    row.attn = SimpleNamespace(wq_a=SimpleNamespace(weight=host), wq_b=SimpleNamespace(weight=host))
    calls[:] = []
    assert V41Layer.pack_shared_input(row) == 0 and calls == []
    # the decoder's FULL layer has no wkv (decoder.entry owns its projection): only the qr packing is asked
    row.attn, row.index, calls[:] = SimpleNamespace(wq_a=SimpleNamespace(weight=cuda), wq_b=SimpleNamespace(weight=cuda)), 2, []
    assert V41Layer.pack_shared_input(row) == 1 and calls == [("qr", 2)]


def test_the_expert_workspace_predicate_is_agreed_by_the_ep_group(monkeypatch):
    """A rank-local predicate in front of the EP group's collective setup (``gate_up.is_cuda``) is all-gathered:
    when any rank says no, every rank keeps the fixed-order exchange (a rank skipping the setup alone would hang
    its peers in the rendezvous)."""
    import torch.distributed as dist

    from isoexec.models.deepseek_v4_1_flash import block
    group = SimpleNamespace()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda g=None: 2)
    for peer, want in ((True, True), (False, False)):
        def gather(out, mine, group=None, peer=peer):
            out[:] = [mine, peer]
        monkeypatch.setattr(dist, "all_gather_object", gather)
        assert block._group_agrees(group, True) is want
    assert block._group_agrees(None, False) is False
