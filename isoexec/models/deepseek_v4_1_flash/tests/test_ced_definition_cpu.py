"""CPU contract of the CED definition: config validation, stacks, layer kinds, handoff, state, admission.

Configs: the released ``config.json``; B14 (``b14_bf16_config.json``: the B14 BF16 checkpoint's config with
``isoexec_subset.source`` removed; the checkpoint's own file is checked too when present); the older 8-row debug
config; mutated copies that are not a Causal Encoder-Decoder.
"""
import copy
import json
import os
import re
from pathlib import Path

import pytest

from isoexec.compiler.realization.weights import _conversion_mapping, _find_mapping, package_ignored_patterns
from isoexec.compiler.source import ModelSource
from isoexec.models.deepseek_v4_1_flash import admission, weight
from isoexec.models.deepseek_v4_1_flash.block import DecoderBlock, DecoderEntry, EncoderBlock
from isoexec.models.deepseek_v4_1_flash.config import (
    BOUNDED_REPLAY,
    FULL_DECODER,
    CedError,
    Csa2Mode,
    ced_geometry,
    program_identity,
)
from isoexec.models.deepseek_v4_1_flash.model import DeepseekV41ForCausalLM
from isoexec.models.deepseek_v4_1_flash.program import DecoderSegments, Turn, ced_states, engine_decoder_rows
from isoexec.models.deepseek_v4_1_flash.reference import checkpoint_shapes, synthetic_config
from isoexec.ops.attention.deepseek_v41_cache_sm100 import spec
from isoexec.schema import Axis, Placement

SOURCE = ModelSource("isoexec.models.deepseek_v4_1_flash")
FIXTURES = Path(__file__).parent / "fixtures"
B14_CHECKPOINT = os.environ.get("ISOEXEC_V41_B14_CHECKPOINT")


def raw(name):
    return json.loads((FIXTURES / name).read_text())


# ---------------------------------------------------------------------------------------------- config
PRODUCTION_DECODER = {20: ("full", 20, "select"), 24: ("reindex", 24, "mask"), 28: ("reindex", 28, "mask"),
                      32: ("reindex", 32, "mask"), 36: ("reindex", 36, "mask")}


def test_production_config_is_the_reports_ced():
    g = ced_geometry(raw("config.json"))
    assert (g.boundary, len(g.encoder), len(g.decoder), g.replay_window, g.storage) == (20, 20, 20, 128, "fp8_fp4")
    modes = [k.mode for k in g.encoder]
    assert modes[:2] == [Csa2Mode.SWA] * 2
    assert [k.layer for k in g.encoder if k.mode is Csa2Mode.FULL] == [2, 8, 14]
    assert g.encoder_groups() == (tuple(range(2, 8)), tuple(range(8, 14)), tuple(range(14, 20)))
    assert all(k.ratio == 2 and k.kv_source == k.index_source and k.candidate == "none" for k in g.encoder[2:])
    assert [k.layer for k in g.encoder if k.engram is not None] == [1, 14]
    assert all(k.stack == "decoder" and k.ratio == 1 and k.kv_source == 20 and k.engram is None for k in g.decoder)
    for k in g.decoder:
        mode, source, cand = PRODUCTION_DECODER.get(k.layer, ("reuse", 20 + (k.layer - 20) // 4 * 4, "none"))
        assert (k.mode.value, k.index_source, k.candidate) == (mode, source, cand), k
        assert k.window_clamp == (k.layer > 20) and k.is_entry == (k.layer == 20)
    assert g.admissible_pipeline_cuts() == (1, 2, 8, 14, 20)
    assert set(g.excluded) == {"vision", "dspark"}


@pytest.mark.parametrize("source", ["fixture", "checkpoint"])
def test_b14_config_keeps_encoder_decoder_and_handoff(source):
    if source == "checkpoint":
        if not B14_CHECKPOINT or not (Path(B14_CHECKPOINT) / "config.json").exists():
            pytest.skip("B14 checkpoint not on this host")
        config = json.loads((Path(B14_CHECKPOINT) / "config.json").read_text())
    else:
        config = raw("b14_bf16_config.json")
    g = ced_geometry(config)
    assert (g.boundary, len(g.encoder), len(g.decoder), g.storage, g.layout) == (8, 8, 6, "bf16", "physical")
    assert [k.checkpoint_layer for k in g.kinds] == [0, 1, 2, 3, 8, 9, 14, 19, 20, 21, 24, 25, 28, 29]
    assert [k.mode.value for k in g.decoder] == ["full", "reuse", "reindex", "reuse", "reindex", "reuse"]
    assert [k.index_source for k in g.decoder] == [8, 8, 10, 10, 12, 12]
    assert [(k.layer, k.engram) for k in g.encoder if k.engram is not None] == [(1, 0), (6, 1)]
    assert g.encoder_groups() == ((2, 3), (4, 5), (6, 7))
    assert g.admissible_pipeline_cuts() == (1, 2, 4, 6, 8)


def _mutated(name, edit):
    config = copy.deepcopy(raw(name))
    edit(config["text_config"], config)
    return config


MUTATIONS = {
    # an encoder Reindex layer (index source without its own keys before the boundary)
    "CED_ENCODER_REINDEX": ("config.json", lambda t, c: t.update(index_source_layer_ids=[2, 5, 8, 14, 20, 24, 28,
                                                                                         32, 36])),
    # Engram in the decoder
    "CED_ENGRAM_IN_DECODER": ("config.json", lambda t, c: t.update(engram_layer_ids=[1, 24])),
    # two decoder KV sources
    "CED_DECODER_KV_SOURCES": ("config.json", lambda t, c: t.update(kv_source_layer_ids=[2, 8, 14, 20, 30],
                                                                    index_source_layer_ids=[2, 8, 14, 20, 24, 28,
                                                                                            30, 32, 36],
                                                                    candidate_source_layer_id=30)),
    # the boundary moved off L/2 at production depth
    "CED_BOUNDARY_NOT_HALF_DEPTH": ("config.json", lambda t, c: t.update(
        compress_ratios=[0, 0] + [2] * 20 + [1] * 18 + [0, 0, 0], kv_source_layer_ids=[2, 8, 14, 22],
        index_source_layer_ids=[2, 8, 14, 22, 24, 28, 32, 36], candidate_source_layer_id=22)),
    # the candidate source is not the boundary
    "CED_BOUNDARY_NOT_CANDIDATE_SOURCE": ("config.json", lambda t, c: t.update(candidate_source_layer_id=24)),
    # a decoder layer without the decoder global KV (sliding window only)
    "CED_DECODER_RATIO": ("config.json", lambda t, c: t["compress_ratios"].__setitem__(39, 0)),
    # a sliding-window-only layer inside the encoder's compressed layers
    "CED_ENCODER_SWA_PREFIX": ("config.json", lambda t, c: t["compress_ratios"].__setitem__(5, 0)),
    # a subset whose boundary row is not production layer 20
    "CED_SUBSET_BOUNDARY": ("b14_bf16_config.json", lambda t, c: c["isoexec_subset"].update(
        layer_sources=[0, 1, 2, 3, 8, 9, 14, 19, 21, 22, 24, 25, 28, 29])),
}


@pytest.mark.parametrize("code", sorted(MUTATIONS))
def test_non_ced_layouts_are_refused_by_name(code):
    name, edit = MUTATIONS[code]
    with pytest.raises(CedError) as caught:
        ced_geometry(_mutated(name, edit))
    assert caught.value.code == code


def test_dspark_and_vision_fields_are_recorded_not_built():
    g = ced_geometry(raw("config.json"))
    assert len(g.compress_ratios) == 40 and g.num_nextn_predict_layers == 3
    assert "dspark" in g.excluded and "vision" in g.excluded
    patterns = weight.ignored_checkpoint_patterns(raw("config.json"))
    for key in ("mtp.0.attn.wq_a.weight", "vision.blocks.0.attn.wo.bias", "image_start",
                "encoder.blocks.3.ffn.gate.bias_vl"):
        assert any(re.search(p, key) for p in patterns), key


# ---------------------------------------------------------------------------------------------- structure
@pytest.fixture(scope="module")
def production():
    return SOURCE.build_meta(raw("config.json"))


@pytest.fixture(scope="module")
def b14():
    return SOURCE.build_meta(raw("b14_bf16_config.json"))


@pytest.mark.parametrize("which", ["production", "b14"])
def test_stacks_layer_kinds_and_handoff_match_the_config(which, request):
    model = request.getfixturevalue(which)
    cfg = model.cfg
    b = cfg.boundary
    assert [blk.index for blk in model.encoder.blocks] == list(range(b))
    assert [blk.index for blk in model.decoder.blocks] == list(range(b, cfg.num_hidden_layers))
    assert all(type(blk) is EncoderBlock for blk in model.encoder.blocks)
    assert all(type(blk) is DecoderBlock for blk in model.decoder.blocks)
    assert type(model.decoder.entry) is DecoderEntry and model.decoder.entry.index == b
    for blk in model.layers():
        k = cfg.kind(blk.index)
        assert blk.kind == k and blk.facts()["mode"] == k.mode.value and blk.facts()["stack"] == k.stack
        assert (blk.engram is not None) == (k.engram is not None)
        assert hasattr(blk.attn, "compressor") == (k.mode is Csa2Mode.FULL and not k.is_entry)
        assert hasattr(blk.attn, "indexer") == (k.index_source == blk.index)
    # the handoff: decoder.entry owns the boundary layer's attention boundary and token-wise projection;
    # the boundary layer's block keeps the query side, the sink and the output projection
    entry = {n for n, _ in model.decoder.entry.named_parameters()}
    assert entry == {"hc_attn_fn", "hc_attn_base", "hc_attn_scale", "attn_norm.weight", "attn.wkv.weight",
                     "attn.kv_norm.weight", "attn.compressor.wkv.weight", "attn.compressor.norm.weight",
                     "attn.indexer.wk.weight", "attn.indexer.k_norm.weight"}
    first = {n for n, _ in model.decoder.blocks[0].named_parameters() if not n.startswith("ffn.")}
    assert first == {"hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale", "ffn_norm.weight", "attn.attn_sink",
                     "attn.wq_a.weight", "attn.q_norm.weight", "attn.wq_b.weight", "attn.wo_a.weight",
                     "attn.wo_b.weight", "attn.indexer.wq_b.weight", "attn.indexer.weights_proj.weight"}
    # encoder CSA2 groups are wired FULL -> REUSE; the decoder wires at stack level
    for blk, (consumes, produces) in zip(model.encoder.blocks, model.encoder.state_routes):
        k = blk.kind
        assert consumes == ({Csa2Mode.REUSE: ("kv", "topk")}.get(k.mode, ()))
        assert produces == ({Csa2Mode.FULL: ("kv", "index_codes", "index_sf", "topk")}.get(k.mode, ()))
    assert [type(u) for u in model.pipeline_layers()] == [EncoderBlock] * b + [type(model.decoder)]


def test_module_paths_and_checkpoint_names_are_inverse(production, b14):
    for model in (production, b14):
        cfg = model.cfg
        for name, _ in [*model.named_parameters(), *model.named_buffers()]:
            ckpt = weight.checkpoint_name(name, cfg)
            assert weight.stack_name(ckpt, cfg) == name
            assert not ckpt.startswith(("encoder.", "decoder."))


def test_reference_program_names_exactly_the_definitions_parameters():
    """The CPU reference program reads exactly the checkpoint tensors the Component tree owns, same shapes (the
    reference holds the Engram FP8 rows decoded, without their scales)."""
    model = DeepseekV41ForCausalLM(synthetic_config(), native=False)
    owned = {weight.checkpoint_name(n, model.cfg): tuple(t.shape)
             for n, t in [*model.named_parameters(), *model.named_buffers()] if not n.endswith(".embed.scale")}
    assert owned == {k: v[0] for k, v in checkpoint_shapes(model.cfg).items()}


def _mapped(config, keys):
    renamings, converters = _conversion_mapping(SOURCE, config)
    mapped = {}
    for key in keys:
        names = _find_mapping(key, renamings, converters)[2]
        assert len(names) == 1
        mapped.setdefault(names[0], []).append(key)
    return mapped, package_ignored_patterns(SOURCE, config)


def test_bf16_physical_checkpoint_keys_land_on_every_parameter(b14):
    """B14 (BF16, rows stored as ``layers.{row}``): one key per parameter, no decode; checkpoint names from the
    model's own inverse map plus the excluded keys a real checkpoint carries."""
    config = SOURCE.build_config(raw("b14_bf16_config.json"))
    registered = {**dict(b14.named_parameters()), **dict(b14.named_buffers())}
    keys = [weight.checkpoint_name(n, b14.cfg) for n in registered] + ["layers.3.ffn.gate.bias_vl", "mtp.0.norm.weight"]
    if B14_CHECKPOINT and (Path(B14_CHECKPOINT) / "model.safetensors.index.json").exists():
        index = json.loads((Path(B14_CHECKPOINT) / "model.safetensors.index.json").read_text())["weight_map"]
        keys = sorted(set(keys) | set(index))
    mapped, ignored = _mapped(config, keys)
    assert {n for n in mapped if n in registered} == set(registered)
    assert all(any(re.search(p, n) for p in ignored) for n in mapped.keys() - registered.keys())
    assert all(len(v) == 1 for n, v in mapped.items() if n in registered)


# ---------------------------------------------------------------------------------------------- state
def test_state_classes_by_stack_with_the_cache_packages_record_widths():
    cfg = ced_geometry(raw("config.json"))
    states = ced_states(cfg)
    by_owner = {}
    for st in states:
        by_owner.setdefault(st.owner, {})[st.name] = st
    for st in states:
        if st.name in ("swa_kv", "global_kv"):
            assert st.record_bytes == spec.KV_RECORD_BYTES
        if st.name == "index_k":
            assert st.record_bytes == spec.INDEX_RECORD_BYTES
    # encoder: window per layer, global KV + index keys + the pending ratio-2 group on FULL layers
    assert all(by_owner[f"encoder.blocks.{i}"]["swa_kv"].lifetime == "request" for i in range(20))
    assert sorted(i for i in range(20) if "global_kv" in by_owner[f"encoder.blocks.{i}"]) == [2, 8, 14]
    assert by_owner["encoder.blocks.8"]["global_kv"].tokens_per_entry == 2
    assert by_owner["encoder.blocks.8"]["compressor_state"].lifetime == "group"
    # decoder: ONE global KV (the entry's, every token), the entry's window keys, segment-scoped decoder windows
    entry = by_owner["decoder.entry"]
    assert set(entry) == {"global_kv", "index_k", "swa_kv"} and entry["global_kv"].tokens_per_entry == 1
    assert all(st.lifetime == "request" for st in entry.values())
    for j in range(1, 20):
        assert set(by_owner[f"decoder.blocks.{j}"]) == {"swa_kv"}
        assert by_owner[f"decoder.blocks.{j}"]["swa_kv"].lifetime == "segment"
    assert "decoder.blocks.0" not in by_owner
    assert by_owner["model"]["segment_start"].lifetime == "segment"
    assert all(st.axis in ("token", "request") for st in states)       # nothing assumes one rank per sequence
    # what the components declare to the runtime is the same table
    model = SOURCE.build_meta(raw("b14_bf16_config.json"))
    assert {s.name for s in model.decoder.entry.state_specs()} == {"global_kv", "index_k", "swa_kv"}
    assert {s.name for s in model.decoder.blocks[1].state_specs()} == {"swa_kv"}
    assert model.decoder.blocks[0].state_specs() == ()
    # the portable contract carries the lifetimes: segment-scoped decoder windows, request-scoped the rest
    from isoexec.state.validation import validate_state_spec
    for st in states:
        portable = st.portable()
        validate_state_spec(portable)
        assert (portable.scope == "segment") == (st.lifetime == "segment" and st.name == "swa_kv")


# ---------------------------------------------------------------------------------------------- admission
def _placement(tp=1, ep=1, pp=1, cp=1, side="trainer"):
    group = object()
    one = Axis(1, 0, None)
    return Placement(tp=Axis(tp, 0, group), ep=Axis(ep, 0, group if ep > 1 else None), dp=one,
                     pp=Axis(pp, 0, group) if pp > 1 else one, etp=Axis(tp, 0, group) if ep == 1 else one,
                     cp=Axis(cp, 0, group) if cp > 1 else one, side=side, leaves=8)


def test_admission_names_every_refusal_and_admits_the_ced_cuts():
    cfg = ced_geometry(raw("config.json"))
    for tp in (1, 2, 4, 8):
        admission.admit(cfg, _placement(tp=tp, side="engine"), "m")
    admission.admit(cfg, _placement(tp=8, ep=8, side="engine"), "m")
    admission.admit(cfg, _placement(tp=1, ep=8, side="engine"), "m")
    admission.admit(cfg, _placement(tp=1, ep=8, side="trainer"), "m")       # the recipe's trainer layout
    for starts in ((0, 20), (0, 2, 8, 14, 20), (0, 1), (0, 14)):
        admission.admit(cfg, _placement(tp=8, pp=len(starts), side="engine"), "m", stage_starts=starts)
    cases = [
        (admission.TP_TREE, dict(placement=_placement(tp=3))),
        (admission.EP_GROUP, dict(placement=_placement(tp=2, ep=4))),
        (admission.EP_EXPERT_SPLIT, dict(placement=_placement(ep=5, side="engine"))),
        (admission.CONTEXT_PARALLEL, dict(placement=_placement(cp=2))),
        (admission.PP_STAGES_UNDECLARED, dict(placement=_placement(pp=2))),
        (admission.PP_CUT_INSIDE_DECODER, dict(placement=_placement(pp=2), stage_starts=(0, 24))),
        (admission.PP_CUT_SPLITS_GROUP, dict(placement=_placement(pp=2), stage_starts=(0, 5))),
        (admission.PROGRAM_UNKNOWN, dict(placement=_placement(), program="vllm_chunk_replay")),
    ]
    for code, kw in cases:
        p = kw.pop("placement")
        with pytest.raises(admission.Refusal) as caught:
            admission.admit(cfg, p, "m", **kw)
        assert caught.value.code == code


def test_program_identity_names_the_ced_mode():
    cfg = ced_geometry(raw("config.json"))
    replay, exact = program_identity(cfg, BOUNDED_REPLAY), program_identity(cfg, FULL_DECODER)
    assert replay != exact and replay["ced_prefill"] == "bounded_replay" and replay["replay_window"] == 128
    assert replay["cache_records"] == {"kv": spec.KV_RECORD_BYTES, "index_key": spec.INDEX_RECORD_BYTES}
    assert replay["index_query_format"] == spec.INDEX_QUERY_FORMAT and replay["moe"]
    model = SOURCE.build_meta(raw("b14_bf16_config.json"))
    assert model.ced_prefill == BOUNDED_REPLAY == model.program_identity()["ced_prefill"]
    assert all(m.ced_prefill == BOUNDED_REPLAY for m in model.members())


# ---------------------------------------------------------------------------------------------- segments
@pytest.mark.parametrize("prompt", [1, 127, 128, 129, 1000])
def test_replay_rule_depends_only_on_the_prompt_length(prompt):
    W = 128
    want = range(max(0, prompt - W), prompt)
    for chunks in ([prompt], [1] * min(prompt, 300) + ([prompt - 300] if prompt > 300 else []),
                   [64] * (prompt // 64) + ([prompt % 64] if prompt % 64 else [])):
        rows, a = [], 0
        for n in chunks:
            rows += list(engine_decoder_rows(prompt, a, a + n, W, BOUNDED_REPLAY))
            a += n
        assert rows == list(want)
    segs = DecoderSegments.replay([0, prompt + 5], [[Turn(prompt, prompt + 5)]], W)
    assert segs.positions.tolist() == list(range(max(0, prompt - W), prompt + 5))
    assert set(segs.seg_start.tolist()) == {max(0, prompt - W)}
    assert segs.positions[segs.score].tolist() == list(range(prompt - 1, prompt + 4))
    exact = DecoderSegments.all_rows([0, prompt + 5], [[Turn(prompt, prompt + 5)]])
    assert exact.positions.tolist() == list(range(prompt + 5)) and set(exact.seg_start.tolist()) == {0}


def test_decoder_gpu_composition_waits_for_the_attention_entry_points():
    import importlib.util

    from isoexec.models.deepseek_v4_1_flash.block import DECODER_GPU_PENDING, ced_ops
    if importlib.util.find_spec("isoexec.ops.attention.deepseek_v41_cache_sm100.ced") is not None:
        pytest.skip("the attention package's CED entry points exist")
    with pytest.raises(NotImplementedError, match=re.escape(DECODER_GPU_PENDING)):
        ced_ops()


def test_capture_calls_the_ced_components_with_per_token_shapes():
    """The captured program: embed, the encoder blocks, decoder.entry (the handoff + global KV), the decoder
    blocks, the head stage and the head, each with per-token abstract shapes (the decoder rows are computed
    inside the decoder components and written back), and the decoder as one pipeline unit."""
    import torch

    from isoexec.compiler.capture.capture import capture
    model = SOURCE.build_meta(raw("debug_config.json"))
    cfg, T = model.cfg, 16
    graph = capture(model, (torch.zeros(1, T, dtype=torch.long),))
    calls = [n for n in graph.program.graph.nodes if n.op == "call_module" and not n.meta.get("isoexec_guard")]
    names = ["embed", *(f"encoder.blocks.{i}" for i in range(cfg.boundary)), "decoder.entry",
             *(f"decoder.blocks.{j}" for j in range(len(cfg.decoder))), "norm", "head"]
    assert [str(n.target) for n in calls] == names
    assert [layer.path for layer in graph.layers] == names[1:cfg.boundary + 1] + ["decoder"]
    S, H = cfg.hc_mult, cfg.hidden_size
    entry = next(n for n in calls if str(n.target) == "decoder.entry")
    assert [tuple(v.shape) for v in entry.meta["val"]] == [(1, T, S, H), (1, T, S), (1, T, H), (1, T, S), (1, T, S, S),
                                                           (T, 512), (128, 128), (128,), (T, 512)]
    block = next(n for n in calls if str(n.target) == "decoder.blocks.1")
    assert [tuple(v.shape) for v in block.meta["val"]] == [(1, T, S, H), (1, T, S), (1, T, H), (1, T, S), (1, T, S, S)]


def test_vllm_partitions_map_to_stage_starts():
    """``VLLM_PP_LAYER_PARTITION`` counts backbone layers per stage, as for native DeepSeek V4.1."""
    cfg = ced_geometry(raw("config.json"))
    assert admission.stage_starts_from_partition(cfg, 2, "20,20") == (0, 20)       # the CED cut
    assert admission.stage_starts_from_partition(cfg, 2, None) == (0, 20)          # native's default: the CED cut
    assert admission.stage_starts_from_partition(cfg, 2, "11,29") == (0, 11)
    with pytest.raises(admission.Refusal):
        admission.admit_pipeline(cfg, admission.stage_starts_from_partition(cfg, 2, "11,29"))
    b14 = ced_geometry(raw("b14_bf16_config.json"))
    assert admission.stage_starts_from_partition(b14, 2, "8,6") == (0, 8)
    with pytest.raises(admission.Refusal):                                          # 7 / 7: inside a group
        admission.admit_pipeline(b14, admission.stage_starts_from_partition(b14, 2, None))
    with pytest.raises(admission.Refusal, match="PP_STAGES_UNDECLARED"):
        admission.stage_starts_from_partition(b14, 2, "8,1")


@pytest.mark.parametrize("edit,refuser", [
    (lambda t: t.update(hidden_size=256, o_lora_rank=256), "residual.deepseek_v41.mhc"),   # the DeepSeek mHC width
    (lambda t: t.update(moe_intermediate_size=96), "MegaMoE"),                           # BF16 MegaMoE tiles
    (lambda t: t.update(q_lora_rank=1024), "q_lora_rank"),                               # the attention composition
])
def test_native_admission_asks_every_op_package(edit, refuser):
    """``ced_geometry`` refuses a geometry any SM100 op package does not serve, naming that package's own rule
    (the tiny smoke checkpoint at hidden 256 passed the attention check alone and failed in the mHC kernel)."""
    config = copy.deepcopy(raw("config.json"))
    edit(config["text_config"])
    with pytest.raises(ValueError, match=refuser):
        ced_geometry(config)
    ced_geometry(config, native=False)                  # the CPU reference program has no kernel limits


def test_engine_capture_hands_the_head_one_row_per_token():
    """The engine passes packed ids ``[T]``; vLLM picks its sampled rows from the body output's leading axis, so the
    head's input must be ``[T, H]`` (a ``[1, T, H]`` body output made vLLM gather ``[R, T, H]`` and the head run on
    R x T rows: smoke2, defect 5)."""
    import torch

    from isoexec.compiler.capture.capture import capture
    model = SOURCE.build_meta(raw("debug_config.json"))
    T = 16
    graph = capture(model, (torch.zeros(T, dtype=torch.long),))
    calls = {str(n.target): n for n in graph.program.graph.nodes if n.op == "call_module"}
    # every sequence-context component receives the root seq (the embed's Engram hash needs it: smoke5)
    for name in ("embed", "encoder.blocks.0", "decoder.entry", "decoder.blocks.0"):
        node = calls[name]
        assert "seq" in node.kwargs or any("seq" in str(a) for a in node.args), (name, node.args, node.kwargs)
    head_in = calls["head"].args[0].meta["val"]
    assert tuple(head_in.shape) == (T, model.cfg.hidden_size)
    assert tuple(calls["head"].meta["val"].shape) == (T, model.cfg.vocab_size)
