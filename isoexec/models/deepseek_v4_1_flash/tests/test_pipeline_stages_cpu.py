"""Engine pipeline stages of the CED definition (CPU): the admitted stage layout is the one vLLM builds, each
stage registers its rows' caches and builds its paged step from its own layers only, and the values crossing a
cut are the single-stage program's (defect 13: an even split that vLLM placed inside a CSA2 group was admitted
by a re-derived partition and failed at cache registration on the second stage).

Native vLLM (``vllm/models/deepseek_v41/attention.py``, ``DeepseekV4Attention.__init__``) refuses a PP split
inside a KV-sharing group (``NotImplementedError``); nothing of a source layer's caches or per-step top-k is
carried across a stage there. The CED definition does the same: only cuts where the residual carry alone
crosses are admitted (``CedGeometry.admissible_pipeline_cuts``), every other one is refused by name.
"""
import hashlib
import json
import math

import numpy as np
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import fx

pytest.importorskip("vllm")

from isoexec.compiler.source import ModelSource  # noqa: E402
from isoexec.models.deepseek_v4_1_flash import admission, engine  # noqa: E402
from isoexec.models.deepseek_v4_1_flash.config import text_geometry  # noqa: E402
from isoexec.schema import Axis, Placement  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE = ModelSource("isoexec.models.deepseek_v4_1_flash")
CONFIGS = ["debug_config.json", "b14_bf16_config.json", "config.json"]


def raw(name):
    return json.loads((FIXTURES / name).read_text())


def vllm_ranges(count, pp):
    from vllm.distributed.utils import get_pp_indices
    return [get_pp_indices(count, rank, pp) for rank in range(pp)]


def stage_layers(cfg, ranges, rank):
    """Backbone layers of stage ``rank`` of half-open LAYER ranges (the decoder is never split: admitted layouts
    start every stage at or before the boundary)."""
    lo, hi = ranges[rank]
    return list(range(lo, cfg.num_hidden_layers if hi > cfg.boundary else hi))


def cut_ranges(cfg, cut):
    return ((0, cut), (cut, cfg.num_hidden_layers))


@pytest.fixture
def no_partition(monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    return monkeypatch


def test_default_split_is_vllms(no_partition):
    """``even_partition`` is ``get_pp_indices``' rule (the remainder goes to the stages before the last)."""
    for pp in range(1, 9):
        for count in range(pp, 46):
            counts = [hi - lo for lo, hi in vllm_ranges(count, pp)]
            assert admission.even_partition(count, pp) == counts, (count, pp)


def test_the_native_default_is_the_ced_cut_at_production_depth(no_partition):
    """Native DeepSeek V4.1 lays out its 40 backbone layers with get_pp_indices: 20 / 20 at PP2, the CED cut. The
    pipeline units (20 encoder blocks + the decoder = 21) split evenly would be 11 / 10, inside a group: the model's
    stage split is native's layer split mapped onto the units, so the default needs no environment variable."""
    model = SOURCE.build_meta(raw("config.json"))
    cfg = model.cfg
    units = len(model.pipeline_layers())
    assert units == 21 and [hi - lo for lo, hi in vllm_ranges(units, 2)] == [11, 10]
    assert vllm_ranges(cfg.num_hidden_layers, 2) == [(0, 20), (20, 40)] and cfg.boundary == 20
    assert model.pipeline_assignments(units, 2) == ((0, 20), (20, 21))
    assert admission.stage_starts_from_partition(cfg, 2) == (0, cfg.boundary)
    # PP4 by default (10 / 10 / 10 / 10) cuts inside an encoder group (10) and the decoder (30): refused by name
    with pytest.raises(admission.Refusal, match="PP_CUT_SPLITS_GROUP"):
        model.pipeline_assignments(units, 4)


@pytest.mark.parametrize("name", CONFIGS)
def test_admitted_stages_are_the_stages_vllm_builds(name, no_partition):
    model = SOURCE.build_meta(raw(name))
    cfg = model.cfg
    n, units = cfg.num_hidden_layers, len(model.pipeline_layers())
    partitions = admission.admissible_partitions(cfg, 2)
    assert f"{cfg.boundary},{n - cfg.boundary}" in partitions
    assert len(partitions) == len(cfg.admissible_pipeline_cuts())
    for partition in partitions:
        no_partition.setenv("VLLM_PP_LAYER_PARTITION", partition)
        layers = admission.vllm_layer_partition(cfg, 2)
        assert tuple(map(tuple, vllm_ranges(n, 2))) == layers == admission.layer_partition(cfg, 2, partition)
        starts = admission.stage_starts_from_partition(cfg, 2, partition)
        assert starts[1] in cfg.admissible_pipeline_cuts()
        assigned = model.pipeline_assignments(units, 2)
        assert assigned[0] == (0, starts[1]) and assigned[1] == (starts[1], units)
    # a cut separating a REUSE layer from its FULL source: refused by name, naming the admissible partitions
    reuse = next(i for i in range(1, cfg.boundary) if cfg.compress_ratios[i] and cfg.source_layer(i, "kv") < i)
    no_partition.setenv("VLLM_PP_LAYER_PARTITION", f"{reuse},{n - reuse}")
    with pytest.raises(admission.Refusal, match="PP_CUT_SPLITS_GROUP") as refused:
        model.pipeline_assignments(units, 2)
    assert f"{cfg.boundary},{n - cfg.boundary}" in str(refused.value)
    # a cut inside the decoder
    no_partition.setenv("VLLM_PP_LAYER_PARTITION", f"{cfg.boundary + 1},{n - cfg.boundary - 1}")
    with pytest.raises(admission.Refusal, match="PP_CUT_INSIDE_DECODER"):
        model.pipeline_assignments(units, 2)


def test_qwen_stage_split_is_unchanged_by_the_author_hook(no_partition):
    """The factory's split: an author without ``pipeline_assignments`` (every Qwen model) takes get_pp_indices over
    its captured layers exactly as before; the V4.1 author takes its hook."""
    from isoexec.runtimes.vllm.factory import stage_assignments
    from isoexec.compiler.realization.tests.test_weights_cpu import qwen_config
    from isoexec.models.qwen3_5.model import Qwen3_5ForConditionalGeneration

    qwen = Qwen3_5ForConditionalGeneration(qwen_config()).eval()
    assert getattr(qwen, "pipeline_assignments", None) is None
    for pp in (1, 2, 4):
        for count in range(pp, 65):
            assert stage_assignments(qwen, count, pp) == tuple(vllm_ranges(count, pp))
    no_partition.setenv("VLLM_PP_LAYER_PARTITION", "3,5")
    assert stage_assignments(qwen, 8, 2) == ((0, 3), (3, 8))
    v41 = SOURCE.build_meta(raw("config.json"))
    no_partition.delenv("VLLM_PP_LAYER_PARTITION")
    assert stage_assignments(v41, 21, 2) == ((0, 20), (20, 21))


def _engine_placement(pp_rank):
    group, one = object(), Axis(1, 0, None)
    return Placement(tp=one, ep=one, dp=one, pp=Axis(2, pp_rank, group), etp=one, cp=one, side="engine", leaves=8)


def test_engine_rows_refuse_a_split_inside_a_group_before_any_registration(no_partition):
    """The rows admit the layout vLLM builds: a partition starting stage 1 at a REUSE layer (debug: layer 3 reads
    layer 2's compressed KV) is refused at placement, by name; the default (4 / 4 here) is admitted."""
    import copy
    model = SOURCE.build_meta(raw("debug_config.json"))
    cfg = model.cfg
    no_partition.setenv("VLLM_PP_LAYER_PARTITION", f"3,{cfg.num_hidden_layers - 3}")
    assert cfg.source_layer(3, "kv") == 2
    with pytest.raises(admission.Refusal, match="PP_CUT_SPLITS_GROUP"):
        copy.deepcopy(model.layers()[3]).apply_placement(_engine_placement(1), site="layers.3")
    no_partition.delenv("VLLM_PP_LAYER_PARTITION")
    copy.deepcopy(model.layers()[3]).apply_placement(_engine_placement(0), site="layers.3")
    copy.deepcopy(model.decoder.entry).apply_placement(_engine_placement(1), site="decoder.entry")


# ------------------------------------------------------------------------------------------ registration
@pytest.fixture(scope="module")
def vllm_config(tmp_path_factory):
    from vllm.engine.arg_utils import EngineArgs
    root = tmp_path_factory.mktemp("ckpt")
    (root / "config.json").write_text(json.dumps(raw("debug_config.json")))
    settings = SOURCE.build_meta(raw("debug_config.json")).engine_settings()
    args = EngineArgs(model=str(root), dtype="bfloat16", kv_cache_dtype=settings["kv_cache_dtype"],
                      block_size=settings["block_size"], enable_chunked_prefill=True, max_model_len=8192,
                      max_num_seqs=64, max_num_batched_tokens=2048, enable_prefix_caching=False, enforce_eager=True,
                      skip_tokenizer_init=True)
    return args.create_engine_config()


@pytest.fixture
def sm100(monkeypatch):
    """The SM100 record formats the layers declare (this test runs without a device)."""
    from vllm.models.deepseek_v41 import attention
    from vllm.platforms import current_platform
    monkeypatch.setattr(attention, "_use_v41_mxfp8_kv_record", lambda: True)
    monkeypatch.setattr(type(current_platform), "is_device_capability_family",
                        lambda self, *a, **k: True, raising=False)


def register(vllm_config, cfg, layers, blocks=40):
    """One rank's registration of ``layers`` (fresh forward context), vLLM's planner and allocator, binding."""
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
    from vllm.v1.worker.utils import allocate_kv_cache

    context = vllm_config.compilation_config.static_forward_context
    context.clear()
    vllm_config.cache_config.kv_cache_layout = None
    with set_current_vllm_config(vllm_config):
        rows = {i: engine.bind_row_caches(cfg, i, vllm_config, f"body.layers.{i}") for i in layers}
        specs = {name: layer.get_kv_cache_spec(vllm_config) for name, layer in context.items()}
        specs = {name: spec for name, spec in specs.items() if spec is not None}
        backends = list({layer.get_attn_backend() for layer in context.values()})
        names = [layout.name for layout in get_supported_kv_cache_layouts(backends)]
        layout = resolve_kv_cache_layout(vllm_config, [names], specs.values())
        builder = DefaultKVCacheConfigBuilder()
        groups = builder.get_kv_cache_groups(vllm_config, specs)
        config = builder.get_kv_cache_config_from_groups(vllm_config, groups, blocks)
        caches = allocate_kv_cache(config, torch.device("cpu"), layout)
    for name, tensor in caches.items():
        context[name].bind_kv_cache(tensor)
    return SimpleNamespace(cfg=cfg, rows=rows, context=dict(context), groups=groups)


def metadata(b, comp, ring, swa_table, starts, seq_lens):
    """vLLM-shaped metadata of one mixed step on rank ``b``; the block tables are logical (the same request
    blocks on every rank), as vLLM's scheduler gives every pipeline stage the same tables."""
    N, R, W = starts[-1], len(seq_lens), b.cfg.sliding_window
    tok = [s for s in range(R) for _ in range(starts[s + 1] - starts[s])]
    pos = [seq_lens[s] - (starts[s + 1] - starts[s]) + j for s in range(R) for j in range(starts[s + 1] - starts[s])]
    md = {}
    for name in b.context:
        if name.endswith(".swa_cache"):
            slots = torch.tensor([swa_table[s][p // 32] * 32 + p % 32 for s, p in zip(tok, pos)], dtype=torch.int64)
            lists = torch.full((N, 1, W), -1, dtype=torch.int32)
            for t, (s, p) in enumerate(zip(tok, pos)):
                for k, q in enumerate(range(max(p - W + 1, 0), p + 1)):
                    lists[t, 0, k] = swa_table[s][q // 32] * 32 + q % 32
            md[name] = SimpleNamespace(
                prefix=name, query_start_loc=torch.tensor(starts, dtype=torch.int32),
                query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
                seq_lens=torch.tensor(seq_lens, dtype=torch.int32), token_to_req_indices=torch.tensor(tok, dtype=torch.int32),
                is_valid_token=slots >= 0, slot_mapping=slots, num_decode_tokens=0, num_prefill_tokens=N,
                decode_swa_indices=lists[:0], prefill_swa_indices=lists)
        elif name.endswith(".compressor.state_cache"):
            md[name] = SimpleNamespace(slot_mapping=torch.tensor([ring[s] * 8 + p % 8 for s, p in zip(tok, pos)]))
        elif name.endswith(".attn"):
            md[name] = SimpleNamespace(block_table=torch.tensor(comp, dtype=torch.int32))
    return md, torch.tensor(pos, dtype=torch.int64)


def step_of(b, md, positions):
    first = min(b.rows)
    step = engine.EngineStep(b.cfg, b.rows[first], md, positions, graph=False)
    for i, row in b.rows.items():
        step.bind_window(i, row, md)
    return step


def logical(step, b, i):
    """Row ``i``'s window rows and lists as (block, slot) pairs of its own layer view (allocation-free)."""
    cache = b.rows[i].swa.kv_cache
    per, off = cache.stride(0) // engine.WINDOW_BYTES, engine.record_rows(cache, engine.WINDOW_BYTES)[1]

    def back(t):
        return torch.where(t >= 0, (t - off) // per * 32 + (t - off) % per, -1)
    return back(step.batch.window_rows[i]), back(step.batch.window_lists[i])


@pytest.mark.parametrize("name", ["debug_config.json", "b14_bf16_config.json"])
def test_each_stage_registers_its_rows_and_steps_like_the_single_stage(vllm_config, sm100, no_partition, name):
    cfg = text_geometry(raw(name))
    n = cfg.num_hidden_layers
    starts, seq_lens = [0, 5, 6, 40], [5, 260, 34]
    comp, ring = [[3, 7, 9], [11, 2, 30], [5, 6, 8]], [13, 14, 15]
    swa = [[(s * 11 + j * 3) % 39 + 1 for j in range(10)] for s in range(3)]
    whole = register(vllm_config, cfg, range(n))
    md, positions = metadata(whole, comp, ring, swa, starts, seq_lens)
    ref = step_of(whole, md, positions)
    for partition in admission.admissible_partitions(cfg, 2):
        no_partition.setenv("VLLM_PP_LAYER_PARTITION", partition)
        ranges = vllm_ranges(n, 2)
        for rank in range(2):
            layers = stage_layers(cfg, ranges, rank)
            b = register(vllm_config, cfg, layers)          # every row finds its source on its own stage
            md, positions = metadata(b, comp, ring, swa, starts, seq_lens)
            step = step_of(b, md, positions)
            sources = {cfg.source_layer(i, "kv") for i in layers if cfg.compress_ratios[i]}
            assert set(step.batch.comp_bt_by_source) == set(step.pools.comp) == sources
            for key in ("tok_seq", "tok_pos", "p0"):
                assert torch.equal(getattr(step.batch, key), getattr(ref.batch, key))
            for r in {cfg.compress_ratios[s] for s in sources}:
                assert torch.equal(step.batch.index_bt[r], ref.batch.index_bt[r])
                assert torch.equal(step.batch.index_read_bt[r], ref.batch.index_read_bt[r])
            if any(cfg.compress_ratios[s] > 1 for s in sources):
                assert torch.equal(step.batch.state_slot, ref.batch.state_slot)
            for i in layers:
                assert all(torch.equal(a, c) for a, c in zip(logical(step, b, i), logical(ref, whole, i)))


def _layouts():
    out = []
    for name in CONFIGS:
        cfg = text_geometry(raw(name))
        for cut in cfg.admissible_pipeline_cuts():
            for rank in range(2):
                out.append((name, cut, rank))
    return out


# Defect 13b: vLLM's allocation of a stage holding only the SWA layer 0 (cut 1) or only the SWA prefix and ONE
# encoder group (cut at the second group start) can end within a page after a layer's last record: no whole 64-record
# page view of it exists at most block counts. ``record_rows`` then gives the row-bounded view (the dual-rail map is
# bounded at its last row). Every other layout, PP1 included, keeps its whole-page view (unchanged).
ROW_BOUNDED = {(name, cut, 0) for name, cut in (("debug_config.json", 1), ("debug_config.json", 4),
                                                ("b14_bf16_config.json", 1), ("b14_bf16_config.json", 4),
                                                ("config.json", 1), ("config.json", 8))}


def _views(b):
    """``{layer: (view, offset)}`` of every window / compressed pool, checked to address exactly its records."""
    out = {}
    for name, layer in b.context.items():
        if not name.endswith((".swa_cache", ".attn")):
            continue
        cache = layer.kv_cache
        view, offset = engine.record_rows(cache, engine.WINDOW_BYTES)
        per = cache.stride(0) // engine.WINDOW_BYTES
        rows = view.numel() // engine.WINDOW_BYTES
        last = offset + (cache.shape[0] - 1) * per + cache.shape[1] - 1           # the layer's last record
        assert view.data_ptr() + offset * engine.WINDOW_BYTES == cache.data_ptr() and last < rows
        storage = cache.untyped_storage()
        assert view.data_ptr() + view.numel() <= storage.data_ptr() + storage.nbytes()
        out[name] = view, offset
    return out


@pytest.mark.parametrize("name,cut,rank", _layouts())
def test_stage_cache_views_address_every_record_at_every_block_count(vllm_config, sm100, name, cut, rank):
    cfg = text_geometry(raw(name))
    layers = stage_layers(cfg, cut_ranges(cfg, cut), rank)
    bounded = 0
    for blocks in range(2, 2 + 64 * 2, 3):          # the view's fit is periodic in the block count (mod 64)
        views = _views(register(vllm_config, cfg, layers, blocks))
        bounded += any(view.dim() == 2 for view, _ in views.values())
    assert (bounded > 0) == ((name, cut, rank) in ROW_BOUNDED)


@pytest.mark.parametrize("name", CONFIGS)
def test_single_stage_views_keep_whole_pages(vllm_config, sm100, name):
    """PP1 never takes the row-bounded form: its views (and so its kernels' maps) are the ones it always had."""
    cfg = text_geometry(raw(name))
    for blocks in range(2, 2 + 64 * 2, 3):
        views = _views(register(vllm_config, cfg, range(cfg.num_hidden_layers), blocks))
        assert all(view.dim() == 3 and view.shape[1] == 64 for view, _ in views.values())


@pytest.mark.parametrize("name", ["debug_config.json", "b14_bf16_config.json"])
def test_a_stage_without_its_rows_source_is_refused_by_name(vllm_config, sm100, no_partition, name):
    """A stage starting at a REUSE layer (admission refuses it first): its first row has no compressed-KV source."""
    cfg = text_geometry(raw(name))
    reuse = next(i for i in range(1, cfg.boundary) if cfg.compress_ratios[i] and cfg.source_layer(i, "kv") < i)
    layers = stage_layers(cfg, cut_ranges(cfg, reuse), 1)
    assert cfg.source_layer(layers[0], "kv") not in layers
    with pytest.raises(admission.Refusal, match="PP_CUT_SPLITS_GROUP"):
        register(vllm_config, cfg, layers)


# ------------------------------------------------------------------------------------------ the carry
def _concrete(shape, tokens):
    return [int(s.node.expr.subs({x: tokens for x in s.node.expr.free_symbols})) if isinstance(s, torch.SymInt)
            else int(s) for s in shape]


def _digest(value, h):
    if isinstance(value, torch.Tensor):
        h.update(repr((value.dtype, tuple(value.shape))).encode())
        h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    elif isinstance(value, (tuple, list)):
        for v in value:
            _digest(v, h)
    else:
        h.update(repr(type(value)).encode())


class StandIns(fx.Interpreter):
    """Runs a stage graph with every component replaced by a function of the BITS of all its tensor inputs
    (a SHA-256 seeded fill of the captured output shapes): equal outputs mean every component received the same
    bits; the graph's own moves (views, the carry's squeeze / unsqueeze, sizes) run as captured."""

    def __init__(self, module, tokens):
        super().__init__(module)
        self.tokens = tokens

    def run_node(self, n):
        self._node = n
        return super().run_node(n)

    def call_module(self, target, args, kwargs):
        if target == "_isoexec_export_input_guard":
            return None
        h = hashlib.sha256(str(target).encode())
        _digest([a for a in (*args, *kwargs.values()) if isinstance(a, (torch.Tensor, tuple, list))], h)
        g = torch.Generator().manual_seed(int.from_bytes(h.digest()[:7], "little"))

        def fill(v):
            if isinstance(v, torch.Tensor):
                shape = _concrete(v.shape, self.tokens)
                if v.dtype.is_floating_point:
                    return torch.randn(shape, generator=g).to(v.dtype)
                if v.dtype == torch.bool:
                    return torch.randint(0, 2, shape, generator=g).bool()
                return torch.randint(0, min(1 << 20, torch.iinfo(v.dtype).max), shape, generator=g, dtype=v.dtype)
            if isinstance(v, (tuple, list)):
                return type(v)(fill(x) for x in v)
            return v
        return fill(self._node.meta["val"])


def test_pp2_carries_the_single_stage_bits_at_every_admissible_cut():
    from isoexec.compiler.capture.capture import capture
    from isoexec.compiler.partition import define_stage
    from isoexec.compiler.realization.graph import clone_graph_module
    from isoexec.compiler.realization.pp import crosses_several_tensors, prepare_pipeline_graph
    from isoexec.compiler.text import text_inputs

    model = SOURCE.build_meta(raw("debug_config.json"))
    cfg = model.cfg
    inputs = text_inputs(model, device="cpu", max_tokens=16, profile="trainer")
    captured = capture(model, inputs.args, inputs.kwargs, dynamic_shapes=inputs.dynamic_shapes,
                       root_layouts=inputs.root_layouts, seq=inputs.seq)
    seq = inputs.args[1]
    ids, T = seq.input_ids, int(seq.input_ids.numel())
    units = len(captured.layers)
    single = StandIns(clone_graph_module(define_stage(captured, ((0, units),), pp_rank=0).module), T).run(ids, seq)
    assert single.dtype == torch.float32 and single.shape == (T, cfg.vocab_size)
    for cut in cfg.admissible_pipeline_cuts():
        ranges = ((0, cut), (cut, units))
        first = clone_graph_module(define_stage(captured, ranges, pp_rank=0).module)
        last = clone_graph_module(define_stage(captured, ranges, pp_rank=1).module)
        assert crosses_several_tensors(first)
        _, sent = prepare_pipeline_graph(first, named_carries=True, final=False)
        received, _ = prepare_pipeline_graph(last, named_carries=True, final=True)
        wire = StandIns(first, T).run(ids, seq)
        assert [tuple(w.shape) for w in wire] == [(T, *c.tail) for c in sent]       # token-leading, as vLLM slices
        assert [w.dtype for w in wire] == [c.dtype for c in received]
        # vLLM's wire: a buffer of the step's token count per key, filled from the sender (a pure byte move)
        out = StandIns(last, T).run(ids, seq, *[w.clone() for w in wire])
        assert torch.equal(out.view(torch.int32), single.view(torch.int32)), cut
        carried = math.fsum(w.nbytes for w in wire) / T
        assert carried >= 2 * (cfg.hc_mult * cfg.hidden_size)      # at least the BF16 streams cross


# ------------------------------------------------------------------------------------------ weight ownership
def _captured(name):
    from isoexec.compiler.capture.capture import capture
    from isoexec.compiler.text import text_inputs
    model = SOURCE.build_meta(raw(name))
    inputs = text_inputs(model, device="cpu", max_tokens=16, profile="trainer")
    return model, capture(model, inputs.args, inputs.kwargs, dynamic_shapes=inputs.dynamic_shapes,
                          root_layouts=inputs.root_layouts, seq=inputs.seq)


@pytest.mark.parametrize("name", ["debug_config.json", "config.json"])
def test_every_stage_owns_skips_or_aliases_every_checkpoint_tensor(name):
    """Every global tensor name (Parameters AND checkpoint buffers, e.g. the frozen Engram tables) is registered on
    the stage, aliased, or remote, at every stage of every admissible cut; and every checkpoint key of the released
    index maps on each stage to names it registers or skips (the GPU failure: ``layers.2.engram.embed.scale`` on the
    stage without layer 2 was neither)."""
    from isoexec.compiler.partition import define_stage
    import re

    from isoexec.compiler.realization.weights import (_conversion_mapping, _find_mapping, ignored_patterns,
                                                      package_ignored_patterns)

    model, captured = _captured(name)
    cfg = model.cfg
    names = (set(dict(captured.program.named_parameters(remove_duplicate=False)))
             | set(dict(captured.program.named_buffers(remove_duplicate=False))))
    assert any(".engram.embed." in n for n in names)
    hf_config = SOURCE.build_config(raw(name))
    renamings, converters = _conversion_mapping(SOURCE, hf_config)
    ignored = ignored_patterns(hf_config) + package_ignored_patterns(SOURCE, hf_config)
    keys = None
    if name == "config.json":
        keys = list(raw("checkpoint_index.json")["tensors"])
    units = len(captured.layers)
    for cut in cfg.admissible_pipeline_cuts():
        assigned = ((0, cut), (cut, units))
        for rank in range(2):
            d = define_stage(captured, assigned, pp_rank=rank)
            local = (set(dict(d.module.named_parameters(remove_duplicate=False)))
                     | set(dict(d.module.named_buffers(remove_duplicate=False))))
            assert local | d.remote_parameters | set(d.parameter_aliases) == names, (cut, rank)
            assert not local & d.remote_parameters
            for key in keys or ():
                if not key.startswith("layers.") or int(key.split(".")[1]) >= cfg.num_hidden_layers:
                    continue
                _, _, logical = _find_mapping(key, renamings, converters)
                missing = [n for n in logical if n not in local and n not in d.remote_parameters
                           and n not in d.parameter_aliases]
                # the loader's rule (load_weights): skipped when every unplaced name is declared ignored
                assert not missing or (len(missing) == len(logical) and all(
                    any(re.search(p, n) for p in ignored) for n in missing)), (cut, rank, key, logical)


def test_qwen_stages_keep_their_parameter_only_remote_sets():
    """Qwen registers no buffers: its remote sets are the Parameters of the other stages, as before."""
    from isoexec.compiler.capture.capture import capture
    from isoexec.compiler.partition import define_stage
    from isoexec.compiler.realization.tests.test_weights_cpu import qwen_config
    from isoexec.compiler.text import text_inputs
    from isoexec.models.qwen3_5.model import Qwen3_5ForConditionalGeneration

    author = Qwen3_5ForConditionalGeneration(qwen_config()).eval()
    inputs = text_inputs(author, device="cpu", max_tokens=8, profile="trainer")
    captured = capture(author, inputs.args, inputs.kwargs, dynamic_shapes=inputs.dynamic_shapes,
                       root_layouts=inputs.root_layouts, seq=inputs.seq)
    params = set(dict(captured.program.named_parameters(remove_duplicate=False)))
    assert not list(captured.program.named_buffers())
    for rank in range(2):
        d = define_stage(captured, ((0, 1), (1, 2)), pp_rank=rank)
        local = set(dict(d.module.named_parameters(remove_duplicate=False)))
        assert d.remote_parameters == params - local - set(d.parameter_aliases)


# ------------------------------------------------------------------------------------------ vLLM's PP cache planning
@pytest.mark.parametrize("name", ["debug_config.json", "b14_bf16_config.json"])
def test_every_stage_resolves_one_layout_and_allocates_only_its_caches(vllm_config, sm100, name):
    """vLLM's engine-core planning across stages (as ``EngineCore._initialize_kv_caches``): every stage declares
    the same cache layouts (an SWA-only stage used to declare all of them: "Workers disagree"), the global groups
    projected per stage allocate only that stage's tensors (vLLM 0.33 listed other stages' tensors of packed
    groups: StopIteration in ``allocate_kv_cache``; ``vllm_patches._local_kv_cache_tensors``), and every pool view
    addresses its records. Single stage: the patch returns the config unchanged."""
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups, get_kv_cache_configs
    from vllm.v1.worker.utils import allocate_kv_cache

    from isoexec.runtimes.vllm.vllm_patches import _local_kv_cache_tensors

    cfg = text_geometry(raw(name))
    n = cfg.num_hidden_layers
    context = vllm_config.compilation_config.static_forward_context
    for cut in (None, *cfg.admissible_pipeline_cuts()):
        stages = [list(range(n))] if cut is None else [list(range(cut)), list(range(cut, n))]
        specs, contexts, layouts = [], [], []
        for layers in stages:
            context.clear()
            vllm_config.cache_config.kv_cache_layout = None
            with set_current_vllm_config(vllm_config):
                for i in layers:
                    engine.bind_row_caches(cfg, i, vllm_config, f"body.layers.{i}")
                spec = {k: layer.get_kv_cache_spec(vllm_config) for k, layer in context.items()}
                specs.append({k: v for k, v in spec.items() if v is not None})
                contexts.append(dict(context))
                layouts.append([x.name for x in get_supported_kv_cache_layouts(
                    list({layer.get_attn_backend() for layer in context.values()}))])
        assert all(x == layouts[0] for x in layouts), (cut, layouts)
        with set_current_vllm_config(vllm_config):
            resolve_kv_cache_layout(vllm_config, layouts, [v for s in specs for v in s.values()])
            groups = get_global_kv_cache_groups(vllm_config, specs)
            configs = get_kv_cache_configs(vllm_config, specs, [1 << 27] * len(specs), groups)
            for config, stage_context in zip(configs, contexts):
                before = list(config.kv_cache_tensors)
                local = _local_kv_cache_tensors(config)
                if cut is None:
                    assert local.kv_cache_tensors == before
                caches = allocate_kv_cache(local, torch.device("cpu"), vllm_config.cache_config.get_resolved_kv_cache_layout())
                assert set(caches) == set(stage_context)
                for key, tensor in caches.items():
                    stage_context[key].bind_kv_cache(tensor)
                _views(SimpleNamespace(context=stage_context))


# ------------------------------------------------------------------------------------------ per-stage host state
def test_a_stage_without_the_embedding_plans_with_the_replay_window():
    """The engine model state of a later stage (no embedding, only rows) reads the geometry from its rows: the
    decoder plan's replay window is the model's (it was 0 there: PP2 logits differed from PP1 at every cut)."""
    import copy

    from torch import nn

    from isoexec.models.deepseek_v4_1_flash import engine_ced
    from isoexec.models.deepseek_v4_1_flash.engine_state import _geometry

    model = SOURCE.build_meta(raw("debug_config.json"))
    cfg = model.cfg
    stage = nn.Module()                                   # a later stage: decoder rows only
    stage.blocks = nn.ModuleList([copy.deepcopy(b) for b in model.decoder.blocks])
    assert _geometry(stage) is not None and _geometry(stage).replay_window == cfg.replay_window == cfg.sliding_window
    assert _geometry(model).replay_window == cfg.replay_window
    assert _geometry(nn.Module()) is None
    prompt = np.array([300, 40]); computed = np.array([0, 0]); scheduled = np.array([300, 40])
    full = engine_ced.plan_step(prompt, computed, scheduled, window=_geometry(stage).replay_window,
                                mode=engine_ced.BOUNDED_REPLAY)
    assert full.segment_start.tolist() == [300 - cfg.replay_window, 0]


def test_replicated_expert_rows_of_an_empty_step_skip_the_collective(monkeypatch):
    """EP > 1 with replicated rows: a step with no rows for a block returns no rows on every rank (the same count
    everywhere) instead of calling MegaMoE with width 0 (refused: TP2 x PP2 crashed on one rank and hung)."""
    from types import SimpleNamespace

    from isoexec.models.deepseek_v4_1_flash import providers
    from isoexec.ops.moe.deepseek_v41_sm100 import ep as ep_ops

    called = []
    monkeypatch.setattr(ep_ops, "mega_ep_forward", lambda *a, **k: called.append(1))
    monkeypatch.setattr(providers, "_tracked", lambda *a: False)
    w = torch.zeros(4, 4)
    block = SimpleNamespace(ep=SimpleNamespace(mega=object(), partitioned=False, size=2, rank=1, expert_map=None),
                            ffn=SimpleNamespace(gate=SimpleNamespace(weight=w, bias=None)),
                            packed_experts=lambda: (w, w), cfg=SOURCE.build_meta(raw("debug_config.json")).cfg)
    out = providers._engine_ep_moe(block, torch.zeros((0, 8), dtype=torch.bfloat16))
    assert out.shape == (0, 8) and out.dtype == torch.bfloat16 and not called
