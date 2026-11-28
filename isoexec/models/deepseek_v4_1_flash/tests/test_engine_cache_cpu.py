"""The rows' native vLLM cache layers and the kernels' addressing of vLLM's own allocation (CPU).

Bit neutrality of the engine adapters is an addressing property: every record the kernels address for a block
lies inside that block's bytes of the layer's page, and no two layers (of exclusively owned blocks) address the
same byte, so each layer reads back exactly what it wrote. Checked on vLLM's planner and allocator with native
DeepSeek V4.1 cache layers, through the adapters the engine step uses (``engine.py``) and the kernels' address
formulas: window and compressed rows ``row * KV_RECORD_BYTES`` (528); index keys ``page * stride + row * 128``
(codes) and ``page * stride + P * 128 + row * 4`` (scale word); compressor state ``slot * stride + j * 4096``.
"""
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from isoexec.compiler.source import ModelSource  # noqa: E402
from isoexec.models.deepseek_v4_1_flash import engine  # noqa: E402
from isoexec.models.deepseek_v4_1_flash.config import text_geometry  # noqa: E402
from isoexec.ops.attention.deepseek_v41_cache_sm100.spec import INDEX_RECORD_BYTES as IREC  # noqa: E402
from isoexec.ops.attention.deepseek_v41_cache_sm100.spec import KV_RECORD_BYTES as REC  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE = ModelSource("isoexec.models.deepseek_v4_1_flash")


def raw(name):
    return json.loads((FIXTURES / name).read_text())


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


def build(vllm_config, config_name, blocks, device="cpu"):
    """Register every row's layers, plan, allocate on ``device`` (the host by default) and bind like vLLM's
    worker."""
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
    from vllm.v1.worker.utils import allocate_kv_cache

    cfg = text_geometry(raw(config_name))
    context = vllm_config.compilation_config.static_forward_context
    context.clear()
    vllm_config.cache_config.kv_cache_layout = None
    with set_current_vllm_config(vllm_config):
        rows = [engine.bind_row_caches(cfg, i, vllm_config, f"body.layers.{i}") for i in range(cfg.num_hidden_layers)]
        specs = {name: layer.get_kv_cache_spec(vllm_config) for name, layer in context.items()}
        specs = {name: spec for name, spec in specs.items() if spec is not None}
        backends = list({layer.get_attn_backend() for layer in context.values()})
        names = [layout.name for layout in get_supported_kv_cache_layouts(backends)]
        layout = resolve_kv_cache_layout(vllm_config, [names], specs.values())
        builder = DefaultKVCacheConfigBuilder()
        groups = builder.get_kv_cache_groups(vllm_config, specs)
        config = builder.get_kv_cache_config_from_groups(vllm_config, groups, blocks)
        caches = allocate_kv_cache(config, torch.device(device), layout)
    for name, tensor in caches.items():
        context[name].bind_kv_cache(tensor)
    base = next(iter(caches.values())).untyped_storage().data_ptr()
    stride = {t.stride(0) * t.element_size() for t in caches.values()}
    assert len(stride) == 1
    return SimpleNamespace(cfg=cfg, rows=rows, context=context, groups=groups, config=config, base=base,
                           block=stride.pop(), blocks=config.num_blocks)


@pytest.mark.parametrize("config_name", ["debug_config.json", "config.json"])
def test_rows_register_native_layers_specs_and_sharing(vllm_config, sm100, config_name):
    from vllm.models.deepseek_v41.attention import DeepseekV4IndexerCache
    from vllm.models.deepseek_v41.compressor import CompressorStateCache
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
    from vllm.v1.kv_cache_interface import CircularBufferSpec, MLAAttentionSpec, SlidingWindowMLASpec
    b = build(vllm_config, config_name, 4)
    cfg = b.cfg
    for i, row in enumerate(b.rows):
        r = cfg.compress_ratios[i]
        assert row.attn == f"body.layers.{i}.attn" and isinstance(row.swa, DeepseekV4SWACache)
        spec = row.swa.get_kv_cache_spec(vllm_config)
        assert isinstance(spec, SlidingWindowMLASpec) and spec.block_size == 32
        assert spec.page_size_bytes == 32 * REC
        assert issubclass(row.swa.get_attn_backend(), engine.native_attention().swa_backend_cls)
        for layer in (row.swa, row.comp, row.index, row.state):    # metadata only: batch-invariant engine
            assert layer is None or layer.get_attn_backend().supports_batch_invariance()
        if not r:
            assert row.source is None and row.comp is None
            continue
        src = cfg.source_layer(i, "kv")
        assert row.source == f"body.layers.{src}.attn"
        if i != src:          # consumers own no compressed or index pages: they read the source's
            assert row.comp is row.index is row.state is None
            continue
        spec = row.comp.get_kv_cache_spec(vllm_config)
        assert isinstance(spec, MLAAttentionSpec) and spec.tokens_per_state == r
        assert spec.page_size_bytes == 128 // r * REC
        assert issubclass(row.comp.get_attn_backend(), engine.native_attention().backend_cls)
        assert isinstance(row.index, DeepseekV4IndexerCache)
        assert row.index.get_kv_cache_spec(vllm_config).page_size_bytes >= 128 // r * IREC
        assert (row.state is None) == (r == 1)
        if r > 1:
            assert isinstance(row.state, CompressorStateCache)
            assert isinstance(row.state.get_kv_cache_spec(vllm_config), CircularBufferSpec)


def addressed(b, name, block):
    """``(lo, hi)`` byte extents of every record the kernels address for layer ``name`` in ``block``."""
    layer = b.context[name]
    cache = layer.kv_cache
    base = cache.data_ptr() - b.base
    if name.endswith(".swa_cache"):
        view, offset = engine.record_rows(cache, REC)
        slots = torch.arange(32) + block * 32
        rows = engine.slot_rows(slots, cache.stride(0) // REC, offset)
        start = view.data_ptr() - b.base
        return [(start + int(r) * REC, start + int(r) * REC + REC) for r in rows]
    if name.endswith(".indexer.k_cache"):
        pages = engine.index_pages(cache)
        P = pages.shape[1]
        units = engine.split_pages(torch.tensor([[block]], dtype=torch.int32), P // 64)[0].tolist()
        out = []
        for u in units:                          # writes: dest = unit * 64 + k; page = dest // P, row = dest % P
            for k in range(64):
                dest = u * 64 + k
                page, row = dest // P, dest % P
                lo = base + page * pages.stride(0)
                out += [(lo + row * 128, lo + row * 128 + 128), (lo + P * 128 + row * 4, lo + P * 128 + row * 4 + 4)]
        return out
    if name.endswith(".compressor.state_cache"):
        r = int(name.split(".layers.")[1].split(".")[0])
        ratio = b.cfg.compress_ratios[r]
        state = engine.state_rows(cache, ratio)
        lo = state.data_ptr() - b.base + block * state.stride(0) * 4
        return [(lo + j * 4096, lo + (j + 1) * 4096) for j in range(ratio)]
    view, offset = engine.record_rows(cache, REC)
    ratio = layer.compress_ratio
    entries = engine.expand_pages(torch.tensor([[block]], dtype=torch.int32), cache.stride(0) // REC,
                                  128 // ratio // 64, offset)[0].tolist()
    start = view.data_ptr() - b.base
    return [(start + (e + k) * REC, start + (e + k + 1) * REC) for e in entries for k in range(64)]


@pytest.mark.parametrize("config_name", ["debug_config.json", "config.json"])
@pytest.mark.parametrize("seed", range(3))
def test_kernel_addressing_is_byte_exclusive_on_vllms_allocator(vllm_config, sm100, config_name, seed):
    rng = random.Random(seed)
    b = build(vllm_config, config_name, 7)
    raw_bytes = None
    for name in b.context:
        raw_bytes = b.context[name].kv_cache.untyped_storage()
        break
    flat = torch.empty(0, dtype=torch.uint8).set_(raw_bytes, 0, (raw_bytes.nbytes(),), (1,))
    flat.zero_()
    # vLLM's block manager: block 0 is the null block; every other block id is owned by one group at a time
    owner = {block: rng.randrange(len(b.groups)) for block in range(1, b.blocks)}
    written = []
    for block, g in owner.items():
        for name in b.groups[g].layer_names:
            tag = (hash((name, block)) % 251) + 1
            for lo, hi in addressed(b, name, block):
                assert block * b.block <= lo and hi <= (block + 1) * b.block, (name, block)
                flat[lo:hi] = tag
            written.append((name, block, tag))
    for name, block, tag in written:
        for lo, hi in addressed(b, name, block):
            assert bool((flat[lo:hi] == tag).all()), (name, block)


def test_engine_record_widths_are_the_cache_packages_declared_records():
    assert (engine.WINDOW_BYTES, engine.COMP_BYTES, engine.INDEX_BYTES) == (REC, REC,
                                                                         IREC)
    assert engine.KV_CACHE_DTYPE == "fp8_ds_mla"


def test_kernel_views_cover_whole_pages(vllm_config, sm100):
    b = build(vllm_config, "config.json", 5)
    for name, layer in b.context.items():
        cache = layer.kv_cache
        if name.endswith(".swa_cache") or name.endswith(".attn"):
            record = REC            # window and compressed pools: the one FP8 record
            view, offset = engine.record_rows(cache, record)
            assert view.is_contiguous() and view.shape[1:] == (64, record) and 0 <= offset < 64
            assert view.data_ptr() + offset * record == cache.data_ptr()
            last = (cache.shape[0] - 1) * (cache.stride(0) // record) + cache.shape[1]
            assert view.shape[0] * 64 >= offset + last
            assert view.data_ptr() + view.numel() <= cache.untyped_storage().data_ptr() + cache.untyped_storage().nbytes()


def _metadata(b, block_tables, swa_tables, starts, seq_lens, graph):
    """vLLM-shaped metadata of one step: requests with host boundaries ``starts`` and lengths ``seq_lens``."""
    N = starts[-1]
    R = len(seq_lens)
    tok = [s for s in range(R) for _ in range(starts[s + 1] - starts[s])]
    pos = [seq_lens[s] - (starts[s + 1] - starts[s]) + j for s in range(R) for j in range(starts[s + 1] - starts[s])]
    W = b.cfg.sliding_window
    md = {}
    for name, layer in b.context.items():
        if name.endswith(".swa_cache"):
            group = next(g for g, grp in enumerate(b.groups) if name in grp.layer_names)
            bt = swa_tables[group]
            slots = torch.tensor([bt[s][p // 32] * 32 + p % 32 for s, p in zip(tok, pos)], dtype=torch.int64)
            lists = torch.full((N, 1, W), -1, dtype=torch.int32)
            for t, (s, p) in enumerate(zip(tok, pos)):
                lo = max(p - W + 1, 0)
                for k, q in enumerate(range(lo, p + 1)):
                    lists[t, 0, k] = bt[s][q // 32] * 32 + q % 32
            decode = graph or all(starts[s + 1] - starts[s] == 1 for s in range(R))
            md[name] = SimpleNamespace(
                prefix=name, query_start_loc=torch.tensor(starts, dtype=torch.int32),
                query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
                seq_lens=torch.tensor(seq_lens, dtype=torch.int32), token_to_req_indices=torch.tensor(tok, dtype=torch.int32),
                is_valid_token=slots >= 0, slot_mapping=slots, num_decode_tokens=N if decode else 0,
                num_prefill_tokens=0 if decode else N, decode_swa_indices=lists if decode else lists[:0],
                prefill_swa_indices=None if decode else lists)
        elif name.endswith(".compressor.state_cache"):
            ring = block_tables["state"]
            md[name] = SimpleNamespace(slot_mapping=torch.tensor([ring[s] * 8 + p % 8 for s, p in zip(tok, pos)]))
        elif name.endswith(".attn"):
            md[name] = SimpleNamespace(block_table=torch.tensor(block_tables["comp"], dtype=torch.int32))
    return md, torch.tensor(pos, dtype=torch.int64), tok, pos


@pytest.mark.parametrize("graph", [False, True])
def test_engine_step_adapters_from_native_metadata(vllm_config, sm100, graph):
    b = build(vllm_config, "debug_config.json", 40)
    cfg = b.cfg
    if graph:
        starts, seq_lens = [0, 1, 2, 3], [300, 1, 129]
    else:
        starts, seq_lens = [0, 5, 6, 40], [5, 260, 34]
    R = len(seq_lens)
    comp = [[3, 7, 9], [11, 2, 30], [5, 6, 8]]
    tables = {"comp": comp, "state": [13, 14, 15]}
    swa_tables = {g: [[(g * 7 + s * 11 + j * 3) % 39 + 1 for j in range(10)] for s in range(R)]
                  for g in range(len(b.groups))}
    md, positions, tok, pos = _metadata(b, tables, swa_tables, starts, seq_lens, graph)
    rows = b.rows
    step = engine.EngineStep(cfg, rows[0], md, positions, graph, capturing=graph)
    for i, row in enumerate(rows):
        step.bind_window(i, row, md)
    batch = step.batch
    assert batch.tok_seq.tolist() == tok and batch.tok_pos.tolist() == pos
    assert batch.p0.tolist() == [seq_lens[s] - (starts[s + 1] - starts[s]) for s in range(R)]
    assert batch.state_slot.tolist() == tables["state"]
    for i, row in enumerate(rows):
        per_block = row.swa.kv_cache.stride(0) // REC
        off = engine.record_rows(row.swa.kv_cache, REC)[1]
        assert step.pools.window[i].data_ptr() + off * REC == row.swa.kv_cache.data_ptr()
        group = next(g for g, grp in enumerate(b.groups) if row.swa.prefix in grp.layer_names)
        bt = swa_tables[group]
        want_rows = [bt[s][p // 32] * per_block + p % 32 + off for s, p in zip(tok, pos)]
        assert batch.window_rows[i].tolist() == want_rows
        lists = batch.window_lists[i]
        for t, (s, p) in enumerate(zip(tok, pos)):
            lo = max(p - cfg.sliding_window + 1, 0)
            want = [bt[s][q // 32] * per_block + q % 32 + off for q in range(lo, p + 1)]
            assert lists[t].tolist() == want + [-1] * (cfg.sliding_window - len(want))
    for r in (1, 2):
        src = next(i for i in cfg.kv_source_layer_ids if cfg.compress_ratios[i] == r)
        cache = b.context[f"body.layers.{src}.attn"].kv_cache
        comp_rows, off = cache.stride(0) // REC, engine.record_rows(cache, REC)[1]
        pages = 128 // r // 64
        assert batch.comp_bt_by_source[src].tolist() == [[blk * comp_rows + 64 * h + off for blk in req for h in range(pages)]
                                             for req in comp]
        assert batch.index_bt[r].tolist() == [[blk * pages + h for blk in req for h in range(pages)] for req in comp]
        width = len(comp[0]) if not graph else min(len(comp[0]), -(-step.capacity // 128))
        assert batch.index_read_bt[r].tolist() == [req[:width] for req in comp]
        assert step.pools.index[src].shape[1] == 128 // r
    if graph:
        assert batch.groups[2][0].tolist() == list(range(R)) and batch.groups[2][1].tolist() == [0] * R
    else:
        counts = [starts[s + 1] - starts[s] for s in range(R)]
        assert batch.groups[2][0].tolist() == [s for s in range(R) for _ in range(-(-counts[s] // 2))]


def test_graph_context_capacity_budget():
    assert engine.graph_context_capacity(64, 8192, None) == 8192
    assert engine.graph_context_capacity(64, 1 << 20, 1 << 24) == 262144
    assert engine.graph_context_capacity(2048, 1 << 20, 1 << 24) == 8192
    with pytest.raises(ValueError, match="one block"):
        engine.graph_context_capacity(1 << 20, 1 << 20, 1 << 24)
