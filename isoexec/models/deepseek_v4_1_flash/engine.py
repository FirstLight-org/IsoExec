"""vLLM engine binding of the V4.1 rows: native cache layers, the paged step and the per-step host state.

Cache memory and per-step metadata are vLLM's own DeepSeek V4.1 ones (``vllm.models.deepseek_v41``): each
row registers the cache-holding layers native vLLM registers, under the same relative prefixes, with the
same specs and metadata backends (FlashMLA sparse family), so vLLM sizes, groups, allocates and shares the
pages exactly as for its own model:

* ``{row}.attn.swa_cache``: ``DeepseekV4SWACache``, the sliding-window FP8 records (528 B, 32-token blocks);
* ``{row}.attn``: the compressed-KV source's FP8 records (528 B), ``tokens_per_state = ratio``
  (``CompressedKVSource``, native's ``MLAAttentionSpec``); consumers read their source's pages;
* ``{row}.attn.indexer.k_cache``: ``DeepseekV4IndexerCache``, the source's MXFP8 index keys (132 B);
* ``{row}.attn.compressor.state_cache``: ``CompressorStateCache``, the per-request FP32 ring of a ratio > 1
  source.

The kernels address those pages through bit-neutral adapters (``EngineStep``): window rows and main lists
from native's sliding-window slots (``block * 32 + offset`` -> physical 528 B rows of the layer's view, the
same ascending-position, -1-tail list contract); the compressed rail as 64-row pages of a block's
``128 / ratio`` records; index keys as pages of ``128 / ratio`` entries (``[P x 128][P x 4]``); the
compressor state in rows ``[0, ratio)`` of the request's ring block. Records are the kernels' own formats.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from isoexec.ops.attention.deepseek_v41_cache_sm100.spec import INDEX_RECORD_BYTES, KV_RECORD_BYTES

#: The engine settings this binding implements (``DeepseekV41ForCausalLM.engine_settings``).
KV_CACHE_DTYPE = "fp8_ds_mla"         # vLLM FLASHMLA_SPARSE_DSV41's decided record program (CACHE-SPEC)
BLOCK_SIZE = 128
SWA_BLOCK = 32
#: Record widths: the cache package's declared constants, never literals here (window and compressed KV share
#: the 528 B FP8 record).
WINDOW_BYTES, COMP_BYTES, INDEX_BYTES = KV_RECORD_BYTES, KV_RECORD_BYTES, INDEX_RECORD_BYTES
PAGE = 64
#: ``ISOEXEC_V41_WINDOW_GLUE`` (default 1): the step's window rows / lists in one launch (``EngineStep._bind_windows``)
#: and the decoder rows' in one more; 0 = the per-layer torch index glue. Launch structure only: the same integers.
WINDOW_GLUE = os.environ.get("ISOEXEC_V41_WINDOW_GLUE", "1").strip() != "0"
WINDOW_GLUE_MAX_LAYERS = 64          # csrc/paged.cu kWinMaxLayers
PAGE_TABLES_MAX = 16                 # csrc/paged.cu kMaxTables
#: ``additional_config.isoexec`` key: score elements of one captured decode step (``tokens * context``).
SCORE_BUDGET_KEY = "decode_graph_score_element_budget"


def native_attention():
    """The native SM100 attention class whose metadata backends the cache layers use (FlashMLA sparse)."""
    from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
    return DeepseekV4FlashMLAAttention


_METADATA_ONLY: dict = {}


def model_kv_cache_layouts():
    """The cache layouts of the whole V4.1 model: the index-key cache's (vLLM ``DeepseekV41IndexerBackend``, the
    one backend of these layers that declares any; every other one serves all layouts). Single-stage, vLLM
    resolves the layout among exactly these."""
    from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
    return DeepseekV41IndexerBackend.supported_kv_cache_layouts()


def metadata_only(backend):
    """``backend`` as these layers use it: its metadata builder only (no attention kernel of the backend
    runs; the rows' own batch-invariant kernels read the pages), so vLLM's batch-invariance requirement
    (``VLLM_BATCH_INVARIANT``) holds for it. One subclass per backend class.

    A backend that serves every layout declares the model's layouts (``model_kv_cache_layouts``): vLLM requires
    every pipeline stage to declare the same set, and a stage of SWA-only rows would otherwise declare all
    (engine PP2 at an encoder cut: "Workers disagree on supported KV cache layouts"). Single-stage the
    resolution is unchanged (vLLM already intersected to these)."""
    hit = _METADATA_ONLY.get(backend)
    if hit is None:
        declared = backend.supported_kv_cache_layouts()
        layouts = model_kv_cache_layouts() if declared is None else declared
        hit = _METADATA_ONLY[backend] = type(backend.__name__, (backend,), {
            "__doc__": f"{backend.__name__} metadata only (rows run their own kernels)",
            "supports_batch_invariance": classmethod(lambda cls: True),
            "supported_kv_cache_layouts": classmethod(lambda cls: layouts)})
    return hit


def _metadata_backend_of(layer) -> None:
    resolve = layer.get_attn_backend
    layer.get_attn_backend = lambda: metadata_only(resolve())


_SOURCE_CLASS: list = []


def compressed_kv_source_class():
    """``CompressedKVSource``: the compressed-KV cache of a source row, with native's spec and backend."""
    if _SOURCE_CLASS:
        return _SOURCE_CLASS[0]
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    class CompressedKVSource(nn.Module, AttentionLayerBase):
        """Registered at the row's attention prefix like native ``DeepseekV4Attention`` of a KV source;
        consumer rows read ``kv_cache`` of their source through the forward context."""

        def __init__(self, prefix: str, ratio: int, cache_config, compilation_config) -> None:
            super().__init__()
            self.prefix, self.layer_name, self.compress_ratio = prefix, prefix, ratio
            self.cache_config = cache_config
            self.kv_cache = torch.tensor([])
            context = compilation_config.static_forward_context
            if prefix in context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            context[prefix] = self

        def get_kv_cache_spec(self, vllm_config):
            from vllm.v1.kv_cache_interface import MLAAttentionSpec, get_kv_quant_mode
            return MLAAttentionSpec(block_size=self.cache_config.block_size, num_kv_heads=1, head_size=512,
                                    dtype=torch.uint8, tokens_per_state=self.compress_ratio,
                                    cache_dtype_str=KV_CACHE_DTYPE, alignment=512, model_version="deepseek_v4",
                                    kv_quant_mode=get_kv_quant_mode(KV_CACHE_DTYPE),
                                    state_content_bytes=COMP_BYTES)

        def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
            self.kv_cache = kv_cache.squeeze(1)          # [blocks, 1, records, 288] -> [blocks, records, 288]

        def get_attn_backend(self):
            return metadata_only(native_attention().backend_cls)

        def forward(self):
            raise RuntimeError("the compressed-KV cache layer holds pages only")

    _SOURCE_CLASS.append(CompressedKVSource)
    return CompressedKVSource


def _layer_prefix(prefix: str, layer: int) -> str:
    out, n = re.subn(r"(^|\.)layers\.\d+(\.|$)", rf"\g<1>layers.{layer}\g<2>", prefix)
    if n != 1:
        raise ValueError(f"cannot locate the row index in {prefix!r}")
    return out


@dataclass
class RowCaches:
    """The native cache layers a row owns and the prefixes of the layers it reads."""

    attn: str                       # this row's attention prefix
    swa: Any
    source: str | None              # attention prefix of the compressed-KV / index-key source (None: window only)
    comp: Any = None                # own layers (sources only)
    index: Any = None
    state: Any = None
    context: dict = field(default_factory=dict, repr=False)
    max_len: int = 0
    score_budget: int | None = None
    graphs: bool = False
    max_capture: int = 0

    def layer(self, prefix: str):
        return self.context[prefix]


def graphs_enabled(vllm_config) -> bool:
    from vllm.config.compilation import CUDAGraphMode
    mode = getattr(vllm_config.compilation_config, "cudagraph_mode", None)
    return mode is not None and mode != CUDAGraphMode.NONE


def bind_row_caches(cfg, index: int, vllm_config, layer_prefix: str) -> RowCaches:
    """Register row ``index``'s native cache layers (``{layer_prefix}.attn...``) in vLLM's forward context."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.models.deepseek_v41.attention import DeepseekV4IndexerCache
    from vllm.models.deepseek_v41.compressor import CompressorStateCache
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

    cc = vllm_config.cache_config
    if cc.cache_dtype != KV_CACHE_DTYPE or cc.block_size != BLOCK_SIZE:
        raise ValueError(f"DeepSeek V4.1 engine caches need kv_cache_dtype={KV_CACHE_DTYPE!r} and block_size "
                         f"{BLOCK_SIZE} (engine_settings), got {cc.cache_dtype!r} / {cc.block_size}")
    if cc.enable_prefix_caching:
        raise ValueError("DeepSeek V4.1 engine caches are served without prefix caching")
    if not current_platform.is_device_capability_family(100):
        raise ValueError("the V4.1 cache records (FP8 window, FP4 compressed) are SM100 formats")
    pc = vllm_config.parallel_config
    if any(getattr(pc, key, 1) != 1 for key in ("prefill_context_parallel_size", "decode_context_parallel_size")):
        raise ValueError("DeepSeek V4.1 rows read the whole request context: context parallelism is refused")
    if vllm_config.speculative_config is not None:
        raise ValueError("speculative decoding is not served by the V4.1 rows")
    attn = f"{layer_prefix}.attn"
    r = cfg.compress_ratios[index]
    swa = DeepseekV4SWACache(head_dim=cfg.head_dim, window_size=cfg.sliding_window, dtype=torch.uint8,
                             prefix=f"{attn}.swa_cache", cache_config=cc,
                             backend_cls=metadata_only(native_attention().swa_backend_cls), block_size=SWA_BLOCK,
                             packed_bytes_per_token=WINDOW_BYTES, packed_page_alignment=512,
                             packed_block_stride_alignment=math.lcm(WINDOW_BYTES, COMP_BYTES),
                             bounded_replay=False)
    source = None if not r else _layer_prefix(layer_prefix, cfg.source_layer(index, "kv")) + ".attn"
    caches = RowCaches(attn=attn, swa=swa, source=source,
                       context=vllm_config.compilation_config.static_forward_context,
                       max_len=vllm_config.model_config.max_model_len)
    settings = (vllm_config.additional_config or {}).get("isoexec") or {}
    budget = settings.get(SCORE_BUDGET_KEY)
    if budget is not None and (type(budget) is not int or budget <= 0 or budget % BLOCK_SIZE):
        raise ValueError(f"{SCORE_BUDGET_KEY} must be a positive multiple of {BLOCK_SIZE}")
    caches.score_budget = budget
    mode = getattr(vllm_config.compilation_config, "cudagraph_mode", None)
    # FULL_AND_PIECEWISE: every V4.1 component is an eager break of the breakable PIECEWISE graphs (piecewise.py),
    # so nothing that reads the step's metadata or host plan is frozen into a segment.
    caches.graphs = mode is not None and mode != CUDAGraphMode.NONE
    if caches.graphs:
        sizes = list(getattr(vllm_config.compilation_config, "cudagraph_capture_sizes", None) or ())
        caches.max_capture = max(sizes) if sizes else int(getattr(vllm_config.compilation_config,
                                                                   "max_cudagraph_capture_size", 0) or 0)
    if r and index in cfg.kv_source_layer_ids:
        cls = compressed_kv_source_class()
        caches.comp = cls(attn, r, cc, vllm_config.compilation_config)
        caches.index = DeepseekV4IndexerCache(head_dim=cfg.index_head_dim + cfg.index_head_dim // 128 * 4,
                                              dtype=torch.uint8, prefix=f"{attn}.indexer.k_cache",
                                              cache_config=cc, compress_ratio=r)
        _metadata_backend_of(caches.index)
        if tuple(caches.index.get_attn_backend().supported_kv_cache_layouts() or ()) != tuple(model_kv_cache_layouts()):
            raise ValueError(f"{attn}: the index-key cache's layouts are not the model's declared cache layouts")
        if r > 1:
            caches.state = CompressorStateCache(state_dim=2 * cfg.head_dim, dtype=torch.float32,
                                                prefix=f"{attn}.compressor.state_cache")
            _metadata_backend_of(caches.state)
    elif source is not None and source not in caches.context:
        # admission refuses such a stage layout first (``admit_pipeline``); this is the registration's own check,
        # with native vLLM's rule (``DeepseekV4Attention``: no PP split inside a KV-sharing group)
        from .admission import PP_CUT_SPLITS_GROUP, Refusal
        raise Refusal(PP_CUT_SPLITS_GROUP, attn, f"compressed-KV source {source} is not registered on this rank: "
                                                 f"the pipeline stage splits a KV-sharing group")
    return caches


def cache_tensors(caches: RowCaches) -> list[torch.Tensor]:
    """The engine-owned cache tensors of a row's own layers."""
    return [layer.kv_cache for layer in (caches.swa, caches.comp, caches.index, caches.state)
            if layer is not None and layer.kv_cache.numel()]


# ------------------------------------------------------------------------------------------ views
def record_rows(cache: torch.Tensor, record: int) -> tuple[torch.Tensor, int]:
    """``(rows, offset)``: the layer's records as rows of one dense uint8 view of the allocation, in which record
    ``j`` of block ``block`` is row ``offset + block * (stride / record) + j``. Whole 64-row pages
    ``[pages, 64, record]`` when the allocation has room for them: the view ends at the layer's last record when
    the bytes before the layer allow it, else on the first whole page after it. Otherwise (a pipeline stage whose
    allocation ends within a page after the layer's last record, defect 13b) the row-bounded view
    ``[rows, record]`` from the layer's first record to its last: the kernels gather single rows by address
    (the dual-rail map is bounded at the view's last row), so both views give every record the same bytes. It
    never leaves the allocation."""
    blocks, entries = cache.shape[0], cache.shape[1]
    stride = cache.stride(0)
    if cache.dtype != torch.uint8 or cache.stride(-1) != 1 or cache.stride(-2) != record or stride % record:
        raise RuntimeError(f"cache pages of {record} B records must be dense rows in record-aligned blocks")
    storage = cache.untyped_storage()
    start = cache.storage_offset()
    need = (blocks - 1) * (stride // record) + entries
    offset = -need % PAGE
    if offset * record <= start:              # start the view ``offset`` rows early: it ends at the last record
        start -= offset * record
        rows = need + offset
    else:                                     # from the layer's first byte, rounded up to a whole page
        offset, rows = 0, -(-need // PAGE) * PAGE
    if start + rows * record > storage.nbytes():
        start, offset = cache.storage_offset(), 0
        if start + need * record > storage.nbytes():
            raise RuntimeError("the cache records do not fit inside the allocation")
        flat = torch.empty(0, dtype=torch.uint8, device=cache.device).set_(storage, start, (need * record,), (1,))
        return flat.view(need, record), 0
    flat = torch.empty(0, dtype=torch.uint8, device=cache.device).set_(storage, start, (rows * record,), (1,))
    return flat.view(rows // PAGE, PAGE, record), offset


def index_pages(cache: torch.Tensor) -> torch.Tensor:
    """``[blocks, P, 1, 132]`` index-key pages (``P = 128 / ratio`` entries: codes, then scale words)."""
    blocks, entries = cache.shape[0], cache.shape[1]
    if cache.dtype != torch.uint8 or cache.shape[-1] != INDEX_BYTES or cache.stride(1) != INDEX_BYTES:
        raise RuntimeError("index-key pages must be dense 132 B records")
    if entries not in (64, 128) or cache.stride(0) % 16 or cache.stride(0) < entries * INDEX_BYTES:
        raise RuntimeError("index-key pages hold 64 or 128 entries at a 16 B aligned block stride")
    return cache.as_strided((blocks, entries, 1, INDEX_BYTES), (cache.stride(0), INDEX_BYTES, INDEX_BYTES, 1))


def state_rows(cache: torch.Tensor, ratio: int) -> torch.Tensor:
    """``[blocks, ratio, 1024]`` FP32: the first ``ratio`` rows of each request's ring block."""
    if cache.dtype != torch.float32 or cache.dim() != 3 or cache.shape[1] < ratio or cache.stride(1) != cache.shape[2]:
        raise RuntimeError("compressor state rings must be dense FP32 rows")
    return cache[:, :ratio]


def slot_rows(slots: torch.Tensor, rows_per_block: int, offset: int = 0, block: int = SWA_BLOCK) -> torch.Tensor:
    """Native slot ids ``block_id * block + j`` (-1: none) -> rows of the layer's row view (``record_rows``)."""
    return torch.where(slots >= 0, slots // block * rows_per_block + slots % block + offset, -1)


def expand_pages(block_table: torch.Tensor, rows_per_block: int, pages: int, offset: int = 0) -> torch.Tensor:
    """Block table -> one entry per 64-record page: block ``b``'s page ``h`` starts at row
    ``offset + b * rows_per_block + 64 h`` of the row view (table entries are first record rows)."""
    offsets = torch.arange(pages, dtype=torch.int32, device=block_table.device) * PAGE + offset
    return (block_table.to(torch.int32)[..., None] * rows_per_block + offsets).flatten(1).contiguous()


def split_pages(block_table: torch.Tensor, units: int) -> torch.Tensor:
    """Block table of ``64 * units``-entry pages -> one entry per 64-entry unit (``b * units + h``)."""
    if units == 1:
        return block_table.to(torch.int32).contiguous()
    offsets = torch.arange(units, dtype=torch.int32, device=block_table.device)
    return (block_table.to(torch.int32)[..., None] * units + offsets).flatten(1).contiguous()


#: Why steps fell back to the per-layer window glue (``_bind_windows`` exit site -> count): read by the counting runs
#: and the tests; one stderr line the first time a site is hit.
WINDOW_GLUE_MISSES: dict = {}


def _window_glue_miss(site: int):
    if site not in WINDOW_GLUE_MISSES:
        import sys
        print(f"[ISOEXEC-V41] window glue: step falls back to the per-layer glue (exit {site})", file=sys.stderr,
              flush=True)
    WINDOW_GLUE_MISSES[site] = WINDOW_GLUE_MISSES.get(site, 0) + 1
    return None


def _page_tables(tables) -> None:
    """Fill ``dict[key] = expand_pages(...)`` / ``split_pages(...)`` for every ``(dict, key, block table, pages,
    mult, step, offset)`` of a step: one launch (``page_tables``, csrc/paged.cu) with the same int32 values
    (``mult`` = rows per block, ``step`` = 64 for ``expand_pages``; ``mult`` = units, ``step`` = 1, ``offset`` = 0
    for ``split_pages``); the torch formulas when the switch is off or a table is not an int32 row-major table."""
    if not tables:
        return
    from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension
    ext = extension()
    fused = (WINDOW_GLUE and hasattr(ext, "page_tables") and len(tables) <= PAGE_TABLES_MAX
             and len({t[2].shape[0] for t in tables}) == 1
             and all(t[2].is_cuda and t[2].dtype == torch.int32 and t[2].dim() == 2 and t[2].stride(-1) == 1 for t in tables))
    if not fused:
        for dest, key, bt, pages, mult, step, offset in tables:
            dest[key] = (expand_pages(bt, mult, pages, offset) if step == PAGE else split_pages(bt, pages))
        return
    outs = [torch.empty((bt.shape[0], bt.shape[1] * pages), dtype=torch.int32, device=bt.device)
            for _, _, bt, pages, *_ in tables]
    ext.page_tables([t[2] for t in tables], [t[3] for t in tables], [t[4] for t in tables], [t[5] for t in tables],
                    [t[6] for t in tables], outs)
    for (dest, key, *_), out in zip(tables, outs):
        dest[key] = out


def real_tokens(swa, tokens: int) -> int:
    """The tokens vLLM's metadata of this step describes (``token_to_req_indices`` rows: ``num_actual_tokens``),
    at most the model's ``tokens``: equal for eager and FULL steps (vLLM pads a FULL step's metadata to the graph
    size), fewer for a PIECEWISE step whose inputs are padded to its graph size while its attention metadata is
    built for the scheduled tokens (vLLM's ``DefaultModelState.prepare_attn``). The mature attention binding runs the
    attention on these rows and zeroes the padding rows' output (``runtimes/vllm/bindings.py``: ``n =
    md.num_actual_tokens``, ``output[n:].zero_()``)."""
    mapped = getattr(swa, "token_to_req_indices", None)
    if mapped is None:
        return tokens
    return min(tokens, int(mapped.shape[0]))


def graph_context_capacity(tokens: int, max_len: int, budget: int | None) -> int:
    """Context positions a captured decode step of ``tokens`` rows scores: the engine context, bounded by
    the score element budget (``tokens * context <= budget``, whole blocks)."""
    if budget is None:
        return max_len
    blocks = budget // (tokens * BLOCK_SIZE)
    if blocks == 0:
        raise ValueError(f"{SCORE_BUDGET_KEY} cannot fit one block of context per captured row")
    return min(max_len, blocks * BLOCK_SIZE)


# ------------------------------------------------------------------------------------------ the step
class EngineStep:
    """One forward's paged metadata, shared by every row (``PagedBatch`` + kernel views of vLLM's pages).

    ``graph``: the step is a single-token decode batch of a graph bucket size (vLLM captures and replays
    those): its host-bound structure is the decode one (one compression-group slot per request) and EP runs
    the static-capacity routes, the same bits as the eager forms. ``capturing``: the step is being captured;
    its index scores span the bucket's score capacity (``graph_context_capacity``), which the model state
    enforces before every replay. vLLM runs its capture and the warmup before it with the eager runtime
    mode, so neither is read from the forward context."""

    def __init__(self, cfg, caches: RowCaches, metadata: dict, positions: torch.Tensor, graph: bool,
                 capturing: bool = False) -> None:
        from isoexec.ops.attention.deepseek_v41_cache_sm100.paged import PagedBatch

        self.key, self.last = metadata, -1
        self.cfg, self.graph, self.capturing = cfg, graph, capturing
        self.program: dict = {}         # what the rows executed (``note``), recorded by the last row
        swa = metadata[caches.swa.prefix]
        # The step's real tokens: vLLM builds the attention metadata of a PIECEWISE (breakable) step for its scheduled
        # tokens only while the model's inputs are padded to the graph size (as for a FULL step it covers the padding
        # too). The rows read vLLM's metadata, so the step is its real tokens; the components run their attention on
        # those rows and leave the padding rows zero, as the mature attention binding does (``real_tokens``).
        N = real_tokens(metadata[caches.swa.prefix], positions.shape[-1])
        self.padded = positions.shape[-1]
        dev = positions.device
        cu_q = swa.query_start_loc.to(torch.int32)
        R = cu_q.numel() - 1
        n_new = cu_q[1:] - cu_q[:-1]
        self.tokens, self.requests = N, R
        tok_seq = swa.token_to_req_indices[:N].to(torch.int32)
        valid = swa.is_valid_token[:N]
        p0 = (swa.seq_lens[:R] - n_new).to(torch.int32)
        ratios = sorted({r for r in cfg.compress_ratios if r})
        groups = {}
        if graph:
            for r in ratios:
                if r > 1:     # single-token decode rows: one group slot per request
                    groups[r] = (torch.arange(R, dtype=torch.int32, device=dev),
                                 torch.zeros(R, dtype=torch.int32, device=dev))
        else:
            counts = (swa.query_start_loc_cpu[1:R + 1] - swa.query_start_loc_cpu[:R]).tolist()
            for r in ratios:
                if r > 1:
                    gs = [s for s, n in enumerate(counts) for _ in range(-(-n // r))]
                    gk = [k for n in counts for k in range(-(-n // r))]
                    # Stage both index arrays together so prefill can enqueue the copy on its CUDA stream.
                    indices = torch.tensor([gs, gk], dtype=torch.int32, device="cpu", pin_memory=dev.type == "cuda")
                    groups[r] = indices.to(dev, non_blocking=True).unbind(0)
        self.pools = SimpleNamespace(cfg=cfg, max_len=caches.max_len, device=dev, window={}, comp={}, index={},
                                     state={})
        comp_bt, index_bt, index_read = {}, {}, {}
        state_slot = torch.zeros(R, dtype=torch.int32, device=dev)
        self.capacity = graph_context_capacity(N, caches.max_len, caches.score_budget) if capturing else None
        # the KV sources of this pipeline stage: those registered on this rank (``bind_row_caches`` refuses a
        # stage row whose source is not, so every row of the stage finds its source here)
        prefixes = {src: _layer_prefix(caches.attn, src)
                    for src in sorted({cfg.source_layer(i, "kv") for i, r in enumerate(cfg.compress_ratios) if r})}
        sources = [src for src in prefixes if prefixes[src] in caches.context]
        prefixes = {src: prefixes[src] for src in sources}
        stated = False
        tables = []          # (dict, key, block table, pages, mult, step, offset): built in one launch below
        for src in sources:
            r = cfg.compress_ratios[src]
            layer = caches.layer(prefixes[src])
            bt = metadata[prefixes[src]].block_table[:R]
            comp = layer.kv_cache
            self.pools.comp[src], comp_offset = record_rows(comp, COMP_BYTES)
            index_layer = caches.layer(f"{prefixes[src]}.indexer.k_cache")
            self.pools.index[src] = index_pages(index_layer.kv_cache)
            entries = BLOCK_SIZE // r
            tables.append((comp_bt, src, bt, entries // PAGE, comp.stride(0) // COMP_BYTES, PAGE, comp_offset))
            if r not in index_bt:
                if entries // PAGE == 1:
                    index_bt[r] = split_pages(bt, 1)
                else:
                    index_bt[r] = None
                    tables.append((index_bt, r, bt, entries // PAGE, entries // PAGE, 1, 0))
                read = bt.to(torch.int32)
                if capturing:
                    read = read[:, :min(read.shape[1], -(-self.capacity // BLOCK_SIZE))]
                index_read[r] = read.contiguous()
            if r > 1:
                ring = caches.layer(f"{prefixes[src]}.compressor.state_cache")
                self.pools.state[src] = state_rows(ring.kv_cache, r)
                if not stated:
                    slots = metadata[ring.prefix].slot_mapping[:N]
                    first = cu_q[:-1].clamp(max=max(N - 1, 0)).long()
                    head = slots[first]
                    state_slot = torch.where(head >= 0, head // ring.block_size, 0).to(torch.int32)
                    stated = True
        _page_tables(tables)
        self.batch = PagedBatch(tok_seq, positions.reshape(-1)[:N].to(torch.int32), cu_q, p0, None, {},
                                state_slot, groups, valid=valid, index_bt=index_bt, win_per=1, comp_per=1,
                                window_rows={}, window_lists={}, index_read_bt=index_read, comp_bt_by_source=comp_bt)
        self._windows: dict = {}
        # every window layer's write rows and main list of this step in ONE launch (``bind_window`` then only binds
        # them): ``_bind_windows``; None = the per-layer torch glue (switch off, or a step it does not cover)
        self._win = self._bind_windows(caches, metadata) if WINDOW_GLUE else None

    def note(self, key: str, value) -> None:
        """One row's execution choice ``key`` (reducer transport, expert path, EP routes): rows that disagree
        record ``"mixed"``."""
        seen = self.program.get(key)
        self.program[key] = value if seen is None or seen == value else "mixed"

    def _bind_windows(self, caches: RowCaches, metadata: dict):
        """The window write rows and main lists of EVERY window layer of this step (this rank's stage) in one integer
        launch (``window_rows_lists``, csrc/paged.cu): the same values ``bind_window``'s torch glue computes per
        layer (``slot_rows`` of native's slot ids and window indices, -1 on invalid rows), bit for bit
        (tests/test_window_glue_gpu.py). The pattern of the GDN engine state's per-step rows (``_decode_rows``: one
        launch for every pool of the step). Returns ``{swa prefix: (view, rows [N], lists [N, W])}`` with the
        step-wide tensors, or None when the step is outside what the launch covers (the per-layer glue then runs)."""
        from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension
        ext = extension()
        if not hasattr(ext, "window_rows_lists"):
            return _window_glue_miss(1)
        N, W = self.tokens, self.cfg.sliding_window
        valid = self.batch.valid
        if valid is None or not valid.is_cuda or valid.dtype != torch.bool or not valid.is_contiguous() or valid.numel() != N:
            return _window_glue_miss(2)
        found = []
        for key, swa in metadata.items():
            if not key.endswith(".swa_cache"):
                continue
            layer = caches.context.get(key)
            cache = getattr(layer, "kv_cache", None)
            m = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
            if m is None or not isinstance(cache, torch.Tensor) or not cache.numel():
                continue
            found.append((int(m.group(1)), key, swa, cache))
        if not found or len(found) > WINDOW_GLUE_MAX_LAYERS:
            return _window_glue_miss(3)
        found.sort()
        nd, npf = int(found[0][2].num_decode_tokens), int(found[0][2].num_prefill_tokens)
        slots, dec, pre, per, offset, views = [], [], [], [], [], []
        for _, key, swa, cache in found:
            if (int(swa.num_decode_tokens), int(swa.num_prefill_tokens)) != (nd, npf):
                return _window_glue_miss(4)
            base = swa.decode_swa_indices          # [tokens, (1,) W] as native builds it
            sm = swa.slot_mapping[:N]
            if (base.dim() < 2 or base.shape[-1] != W or base.stride(-1) != 1 or not sm.is_contiguous()
                    or sm.numel() != N):
                return _window_glue_miss(5)
            if npf:
                p = swa.prefill_swa_indices[:npf].reshape(npf, W)
                if nd + npf > N or base.shape[0] < nd or p.stride(-1) != 1 or p.dtype != base.dtype:
                    return _window_glue_miss(6)
                base = base[:nd].reshape(nd, W)        # bind_window's torch.cat operands
                pre.append(p)
            else:
                # bind_window's as_strided rows of the decode buffer (every step row, its own row stride)
                base = base.as_strided((base.shape[0], W), (base.stride(0), 1), base.storage_offset())
                pre.append(None)
            view, off = record_rows(cache, WINDOW_BYTES)
            slots.append(sm)
            dec.append(base)
            per.append(cache.stride(0) // WINDOW_BYTES)
            offset.append(off)
            views.append(view)
        dev = valid.device
        rows = torch.empty((len(found), N), dtype=torch.int64, device=dev)
        lists = torch.empty((len(found), N, W), dtype=torch.int32, device=dev)
        ext.window_rows_lists(slots, dec, pre, per, offset, valid, nd, npf, SWA_BLOCK, rows, lists)
        self._win_rows, self._win_lists = rows, lists
        self._win_layer = {index: i for i, (index, *_rest) in enumerate(found)}
        return {key: (views[i], rows[i], lists[i]) for i, (_, key, _s, _c) in enumerate(found)}

    def decoder_windows(self, batch, sel, boundary: int) -> bool:
        """The decoder-row ``batch``'s (``PagedBatch.rows_subset(..., windows=False)``) window rows and main lists of
        every decoder layer ``>= boundary`` in one launch (``decoder_window_lists``): the rows ``sel`` of the step's
        (``_bind_windows``), and for the layers after the boundary the main lists truncated at ``batch.window_floor``
        (``paged.floor_window_lists``' integers); ``batch.lists_floored`` tells the layers not to truncate again.
        False (nothing written) without the step-wide windows."""
        if not self._win:
            return False
        from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension
        layers = sorted(i for i in self._win_layer if i >= boundary)
        if not layers:
            return False
        floor = batch.window_floor
        R, W = sel.numel(), self._win_lists.shape[-1]
        rows = torch.empty((len(layers), R), dtype=torch.int64, device=sel.device)
        lists = torch.empty((len(layers), R, W), dtype=torch.int32, device=sel.device)
        extension().decoder_window_lists(self._win_rows, self._win_lists, [self._win_layer[i] for i in layers],
                                         [int(floor is not None and i > boundary) for i in layers], sel,
                                         batch.tok_pos, floor, rows, lists)
        batch.window_rows = {i: rows[d] for d, i in enumerate(layers)}
        batch.window_lists = {i: lists[d] for d, i in enumerate(layers)}
        batch.lists_floored = True
        return True

    def bind_window(self, index: int, caches: RowCaches, metadata: dict) -> None:
        """Row ``index``'s window pool view, write rows and main list (shared by its sliding-window group)."""
        hit = self._win.get(caches.swa.prefix) if self._win else None
        if hit is not None:
            self.pools.window[index] = hit[0]
            self.batch.window_rows[index] = hit[1]
            self.batch.window_lists[index] = hit[2]
            return
        swa = metadata[caches.swa.prefix]
        cache = caches.swa.kv_cache
        view, offset = record_rows(cache, WINDOW_BYTES)
        key = (id(swa), offset)
        hit = self._windows.get(key)
        if hit is None or hit[0] is not swa:
            N, W = self.tokens, self.cfg.sliding_window
            per_block = cache.stride(0) // WINDOW_BYTES
            rows = slot_rows(swa.slot_mapping[:N].to(torch.int64), per_block, offset)
            nd, npf = int(swa.num_decode_tokens), int(swa.num_prefill_tokens)
            if npf == 0:
                base = swa.decode_swa_indices
                lists = base.as_strided((N, base.shape[-1]), (base.stride(0), 1), base.storage_offset())
            else:
                parts = [swa.decode_swa_indices[:nd].reshape(nd, W), swa.prefill_swa_indices[:npf].reshape(npf, W)]
                if nd + npf < N:
                    parts.append(torch.full((N - nd - npf, W), -1, dtype=torch.int32, device=cache.device))
                lists = torch.cat(parts)
            if lists.shape != (N, W):
                raise RuntimeError("sliding-window lists disagree with the step's tokens or the window")
            lists = torch.where(self.batch.valid[:, None], slot_rows(lists, per_block, offset), -1).to(torch.int32)
            hit = self._windows[key] = (swa, rows.contiguous(), lists.contiguous())
        self.pools.window[index] = view
        self.batch.window_rows[index] = hit[1]
        self.batch.window_lists[index] = hit[2]


_CURRENT: dict = {"step": None}

#: Route modes of the EP engine forwards that ran in Python (eager steps, warmups, captures; replays run
#: no Python): ``{"static": {tokens: forwards}, "captured": {tokens: forwards}, "counted": forwards}``.
ROUTE_STATS: dict = {"static": {}, "captured": {}, "counted": 0}


def record_routes(static: bool, tokens: int | None, capturing: bool) -> None:
    if not static:
        ROUTE_STATS["counted"] += 1
        return
    key = "captured" if capturing else "static"
    ROUTE_STATS[key][tokens] = ROUTE_STATS[key].get(tokens, 0) + 1


#: Program receipts of this rank's engine forwards. Forwards that run Python (eager steps, warmups, captures)
#: record what their rows executed: ``eager`` counts forwards per program, ``captured`` holds each graph
#: bucket's program and the weight version at capture. Replays run no Python: the model state records each
#: one before vLLM replays it (``replays``: per bucket, per weight version the graph read). ``weights`` counts
#: committed weight updates (``VllmModel.finish_weight_update``).
RECEIPTS: dict = {"eager": {}, "captured": {}, "replays": {}, "weights": 0}


def program_id(program: dict) -> str:
    return ",".join(f"{k}={program[k]}" for k in sorted(program))


def record_program(step: EngineStep) -> None:
    pid = program_id(step.program)
    if step.capturing:
        RECEIPTS["captured"][str(step.tokens)] = {"program": pid, "weights": RECEIPTS["weights"]}
    else:
        RECEIPTS["eager"][pid] = RECEIPTS["eager"].get(pid, 0) + 1


def record_replay(tokens: int) -> None:
    """A graph replay of bucket ``tokens`` with the current weights (the captured graph reads them in place)."""
    per = RECEIPTS["replays"].setdefault(str(tokens), {})
    version = str(RECEIPTS["weights"])
    per[version] = per.get(version, 0) + 1


def commit_weights() -> None:
    RECEIPTS["weights"] += 1


def current_step() -> EngineStep | None:
    """The step of the forward in progress (built by its first encoder block), else None."""
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available() or get_forward_context().attn_metadata is None:
        return None
    step = _CURRENT["step"]
    return step if step is not None and step.key is get_forward_context().attn_metadata else None


def engine_step(cfg, index: int, caches: RowCaches, seq) -> EngineStep | None:
    """The forward's ``EngineStep`` (built by its first row), or None without paged metadata (profiling)."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    context = get_forward_context()
    metadata = context.attn_metadata
    if metadata is None:
        return None
    if not isinstance(metadata, dict) or seq is None:
        raise RuntimeError("DeepSeek V4.1 rows need vLLM's per-layer attention metadata and the step positions")
    step = _CURRENT["step"]
    if step is None or step.key is not metadata or index <= step.last:
        swa = metadata[caches.swa.prefix]
        tokens = seq.positions.shape[-1]
        capturing = seq.positions.is_cuda and torch.cuda.is_current_stream_capturing()
        decode = int(swa.num_prefill_tokens) == 0 and int(getattr(swa, "max_decode_query_len", 1)) <= 1
        # a captured decode bucket only when vLLM's metadata covers every padded row (FULL); a padded PIECEWISE step
        # runs as the eager step of its real tokens
        graph = (caches.graphs and decode and tokens <= caches.max_capture
                 and real_tokens(swa, tokens) == tokens)
        if capturing and not graph:
            raise RuntimeError("DeepSeek V4.1 rows capture single-token decode batches of the capture sizes only")
        step = _CURRENT["step"] = EngineStep(cfg, caches, metadata, seq.positions, graph, capturing)
    step.last = index
    step.bind_window(index, caches, metadata)
    return step


__all__ = ["BLOCK_SIZE", "EngineStep", "KV_CACHE_DTYPE", "RECEIPTS", "RowCaches", "SCORE_BUDGET_KEY",
           "bind_row_caches", "cache_tensors", "commit_weights", "compressed_kv_source_class", "engine_step",
           "expand_pages", "graph_context_capacity", "index_pages", "native_attention", "program_id",
           "record_program", "record_replay", "record_rows", "slot_rows", "split_pages", "state_rows"]
