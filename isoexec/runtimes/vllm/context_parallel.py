"""Prefill and decode context parallelism for IsoExec softmax attention on vLLM Model Runner V2.

Status: NOT ADMITTED. Engine context parallelism is refused for IsoExec models while vLLM builds its
engine configuration (``validation.install_context_parallel_refusal``, ``register_native``), so
``Settings.from_config`` is always ``None`` at binding time and nothing below runs; ``install_admission``
is not installed. Kept as the starting point for engine CP. Known gap found on GPU: under PCP, V2's
MRoPE positions (``RopeState.prepare_positions``) of a local batch restart every chunk at the request's
start, so any MRoPE model would need the global positions gathered to the local rows first.

vLLM 0.33 implements Model Runner V2 PCP (and PCP with DCP) for MLA models only. This module
implements it for IsoExec's GQA/MHA attention with the existing numerical path:

- PCP: V2's ``PCPManager`` hands each rank its DualChunkSwap query segments and restores the
  hidden states before sampling. The KV write uses V2's gathered slot mapping, so the K/V of every
  PCP rank's rows is all-gathered before ``unified_kv_cache_update``.
- DCP: the persistent KV cache is sharded by token over the DCP group with vLLM's own layout
  (``BlockTables.compute_slot_mappings``). Each rank writes only the tokens it owns.
- Attention: each request's shards are all-gathered over the DCP group back into original token
  order in a temporary paged buffer with the cache's page size, and the arch-selected paged core
  runs over it with the local query rows. No partial softmax is merged, so each query row is
  computed by the same kernel over the same keys in the same order as without context
  parallelism.

Prefill steps run eagerly and read the global batch published by ``PCPManager.partition_batch``;
dummy batches (staged by ``prepare_inputs_for_capture_or_profile``, identical on every rank) clear it
and use their own metadata.
Decode-only steps replicate every request on every rank (DCP shards decode KV), so local rows are
the global requests in the same order and the path uses only captured attention metadata and
fixed ``max_model_len`` shapes, which keeps FULL_DECODE_ONLY graphs valid.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

#: Host counters for engagement evidence; captured replays count calls only.
STATS: dict[str, int] = {"calls": 0, "prefill_calls": 0, "decode_calls": 0, "dummy_calls": 0, "captured_calls": 0,
                         "kv_rows_written": 0, "kv_rows_read": 0}


@dataclass(frozen=True)
class Settings:
    """Resolved context-parallel layout, captured when the attention core is bound."""

    pcp_size: int
    dcp_size: int
    interleave: int
    max_model_len: int

    @classmethod
    def from_config(cls, vllm_config) -> "Settings | None":
        parallel = vllm_config.parallel_config
        pcp = int(getattr(parallel, "prefill_context_parallel_size", 1))
        dcp = int(getattr(parallel, "decode_context_parallel_size", 1))
        if pcp == 1 and dcp == 1:
            return None
        return cls(pcp, dcp, int(parallel.cp_kv_cache_interleave_size), int(vllm_config.model_config.max_model_len))


@dataclass(frozen=True)
class GlobalBatch:
    """The step's global requests as every PCP rank sees them, published before the forward."""

    block_tables: tuple[torch.Tensor, ...]  # per KV group, [num_global, max_blocks]
    seq_lens: torch.Tensor  # [num_global] int32
    local_to_global: torch.Tensor  # [num_local_rows] int64
    num_local_rows: int
    max_seq_len: int
    serial: int


_CURRENT: GlobalBatch | None = None
_SERIAL = 0


def publish_global_batch(manager: Any) -> None:
    """Record the global batch of the step ``manager.partition_batch`` just partitioned."""
    global _CURRENT, _SERIAL
    global_batch, local_batch = manager._global_batch, manager._local_batch
    tables = manager._block_tables
    if global_batch is None or local_batch is None or tables is None:
        _CURRENT = None
        return
    index = {int(state): row for row, state in enumerate(global_batch.idx_mapping_np.tolist())}
    local = [index[int(state)] for state in local_batch.idx_mapping_np.tolist()]
    device = global_batch.idx_mapping.device
    _SERIAL += 1
    _CURRENT = GlobalBatch(
        block_tables=tuple(table.gpu.index_select(0, global_batch.idx_mapping) for table in tables.block_tables),
        seq_lens=global_batch.seq_lens[: global_batch.num_reqs].to(torch.int32),
        local_to_global=torch.tensor(local, dtype=torch.int64, device=device),
        num_local_rows=len(local),
        max_seq_len=int(global_batch.seq_lens_cpu_upper_bound[: global_batch.num_reqs].max().item()),
        serial=_SERIAL,
    )


def clear_global_batch() -> None:
    """Forget the published batch: the next forward is a dummy batch staged without partition_batch."""
    global _CURRENT
    _CURRENT = None


def owner_layout(capacity: int, *, dcp_size: int, block_size: int, interleave: int, device) -> tuple[torch.Tensor, ...]:
    """For global positions ``[0, capacity)``: the owning DCP rank and the owner's local position.

    The inverse of vLLM's DCP slot mapping: within each virtual block of ``block_size * dcp_size``
    positions, runs of ``interleave`` positions rotate over the ranks, and each rank packs its runs
    into one ``block_size`` page.
    """
    if capacity % (block_size * dcp_size):
        raise ValueError(f"capacity {capacity} must be a multiple of block_size*dcp_size={block_size * dcp_size}")
    positions = torch.arange(capacity, dtype=torch.int64, device=device)
    virtual = block_size * dcp_size
    block, offset = torch.div(positions, virtual, rounding_mode="floor"), torch.remainder(positions, virtual)
    owner = torch.remainder(torch.div(offset, interleave, rounding_mode="floor"), dcp_size)
    local = (block * block_size + torch.div(offset, interleave * dcp_size, rounding_mode="floor") * interleave
             + torch.remainder(offset, interleave))
    return owner, local


def gather_full_cache(cache: torch.Tensor, block_rows: torch.Tensor, capacity: int, *, all_gather,
                      dcp_size: int, interleave: int) -> torch.Tensor:
    """Pages ``[rows * capacity / block_size, block_size, H, D]`` holding each row's KV in token order.

    ``cache`` is this rank's ``[blocks, block_size, H, D]`` shard and ``block_rows`` the rows'
    block tables, identical on every DCP rank. ``all_gather(tensor)`` concatenates every DCP
    rank's tensor along dim 0 in rank order (vLLM's capture-safe group collective).
    """
    rows, block_size = int(block_rows.shape[0]), int(cache.shape[1])
    local_capacity = capacity // dcp_size
    local_positions = torch.arange(local_capacity, dtype=torch.int64, device=cache.device)
    pages = block_rows.long().index_select(1, torch.div(local_positions, block_size, rounding_mode="floor"))
    shard = cache[pages, torch.remainder(local_positions, block_size).expand(rows, -1)]  # [rows, Nloc, H, D]
    shard = shard.contiguous()
    gathered = all_gather(shard).view((dcp_size,) + tuple(shard.shape))
    owner, local = owner_layout(capacity, dcp_size=dcp_size, block_size=block_size, interleave=interleave,
                                device=cache.device)
    full = gathered.permute(1, 0, 2, 3, 4)[:, owner, local]  # [rows, capacity, H, D]
    return full.reshape(rows * capacity // block_size, block_size, *full.shape[2:])


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def run(attn, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, output: torch.Tensor, *,
        paged_core, settings: Settings) -> None:
    """IsoExec attention for one layer under PCP and/or DCP; writes ``output`` for the local rows."""
    from vllm.distributed import get_dcp_group, get_pcp_group
    from vllm.model_executor.layers.attention.attention import get_attention_context, unified_kv_cache_update

    from isoexec.ops.attention.paged_kv import split_paged_kv_cache

    pcp_size, dcp_size = settings.pcp_size, settings.dcp_size
    capturing = query.is_cuda and torch.cuda.is_current_stream_capturing()
    STATS["calls"] += 1
    STATS["captured_calls"] += int(capturing)
    # 1. KV write: V2's slot mapping covers every PCP rank's rows, rank-major and padded alike.
    if pcp_size > 1:
        pcp = get_pcp_group()
        key_rows, value_rows = pcp.all_gather(key.contiguous(), dim=0), pcp.all_gather(value.contiguous(), dim=0)
    else:
        key_rows, value_rows = key, value
    if not attn.attn_backend.forward_includes_kv_cache_update:
        if attn.use_direct_call:
            unified_kv_cache_update(key_rows, value_rows, attn.layer_name)
        else:
            from vllm.model_executor.layers.attention.attention import (  # pyright: ignore[reportPrivateImportUsage]
                _encode_layer_name,
            )

            torch.ops.vllm.unified_kv_cache_update(key_rows, value_rows, _encode_layer_name(attn.layer_name))
    md, _layer, kv_cache, slot_mapping = get_attention_context(attn.layer_name)
    if md is None:  # metadata-free dummy run: shapes only, identical on every rank
        output.zero_()
        return
    if not getattr(md, "causal", True) or getattr(md, "use_cascade", False):
        raise RuntimeError("context-parallel IsoExec attention requires causal, non-cascade metadata")
    key_cache, value_cache = split_paged_kv_cache(kv_cache, num_kv_heads=attn.num_kv_heads, head_size=attn.head_size)
    block_size = int(key_cache.shape[1])
    dcp = get_dcp_group()
    interleave = settings.interleave
    multiple = block_size * dcp_size * interleave
    n = int(md.num_actual_tokens)
    decode_only = int(md.max_query_len) == 1
    if decode_only:
        # Every request is a local row on every rank, in global order (DCP replicates decodes).
        block_rows = md.block_table
        capacity = _round_up(settings.max_model_len, multiple)
        row_pages = torch.arange(block_rows.shape[0], dtype=torch.int64, device=query.device)
        STATS["decode_calls"] += 1
    elif _CURRENT is None:
        # A dummy batch (FlashInfer autotune, warmup): staged identically on every rank without
        # partition_batch, so its local rows are the same on every rank. Outputs are discarded.
        block_rows = md.block_table
        capacity = _round_up(int(md.max_seq_len), multiple)
        row_pages = torch.arange(block_rows.shape[0], dtype=torch.int64, device=query.device)
        STATS["dummy_calls"] += 1
    else:
        batch = _CURRENT
        rows = int(md.query_start_loc.shape[0]) - 1
        if batch.num_local_rows > rows:
            raise RuntimeError("context-parallel prefill attention: the published global batch has more local rows "
                               f"({batch.num_local_rows}) than this step's metadata ({rows})")
        block_rows = batch.block_tables[_kv_group(attn)]
        capacity = _round_up(batch.max_seq_len, multiple)
        row_pages = torch.zeros(rows, dtype=torch.int64, device=query.device)
        row_pages[: batch.num_local_rows] = batch.local_to_global
        STATS["prefill_calls"] += 1
    gather = (lambda tensor: dcp.all_gather(tensor, dim=0)) if dcp_size > 1 else (lambda tensor: tensor)
    full_k = gather_full_cache(key_cache, block_rows, capacity, all_gather=gather, dcp_size=dcp_size,
                               interleave=interleave)
    full_v = gather_full_cache(value_cache, block_rows, capacity, all_gather=gather, dcp_size=dcp_size,
                               interleave=interleave)
    pages_per_row = capacity // block_size
    table = (row_pages.view(-1, 1) * pages_per_row
             + torch.arange(pages_per_row, dtype=torch.int64, device=query.device)).to(torch.int32)
    paged_core(query[:n], full_k, full_v, cu_seqlens_q=md.query_start_loc, seqused_k=md.seq_lens,
               block_table=table, max_seqlen_q=int(md.max_query_len), max_seqlen_k=int(md.max_seq_len),
               scale=float(attn.impl.scale), out=output[:n])
    if n < output.shape[0]:
        output[n:].zero_()
    if not capturing and os.environ.get("ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR"):
        written = int((slot_mapping >= 0).sum().item()) if isinstance(slot_mapping, torch.Tensor) else -1
        STATS["kv_rows_written"] += written
        STATS["kv_rows_read"] += int(block_rows.shape[0]) * (capacity // dcp_size)
        if STATS["calls"] & (STATS["calls"] - 1) == 0:
            pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
            print(f"[ISOEXEC-CP-ATTN] pid={os.getpid()} pcp_rank={pcp_rank} dcp_rank={dcp.rank_in_group} "
                  f"stats={STATS}", flush=True)


def _kv_group(attn) -> int:
    """This layer's KV cache group; IsoExec attention layers share one full-attention group."""
    return int(getattr(attn, "_isoexec_kv_group", 0))


def install_admission(native_architectures: tuple[str, ...]) -> None:
    """Admit IsoExec native models to V2 PCP/DCP; every other model keeps vLLM's own checks.

    ``PCPManager.validate_config`` admits only MLA models and ``ModelConfig`` admits GQA DCP only
    when TP exceeds the KV head count, because vLLM's own GQA DCP merges per-rank partial
    attention. IsoExec gathers the complete KV instead, so neither restriction applies to it.
    The remaining V2 restrictions (no speculative decoding, LoRA or multimodal inputs, and graphs
    only for decode-only routines) stay enforced.
    """
    from vllm.config.model import ModelConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

    def native(model_config) -> bool:
        return bool(set(getattr(model_config, "architectures", None) or ()) & set(native_architectures))

    verify = ModelConfig.verify_with_parallel_config
    if not getattr(verify, "_isoexec_cp", False):
        def verify_with_parallel_config(self, parallel_config):
            if not native(self) or parallel_config.decode_context_parallel_size == 1:
                return verify(self, parallel_config)
            dcp = parallel_config.decode_context_parallel_size
            parallel_config.decode_context_parallel_size = 1
            try:
                return verify(self, parallel_config)
            finally:
                parallel_config.decode_context_parallel_size = dcp

        setattr(verify_with_parallel_config, "_isoexec_cp", True)
        ModelConfig.verify_with_parallel_config = verify_with_parallel_config

    validate = PCPManager.__dict__["validate_config"].__func__
    if not getattr(validate, "_isoexec_cp", False):
        def validate_config(vllm_config, supports_mm_inputs):
            if not native(vllm_config.model_config):
                return validate(vllm_config, supports_mm_inputs)
            if supports_mm_inputs or vllm_config.lora_config is not None or vllm_config.speculative_config is not None:
                raise NotImplementedError("IsoExec V2 PCP excludes multimodal inputs, LoRA and speculative decoding")
            mode = vllm_config.compilation_config.cudagraph_mode
            if mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY):
                raise NotImplementedError("IsoExec V2 PCP requires cudagraph_mode NONE or FULL_DECODE_ONLY")

        setattr(validate_config, "_isoexec_cp", True)
        setattr(PCPManager, "validate_config", staticmethod(validate_config))

    stage_dummy = PCPManager.prepare_inputs_for_capture_or_profile
    if not getattr(stage_dummy, "_isoexec_cp", False):
        def prepare_inputs_for_capture_or_profile(self, input_batch):
            clear_global_batch()
            return stage_dummy(self, input_batch)

        setattr(prepare_inputs_for_capture_or_profile, "_isoexec_cp", True)
        setattr(PCPManager, "prepare_inputs_for_capture_or_profile", prepare_inputs_for_capture_or_profile)

    partition = PCPManager.partition_batch
    if not getattr(partition, "_isoexec_cp", False):
        def partition_batch(self, input_batch, batch_desc):
            local = partition(self, input_batch, batch_desc)
            publish_global_batch(self)
            return local

        setattr(partition_batch, "_isoexec_cp", True)
        setattr(PCPManager, "partition_batch", partition_batch)


__all__ = ["STATS", "GlobalBatch", "Settings", "clear_global_batch", "gather_full_cache", "install_admission", "owner_layout",
           "publish_global_batch", "run"]
