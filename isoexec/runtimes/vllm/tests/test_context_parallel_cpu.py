"""CPU contracts for IsoExec's Model Runner V2 PCP/DCP attention layout."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isoexec.runtimes.vllm import context_parallel as cp


def _vllm_local_slot(position: int, *, block_row, block_size: int, dcp_size: int, dcp_rank: int, interleave: int):
    """vLLM 0.33 ``_compute_slot_mappings_kernel`` for one DCP-sharded position (None if not local)."""
    virtual = block_size * dcp_size
    block, offset = divmod(position, virtual)
    if (offset // interleave) % dcp_size != dcp_rank:
        return None
    local = block * block_size + (offset // (interleave * dcp_size)) * interleave + offset % interleave
    return block_row[local // block_size] * block_size + local % block_size


@pytest.mark.parametrize(("dcp_size", "interleave"), [(2, 1), (2, 2), (4, 1)])
def test_owner_layout_inverts_the_vllm_dcp_slot_mapping(dcp_size, interleave):
    block_size, capacity = 4, 4 * dcp_size * 3
    row = list(range(10, 10 + capacity // block_size))
    owner, local = cp.owner_layout(capacity, dcp_size=dcp_size, block_size=block_size, interleave=interleave,
                                   device="cpu")
    for position in range(capacity):
        rank = int(owner[position])
        slot = _vllm_local_slot(position, block_row=row, block_size=block_size, dcp_size=dcp_size, dcp_rank=rank,
                                interleave=interleave)
        assert slot == row[int(local[position]) // block_size] * block_size + int(local[position]) % block_size
        others = [r for r in range(dcp_size) if r != rank]
        assert all(_vllm_local_slot(position, block_row=row, block_size=block_size, dcp_size=dcp_size, dcp_rank=r,
                                    interleave=interleave) is None for r in others)


def test_gather_full_cache_restores_token_order_from_dcp_shards():
    dcp_size, block_size, heads, dim, interleave = 2, 4, 2, 3, 1
    lengths = [11, 6]
    capacity = 16  # multiple of block_size * dcp_size
    blocks = 12
    rows = [[1, 2, 3, 0], [5, 7, 0, 0]]  # per-request block tables, identical on both ranks
    caches = [torch.full((blocks, block_size, heads, dim), -1.0) for _ in range(dcp_size)]
    truth = [torch.randn(length, heads, dim) for length in lengths]
    for request, (row, values) in enumerate(zip(rows, truth)):
        for position in range(lengths[request]):
            for rank in range(dcp_size):
                slot = _vllm_local_slot(position, block_row=row, block_size=block_size, dcp_size=dcp_size,
                                        dcp_rank=rank, interleave=interleave)
                if slot is not None:
                    caches[rank].view(-1, heads, dim)[slot] = values[position]
    block_rows = torch.tensor(rows, dtype=torch.int32)
    # Every rank contributes its shard; the gather concatenates them in rank order.
    local_capacity = capacity // dcp_size
    positions = torch.arange(local_capacity)
    per_rank = [caches[r][block_rows.long()[:, positions // block_size], (positions % block_size).expand(2, -1)]
                for r in range(dcp_size)]
    pages = cp.gather_full_cache(caches[0], block_rows, capacity, all_gather=lambda t: torch.cat(per_rank, dim=0),
                                 dcp_size=dcp_size, interleave=interleave)
    full = pages.view(len(rows), capacity, heads, dim)
    for request, values in enumerate(truth):
        assert torch.equal(full[request, : lengths[request]], values)


def test_publish_global_batch_maps_local_rows_to_global_requests():
    tables = SimpleNamespace(block_tables=[SimpleNamespace(gpu=torch.arange(4 * 3, dtype=torch.int32).view(4, 3))])
    global_batch = SimpleNamespace(
        idx_mapping_np=np.array([3, 1]), idx_mapping=torch.tensor([3, 1]), num_reqs=2,
        seq_lens=torch.tensor([9, 5, 0], dtype=torch.int32), seq_lens_cpu_upper_bound=torch.tensor([9, 5, 0]),
    )
    local_batch = SimpleNamespace(idx_mapping_np=np.array([1, 3, 3]))  # decode B, then A's two segments
    manager = SimpleNamespace(_global_batch=global_batch, _local_batch=local_batch, _block_tables=tables)
    cp.publish_global_batch(manager)
    batch = cp._CURRENT
    assert batch.local_to_global.tolist() == [1, 0, 0]
    assert batch.block_tables[0].tolist() == [[9, 10, 11], [3, 4, 5]]
    assert batch.seq_lens.tolist() == [9, 5] and batch.max_seq_len == 9 and batch.num_local_rows == 3


def test_settings_are_none_without_context_parallelism():
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1,
                                        cp_kv_cache_interleave_size=1),
        model_config=SimpleNamespace(max_model_len=2048),
    )
    assert cp.Settings.from_config(config) is None
    config.parallel_config.prefill_context_parallel_size = 2
    config.parallel_config.decode_context_parallel_size = 2
    assert cp.Settings.from_config(config) == cp.Settings(2, 2, 1, 2048)
