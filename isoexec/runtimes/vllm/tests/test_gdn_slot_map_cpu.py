"""Resolved cache geometry must be reserved before its map can enter a graph."""

from types import SimpleNamespace

import pytest
import torch

from isoexec.ops.gdn import RecurrentGDN
from isoexec.runtimes.vllm.gdn_slot_map import wrap_cache_initialization


def make_pool(monkeypatch):
    monkeypatch.setenv("ISOEXEC_GDN_SLOT_MAP_SIZE", "4")
    owner = torch.nn.Module()
    owner.register_parameter("conv", torch.nn.Parameter(torch.zeros((6, 3), dtype=torch.bfloat16, device="cpu")))
    owner.register_parameter("a", torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32, device="cpu")))
    owner.register_parameter("dt", torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32, device="cpu")))
    pool = RecurrentGDN(capacity=2, conv_weight=owner.conv, conv_bias=None, A_log=owner.a, dt_bias=owner.dt,
                        num_k_heads=1, head_k_dim=2, num_v_heads=1, head_v_dim=2, device="cpu",
                        on_alloc=lambda name, tensor: owner.register_buffer(name, tensor, persistent=False))
    return owner, pool


def test_growth_preserves_device_mappings_host_outbox_and_registration(monkeypatch):
    owner, pool = make_pool(monkeypatch)
    row = pool._assign(2)
    pool.flush_slot_map()
    pending_row = pool._assign(1)
    pool.slot2row[1] = 0
    pool._set_slot(1, pending_row)
    old = pool.slot2row
    assert pool.finalize_slot_map(17) == old.numel() * old.element_size()
    assert owner.slot2row is pool.slot2row
    assert pool.slot2row.data_ptr() != old.data_ptr()
    assert pool.slot2row[2].item() == row
    assert pool.slot2row.numel() == 18
    assert torch.count_nonzero(pool.slot2row[4:]).item() == 0
    assert pool.flush_slot_map() == 1
    assert pool.slot2row[1].item() == pending_row
    pointer = pool.slot2row.data_ptr()
    assert pool.finalize_slot_map(17) == 0
    assert pool.finalize_slot_map(5) == 0
    with pytest.raises(RuntimeError, match="rebuild the engine"):
        pool.finalize_slot_map(18)
    assert pool.slot2row.data_ptr() == pointer
    assert "slot2row" not in owner.state_dict()


def test_no_growth_also_freezes_the_graph_address(monkeypatch):
    _, pool = make_pool(monkeypatch)
    pointer = pool.slot2row.data_ptr()
    assert pool.finalize_slot_map(3) == 0
    assert pool.slot2row.data_ptr() == pointer
    with pytest.raises(RuntimeError, match="finalized"):
        pool.finalize_slot_map(5)


def test_original_cache_init_observes_final_map_profiling_does_not_freeze(monkeypatch):
    _, pool = make_pool(monkeypatch)
    from isoexec.ops.gdn import gdn_cpr_state

    monkeypatch.setattr(gdn_cpr_state, "CPR_LAYERS", [pool])
    calls = []
    allocation_context = object()

    def initialize(runner, config, is_profiling=False, kv_cache_allocation_context=None):
        calls.append(
            (
                config.num_blocks,
                pool.slot2row.numel(),
                pool._slot_map_finalized,
                is_profiling,
                kv_cache_allocation_context,
            )
        )
        return "native result"

    wrapped = wrap_cache_initialization(initialize)
    runner = SimpleNamespace()
    assert wrapped(runner, SimpleNamespace(num_blocks=1), is_profiling=True) == "native result"
    assert (
        wrapped(
            runner,
            SimpleNamespace(num_blocks=12),
            kv_cache_allocation_context=allocation_context,
        )
        == "native result"
    )
    assert calls == [
        (1, 4, False, True, None),
        (12, 13, True, False, allocation_context),
    ]
    assert runner._isoexec_slot_map_receipt == {
        "num_blocks": 12, "map_entries": [13], "initialization_copied_bytes": 32,
    }
    with pytest.raises(RuntimeError, match="finalized"):
        wrapped(runner, SimpleNamespace(num_blocks=13))
    assert len(calls) == 2


def test_last_live_block_cannot_alias_positive_padding(monkeypatch):
    _, pool = make_pool(monkeypatch)
    pool.finalize_slot_map(17)
    row = pool._assign(16)
    pool.flush_slot_map()
    query = torch.tensor([16, 17, 999], dtype=torch.int64, device="cpu")
    assert pool._rows(query).tolist() == [row, 0, 0]
    for operation in (lambda: pool._assign(17), lambda: pool.rebind_slot(16, 17)):
        with pytest.raises(RuntimeError, match="padding"):
            operation()
    assert pool._rows(query).tolist() == [row, 0, 0]
