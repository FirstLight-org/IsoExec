"""Discard-only registered state must never enter vLLM's restore snapshot."""
import unittest
from unittest.mock import patch

import torch

from isoexec.ops.gdn.shared import gdn_cpr_sleep as sleep


class BufferSnapshotTest(unittest.TestCase):
    def test_vllm_snapshot_sees_no_arena_view_and_the_model_gets_them_back(self):
        model = torch.nn.Module()
        model.core = torch.nn.Module()
        arena = torch.ones(32, dtype=torch.float32, device="cpu")
        model.core.register_buffer("arena", arena, persistent=False)
        model.core.register_buffer("offset_view", arena[8:16], persistent=False)
        model.core.register_buffer("metadata", torch.tensor([7, 9], dtype=torch.int64, device="cpu"))
        model.register_buffer("top", torch.tensor([1.5]))
        before = list(dict(model.named_buffers()))
        with patch.object(sleep, "_ARENAS", [(None, arena, arena.numel() * arena.element_size())]):
            with sleep._arena_buffers_hidden(model):
                # What vLLM's level-two sleep snapshots.
                self.assertEqual(set(dict(model.named_buffers())), {"core.metadata", "top"})
        self.assertEqual(list(dict(model.named_buffers())), before)
        self.assertEqual(model.core.offset_view.untyped_storage().data_ptr(), arena.untyped_storage().data_ptr())

    def test_without_arena_every_buffer_is_visible(self):
        model = torch.nn.Module()
        model.register_buffer("value", torch.tensor([2.5], dtype=torch.float32, device="cpu"), persistent=False)
        with patch.object(sleep, "_ARENAS", []), sleep._arena_buffers_hidden(model):
            self.assertEqual(set(dict(model.named_buffers())), {"value"})


def test_sleep_reset_discards_ownership_and_wake_zeros_only_the_arena():
    """A CPU record-pool fixture exercises the same reset used by the CPR engine."""
    from types import SimpleNamespace

    arena = torch.full((24,), 7, dtype=torch.uint8)
    layer = SimpleNamespace(
        _flat_arena=arena, ssm_state=arena.view(3, 8),
        slot2row=torch.tensor([0, 2, 1]), last_used=torch.tensor([0, 4, 5]),
        _clock=torch.tensor(5), _null32=torch.tensor(-1, dtype=torch.int32),
        _slot2row={11: 2, 13: 1}, _row2slot={2: 11, 1: 13},
        _slot_pending={11: 2}, _free=[], _free_set=set(),
        _row_pos={1: 64, 2: 37}, _driver_managed=True,
        _last_used_host=[0, 4, 5], _clock_host=5, _host_lru_valid=False,
    )
    sleep._reset_layer(layer, zero_arena=False)
    assert torch.equal(arena, torch.full_like(arena, 7)), "sleep must not write discarded pages"
    assert layer._slot2row == layer._row2slot == layer._slot_pending == layer._row_pos == {}
    assert layer._free == [1, 2] and layer._free_set == {1, 2}
    assert layer._driver_managed is False
    assert not layer.slot2row.any() and not layer.last_used.any() and layer._clock.item() == 0
    assert layer._last_used_host == [0, 0, 0] and layer._clock_host == 0 and layer._host_lru_valid
    assert layer._null32.item() == -1

    sleep._reset_layer(layer, zero_arena=True)
    assert not arena.any() and not layer.ssm_state.any()
    assert layer._null32.item() == -1


if __name__ == "__main__":
    unittest.main()
