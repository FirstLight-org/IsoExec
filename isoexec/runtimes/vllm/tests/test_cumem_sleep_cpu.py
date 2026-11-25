"""CPU tests for releasing cached memory before vLLM sleep unmaps its pool."""

import sys
import types

import pytest
import torch


class _FakeAllocator:
    """``pointer_to_data`` = tracked allocations; ``cached`` = the ones the caching allocator holds free."""

    def __init__(self, live, cached):
        self.pointer_to_data = {ptr: "live" for ptr in live} | {ptr: "cached" for ptr in cached}
        self.cached = list(cached)
        self.unmapped = False
        self.double_unmaps = []
        self.offload_tags = "unset"

    def _python_free_callback(self, ptr):
        self.pointer_to_data.pop(ptr)
        if self.unmapped:
            self.double_unmaps.append(ptr)  # vLLM 0.28 on CUDA: cuMemUnmap of an unmapped range

    def empty_cache(self):
        for ptr in self.cached:
            self._python_free_callback(ptr)
        self.cached = []

    def sleep(self, offload_tags=None):
        self.offload_tags = offload_tags
        self.unmapped = True
        self.empty_cache()  # vLLM's own late gc.collect(); torch.cuda.empty_cache()


@pytest.fixture
def fake_cumem(monkeypatch):
    for name in ("vllm", "vllm.device_allocator"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = types.ModuleType("vllm.device_allocator.cumem")

    class CuMemAllocator(_FakeAllocator):
        pass

    module.CuMemAllocator = CuMemAllocator
    monkeypatch.setitem(sys.modules, "vllm.device_allocator.cumem", module)
    return module


def test_the_defect_the_patch_exists_for(fake_cumem):
    alloc = fake_cumem.CuMemAllocator(live=[1, 2], cached=[3])
    alloc.sleep()
    assert alloc.double_unmaps == [3]


def test_cached_pool_blocks_are_released_before_the_unmap(fake_cumem, monkeypatch, capsys):
    from isoexec.runtimes.vllm.vllm_patches import patch_vllm_cumem_sleep

    alloc = fake_cumem.CuMemAllocator(live=[1, 2], cached=[3, 4])
    monkeypatch.setattr(torch.cuda, "empty_cache", alloc.empty_cache)
    assert patch_vllm_cumem_sleep() is True
    assert patch_vllm_cumem_sleep() is True  # idempotent: the wrapper is not wrapped again
    alloc.sleep(offload_tags=("weights",))
    assert alloc.double_unmaps == [] and alloc.unmapped and alloc.offload_tags == ("weights",)
    assert sorted(alloc.pointer_to_data) == [1, 2]
    assert "released 2 cached pool allocation(s) BEFORE the sleep unmap (4 -> 2 tracked)" in capsys.readouterr().out
    # nothing cached: silent, and the stock sleep still runs
    quiet = fake_cumem.CuMemAllocator(live=[7], cached=[])
    monkeypatch.setattr(torch.cuda, "empty_cache", quiet.empty_cache)
    quiet.sleep()
    assert quiet.unmapped and capsys.readouterr().out == ""


def test_a_moved_allocator_surface_keeps_stock_sleep(fake_cumem, capsys):
    from isoexec.runtimes.vllm.vllm_patches import patch_vllm_cumem_sleep

    def sleep(self, level, offload_tags=None):  # a vLLM that changed the signature
        self.unmapped = True

    fake_cumem.CuMemAllocator.sleep = sleep
    assert patch_vllm_cumem_sleep() is False
    assert fake_cumem.CuMemAllocator.sleep is sleep
    assert "NOT INSTALLED" in capsys.readouterr().out
