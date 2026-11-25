"""cumem_asleep: whether a tensor's bytes are in a vLLM cumem allocation that is asleep (unmapped)."""
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("vllm.device_allocator.cumem")

from vllm.device_allocator import cumem  # noqa: E402

from isoexec.runtimes.vllm.weights_sleep import cumem_asleep  # noqa: E402


class _T:
    def __init__(self, ptr):
        self.ptr = ptr

    def data_ptr(self):
        return self.ptr


def test_reports_the_state_of_the_containing_allocation(monkeypatch):
    alloc = NS(pointer_to_data={1000: NS(handle=(0, 100), is_asleep=True, tag="kv_cache"),
                                5000: NS(handle=(0, 50), is_asleep=False, tag="weights")})
    monkeypatch.setattr(cumem.CuMemAllocator, "instance", alloc, raising=False)
    assert cumem_asleep(_T(1000)) and cumem_asleep(_T(1099))
    assert not cumem_asleep(_T(1100))          # past the allocation: not in the pool
    assert not cumem_asleep(_T(5010))          # awake allocation
    assert not cumem_asleep(_T(10))            # outside the pool


def test_no_sleep_mode_means_awake(monkeypatch):
    monkeypatch.setattr(cumem.CuMemAllocator, "instance", None, raising=False)
    assert not cumem_asleep(_T(1000))
