"""The tensor-map host path establishes the input device's context on fresh threads."""

import ctypes
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from isoexec.core.arch import live_device_arch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or live_device_arch() != 'sm100', reason='requires SM100 through a GPU lease'
)


def test_general_gemm_from_thread_without_current_driver_context():
    from isoexec.ops.mm.mm_sm100.general import ops
    from isoexec.ops.mm.mm_sm100.general._extension import extension

    extension()  # Compile/load on the caller before starting the fresh worker thread.
    a = torch.randn(37, 128, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(128, 256, device='cuda', dtype=torch.bfloat16)
    driver = ctypes.CDLL('libcuda.so.1')
    driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    driver.cuCtxGetCurrent.restype = ctypes.c_int

    def work():
        current = ctypes.c_void_p()
        assert driver.cuCtxGetCurrent(ctypes.byref(current)) == 0
        assert current.value is None, 'the regression must start without a current driver context'
        result = ops.mm(a, b)
        assert driver.cuCtxGetCurrent(ctypes.byref(current)) == 0
        assert current.value is not None
        return result

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(work).result()
    assert torch.equal(result, ops.mm(a, b))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires a two-GPU lease')
def test_general_device_guard_restores_original_device():
    from isoexec.ops.mm.mm_sm100.general import ops

    with torch.cuda.device(1):
        a = torch.randn(37, 128, device='cuda:1', dtype=torch.bfloat16)
        b = torch.randn(128, 256, device='cuda:1', dtype=torch.bfloat16)
        expected = ops.mm(a, b)
    with torch.cuda.device(0):
        result = ops.mm(a, b)
        assert torch.cuda.current_device() == 0
    assert result.device == a.device
    assert torch.equal(result, expected)
