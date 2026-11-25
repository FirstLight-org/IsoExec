from __future__ import annotations

NON_ACCELERATOR_ARCH: str = "cpu"


def is_accelerator_arch(arch: str) -> bool:
    """True iff ``arch`` names a real accelerator, i.e. non-empty and not the sentinel."""
    return bool(arch) and arch != NON_ACCELERATOR_ARCH


def _capability_over_nvml_ctypes(index: int) -> tuple[int, int] | None:
    """Query compute capability through the driver library when pynvml is unavailable."""
    import ctypes

    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        if lib.nvmlInit_v2() != 0:
            return None
        try:
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(handle)) != 0:
                return None
            major, minor = ctypes.c_int(), ctypes.c_int()
            if lib.nvmlDeviceGetCudaComputeCapability(handle, ctypes.byref(major), ctypes.byref(minor)) != 0:
                return None
            return int(major.value), int(minor.value)
        finally:
            lib.nvmlShutdown()
    except Exception:
        return None


def _capability_without_cuda_init(torch) -> tuple[int, int] | None:
    """Read compute capability through NVML without creating a CUDA context."""
    try:
        from torch.cuda import _get_nvml_device_index
    except Exception:
        return None
    try:
        index = _get_nvml_device_index(0)
    except Exception:
        return None
    try:
        import pynvml
    except Exception:
        return _capability_over_nvml_ctypes(index)
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        return int(major), int(minor)
    except Exception:
        return _capability_over_nvml_ctypes(index)


def _arch_tag(major: int, minor: int) -> str:
    if (major, minor) == (9, 0):
        return "sm90"
    if major == 10:
        return "sm100"
    return f"sm{major}{minor}"


def detect_arch() -> str:
    try:
        import torch  # deferred: core stays importable in an environment without torch
    except Exception:
        return NON_ACCELERATOR_ARCH
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        return NON_ACCELERATOR_ARCH
    if not cuda_available:
        return NON_ACCELERATOR_ARCH
    cc = _capability_without_cuda_init(torch)
    if cc is not None:
        return _arch_tag(*cc)
    return _arch_tag(*torch.cuda.get_device_capability())


def live_device_arch() -> str:
    """The arch tag of the CUDA device this process would use right now, or the sentinel."""
    try:
        import torch
    except Exception:
        return NON_ACCELERATOR_ARCH
    if not torch.cuda.is_available():
        return NON_ACCELERATOR_ARCH
    return _arch_tag(*torch.cuda.get_device_capability())


ARCH: str = detect_arch()

