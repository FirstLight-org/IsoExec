"""Build the sm90 projection-backward extension (wgrad accumulate, split-column assembly) lazily (CUTLASS 3.x / CuTe, sm90a); importing builds nothing.

Build directory: ``ISOEXEC_MM_BWD_SM90_BUILD_DIR`` (a CUTLASS-revision tag is appended), else PyTorch's
extension cache keyed by the source bytes. Never inside the repository.
"""

from functools import lru_cache

from isoexec.ops.build_cache import build_sm90_extension

SOURCES = ("bind.cpp", "wgrad.cu", "wgrad_grouped.cu", "columns.cu")

EXTRA_CUDA_FLAGS = ("-DNDEBUG", "-Xptxas=-warn-spills", "-lineinfo")


@lru_cache(None)
def extension():
    """The loaded ``isoexec_mm_bwd_sm90_cuda`` module; compiled on the first call, never on import."""
    return build_sm90_extension(
        "mm_bwd",
        __file__,
        SOURCES,
        include_csrc=True,
        include_cccl=True,
        include_cutlass=True,
        extra_cuda_flags=EXTRA_CUDA_FLAGS,
    )
