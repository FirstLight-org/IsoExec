"""Build the sm90 mm extension lazily (CUTLASS 3.x / CuTe, sm90a); importing this builds nothing.

Build directory: ``ISOEXEC_MM_SM90_BUILD_DIR`` (a CUTLASS-revision tag is appended), else PyTorch's
extension cache keyed by the source bytes. Never inside the repository.
"""

from functools import lru_cache

from isoexec.ops.build_cache import NUMERICS_FLAGS, build_sm90_extension

SOURCES = (
    "mm_sm90_bind.cu",
    *(f"mm_sm90_tu{i}.cu" for i in range(13)),
)

#: Everything beyond the shared sm90 profile (which already pins ``NUMERICS_FLAGS``: no FTZ, no FMA
#: contraction -- the tree adds are separate fp32 adds).
EXTRA_CUDA_FLAGS = ("-DNDEBUG", "-Xptxas=-warn-spills", "-lineinfo", "-DCUTLASS_ENABLE_GDC_FOR_SM90")

assert "--fmad=false" in NUMERICS_FLAGS and "--ftz=false" in NUMERICS_FLAGS


@lru_cache(None)
def extension():
    """The loaded ``isoexec_mm_sm90_cuda`` module; compiled on the first call, never on import."""
    return build_sm90_extension(
        "mm",
        __file__,
        SOURCES,
        include_csrc=True,
        include_cccl=True,
        include_cutlass=True,
        extra_cuda_flags=EXTRA_CUDA_FLAGS,
    )
