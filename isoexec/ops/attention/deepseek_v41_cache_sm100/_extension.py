"""Build the row-kernel extension under the shared SM100 numerics profile.

``build_cache.NUMERICS_FLAGS`` (``--ftz=false --fmad=false``) with nothing appended: the same
profile ``channel_pool_sm100`` and ``quant/checkpoint_sm100`` are built with, which is what makes
the fused pool and quantizer scalar graphs bit-equal to theirs (tests/test_rows_gpu.py proves it).
Compile OUTSIDE a GPU lease; importing this module builds nothing.
"""
from __future__ import annotations

from functools import lru_cache

from isoexec.ops.build_cache import NUMERICS_FLAGS, build_sm100_extension

assert NUMERICS_FLAGS == ("--ftz=false", "--fmad=false"), "the shared numerics profile moved"

SOURCES = ("rows.cu", "rows_bwd.cu", "paged.cu")


@lru_cache(None)
def extension():
    # rms_quant_5120 fuses norms_sm100's RMSNorm: its segment length comes from there (csrc/norm_segments.cuh)
    from isoexec.ops.norms.norms_sm100.schedule import cuda_defines

    return build_sm100_extension("deepseek_v41_rows", __file__, SOURCES, include_csrc=True,
                                 extra_cuda_flags=cuda_defines())
