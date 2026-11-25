"""Retired. The FP8-derived clamped-SwiGLU expert kernels were removed (BF16-only campaign); the V4.1 experts'
BF16 rows and backward live in ``moe/deepseek_v41_sm100/rows.py``.

What remains here is transitional and owned elsewhere:

* ``packing``: an import alias of ``moe/deepseek_v41_sm100/packing`` for callers not yet repointed
  (``models/deepseek_v4_1_flash/block.py``, owned by the hang lane).
* ``csrc/clamped_numerics.cuh``: included by ``collectives/detreduce_sm100/csrc/detreduce.cu`` (hang lane),
  whose GLM FP8 EP-prep kernel uses its activation quantizer. It goes when that include goes.
"""
