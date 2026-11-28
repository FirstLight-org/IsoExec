"""Native SM100 DeepSeek V4.1 attention composition: forward, decode and paged caches, BF16-actor backward.

The recipe identifiers are in ``spec``; the numerical program of the row kernels is ``csrc/numerics.cuh``.
"""
from .spec import BACKWARD_RECIPE, RECIPE

NATIVE_QUALIFIED = False

__all__ = ["RECIPE", "BACKWARD_RECIPE", "NATIVE_QUALIFIED"]
