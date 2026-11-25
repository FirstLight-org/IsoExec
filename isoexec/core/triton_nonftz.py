"""Triton primitives that preserve subnormals and required rounding where libdevice does not. Kept
out of core package initialization to avoid importing Triton eagerly.
"""

from __future__ import annotations

try:
    import triton  # noqa: F401
    import triton.language as tl
    from triton.language.extra import libdevice

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


# rsqrt flushes subnormal inputs; clamp to the smallest normal fp32 value. Replacing it with 1 /
# sqrt(x) would double-round.
SMALLEST_NORMAL_F32 = 1.17549435e-38
# 2**-24: below this, log1p(x) == x after fp32 rounding (the x^2/2 term cannot reach the mantissa).
_LOG1P_LINEAR_BELOW = 5.9604645e-8


if HAVE_TRITON:
    # A jitted kernel can only close over module globals that are ``tl.constexpr`` objects; a plain
    # float raises NameError at compile time inside the caller's kernel. Derived, never re-typed.
    _LOG1P_LINEAR_BELOW_TL = tl.constexpr(_LOG1P_LINEAR_BELOW)

    @triton.jit
    def div_rn(a, b):
        """fp32 ``a / b``, round-to-nearest, subnormal-preserving.

        Replaces ``libdevice.div_rn``, which flushes subnormal quotients, and Triton's ``/``, which
        lowers to an approximate reciprocal.
        """
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def sqrt(x):
        """fp32 ``sqrt(x)``, round-to-nearest, subnormal-preserving.

        This fixes more than FTZ: ``libdevice.sqrt`` also disagrees with ATen on ordinary normal
        inputs, so any kernel taking an L2 or RMS norm is exposed on completely ordinary data.
        """
        return tl.inline_asm_elementwise("sqrt.rn.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def log1p(x):
        """Compute fp32 log1p without flushing subnormals. For |x| < 2**-24, the correctly rounded
        result is x.
        """
        return tl.where(tl.abs(x) < _LOG1P_LINEAR_BELOW_TL, x, libdevice.log1p(x))

    @triton.jit
    def sigmoid(x):
        """fp32 ``sigmoid`` matching ATen bitwise, including where the result is subnormal.

        The same expression built on ``libdevice.div_rn`` flushes for x below about -87.
        """
        return div_rn(1.0, 1.0 + libdevice.exp(-x))

