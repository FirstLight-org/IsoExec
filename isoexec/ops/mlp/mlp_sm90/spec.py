"""Import-light declaration of the SM90 fused ``mlp.act``.

Recipe v1 is the eager trainer program ``F.silu(g) * (u + 0.0)`` (ops/mlp/eager.py) rounded where it
rounds: ``s = rn_bf16(g / (1 + expf(-g)))`` (fp32 IEEE divide, libdevice expf, FTZ off), then
``rn_bf16(fl32(s) * fl32(u + 0.0))``. One launch instead of three; the bits are the eager bits.
"""

from __future__ import annotations

from isoexec.core.registry import ImplSpec

ARCH = "sm90"
RECIPE_VERSION = 1
ACT_IMPL = "sm90_fused_v1"

IMPLS = (ImplSpec(op="mlp.act", impl_id=ACT_IMPL, version=RECIPE_VERSION, supported_archs=frozenset({ARCH})),)
