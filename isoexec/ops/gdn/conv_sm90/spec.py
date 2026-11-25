"""Import-light declarations for the SM90 width-4 causal convolution recipe.
Decode, varlen prefill and trainer forward share one scalar graph; they claim no bitwise equality
with the vendored Triton pair, so both runtimes must switch together."""

from isoexec.core.registry import ImplSpec

#: Taps per channel. ``csrc/common.cuh`` states the same number as the ``w0..w3`` / ``h3, h2, h1, x0``
#: chain of ``preact``; the two must agree.
WIDTH = 4

#: Bump when the numerical recipe changes; ops.py imports this identity.
RECIPE_VERSION = 1

IMPLS: tuple[ImplSpec, ...] = (
    # Hosts bind the GDN row's SiLU; activation is not a declared caller input.
    ImplSpec(op="conv.short", impl_id="sm90_cuda_v1", version=RECIPE_VERSION, supported_archs=frozenset({"sm90"})),
)
