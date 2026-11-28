"""Attention declarations. The GQA core is declared by its arch package (``gqa_sm90``, ``gqa_sm100``)
and the RoPE seam by its own; both are appended by ``__getattr__``. MLA and the output gate are
declared here."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isoexec.core.registry import ANY, ImplSpec
from isoexec.ops._dispatch import family_impls

if TYPE_CHECKING:  # never executed: it names what the lazy ``__getattr__`` below otherwise hides.
    IMPLS: tuple[ImplSpec, ...]

#: One KV-reduction split, so a row's arithmetic does not depend on how the launch was tiled
#: (the FA3 absorbed MLA call; gqa_sm90.spec.NUM_SPLITS is the same pin for its kernels).
NUM_SPLITS = 1

#: The KV cache format the kernel reads (A36). This refuses every fp8/nvfp4 cache.
KV_DTYPE = "bf16"

_SM90 = frozenset({"sm90"})
_SM100 = frozenset({"sm100"})

#: The family's own declarations, written here. The rope and gqa seams' are appended by ``__getattr__``.
_CORE: tuple[ImplSpec, ...] = (
    ImplSpec(op="attention.core.mla", impl_id="fa3_absorbed_ns1", version=1, supported_archs=_SM90),
    # SEG changes the balanced merge tree and is digested; page/cache dtype are constants.
    # This recipe has no bitwise twin claim with FA3's serial KV reduction.
    ImplSpec(
        op="attention.core.mla",
        impl_id="sm90_absorbed_segtree",
        version=1,
        supported_archs=_SM90,
        inputs={"seg_tokens": ANY},
    ),
    ImplSpec(op="attention.core.mla", impl_id="sm100_absorbed_segtree", version=1,
             supported_archs=_SM100, inputs={"seg_tokens": ANY}),
    # Sigmoid and SiLU gates have distinct numerical identities.
    ImplSpec(op="attention.gate", impl_id="compiled_sigmoid_gate_fp32", version=1, supported_archs=_SM90 | _SM100),
    ImplSpec(op="attention.gate", impl_id="eager_swish", version=1, supported_archs=_SM90 | _SM100),
)

__all__ = ("IMPLS", "KV_DTYPE", "NUM_SPLITS")


# Registry.add groups flat declarations by op; family ordering affects presentation only.
#: DeepSeek Sparse Attention on Blackwell (attention/dsa_sm100): three sub-ops, each recipe its own id,
#: no twin class (nothing else implements them). The backward recipes have no atomics and are not
#: required to be bitwise; they are float64-accurate.
_DSA: tuple[ImplSpec, ...] = (
    ImplSpec(op="attention.dsa.indexer", impl_id="sm100_dsa_indexer_v1", version=1, supported_archs=_SM100),
    ImplSpec(op="attention.dsa.topk", impl_id="sm100_dsa_topk_v1", version=1, supported_archs=_SM100),
    ImplSpec(op="attention.dsa.sparse", impl_id="sm100_dsa_sparse_segtree_v1", version=1, supported_archs=_SM100),
    # v2: v1 + the lazy rescale at 8 log2 units; its own bits.
    ImplSpec(op="attention.dsa.sparse", impl_id="sm100_dsa_sparse_v2", version=2, supported_archs=_SM100),
    # the dual-rail leaf (FlashMLA-derived, no split) and its learned-sink form; their own bits.
    ImplSpec(op="attention.dsa.sparse", impl_id="sm100_dsa_sparse_dualrail_v1", version=1, supported_archs=_SM100),
    ImplSpec(op="attention.dsa.sparse", impl_id="sm100_dsa_sparse_dualrail_sink_v1", version=1, supported_archs=_SM100),
    ImplSpec(op="attention.dsa.sparse_bwd", impl_id="sm100_dsa_sparse_bwd_v1", version=1, supported_archs=_SM100),
    ImplSpec(op="attention.dsa.indexer_bwd", impl_id="sm100_dsa_indexer_bwd_v1", version=1, supported_archs=_SM100),
)

__getattr__ = family_impls(__name__, "rope", "gqa", own=_CORE + _DSA)
