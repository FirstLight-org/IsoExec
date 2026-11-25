"""The dense GEMM census of a constructed component tree.

Each executed component contributes the dense 2-D GEMMs it runs through the batch-invariant matmul,
read from its own dimensions, so the census follows whatever model package built the tree.
Row-parallel sites are absent (pik runs their leaf GEMMs itself), as are the routed experts (the
batched/indexed expert kernels consume them). Site names are the shape tables' vocabulary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from isoexec.nn.attention import Attention
from isoexec.nn.gdn import GDN
from isoexec.nn.head import LMHead
from isoexec.nn.inspect import executed_modules
from isoexec.nn.mla import LatentAttention
from isoexec.nn.mlp import MLP
from isoexec.nn.moe import MoE

if TYPE_CHECKING:
    from isoexec.ops.mm.mm_shapes import GemmCensus, GemmSpec


def _component_sites(module: nn.Module) -> tuple[GemmSpec, ...]:
    # Imported per call: constructing or importing a component must load no ops module.
    from isoexec.ops.mm.mm_shapes import GemmSpec, dtype_of

    bf16, fp32 = dtype_of("bf16"), dtype_of("fp32")
    if isinstance(module, Attention):
        width = (module.heads + 2 * module.kv_heads) * module.head_dim
        if module.out_gate is not None:
            width += module.heads * module.head_dim  # the gated output's projection rides the fused qkv
        return (GemmSpec("attn.qkv", module.hidden, width, bf16, "tp"),)
    if isinstance(module, GDN):
        width = 2 * module.key_dim + 2 * module.value_dim + 2 * module.value_heads  # q,k | v,z | beta,alpha
        return (GemmSpec("gdn.in_proj_qkvzba", module.hidden, width, bf16, "tp"),)
    if isinstance(module, LatentAttention):
        return (
            GemmSpec("mla.q_down", module.hidden, module.q_lora_rank, bf16),
            GemmSpec("mla.q_up", module.q_lora_rank, module.heads * module.head_dim, bf16, "tp",
                     granule=module.heads),
            GemmSpec("mla.kv_down", module.hidden, module.kv_lora_rank + module.qk_rope_head_dim, bf16),
        )
    if isinstance(module, MoE):
        sites = [GemmSpec("moe.router", module.hidden, module.num_experts, fp32)]
        if module.shared_intermediate:
            if module.shared_expert_gate:
                sites.append(GemmSpec("shexp.gate", module.hidden, 1, bf16))
            sites.append(GemmSpec("shexp.fc1", module.hidden, 2 * module.shared_intermediate, bf16, "tp"))
        return tuple(sites)
    if isinstance(module, MLP):
        return (GemmSpec("mlp.fc1", module.hidden, 2 * module.intermediate, bf16, "tp"),)
    if isinstance(module, LMHead):
        return (GemmSpec("lm_head", module.hidden, module.padded_vocab, bf16, "tp"),)
    return ()


def gemm_census(model: nn.Module) -> GemmCensus:
    """The parallelism-independent dense GEMM census of ``model``'s executed components.

    Dimensions are the components' global ones, so a placed tree and its unplaced meta original
    give equal censuses. ``GemmCensus.sites(tp, etp)`` instantiates it.
    """
    from isoexec.ops.mm.mm_shapes import GemmCensus

    return GemmCensus(tuple(spec for _, module in executed_modules(model) for spec in _component_sites(module)))


__all__ = ["gemm_census"]
