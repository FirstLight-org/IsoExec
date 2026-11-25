"""Engine-side operation bindings for the vLLM runtime. Bindings retain family-declared numerical
identities and require an explicit host provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch


class BindingError(RuntimeError):
    """An invalid engine deployment, cache geometry, row kind or step fact."""


#: ``logprobs.lm_head_slice`` on an engine: the rows the sampler selected. The other member of the
#: ``logprobs.row_select`` twin class is the trainer's ``identity``, and which rows arrived is
#: exactly what the two sides compare.
SLICE_IMPL = "sampled_rows"


def lm_head_slice_sampled_rows(h: torch.Tensor) -> torch.Tensor:
    """Keep sampled hidden rows unchanged; the engine has already selected them."""
    return h


def gather_vocab(logits_local: torch.Tensor, group, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """All-gather a vocab-parallel logits shard along the vocab dim in rank order (a byte copy).

    Over pik's symmetric memory where the group has P2P: stage the shard, one barrier, and one widening
    copy per peer straight into its vocab columns. Otherwise one NCCL collective into a rank-major
    ``[world, ..., V/world]`` buffer, then ONE copy kernel that both interleaves the shards along the vocab
    dim and, with ``out_dtype``, widens them (bf16 -> fp32 is exact). Bytes only on both paths.
    """
    if group is None:
        return logits_local if out_dtype is None else logits_local.to(out_dtype)
    world = torch.distributed.get_world_size(group=group)
    if world == 1:
        return logits_local if out_dtype is None else logits_local.to(out_dtype)
    local = logits_local.contiguous()
    lead, width = local.shape[:-1], local.shape[-1]
    out = torch.empty((*lead, world * width), dtype=out_dtype or local.dtype, device=local.device)
    if local.is_cuda and _gather_p2p(local, group, out):
        return out
    stacked = torch.empty((world, *local.shape), dtype=local.dtype, device=local.device)
    torch.distributed.all_gather_into_tensor(stacked, local, group=group)
    out.view(*lead, world, width).copy_(stacked.movedim(0, -2))
    return out


def _gather_p2p(local: torch.Tensor, group, out: torch.Tensor) -> bool:
    """Over pik's symmetric memory where the group has it: one barrier and a widening copy per peer instead
    of an NCCL all-gather plus an interleaving copy. Same bytes either way; False means take NCCL."""
    from isoexec.ops.collectives.pik_bootstrap import ensure_pik

    ensure_pik()  # one canonical pik: its pools, setup counter and transport policy are the all-reduce's
    import pik.allreduce as pik_allreduce  # type: ignore

    return pik_allreduce.gather_last_dim(local, group, out)


def _declared_version(op: str, impl: str) -> int | None:
    """Read a recipe's declared version; undeclared recipes fail execution admission."""
    from isoexec.core.registry_build import build_registry

    registry = build_registry()
    if not registry.has_op(op):
        return None
    spec = registry.get_op(op).impls.get(impl)
    return None if spec is None else spec.version


def engine_resolution(
    op: str,
    impl: str,
    fn,
    *,
    inputs: Mapping[str, object] | None = None,
    owns=None,
    provider: str,
):
    """Describe the engine callable and its explicit fused-operation ownership."""
    from isoexec.core.bindings import BoundOp

    return BoundOp(
        op, impl, _declared_version(op, impl), provider,
        dict(inputs or {}), fn, owns=dict(owns or {}),
    )


def gdn_core_resolution(site: str, fn, *, table, provider: str):
    """Bind the state-bearing GDN core using identities from the selected family, including its
    fused convolution, l2 normalization and gating. Chunk length and state dtype are
    kernel-owned properties reported by the pool, not binding inputs.
    """
    core = table.bind("gdn.core", site=f"{site}.core")
    owns = dict(core.owns) or {core.op: core.impl}
    owns[core.op] = core.impl
    owns["conv.short"] = table.bind("conv.short", site=f"{site}.conv").impl
    # Explicit reference builds retain their reference identity. A host adapter
    # cannot turn an undeclared reference recipe into a production implementation
    # merely by changing the provider label.
    return replace(core, fn=fn, owns=owns, provider=core.provider if table.mode == "reference" else provider)


__all__ = [
    "BindingError",
    "SLICE_IMPL",
    "engine_resolution",
    "gather_vocab",
    "gdn_core_resolution",
    "lm_head_slice_sampled_rows",
]
