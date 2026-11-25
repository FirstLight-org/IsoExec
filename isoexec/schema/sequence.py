"""Packed-sequence metadata shared by trainer and engine forwards. Tensor operations and pytree
registration remain explicit to preserve import purity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # type-level only; importing this module never touches torch
    from torch import Tensor


@dataclass(eq=False)  # eq=False: tensor fields; registered as a pytree by register_pytrees()
class SequenceContext:
    """Metadata for one forward: positions are int64 [T] or MRoPE [3,T], and cu_seqlens is int32
    [R+1]. max_q and max_kv are host integers; max_kv may be an upper bound and must not
    determine numerical scheduling.

    cu_seqlens_cpu optionally retains the same boundaries on the host. Trainer packing builds
    it once per microbatch; GDN uses it to size scratch and choose a schedule without a device
    synchronization in every layer. It must describe the same packed batch.

    rope_freqs is populated once per forward; callers must not reuse a context after changing
    positions. rope_cos_sin is reserved for the existing pytree layout and has no current
    writer.
    """

    positions: "Tensor"
    cu_seqlens: "Tensor"
    max_q: int
    max_kv: "int | None" = None
    input_ids: "Tensor | None" = None  # token-conditioned layers (hash routing)
    rope_freqs: "Tensor | None" = None  # filled per forward; see the class docstring
    rope_cos_sin: "tuple[Tensor, Tensor] | None" = None  # the fused kernel's form of rope_freqs
    cu_seqlens_cpu: "Tensor | None" = None


def make_sequence_context(
    positions: "Tensor",
    cu_seqlens: "Tensor",
    *,
    max_q: int,
    max_kv: "int | None" = None,
    input_ids: "Tensor | None" = None,
    cu_seqlens_cpu: "Tensor | None" = None,
) -> SequenceContext:
    """Validate packed sequence metadata without importing torch. Callers supply max_q to avoid a
    device synchronization when deriving it.
    """
    if positions.dim() not in (1, 2):
        raise ValueError(f"positions must be [T] or [3, T], got {tuple(positions.shape)}")
    if positions.dim() == 2 and positions.shape[0] != 3:
        raise ValueError(f"2-D positions are the three MRoPE rows [3, T], got {tuple(positions.shape)}")
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens must be [R+1] with R >= 1, got {tuple(cu_seqlens.shape)}")
    if cu_seqlens_cpu is not None:
        if cu_seqlens_cpu.device.type != "cpu" or cu_seqlens_cpu.dim() != 1:
            raise ValueError("cu_seqlens_cpu must be a one-dimensional CPU tensor")
        if cu_seqlens_cpu.numel() != cu_seqlens.numel():
            raise ValueError("cu_seqlens_cpu must have the same number of boundaries as cu_seqlens")
    if int(max_q) < 1:
        raise ValueError(f"max_q must be a positive host int, got {max_q!r}")
    if max_kv is not None and int(max_kv) < int(max_q):
        raise ValueError(f"max_kv {max_kv} is below max_q {max_q}")
    return SequenceContext(
        positions=positions,
        cu_seqlens=cu_seqlens,
        max_q=int(max_q),
        max_kv=None if max_kv is None else int(max_kv),
        input_ids=input_ids,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )


def num_tokens(seq: SequenceContext) -> int:
    """``T``: the packed token count, read off the positions row."""
    return int(seq.positions.shape[-1])


_PYTREES_REGISTERED = False


def register_pytrees() -> None:
    """Register ``SequenceContext`` as a torch pytree (flatten = field values, spec = field names).
    Called explicitly by a host that needs it; importing this module never touches torch."""
    global _PYTREES_REGISTERED
    if _PYTREES_REGISTERED:
        return
    import dataclasses

    import torch.utils._pytree as pytree

    names = tuple(f.name for f in dataclasses.fields(SequenceContext))

    def _flatten(ctx):
        return [getattr(ctx, n) for n in names], names

    def _unflatten(values, context):
        return SequenceContext(**dict(zip(context, values)))

    def _flatten_with_keys(ctx: SequenceContext) -> tuple[list[tuple[pytree.KeyEntry, Any]], tuple[str, ...]]:
        # PyTorch GetAttrKey implements KeyEntry; its get() parameter is named obj, not parent.
        pairs = [(cast(pytree.KeyEntry, pytree.GetAttrKey(n)), getattr(ctx, n)) for n in names]
        return pairs, names

    if SequenceContext not in getattr(pytree, "SUPPORTED_NODES", {}):
        pytree.register_pytree_node(
            SequenceContext, _flatten, _unflatten,
            flatten_with_keys_fn=_flatten_with_keys,
            serialized_type_name="isoexec.schema.sequence.SequenceContext",
        )
    _PYTREES_REGISTERED = True


__all__ = ["SequenceContext", "make_sequence_context", "num_tokens", "register_pytrees"]
