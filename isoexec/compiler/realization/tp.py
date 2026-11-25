"""Primitives for tensor-parallel (TP) local parameter shapes and weight slices.

Local geometry uses the shard rules in ``isoexec.runtimes.parallel``: below the head granule,
engines replicate KV heads while trainers split them and gather activations. A ``LocalSpec`` is
planned for one tensor at a time and never stored.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import fx, nn

from isoexec.nn import GDN, Attention, LMHead
from isoexec.runtimes.parallel import LocalSpec, axis_local_spec, below_units_policy, units_replicate
from isoexec.schema import ShardAxis
from isoexec.sync.weights import gather_full


def _kind(owner: nn.Module) -> str:
    """The tensor-owner kind that selects a below-units policy."""
    return "attn" if isinstance(owner, Attention) else "gdn" if isinstance(owner, GDN) else "other"


def _degree(owner: nn.Module, role: str, *, tp_size: int, tp_rank: int, etp_size: int | None) -> tuple[int, int]:
    """Return the ranks a tensor splits over and this rank among them: ETP for expert tensors and
    TP otherwise. ``etp_size=None`` means ETP equals TP."""
    if etp_size is None or etp_size == tp_size or owner.tensor_axes()[role].kind != "expert":
        return tp_size, tp_rank
    if etp_size != 1:
        raise ValueError(f"Native expert tensors split over ETP=TP or ETP=1, got ETP={etp_size} at TP={tp_size}")
    return 1, 0


def local_spec(module: nn.Module, logical_name: str, full_shape: tuple[int, ...], *,
               tp_size: int, tp_rank: int, side: str, etp_size: int | None = None) -> LocalSpec:
    """Plan this rank's shard of one registered tensor at its global shape."""
    path, _, role = logical_name.rpartition(".")
    owner = module.get_submodule(path)
    size, rank = _degree(owner, role, tp_size=tp_size, tp_rank=tp_rank, etp_size=etp_size)
    dtype = str(module.get_parameter(logical_name).dtype)
    return axis_local_spec(logical_name, tuple(full_shape), tensor_axis(module, logical_name), dtype,
                           tp=size, rank=rank, side=side, kind=_kind(owner))


def shard_parameter_shapes(module: nn.Module, *, tp_size: int, side: str, etp_size: int | None = None) -> None:
    """Resize global meta parameters to TP-local shapes without replacing Parameters.

    TP1 leaves the module unchanged. Parameter owners supply ``tensor_axes()``.
    Expert tensors split along their declared ETP axis over ``etp_size`` ranks,
    which is TP at EP=1; ETP=1 keeps whole experts. Names, dtypes, trainability,
    ties, buffers, and FX code are preserved. This does not bind operations or
    select weight data for a rank.
    """
    if tp_size == 1:
        return
    shapes: dict[nn.Parameter, tuple[tuple[int, ...], ShardAxis]] = {}
    for path, child in module.named_modules():
        parameters = {name: value for name, value in child._parameters.items() if value is not None}
        if not parameters:
            continue
        if not callable(getattr(child, "tensor_axes", None)):
            raise TypeError(f"{path or '<root>'}: TP parameter shapes require tensor_axes()")
        axes = _tensor_axes(child)
        for name, parameter in parameters.items():
            logical_name = f"{path}.{name}" if path else name
            if not parameter.is_meta:
                raise ValueError(f"{logical_name}: TP parameter shapes require meta parameters")
            if name not in axes:
                raise ValueError(f"{logical_name}: tensor_axes() has no TP declaration")
            # Every rank holds the same local extent, so rank 0 plans the shape.
            size, _ = _degree(child, name, tp_size=tp_size, tp_rank=0, etp_size=etp_size)
            spec = axis_local_spec(logical_name, tuple(parameter.shape), axes[name], str(parameter.dtype),
                                   tp=size, rank=0, side=side, kind=_kind(child))
            declaration = (spec.local_shape, axes[name])
            if parameter in shapes and shapes[parameter] != declaration:
                raise ValueError(f"{logical_name}: shared parameter has incompatible TP declarations")
            shapes[parameter] = declaration

    for parameter, (shape, _) in shapes.items():
        if tuple(parameter.shape) != shape:
            parameter.data = torch.empty(shape, dtype=parameter.dtype, device="meta")


def set_local_dims(module: nn.Module, *, tp_size: int) -> None:
    """Set Attention activation heads with the shard rule at ``tp_size`` and GDN dimensions from weights.

    Below the KV head granule every rank consumes one whole KV head: engines replicate it and
    trainers reassemble it with ``kv_gather``, so the registered KV rows do not determine it.
    """
    for child in module.modules():
        if isinstance(child, Attention):
            child.set_local_heads(tp_size)
        elif isinstance(child, GDN):
            child.key_dim_local, child.value_dim_local = child.q.shape[0], child.v.shape[0]
            child.key_heads_local = child.key_dim_local // child.key_head_dim
            child.value_heads_local = child.value_dim_local // child.value_head_dim


def set_head_output_shape(module: fx.GraphModule) -> None:
    """Make captured LMHead checks expect the registered local vocabulary width."""
    from isoexec.compiler.capture.export_normalize import _checked_output

    changed = False
    for node in module.graph.nodes:
        if node.op != "call_function" or node.target is not _checked_output:
            continue
        head = module.get_submodule(node.args[2])
        if not isinstance(head, LMHead) or head.weight.shape[0] == head.padded_vocab:
            continue
        shape, strides = list(node.args[3]), list(node.args[4])
        shape[-1] = strides[-2] = head.weight.shape[0]
        node.args = (*node.args[:3], shape, strides, *node.args[5:])
        changed = True
    if changed:
        module.recompile()


def _tensor_axes(owner: nn.Module) -> Mapping[str, ShardAxis]:
    """Return the owner's TP axes, resolving expert tensors to their ETP axes."""
    axes = dict(owner.tensor_axes())
    for role, axis in axes.items():
        if axis.kind == "expert":
            axes[role] = owner.tensor_plan_metadata()[role].expert_axis
    return axes


def tensor_axis(module: nn.Module, logical_name: str) -> ShardAxis:
    """Return the TP axis the owner of ``logical_name`` declares, resolving expert tensors to ETP axes."""
    path, _, role = logical_name.rpartition(".")
    return _tensor_axes(module.get_submodule(path))[role]


def global_shape(module: nn.Module, logical_name: str, tensor: torch.Tensor, *,
                 tp_size: int, side: str, etp_size: int | None = None) -> tuple[int, ...]:
    """Return the global shape of a TP-local registered tensor."""
    shape = list(tensor.shape)
    if tp_size == 1:
        return tuple(shape)
    path, _, role = logical_name.rpartition(".")
    owner = module.get_submodule(path)
    size, _ = _degree(owner, role, tp_size=tp_size, tp_rank=0, etp_size=etp_size)
    axis = tensor_axis(module, logical_name)
    if axis.kind == "replicated" or size == 1:
        return tuple(shape)
    dim = cast(int, axis.dim)
    if axis.segments is not None:
        shape[dim] = sum(width for width, _ in axis.segments)
    elif units_replicate(size, axis.units, below_units_policy(side, _kind(owner))):
        # Each replicated rank holds one whole unit.
        shape[dim] *= cast(int, axis.units)
    else:
        shape[dim] *= size
    return tuple(shape)


def slice_weight(module: nn.Module, logical_name: str, tensor: torch.Tensor, *,
                 tp_size: int, tp_rank: int, side: str, etp_size: int | None = None) -> torch.Tensor:
    """Select one rank's shard of a full tensor."""
    if tp_size == 1:
        return tensor
    spec = local_spec(module, logical_name, tuple(tensor.shape), tp_size=tp_size, tp_rank=tp_rank, side=side,
                      etp_size=etp_size)
    return spec.slice(tensor)


def gather_weight(module: nn.Module, logical_name: str, tensor: torch.Tensor, *,
                  tp_group: torch.distributed.ProcessGroup, side: str) -> torch.Tensor:
    """Reconstruct a full tensor on every TP rank from this rank's shard."""
    tp_size = tp_group.size()
    if tp_size == 1:
        return tensor
    full_shape = global_shape(module, logical_name, tensor, tp_size=tp_size, side=side)
    spec = local_spec(module, logical_name, full_shape, tp_size=tp_size, tp_rank=tp_group.rank(), side=side)
    return gather_full(tensor, spec, tp_group)
