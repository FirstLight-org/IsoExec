"""Internal normalization for the standard text-model root input."""

from collections.abc import Callable, Mapping
from typing import Any, cast


def logical_input_contract(value: Any, name: str) -> Any:
    """Keep dtype/rank facts without retaining capture tensors or native devices."""
    import torch
    from isoexec.schema import SequenceContext

    if isinstance(value, torch.Tensor):
        return {"rank": value.ndim, "dtype": "token_index" if name == "input_ids" else str(value.dtype)}
    if isinstance(value, SequenceContext):
        # CPU offsets are a native carrier convenience; logical sequence dtypes
        # agree even when trainer and engine capture different optional carriers.
        return {"type": "SequenceContext", "positions": logical_input_contract(value.positions, "positions"),
                "cu_seqlens": logical_input_contract(value.cu_seqlens, "cu_seqlens"),
                "input_ids": logical_input_contract(value.input_ids, "input_ids")}
    if isinstance(value, (tuple, list)):
        return [logical_input_contract(child, name) for child in value]
    if isinstance(value, Mapping):
        return {key: logical_input_contract(child, str(key)) for key, child in sorted(value.items())}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _normalize_token_ids(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Accept int32/int64 input_ids and present int64 to capture and execution."""
    result = dict(arguments)
    if "input_ids" not in result:
        return result
    import torch

    value = result["input_ids"]
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must be a Tensor with dtype int32 or int64")
    if value.dtype == torch.int64:
        return result

    from torch.utils import _pytree as pytree
    from isoexec.schema.sequence import register_pytrees

    register_pytrees()
    is_alias = cast(Callable[[torch.Tensor, torch.Tensor], bool], getattr(torch._C, "_is_alias_of"))
    leaves, _ = pytree.tree_flatten(result)
    if any(isinstance(other, torch.Tensor) and other is not value and is_alias(value, other) for other in leaves):
        raise ValueError("input_ids has distinct storage aliases; provide independent token tensors")
    replacements: dict[int, Any] = {id(value): value.to(torch.int64)}

    # Preserve repeated Tensor references and leave unaffected containers intact.
    def replace(item: Any) -> Any:
        if id(item) in replacements:
            return replacements[id(item)]
        children, spec = pytree.tree_flatten(item, is_leaf=lambda child: child is not item)
        if spec.is_leaf():
            return item
        updated = [replace(child) for child in children]
        mapped = item if all(a is b for a, b in zip(children, updated, strict=True)) else spec.unflatten(updated)
        replacements[id(item)] = mapped
        return mapped

    return replace(result)
