"""Run component forward on an isolated logical copy with abstract OP bindings.

Inner metadata operations are excluded from the surrounding export graph.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import fields, is_dataclass
from typing import Any, cast

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils import _pytree as pytree

from isoexec.nn._slots import OpSlots
from isoexec.nn.component import Component
from isoexec.nn.inspect import place
from isoexec.ops.abstract import AbstractOpTable
from isoexec.schema import Axis, Placement
from isoexec.nn.capture_state import StateCopy, module_tree_state, tensor_state


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        return [tensor for item in value.values() for tensor in _tensor_leaves(item)]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensor_leaves(item)]
    if is_dataclass(value) and not isinstance(value, type):
        return [tensor for item in fields(value) for tensor in _tensor_leaves(vars(value)[item.name])]
    return []


class _Copy:
    def __init__(self, mode: FakeTensorMode, device: torch.device) -> None:
        self.mode = mode
        self.device = device
        self.copier = StateCopy(self.tensor, self.module)
        self.memo = self.copier.memo

    def tensor(self, value: torch.Tensor) -> torch.Tensor:
        if isinstance(value, FakeTensor):
            if value.fake_mode is not self.mode:
                raise ValueError("Component abstract inputs must share one FakeTensorMode")
            return value
        # Meta parameters inherit the input's logical device under FakeTensorMode.
        if value.device.type == "meta":
            return torch.empty_strided(value.shape, value.stride(), dtype=value.dtype,
                                       device=self.device, requires_grad=value.requires_grad)
        return self.mode.from_tensor(value, static_shapes=True)

    def value(self, value: Any) -> Any:
        return self.copier.value(value)

    def module(self, module: nn.Module) -> nn.Module:
        if module._forward_hooks or module._forward_pre_hooks:
            raise ValueError("Component abstract inference does not execute module hooks")
        if isinstance(module, Component) and module.placed:
            raise ValueError("Component abstract inference requires an unplaced logical component")
        result = copy(module)
        self.memo[id(module)] = result
        result.__dict__ = {name: self.value(value) for name, value in vars(module).items() if name != "slots"}
        if isinstance(result, Component) and isinstance(module, Component):
            result.slots = OpSlots({name: module.slots.op_of(name) for name in module.slots},
                                   site=f"<abstract {type(module).__name__}>")
        return result


def _state(value: Any) -> Any:
    """Record observable mutable input state without reading any tensor data."""
    if isinstance(value, torch.Tensor):
        return tensor_state(value)
    if isinstance(value, Mapping):
        return tuple((key, _state(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return type(value), tuple(_state(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value), tuple((item.name, _state(vars(value)[item.name])) for item in fields(value))
    return value


class _OwnedWrites(TorchDispatchMode):
    """Reject writes to owned and input storage, including detached/view aliases."""

    def __init__(self, component: nn.Module, inputs: list[torch.Tensor]) -> None:
        # Stable alias handles retain the original allocations even if forward
        # temporarily replaces an owner's .data and later restores it.
        with disable_proxy_modes_tracing():
            self.owned = tuple(value.detach() for value in (*component.parameters(), *component.buffers()))
            self.inputs = tuple(value.detach() for value in inputs)

    def __torch_dispatch__(self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None) -> Any:
        kwargs = {} if kwargs is None else kwargs
        schema = func._schema
        if schema.is_mutable:
            alias = cast(Callable[[torch.Tensor, torch.Tensor], bool], getattr(torch._C, '_is_alias_of'))
            for index, argument in enumerate(schema.arguments):
                if argument.alias_info is None or not argument.alias_info.is_write:
                    continue
                value = args[index] if index < len(args) else kwargs.get(argument.name)
                if any(alias(tensor, owned) for tensor in _tensor_leaves(value) for owned in self.owned):
                    raise ValueError('Component abstract forward writes an owned Parameter/Buffer; '
                                     'an explicit effect contract is required before a pure export marker')
                if any(alias(tensor, source) for tensor in _tensor_leaves(value) for source in self.inputs):
                    raise ValueError('Component abstract forward writes an argument; '
                                     'an explicit effect contract is required before a pure export marker')
        return func(*args, **kwargs)


def abstract_output(component: Component, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    """Infer fresh output tensors through the component forward and abstract OPs."""
    tensors = _tensor_leaves((args, kwargs))
    if not tensors:
        raise ValueError("Component abstract inference needs at least one tensor input")
    if torch.is_inference_mode_enabled() or any(value.is_inference() for value in tensors):
        raise ValueError(
            "Capture component metadata outside torch.inference_mode() with non-inference example tensors; "
            "torch.no_grad() is supported. Inference tensors have no mutation version counter. "
            "This restriction does not prevent later inference execution."
        )
    modes = {value.fake_mode for value in tensors if isinstance(value, FakeTensor)}
    if len(modes) > 1:
        raise ValueError("Component abstract inputs must share one FakeTensorMode")
    mode = next(iter(modes)) if modes else FakeTensorMode()
    with disable_proxy_modes_tracing(), mode:
        copier = _Copy(mode, tensors[0].device)
        clone = cast(Component, copier.value(component))
        fake_args, fake_kwargs = copier.value(args), copier.value(dict(kwargs))
        axis = Axis(1, 0)
        placement = Placement(tp=axis, ep=axis, dp=axis, pp=axis, etp=axis, cp=axis,
                              side="trainer", leaves=1)
        place(clone, placement, prefix="abstract", op_table=AbstractOpTable())
        before = _state((fake_args, fake_kwargs))
        owned_before = module_tree_state(clone, objects=True)
        with _OwnedWrites(clone, _tensor_leaves((fake_args, fake_kwargs))):
            result = clone(*fake_args, **fake_kwargs)
        if module_tree_state(clone, objects=True) != owned_before:
            raise ValueError('Component abstract forward changed owned parameters, buffers or persistent Python state; '
                             'an explicit effect contract is required before a pure export marker')
        if _state((fake_args, fake_kwargs)) != before:
            raise ValueError("Component abstract forward writes an argument; an explicit effect contract is required")
        outputs, _ = pytree.tree_flatten(result)
        if not outputs or not all(isinstance(value, FakeTensor) for value in outputs):
            raise TypeError("Component abstract forward must return a Tensor or a pytree of Tensors")
        sources = _tensor_leaves((fake_args, fake_kwargs)) + list(clone.parameters()) + list(clone.buffers())
        is_alias = cast(Callable[[torch.Tensor, torch.Tensor], bool], getattr(torch._C, "_is_alias_of"))
        for output in outputs:
            if output.layout != torch.strided or output.is_conj() or output.is_neg():
                raise ValueError("Component output needs strided layout without lazy conjugate/negative views")
            if output.storage_offset() != 0:
                raise ValueError("Component output needs storage offset zero for this capture envelope")
            if any(is_alias(output, source) for source in sources):
                raise ValueError("Component output aliases an input, parameter or another output; an explicit alias contract is required")
            sources.append(output)
        return result


__all__ = ["abstract_output"]
