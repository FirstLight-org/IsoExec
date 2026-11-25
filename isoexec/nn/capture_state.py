"""Isolated supported Python state and metadata-only capture purity checks."""

from collections import OrderedDict, defaultdict
from collections.abc import Callable
from copy import copy
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from pathlib import PurePath
import struct
from types import BuiltinFunctionType, CellType, FunctionType, MappingProxyType, MethodType, ModuleType, SimpleNamespace
from typing import Any

import torch
from torch import nn

_CALLABLES = (FunctionType, BuiltinFunctionType, MethodType, type)
_SYMBOLS = (torch.SymInt, torch.SymFloat, torch.SymBool)
_SCALARS = (str, int, float, bool, bytes, type(None), torch.dtype, torch.device, Enum, PurePath, range, *_SYMBOLS)
_EXPORT_HOOK_REGISTRIES = frozenset({'_forward_hooks', '_forward_pre_hooks', '_forward_hooks_with_kwargs',
                                   '_forward_pre_hooks_with_kwargs', '_forward_hooks_always_called'})


def scalar_identity(value: Any) -> Any:
    """Identity for fixed Python values, including IEEE signed zero/NaN bits."""
    if isinstance(value, _SYMBOLS):
        return type(value), id(value.node)
    if isinstance(value, range):
        return range, value.start, value.stop, value.step
    return type(value), struct.pack('!d', value) if isinstance(value, float) else value


def _field(value: Any, name: str) -> Any:
    """Inspect metadata storage without invoking authored runtime-field guards."""
    state = vars(value) if hasattr(value, '__dict__') else {}
    return state[name] if name in state else object.__getattribute__(value, name)


class StateCopy:
    """Copy only supported containers; runtime resources require explicit ports.

    One memo spans the complete module tree, so repeated container, module and
    Tensor registrations preserve their identity relationships on the copy.
    """

    def __init__(self, tensor: Callable[[torch.Tensor], torch.Tensor], module: Callable[[nn.Module], nn.Module]) -> None:
        self.tensor, self.module = tensor, module
        self.memo: dict[int, Any] = {}
        self.active: set[int] = set()

    def value(self, value: Any) -> Any:
        if id(value) in self.memo:
            return self.memo[id(value)]
        if isinstance(value, MethodType):
            owner = value.__self__
            if not isinstance(owner, nn.Module) and not is_dataclass(owner):
                raise TypeError('Bound runtime methods need an explicit capture resource port')
            result = MethodType(self.value(value.__func__), self.value(owner))
            self.memo[id(value)] = result
            return result
        if isinstance(value, FunctionType):
            # Clone function-owned supported state, not the runtime/global
            # module graph. Imported modules/types remain static code owners.
            def cell(item: Any) -> CellType:
                cells = (lambda: item).__closure__
                assert cells is not None
                return cells[0]
            original_closure = value.__closure__
            closure = None if original_closure is None else tuple(cell(None) for _ in original_closure)
            namespace = dict(value.__globals__)
            result = FunctionType(value.__code__, namespace, value.__name__, None, closure)
            self.memo[id(value)] = result
            result.__qualname__ = value.__qualname__
            result.__module__ = value.__module__
            result.__defaults__ = self.value(value.__defaults__)
            result.__kwdefaults__ = self.value(value.__kwdefaults__)
            result.__annotations__ = dict(value.__annotations__)
            result.__dict__.update(self.value(vars(value)))
            if closure is not None:
                assert original_closure is not None
                for cloned, original in zip(closure, original_closure, strict=True):
                    cloned.cell_contents = self.value(original.cell_contents)
            for name in value.__code__.co_names:
                if name in namespace and not isinstance(namespace[name], (ModuleType, type, BuiltinFunctionType)):
                    namespace[name] = self.value(namespace[name])
            return result
        if isinstance(value, _SCALARS + _CALLABLES):
            return value
        if isinstance(value, torch.Tensor):
            result = self.tensor(value)
        elif isinstance(value, nn.Module):
            return self.module(value)
        elif type(value) in (dict, OrderedDict, defaultdict):
            result = defaultdict(None) if type(value) is defaultdict else type(value)()
            self.memo[id(value)] = result
            if type(value) is defaultdict:
                result.default_factory = self.value(value.default_factory)
            for key, item in value.items():
                result[self.value(key)] = self.value(item)
        elif type(value) is SimpleNamespace:
            result = SimpleNamespace()
            self.memo[id(value)] = result
            result.__dict__.update(self.value(vars(value)))
        elif type(value) is partial:
            if id(value) in self.active:
                raise TypeError('Cyclic partial capture state needs an explicit state contract')
            self.active.add(id(value))
            try:
                result = partial(self.value(value.func), *self.value(value.args), **self.value(value.keywords))
                self.memo[id(value)] = result
                result.__dict__.update(self.value(vars(value)))
            finally:
                self.active.remove(id(value))
        elif type(value) is list:
            result = []
            self.memo[id(value)] = result
            result.extend(self.value(item) for item in value)
        elif isinstance(value, (tuple, set, frozenset, MappingProxyType)):
            if id(value) in self.active:
                raise TypeError('Cyclic immutable capture state needs an explicit state contract')
            self.active.add(id(value))
            try:
                if isinstance(value, MappingProxyType):
                    result = MappingProxyType({self.value(key): self.value(item) for key, item in value.items()})
                else:
                    items = tuple(self.value(item) for item in value)
                    result = (type(value)(*items) if isinstance(value, tuple) and hasattr(value, '_fields')
                              else type(value)(items))
            finally:
                self.active.remove(id(value))
        elif is_dataclass(value) and not isinstance(value, type):
            result = copy(value)
            self.memo[id(value)] = result
            for field in fields(value):
                object.__setattr__(result, field.name, self.value(_field(value, field.name)))
        else:
            raise TypeError(f'Unsupported capture state {type(value).__name__}; use plain containers, '
                            'dataclass configuration, or an explicit runtime resource port')
        self.memo[id(value)] = result
        return result


def tensor_state(value: torch.Tensor) -> tuple[Any, ...]:
    storage = value.untyped_storage()
    return (id(value), None if value.is_inference() else scalar_identity(value._version),
            tuple(scalar_identity(item) for item in value.shape), tuple(scalar_identity(item) for item in value.stride()),
            value.dtype, value.device, value.layout, scalar_identity(value.storage_offset()), value.requires_grad,
            value.is_conj(), value.is_neg(), storage._cdata, scalar_identity(storage.nbytes()))


def state_identity(value: Any, *, memo: dict[int, int] | None = None, objects: bool = False) -> Any:
    """Freeze nested metadata, alias identity and tensor versions without data reads."""
    if isinstance(value, _SCALARS):
        return scalar_identity(value)
    if isinstance(value, (BuiltinFunctionType, type, ModuleType)) or isinstance(value, nn.Module):
        return type(value), id(value)
    if isinstance(value, torch.Tensor):
        return tensor_state(value)
    memo = {} if memo is None else memo
    if id(value) in memo:
        return 'alias', memo[id(value)]
    memo[id(value)] = len(memo)
    if isinstance(value, MethodType):
        children = (state_identity(value.__func__, memo=memo, objects=objects), id(value.__self__))
    elif isinstance(value, FunctionType):
        closed = () if value.__closure__ is None else tuple(state_identity(cell.cell_contents, memo=memo, objects=objects)
                                                          for cell in value.__closure__)
        namespace = tuple((name, value.__globals__[name]) for name in value.__code__.co_names
                          if name in value.__globals__ and not isinstance(value.__globals__[name], (ModuleType, type, BuiltinFunctionType)))
        children = (state_identity(value.__defaults__, memo=memo, objects=objects),
                    state_identity(value.__kwdefaults__, memo=memo, objects=objects),
                    closed,
                    tuple((name, state_identity(item, memo=memo, objects=objects)) for name, item in namespace))
    elif type(value) is partial:
        children = (state_identity(value.func, memo=memo, objects=objects),
                    state_identity(value.args, memo=memo, objects=objects),
                    state_identity(value.keywords, memo=memo, objects=objects),
                    state_identity(vars(value), memo=memo, objects=objects))
    elif type(value) is SimpleNamespace:
        children = state_identity(vars(value), memo=memo, objects=objects)
    elif type(value) in (dict, OrderedDict, defaultdict) or isinstance(value, MappingProxyType):
        children = tuple((state_identity(key, memo=memo, objects=objects), state_identity(item, memo=memo, objects=objects))
                         for key, item in value.items())
        if type(value) is defaultdict:
            children = (state_identity(value.default_factory, memo=memo, objects=objects), children)
    elif isinstance(value, (tuple, list, set, frozenset)):
        children = tuple(state_identity(item, memo=memo, objects=objects) for item in value)
    elif is_dataclass(value) and not isinstance(value, type):
        children = tuple((field.name, state_identity(_field(value, field.name), memo=memo, objects=objects))
                         for field in fields(value))
    elif objects and hasattr(value, '__dict__'):
        children = tuple((name, state_identity(item, memo=memo, objects=True)) for name, item in vars(value).items())
    else:
        raise TypeError(f'Unsupported capture state {type(value).__name__}; an explicit state contract is required')
    return type(value), id(value), children


def module_state(module: nn.Module, *, exclude: frozenset[str] = frozenset(), objects: bool = False) -> Any:
    registrations = tuple((kind, tuple((name, id(value)) for name, value in entries.items()))
                          for kind, entries in (('modules', module._modules), ('parameters', module._parameters),
                                                ('buffers', module._buffers)))
    tensors = tuple((kind, name, tensor_state(value))
                    for kind, entries in (('parameter', module._parameters), ('buffer', module._buffers))
                    for name, value in entries.items() if value is not None)
    memo: dict[int, int] = {}
    def attribute(name: str, value: Any) -> Any:
        if name in _EXPORT_HOOK_REGISTRIES:
            return tuple((key, id(hook)) for key, hook in value.items())
        return state_identity(value, memo=memo, objects=objects)
    attributes = tuple((name, attribute(name, value)) for name, value in vars(module).items()
                       if name not in {'_parameters', '_buffers', '_modules', *exclude})
    return registrations, tensors, attributes


def module_tree_state(module: nn.Module, *, objects: bool = False) -> Any:
    return tuple((path, id(child), module_state(child, objects=objects))
                 for path, child in module.named_modules(remove_duplicate=False))


__all__ = ['StateCopy', 'scalar_identity', 'state_identity', 'tensor_state', 'module_state', 'module_tree_state']
