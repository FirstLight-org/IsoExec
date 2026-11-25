"""Refuse backward-only author effects that an ATen forward graph cannot retain."""

from collections.abc import Iterator
from contextlib import contextmanager
import sys
from types import FrameType
from typing import Any, Literal, Protocol

import torch
from torch.overrides import TorchFunctionMode


class _Session(Protocol):
    @property
    def opaque(self) -> bool: ...
    @property
    def authored(self) -> bool: ...
    modes: tuple[Any, ...]


def execution_modes() -> tuple[Any, ...]:
    return (torch.is_grad_enabled(), torch.is_inference_mode_enabled(), torch.get_default_dtype(),
            tuple((device, torch.is_autocast_enabled(device), torch.get_autocast_dtype(device))
                  for device in ("cpu", "cuda")))


class _ModeEffects(TorchFunctionMode):
    def __init__(self, session: _Session) -> None:
        self.session = session

    def __torch_function__(self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None) -> Any:
        if self.session.authored and not self.session.opaque and execution_modes() != self.session.modes:
            raise ValueError("Authored grad/inference/autocast/default-dtype regions need an explicit capture "
                             "effect contract; use explicit tensor operations instead of changing execution modes")
        return func(*args, **({} if kwargs is None else kwargs))


@contextmanager
def capture_effects(session: _Session) -> Iterator[None]:
    """Inspect only this thread's ordinary authored forward, without global patches.

    Opaque Components execute their original backward at runtime, so their own
    autograd implementations are not decomposed and are outside this refusal.
    """
    previous = sys.getprofile()
    if previous is not None and not callable(previous):
        raise ValueError("Capture effect inspection requires disabling the active Python profiler during capture")
    forbidden = {
        torch.Tensor.register_hook.__code__: "tensor backward hooks",
        torch.Tensor.register_post_accumulate_grad_hook.__code__: "tensor backward hooks",
    }
    # torch <= 2.13 wraps Function.apply in Python, so entering it is a "call" of that code object.
    # torch 2.14 binds torch._C._FunctionBase.apply directly: no Python frame is entered, and the
    # profiler reports a "c_call" whose arg is the builtin bound to the Function subclass.
    apply = torch.autograd.Function.apply
    if hasattr(apply, "__func__") and hasattr(apply.__func__, "__code__"):
        forbidden[apply.__func__.__code__] = "custom autograd.Function backward"

    def _is_function_apply(arg: Any) -> bool:
        owner = getattr(arg, "__self__", None)
        return (getattr(arg, "__name__", None) == "apply" and isinstance(owner, type)
                and issubclass(owner, torch.autograd.Function))

    def observe(frame: FrameType, event: Literal["call", "return", "c_call", "c_return", "c_exception"], arg: Any) -> None:
        if session.authored and not session.opaque:
            effect = (forbidden.get(frame.f_code) if event == "call"
                      else "custom autograd.Function backward" if event == "c_call" and _is_function_apply(arg)
                      else None)
            if effect is not None:
                raise ValueError(f"Authored {effect} needs an explicit capture effect contract; "
                                 "compose existing Components or an explicit supported tensor operation")
        if previous is not None:
            previous(frame, event, arg)

    try:
        sys.setprofile(observe)
        with _ModeEffects(session):
            yield
    finally:
        sys.setprofile(previous)
