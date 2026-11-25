"""Standard packed-text examples and native root binding for model adapters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import inspect
from typing import Any, Literal

import torch
from torch import nn

from isoexec.checks.config import component_facts
from isoexec.schema import SequenceContext, make_sequence_context

from isoexec.compiler.capture.input_policy import _normalize_token_ids
from isoexec.compiler.capture.rootspec import RootTensorLayout


@dataclass(frozen=True)
class CaptureInputs:
    """Worker-local authored arguments and optional internal sequence input.

    dynamic_shapes describes authored parameters in signature order, followed by
    seq when supplied here. No model, tensors, or graph cross workers.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
    dynamic_shapes: tuple[Any, ...] | None = None
    root_layouts: tuple[RootTensorLayout, ...] = ()
    seq: SequenceContext | None = None


def text_roots(input_ids: torch.Tensor, seq: SequenceContext) -> dict[str, Any]:
    """Bind packed token IDs without changing the caller's sequence metadata.

    Both standard token references point to the same canonical int64 Tensor.
    Other field values, devices, and optional-field structure remain unchanged.
    Native adapters own sequence validation and movement before this binding.
    """
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
        raise ValueError("Standard text input_ids must be a packed one-dimensional Tensor")
    if not isinstance(seq, SequenceContext):
        raise TypeError("Standard text seq must be a SequenceContext")
    if type(seq.max_q) is not int or type(seq.max_kv) is not int:
        raise ValueError("Standard text max_q and max_kv must be host integers")
    if seq.max_q < 1 or seq.max_kv < seq.max_q:
        raise ValueError("Standard text lengths require 1 <= max_q <= max_kv")
    return _normalize_token_ids({"input_ids": input_ids, "seq": replace(seq, input_ids=input_ids)})


def text_inputs(
    model: nn.Module, *, device: str | torch.device,
    tokens: int = 4, max_tokens: int = 4096,
    profile: Literal["trainer", "engine"] = "trainer",
) -> CaptureInputs:
    """Provide packed-text inputs and an internal seq when forward omits it.

    Token and packed-request counts vary independently; host length integers use
    export's dynamic scalar guards. Required roots beyond input_ids/seq need
    explicit capture examples, rather than inferred values or dtypes.
    """
    if profile not in ("trainer", "engine"):
        raise ValueError("Text input profile must be 'trainer' or 'engine'")
    if type(tokens) is not int or type(max_tokens) is not int or not 2 <= tokens <= max_tokens:
        raise ValueError("Text capture requires 2 <= tokens <= max_tokens; the dynamic range includes runtime T=1")
    signature = inspect.signature(model.forward)
    required = [name for name, parameter in signature.parameters.items()
                if parameter.default is inspect.Parameter.empty
                and name not in ("input_ids", "seq")
                and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
    if required or "input_ids" not in signature.parameters:
        raise ValueError(f"Standard text inputs require forward parameter input_ids; "
                         f"other required roots {required} need explicit CaptureInputs")
    if any(parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
           for parameter in signature.parameters.values()):
        raise ValueError("A variadic model forward needs explicit CaptureInputs")

    target = torch.device(device)
    ids = torch.zeros(tokens, dtype=torch.int64, device=target)
    # Use two packed requests so request-count capture is genuinely dynamic.
    split = tokens // 2
    cu_cpu = torch.tensor([0, split, tokens], dtype=torch.int32)
    positions = torch.cat((torch.arange(split), torch.arange(tokens - split))).to(device=target)
    mrope = any(facts.get("mrope_section") is not None for facts in component_facts(model).values())
    if mrope:
        positions = positions.expand(3, -1).clone()
    seq = make_sequence_context(positions, cu_cpu.to(device=target, copy=True), max_q=tokens - split,
                                max_kv=tokens - split, input_ids=ids,
                                cu_seqlens_cpu=cu_cpu if profile == "trainer" else None)
    roots = text_roots(ids, seq)
    token_dim = torch.export.Dim("tokens", min=1, max=max_tokens)
    offsets_dim = torch.export.Dim("packed_offsets", min=2, max=max_tokens + 1)
    sequence_shapes = [
        {1 if mrope else 0: token_dim}, {0: offsets_dim}, torch.export.Dim.DYNAMIC,
        torch.export.Dim.DYNAMIC, {0: token_dim}, None, None,
        {0: offsets_dim} if profile == "trainer" else None,
    ]
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    shapes: list[Any] = []
    for name, parameter in signature.parameters.items():
        value = roots[name] if name in roots else parameter.default
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[name] = value
        else:
            args.append(value)
        shapes.append({0: token_dim} if name == "input_ids" else sequence_shapes if name == "seq" else None)
    internal_seq = None if "seq" in signature.parameters else roots["seq"]
    if internal_seq is not None:
        shapes.append(sequence_shapes)
    locality = "cp_local" if profile == "trainer" else "global"
    layouts = (
        RootTensorLayout("input_ids", locality, token_axis=0 if locality == "cp_local" else None),
        RootTensorLayout("seq", locality, token_axis=-1 if locality == "cp_local" else None,
                         path=(("attr", "positions"),)),
        RootTensorLayout("seq", "global", path=(("attr", "cu_seqlens"),)),
    )
    if profile == "trainer":
        layouts += (RootTensorLayout("seq", "global", path=(("attr", "cu_seqlens_cpu"),)),)
    return CaptureInputs(tuple(args), kwargs, tuple(shapes), layouts, internal_seq)


__all__ = ["CaptureInputs", "text_inputs", "text_roots"]
