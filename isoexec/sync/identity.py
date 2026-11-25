"""The forward program's identity, carried with every weight update from the trainer to the engine.

A model package whose computed function is not fully determined by the weights and the shared configuration
(DeepSeek V4.1: the CED prefill program, its replay window, the index-score fold, the cache record programs)
gives its placed components ``program_identity()`` (compared) and optionally ``program_recorded()`` (switches
that select a transport or a backward but not the forward bits: logged next to the identity, never compared).

The trainer's weight stream starts with one extra entry, ``NAME``: the canonical JSON of ``{"identity": ...,
"recorded": ...}`` as a 1-D BF16 tensor of its UTF-8 bytes (every byte value 0..255 is exact in BF16, so the
stream's dtype handling cannot change it). The engine compares it with its own identity at every update and
refuses a mismatch by name, listing each differing entry; an update of such a model without the entry is
refused too. Models without an identity stream nothing extra.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import torch

NAME = "isoexec.program_identity"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"program identity entries must be plain data, got {type(value).__name__}")


def stage_identity(module: torch.nn.Module) -> dict | None:
    """``{"identity", "recorded"}`` of the first module that states a program identity (every placed member of a
    model states the same one), or None."""
    walk = getattr(module, "modules", None)
    for child in (walk() if callable(walk) else ()):
        identity = getattr(child, "program_identity", None)
        if callable(identity):
            recorded = getattr(child, "program_recorded", None)
            return {"identity": _plain(identity()), "recorded": _plain(recorded() if callable(recorded) else {})}
    return None


def encode(payload: Mapping) -> torch.Tensor:
    data = json.dumps(_plain(payload), sort_keys=True, separators=(",", ":")).encode()
    return torch.tensor(list(data), dtype=torch.bfloat16)


def decode(tensor: torch.Tensor) -> dict:
    values = tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if values.numel() == 0 or bool(((values < 0) | (values > 255) | (values != values.round())).any()):
        raise ValueError(f"{NAME}: not a byte string")
    return json.loads(bytes(int(v) for v in values.tolist()).decode())


def differences(engine: Mapping, trainer: Mapping, prefix: str = "") -> list[str]:
    out = []
    for key in sorted(set(engine) | set(trainer)):
        a, b = engine.get(key, "<absent>"), trainer.get(key, "<absent>")
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            out += differences(a, b, f"{prefix}{key}.")
        elif a != b:
            out.append(f"{prefix}{key}: engine={a!r} trainer={b!r}")
    return out


def check(engine: Mapping, received: Mapping) -> None:
    """Refuse a trainer whose forward program differs from the engine's (``recorded`` entries are not compared)."""
    diff = differences(engine["identity"], received.get("identity", {}))
    if diff:
        raise ValueError("PROGRAM_IDENTITY_MISMATCH: the trainer's forward program differs from the engine's: "
                         + "; ".join(diff) + f" (recorded, not compared: engine {engine.get('recorded')}, "
                         f"trainer {received.get('recorded')})")


__all__ = ["NAME", "check", "decode", "differences", "encode", "stage_identity"]
