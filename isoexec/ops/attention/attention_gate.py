# _apply_output_gate is copied from Megatron-LM 71e418ea7, megatron/core/transformer/attention.py:
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Used under the BSD 3-Clause License reproduced in NOTICE. The rest of this file is IsoExec's (Apache-2.0).
"""Attention output gates: compiled fp32 sigmoid and eager fp32 SiLU, each with one output-dtype round.
The sigmoid method is vendored from megatron-core; its torch.compile decorator is part of the rounding schedule.
Provenance and symbol digests below are checked by test_attention_gate_provenance_cpu.py on every CI run and wheel bump."""

from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

#: Upstream provenance and recorded symbol digests.
_VENDORED_FROM = {
    "license": "BSD-3-Clause",
    "project": "megatron-core",
    "module": "megatron.core.transformer.attention",
    "version": "0.19.0+71e418ea7",
    "commit": "71e418ea7",
    "wheel": "megatron_core-0.19.0+71e418ea7 (isoexec runtime venv; NVIDIA/Megatron-LM)",
    "url": "https://github.com/NVIDIA/Megatron-LM/blob/71e418ea7/megatron/core/transformer/attention.py",
    "vendored_on": "2026-09-04",
    "upstream_total_lines": 1377,
    "vendored_lines": 8,
#: Keep the method at upstream indentation for byte-for-byte source hashing.
    "symbols": {
        "Attention._apply_output_gate": "bf3f16356bbfac10e233d0f611d39ab15d9492da69eb93f6e1fbcef5bcc59003",
    },
    "vendored_symbol_holder": "_McoreAttention",
    "substitutions": [
        (
            "megatron.core.jit.jit_fuser (torch.compile via enable_jit_fuser)",
            "jit_fuser = torch.compile, with the same torch>=2.2.0a0 predicate, refusing below it",
            "The decorator IS part of the rounding schedule: inductor's sigmoid differs from ATen's "
            "by one ulp on some inputs, so an undecorated copy is a different function. mcore's "
            "sub-2.2 fallbacks (torch.jit.script / no-op) are two more different functions, so we "
            "raise instead of taking one.",
        ),
    ],
}

#: Pin the upstream decorator-selection statement in the provenance test.
_JIT_FUSER_UPSTREAM_STMT = '        if is_torch_min_version("2.2.0a0"):\n            jit_fuser = torch.compile'

#: mcore's own threshold, reproduced rather than approximated.
_MIN_TORCH_FOR_COMPILE = "2.2.0a0"


def _torch_is_at_least(version: str) -> bool:
    """``megatron.core.utils.is_torch_min_version`` -- ``Version(torch.__version__) >= Version(v)``."""
    from packaging.version import Version

    return Version(torch.__version__.split("+")[0]) >= Version(version)


if not _torch_is_at_least(_MIN_TORCH_FOR_COMPILE):  # pragma: no cover -- pinned wheels are far above
    raise RuntimeError(
        f"[isoexec] attention.gate: torch {torch.__version__} is below {_MIN_TORCH_FOR_COMPILE}, where "
        "megatron's jit_fuser stops being torch.compile. The vendored gate is bitwise-equal to the "
        "COMPILED function only; torch.jit.script and the no-op decorator are different rounding "
        "schedules. Refusing rather than matching the wrong reference."
    )

#: What ``megatron/core/jit.py`` resolves to on every torch this repo pins. See the substitution.
jit_fuser = torch.compile


def provenance() -> dict:
    """The recorded upstream identity of this copy."""
    return dict(_VENDORED_FROM)


def source_digests() -> dict:
    """sha256 of the vendored method's decorated source text, recomputed from THIS file.

    Sliced out of ``_McoreAttention`` and keyed by the UPSTREAM name (``Attention.
    _apply_output_gate``), so the digest is directly comparable with the one the provenance test
    slices out of megatron.
    """
    import ast
    import pathlib

    text = pathlib.Path(__file__).read_text()
    lines = text.split("\n")
    holder = _VENDORED_FROM["vendored_symbol_holder"]
    out = {}
    for node in ast.parse(text).body:
        if not isinstance(node, ast.ClassDef) or node.name != holder:
            continue
        for sub in node.body:
            if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            key = f"Attention.{sub.name}"
            if key not in _VENDORED_FROM["symbols"]:
                continue
            first = min([d.lineno for d in sub.decorator_list] + [sub.lineno])
            body = "\n".join(lines[first - 1 : sub.end_lineno]) + "\n"
            out[key] = hashlib.sha256(body.encode()).hexdigest()
    return out


# VENDORED REGION -- BYTE FOR BYTE. DO NOT EDIT, DO NOT REFORMAT.
#
# Preserve upstream layout for the digest; the shim method is called unbound with self=None.
class _McoreAttention:
    """Shim holder for the vendored method. Not a model class; see the note above."""

    # fmt: off
    @jit_fuser
    def _apply_output_gate(self, x, gate):
        x_dtype = x.dtype
        gate = gate.contiguous()
        gate = gate.view(*x.shape)
        x = x * torch.sigmoid(gate.float())
        x = x.to(x_dtype)
        return x
    # fmt: on


# END VENDORED REGION


def attention_gate_sigmoid(o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply compiled fp32 x * sigmoid(gate.float()) with one bf16 round.
    The vendored call uses Megatron's [T, 1, hq*dv] view, then restores the caller's shape."""
    T = o.shape[0]
    x = o.reshape(T, 1, -1)
    g = gate.reshape(T, 1, -1)
    y = _McoreAttention._apply_output_gate(None, x, g)
    return y.reshape(o.shape)


def attention_gate_swish(o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply Qwen3.6's fp32 x * silu(gate.float()) with one bf16 round.
    This path remains eager: compiling it would change its recorded rounding schedule."""
    y = o.to(torch.float32) * F.silu(gate.to(torch.float32))
    return y.to(o.dtype)


__all__ = ["attention_gate_sigmoid", "attention_gate_swish", "provenance", "source_digests"]
