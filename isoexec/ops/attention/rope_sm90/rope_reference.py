# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# The vendored functions below are copied from Megatron-LM 71e418ea7,
# megatron/core/models/common/embeddings/rope_utils.py, under the BSD 3-Clause License
# reproduced in NOTICE.
"""Megatron rotate-half RoPE vendored as the eager rope.apply reference.
The recipe rounds fp32 cos/sin to the operand dtype and uses three bf16 multiply/add rounds;
the fp32 single-round recipe is numerically distinct.
Symbol provenance below is citation; test_rope_vendor_provenance_cpu.py checks this copy against its recorded digests."""

from __future__ import annotations

import hashlib
import warnings
from typing import Optional

import torch
from torch import Tensor

#: Digests cover each vendored symbol's decorated source text on both sides.
_VENDORED_FROM = {
    "license": "BSD-3-Clause",
    "project": "megatron-core",
    "module": "megatron.core.models.common.embeddings.rope_utils",
    "version": "0.19.0+71e418ea7",
    "commit": "71e418ea7",
    "wheel": "megatron_core-0.19.0+71e418ea7 (isoexec runtime venv; NVIDIA/Megatron-LM)",
    "url": "https://github.com/NVIDIA/Megatron-LM/blob/71e418ea7/megatron/core/models/common/embeddings/rope_utils.py",
    "vendored_on": "2026-09-04",
    "upstream_total_lines": 461,
    "vendored_lines": 51,
    "symbols": {
        "_rotate_half": "de7b8b4e63615eb85f5334de600caf51c665f0ac17d98288bbfbb33c451015e8",
        "_apply_rotary_pos_emb_bshd": "4a54b66e47ae8722278401fdd948138540678e67e2331067685ef94cdc801282",
    },
    # No substitutions are needed; provenance tests assert the empty table.
    "substitutions": [],
}


def provenance() -> dict:
    """The recorded upstream identity of this copy."""
    return dict(_VENDORED_FROM)


def source_digests() -> dict:
    """sha256 per vendored symbol, recomputed from THIS file.

    Comparing only "upstream now" against "the record" would let an edit to the copy pass as
    agreement; this is the third leg that closes that.
    """
    import ast
    import pathlib

    text = pathlib.Path(__file__).read_text()
    lines = text.split("\n")
    out = {}
    for node in ast.parse(text).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in _VENDORED_FROM["symbols"]:
            continue
        first = min([d.lineno for d in node.decorator_list] + [node.lineno])
        body = "\n".join(lines[first - 1 : node.end_lineno]) + "\n"
        out[node.name] = hashlib.sha256(body.encode()).hexdigest()
    return out


# VENDORED REGION -- BYTE FOR BYTE. DO NOT EDIT, DO NOT REFORMAT.
#
# Preserve upstream formatting within the byte-checked symbol region.
# fmt: off
def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
    multi_latent_attention: Optional[bool] = None,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]
        rotary_interleaved (bool): Whether to apply interleaving in the rotate half function.
        mla_rotary_interleaved (bool): Whether to apply MLA-style interleaving for RoPE.
        mscale (float): The scaling factor for the RoPE.

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    if multi_latent_attention is not None:
        warnings.warn(
            "multi_latent_attention is deprecated. Please use mla_rotary_interleaved instead.",
            DeprecationWarning,
        )
        mla_rotary_interleaved = multi_latent_attention

    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if mla_rotary_interleaved:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)

    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)
# fmt: on
# END VENDORED REGION


def rope_apply_eager(q, k, freqs, *, style: str, rotary_dim: int, seq=None):
    """Apply vendored RoPE to q/k [T, heads, head_dim] through a B=1 bshd view.
    Doubled freqs form [T,1,1,rot]; style selects the upstream pair layout and partial rotary
    leaves the tail untouched. The call-convention argument seq is unused."""
    if freqs.shape[-1] * 2 != rotary_dim:
        raise ValueError(f"freqs [T,{freqs.shape[-1]}] do not match rotary_dim {rotary_dim}")
    emb = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]  # RotaryEmbedding.get_emb layout [T,1,1,rot]
    inter = style == "interleaved"
    q4 = _apply_rotary_pos_emb_bshd(q.unsqueeze(1), emb, rotary_interleaved=inter, mscale=1.0)
    k4 = _apply_rotary_pos_emb_bshd(k.unsqueeze(1), emb, rotary_interleaved=inter, mscale=1.0)
    return q4.squeeze(1), k4.squeeze(1)


__all__ = ["rope_apply_eager", "provenance", "source_digests"]
