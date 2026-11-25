# ---------------------------------------------------------------------------------------------
# VENDORED KERNEL -- DO NOT EDIT BY HAND.
# upstream_module: vllm.model_executor.layers.fla.ops.index
# upstream_dist:   vllm 1.0.0.dev20260620+cu130  (scm 0.22.1rc1.dev436+g6f573f486, commit 6f573f486)
# upstream_url:    https://github.com/vllm-project/vllm (Apache-2.0; see NOTICE)
# provenance:      ops/gdn/shared/vendored_fla/PROVENANCE.json
# local changes:   only the rewrite_rules recorded in PROVENANCE.json; vendored_sha256 pins this file.
# why vendored:    kernel-ownership.md 1 -- this kernel DETERMINES a rounding schedule the
#                  IsoExec arithmetic contract pins, and its upstream is an internal module of a
#                  fast-moving project with no API-stability contract. Test 2 says: vendor it.
# ---------------------------------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch

from ._compat import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
