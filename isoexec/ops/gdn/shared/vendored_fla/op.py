# ---------------------------------------------------------------------------------------------
# VENDORED KERNEL -- DO NOT EDIT BY HAND.
# upstream_module: vllm.model_executor.layers.fla.ops.op
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

import os

from ._compat import tl, tldevice, triton

from .utils import is_gather_supported

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
    exp2 = tl.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    exp = tl.exp
    exp2 = tl.exp2
    log = tl.log
    log2 = tl.log2


if not is_gather_supported:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Gather operation that works when tl.gather is not supported.
        This is a fallback implementation that returns None.
        Just to make triton compiler happy.
        """
        return None
else:
    gather = tl.gather

if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    # For Triton 3.3.x
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, "make_tensor_descriptor"):
    # For Triton 3.4.x and later
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    """
    Fallback implementation when TMA is not supported.
    Returns None to indicate TMA descriptors are unavailable.
    Just make triton compiler happy.
    """

    @triton.jit
    def make_tensor_descriptor(
        base,
        shape,
        strides,
        block_shape,
        _builder=None,
    ):
        return None
