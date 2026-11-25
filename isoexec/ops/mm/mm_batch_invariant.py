# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The vendored functions below are copied from vLLM 6f573f486,
# vllm/model_executor/layers/batch_invariant.py (Apache-2.0); see NOTICE.
"""VENDORED: the batch-invariant dense-GEMM kernels that ``mm`` binds, copied from vLLM.

``mm`` is every dense GEMM in every composition entry, on every step of every host, and its
rounding schedule is pinned by the IsoExec contract. Under kernel-ownership.md §1 that makes it
ours to own: it determines a pinned rounding schedule (test 1), and its upstream --
``vllm.model_executor.layers.batch_invariant`` -- is an INTERNAL module of a fast-moving project
with no API-stability promise (test 2), not a released versioned artifact whose schedule we fix by
the parameters we pass. So it is vendored, not depended on. FA3 sits on the other side of that same
line and stays external.

WHAT THIS FILE IS. Nine functions, 412 lines, copied BYTE FOR BYTE out of the pinned wheel's
``batch_invariant.py`` (983 lines / 28 functions) -- the subset reachable from the bindings
``ops/`` actually has, established by following the live call paths rather than by copying the
file wholesale:

    aten::mm      -> mm_batch_invariant      -> matmul_persistent -> matmul_kernel_persistent
    aten::addmm   -> addmm_batch_invariant   -> matmul_persistent -> matmul_kernel_persistent
    aten::bmm     -> bmm_batch_invariant     -> bmm_kernel
    aten::softmax -> softmax_batch_invariant                       (a python composition over aten)

On sm90, ``matmul_persistent`` is wrapped by the segment-tree GEMM (``ops/mm/mm_sm90``), which serves
every bf16 shape it expresses; this kernel serves the rest (fp32 router, N = 1 gate, bias, K % 64).

``softmax_batch_invariant`` is the MoE router's fp32 softmax and the ninth symbol. It was the last
runtime import on the reference router's path: the reference trainer bound it through a
forwarder in ``ops/moe/moe_sm90/moe_batch_invariant.py`` that did ``from vllm... import batch_invariant``,
so the reference trainer -- the thing every gate compares against -- could not be built in a process
without vLLM installed. Vendoring it here closes that (kernel-ownership.md section 2a) and is what
``ops/mm/tests/test_softmax_vendor_no_vllm_cpu.py`` asserts, by running the reference router in a
subprocess with ``sys.modules["vllm"] = None``. It is five lines of pure torch and it is NOT a
transcription: the digest below is computed from the upstream file and from this copy by the same
``ast`` slice, and they agree. Note it accepts and IGNORES ``dtype`` -- upstream does, the aten
``softmax`` override is called with it, and reproducing that faithfully is the whole point of a
verbatim copy.

Nothing here is invented. The copy is verbatim so ``_VENDORED_FROM["symbols"]`` can carry a sha256
per function and ``tests/test_mm_vendor_provenance_cpu.py`` can detect any local edit to the
recorded vLLM 0.22 copy. The copy was qualified for determinism when it was vendored and is not
compared with whatever vLLM is installed.

# TODO(jiangjiawei1103): Replace the entire vendored kernel implementation with a verbatim copy from vLLM 0.30.0.

WHAT IS NOT VERBATIM, and why. Exactly three module-level substitutions, each recorded in
``_VENDORED_FROM["substitutions"]`` and each checked by the provenance test:

  * ``from vllm.triton_utils import tl, triton`` -> plain ``triton`` / ``triton.language``.
    vLLM's shim only supplies a placeholder when triton is missing; the pinned SkyRL lock installs
    triton and the A23 toolchain fingerprint compares its version, so importing it directly is the
    same object with one fewer dependency.
  * ``vllm.utils.platform_utils.num_compute_units`` -> the two-line torch expression it resolves to
    on CUDA (``torch.cuda.get_device_properties(i).multi_processor_count``). This value is a
    ``tl.constexpr`` (``NUM_SMS``) and therefore part of the launch geometry, NOT of the K-reduction
    order; it is reproduced exactly rather than approximated.
  * ``_fp16_block_size_n``, which upstream assigns inside ``enable_batch_invariant_mode`` from
    ``get_max_shared_memory_bytes()``. Reproduced as ``configure_fp16_block_size_n()`` over torch's
    own ``shared_memory_per_block_optin`` attribute. fp16 only -- both models here are bf16/fp32, so
    this constant is off every live path -- but it is reproduced so the copy is complete rather than
    subtly narrower than what it replaces.

The provenance test protects the recorded 0.22 source identity and checks the installed 0.30
function names and re-expressed device rules. The bitwise gate validates the retained numerical
behavior against installed vLLM 0.30 on its stated cases. Neither measures throughput regressions.

THE SEAM. This module is pure torch+triton and imports no runtime. Installing it into a live vLLM
process -- the inverted direction, where we write OUR numerics into the engine -- is
``mm_runtime_seam.py``, which is the one file under ``ops/mm/`` that still touches ``vllm`` and is
§2c plumbing by construction.

Rounding schedules recorded on this module's ops (moved verbatim from ``ImplSpec.rounding.documentary``, which
M6.4 deleted; the registry keeps only the pins a host must set):

``mm:triton_batch_invariant``
    batch-invariant persistent matmul; fixed BLOCK_SIZE_K, no split-K, so element (m,n) reduces in one fixed K
    order regardless of M (batch).
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable
from typing import Any

import torch
import triton
import triton.language as tl

#: Machine-readable origin of this copy. ``tests/test_mm_vendor_provenance_cpu.py`` verifies that
#: these local functions still match the recorded vLLM 0.22 hashes. The record is citation; the copy
#: is not compared with the installed vLLM.
_VENDORED_FROM = {
    "license": "Apache-2.0",
    "project": "vllm",
    "module": "vllm.model_executor.layers.batch_invariant",
    "version": "0.22.1rc1.dev436+g6f573f486",
    "commit": "g6f573f486",
    "wheel": "pinned-wheels/vllm-1.0.0.dev20260620+cu130-cp312-cp312-linux_x86_64.whl",
    "url": "https://github.com/vllm-project/vllm/blob/6f573f486/vllm/model_executor/layers/batch_invariant.py",
    "vendored_on": "2026-09-04 (softmax_batch_invariant added 2026-09-07)",
    "upstream_total_lines": 983,
    "vendored_lines": 412,
    # sha256 of each symbol's decorated source text, exactly as `ast` slices it out of the upstream
    # module file. Verbatim copies, so these digests are computable from BOTH sides.
    "symbols": {
        "_matmul_launch_metadata": "3fca2cf65c17782f2323877c6d2c4e7298d4646f66a5ccddaa041d5721cb848e",
        "_compute_pid": "acc2e58a3fcc4cdcfa97b6b0b2053b56c6064ea2ae480090ea0da3f5244e7c3b",
        "matmul_kernel_persistent": "a59df4aac92de654ffb0a07e2a3714c7fcc9a684d96dfdfbdc993eb4909828f5",
        "matmul_persistent": "4ea941a86dc63bf650c3f1d741c39f6765428729b3762f6d2d430743688ba320",
        "bmm_kernel": "61f3d3a1f9f8d0955777211e408c17d5eb69ad475086cafe5e8dd7d89a1cad3e",
        "bmm_batch_invariant": "acccb02b00a3ae3b5124253cce758df8e3f42f4c36ebb2894b49494f1539ae46",
        "mm_batch_invariant": "9486abaa90312dd94208755fefe18d45762aa9a76bf747f83a349031377782bb",
        "addmm_batch_invariant": "7c512163d38c63e2a24c4a60c69a9ac485d95fafadf988011111be159a64dc55",
        "softmax_batch_invariant": "480fe74cf2eb6abda52e722108026a4741178bf46c565b5960f08e929d0ea6b6",
    },
    # The three deliberate module-level departures, as (what upstream does, what we do, why).
    "substitutions": [
        (
            "from vllm.triton_utils import tl, triton",
            "import triton; import triton.language as tl",
            "vLLM's shim only stubs triton when it is absent; the pinned SkyRL lock installs it and "
            "the A23 toolchain fingerprint compares its version.",
        ),
        (
            "from vllm.utils.platform_utils import num_compute_units",
            "num_compute_units() below",
            "resolves on CUDA to torch.cuda.get_device_properties(i).multi_processor_count; "
            "reproduced exactly. NUM_SMS is launch geometry, not K-reduction order.",
        ),
        (
            "_fp16_block_size_n = 256 if get_max_shared_memory_bytes() > 106496 else 128",
            "configure_fp16_block_size_n()",
            "same rule over torch's shared_memory_per_block_optin. fp16 only; off every live path "
            "here, reproduced for completeness.",
        ),
    ],
}

#: The upstream statement ``configure_fp16_block_size_n`` re-expresses, asserted verbatim by the
#: provenance test so this one non-verbatim piece cannot drift unnoticed either.
_FP16_BLOCK_SIZE_N_UPSTREAM_STMT = "_fp16_block_size_n = 256 if get_max_shared_memory_bytes() > 106496 else 128"


def provenance() -> dict:
    """The recorded upstream identity of this copy. Read by the provenance test and by the
    install banner, so a running process can say which upstream commit its ``mm`` came from."""
    return dict(_VENDORED_FROM)


def source_digests() -> dict:
    """sha256 per vendored symbol, recomputed from THIS file.

    The provenance test compares three things and they must all agree: what upstream has now, what
    ``_VENDORED_FROM["symbols"]`` records, and what this file actually contains. Comparing only the
    first two would let an edit to the copy pass unnoticed.
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


# ---------------------------------------------------------------------------------------------
# Substituted helpers (NOT verbatim -- see _VENDORED_FROM["substitutions"]).
# ---------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def num_compute_units(device_id: int = 0) -> int:
    """SM count for ``device_id``.

    Upstream reaches this through ``vllm.utils.platform_utils.num_compute_units`` ->
    ``CudaPlatform.num_compute_units`` -> exactly this expression. It feeds ``NUM_SMS``, a
    ``tl.constexpr`` that decides how many persistent programs launch and therefore which tiles a
    program walks -- it does NOT change the order in which one output element reduces over K, which
    is what batch invariance is about. Cached because it is read on every ``matmul_persistent``.
    """
    if device_id is None:
        device_id = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device_id).multi_processor_count


#: fp16 N-tile size. Upstream sets this inside ``enable_batch_invariant_mode``; both models on this
#: branch are bf16/fp32 so it is off every live path, but the default and the rule are reproduced.
_fp16_block_size_n = 256


def configure_fp16_block_size_n(device_id: int = 0) -> int:
    """Reproduce upstream's ``_fp16_block_size_n`` assignment without importing vLLM.

    Upstream: ``256 if get_max_shared_memory_bytes() > 106496 else 128``, where
    ``get_max_shared_memory_bytes`` is ``cudaDevAttrMaxSharedMemoryPerBlockOptin``. torch exposes
    the same attribute as ``shared_memory_per_block_optin``. Never raises: a device that cannot
    answer keeps the 256 default, which is what upstream's module-level default is too.
    """
    global _fp16_block_size_n
    try:
        props = torch.cuda.get_device_properties(device_id)
        optin = getattr(props, "shared_memory_per_block_optin", None)
        if optin is None:
            return _fp16_block_size_n
        _fp16_block_size_n = 256 if int(optin) > 106496 else 128
    except Exception:  # noqa: BLE001 -- an unqueryable device keeps the upstream default
        pass
    return _fp16_block_size_n


# ---------------------------------------------------------------------------------------------
# VERBATIM from vllm.model_executor.layers.batch_invariant @ 6f573f486.
# Do not edit below this line. Edits break the provenance digests, which is the point: the copy is
# only trustworthy while it is provably a copy. To take upstream's improvements, re-run the
# extraction and bump _VENDORED_FROM, with the bitwise gate re-run as the evidence.
# ---------------------------------------------------------------------------------------------

# fmt: off
# The formatter is switched off for the rest of the file ON PURPOSE: black would reflow upstream's
# 80-column layout to this repo's 120 and every provenance digest below would stop matching the
# wheel it was copied from. A verbatim copy has to stay byte-for-byte verbatim or the recorded
# digests stop identifying it. Nothing after this marker is ours to format.


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: dict[str, Any]
) -> dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"

    bytes_per_elem = args["c_ptr"].element_size()
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1, (
        "Currently assuming bias is 1D, let Horace know if you run into this"
    )
    NUM_SMS = num_compute_units(a.device.index)
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": _fp16_block_size_n,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        bias,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs[dtype],
    )
    return c


@triton.jit
def bmm_kernel(
    a_ptr,  # (*, ) pointer to A, (B, M, K)
    b_ptr,  # (*, ) pointer to B, (B, K, N)
    c_ptr,  # (*, ) pointer to C, (B, M, N)
    B,  # int, batch size
    M,  # int, output rows
    N,  # int, output cols
    K,  # int, reduction dim
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    """Batched GEMM: (B, M, K) x (B, K, N) -> (B, M, N)

    Each program computes one (batch_idx, tile_m, tile_n) tile, accumulating
    along K in a fixed order to preserve batch invariance.
    """
    pid_b = tl.program_id(0)
    pid = tl.program_id(1)

    if pid_b >= B:
        return

    # number of tiles along M / N
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return

    # offs_m / offs_n: raw global row/col indices for this tile
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # masks for valid logical rows/cols within (M, N)
    mask_m = offs_m < M  # [BLOCK_SIZE_M]
    mask_n = offs_n < N  # [BLOCK_SIZE_N]

    if A_LARGE or B_LARGE or C_LARGE:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)

    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = tl.where(mask_n, offs_n, 0)

    # hint for triton contiguous memory
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # base pointers for current batch, shape-wise:
    #   a_batch_ptr points to A[pid_b, 0, 0]
    #   b_batch_ptr points to B[pid_b, 0, 0]
    #   c_batch_ptr points to C[pid_b, 0, 0]
    a_batch_ptr = a_ptr + pid_b * stride_ab
    b_batch_ptr = b_ptr + pid_b * stride_bb
    c_batch_ptr = c_ptr + pid_b * stride_cb

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # number of K-blocks this tile iterates over
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_k_mask = tl.arange(0, BLOCK_SIZE_K)

    for ki in range(k_tiles):
        if A_LARGE or B_LARGE:
            # offs_k: [BLOCK_SIZE_K], global K indices
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        else:
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # a_ptrs: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        #   element (i, j) points to A[pid_b, offs_m[i], offs_k[j]]
        a_ptrs = a_batch_ptr + (
            offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
        # b_ptrs: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        #   element (i, j) points to B[pid_b, offs_k[i], offs_n[j]]
        b_ptrs = b_batch_ptr + (
            offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )

        # valid K lanes for this block
        k_valid = offs_k_mask < (K - ki * BLOCK_SIZE_K)
        # A mask within (M, K): [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_mask = mask_m[:, None] & k_valid[None, :]
        # B mask within (K, N): [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_mask = k_valid[:, None] & mask_n[None, :]

        # a: [BLOCK_SIZE_M, BLOCK_SIZE_K] from A[offs_m, offs_k]
        a = tl.load(
            a_ptrs,
            mask=a_mask,
            other=0.0,
        )
        # b: [BLOCK_SIZE_K, BLOCK_SIZE_N] from B[offs_k, offs_n]
        b = tl.load(
            b_ptrs,
            mask=b_mask,
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)

    # c_m / c_n: [BLOCK_SIZE_M] / [BLOCK_SIZE_N], row/col indices for C
    c_m = offs_m
    c_n = offs_n
    if C_LARGE:
        c_m = c_m.to(tl.int64)
        c_n = c_n.to(tl.int64)

    # c_ptrs: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    #   element (i, j) points to C[pid_b, c_m[i], c_n[j]]
    c_ptrs = c_batch_ptr + stride_cm * c_m[:, None] + stride_cn * c_n[None, :]
    # mask out elements that fall outside logical (M, N) range
    c_mask = mask_m[:, None] & mask_n[None, :]
    # cast FP32 accumulator back to original dtype of C
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


def bmm_batch_invariant(a, b, *, out=None):
    # Batched matrix multiply: (B, M, K) x (B, K, N) -> (B, M, N)
    if not (a.ndim == 3 and b.ndim == 3):
        raise ValueError(
            f"bmm_batch_invariant expects 3D tensors, "
            f"got shapes {a.shape} and {b.shape}"
        )

    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Batch dimensions of tensors must match, "
            f"but got {a.shape[0]} and {b.shape[0]}."
        )
    if a.shape[2] != b.shape[1]:
        raise ValueError(
            f"Incompatible inner dimensions for matmul: got {a.shape} and {b.shape}."
        )
    if a.dtype != b.dtype:
        raise ValueError(f"Incompatible dtypes: got {a.dtype} and {b.dtype}.")

    B, M, K = a.shape
    _, _, N = b.shape
    dtype = a.dtype

    if out is None:
        c = torch.empty((B, M, N), device=a.device, dtype=dtype)
    else:
        assert out.shape == (B, M, N), "out tensor has incorrect shape"
        assert out.dtype == dtype and out.device == a.device, "out tensor mismatch"
        c = out

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": _fp16_block_size_n,
            "BLOCK_SIZE_K": 64,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "num_stages": 3,
            "num_warps": 8,
        },
    }

    cfg = configs[dtype]
    # grid = (B, num_tiles_per_matrix)
    grid = (
        B,
        triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),
    )

    bmm_kernel[grid](
        a,
        b,
        c,
        B,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        **cfg,
    )

    return c


def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)


def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)


def softmax_batch_invariant(input, dim, dtype=None):
    # Compute softmax in a deterministic way
    # First subtract max for numerical stability (standard practice)
    input_max = torch.amax(input, dim=dim, keepdim=True)
    input = input - input_max
    exp_x = torch.exp(input)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x


# fmt: on

__all__ = [
    "addmm_batch_invariant",
    "bmm_batch_invariant",
    "bmm_kernel",
    "configure_fp16_block_size_n",
    "matmul_kernel_persistent",
    "matmul_persistent",
    "mm_batch_invariant",
    "num_compute_units",
    "provenance",
    "softmax_batch_invariant",
    "source_digests",
]
