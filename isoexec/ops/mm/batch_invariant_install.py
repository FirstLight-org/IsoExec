"""Install the owned CUDA reductions explicitly in trainer processes."""

from __future__ import annotations

import os

import torch

from . import batch_invariant_reductions as reductions
from . import mm_batch_invariant as gemm

_library = None


def installed() -> bool:
    return _library is not None


def install() -> None:
    global _library
    if installed():
        return
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("IsoExec trainer installation requires a certified SM90 or newer CUDA device")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"
    # Setting the environment cannot repair parsers initialized by an earlier GEMM.
    workspace = (
        torch.backends.cuda.cublas_workspace_size(),
        torch.backends.cuda.cublaslt_workspace_size(),
    )
    if workspace != (131072, 1024):
        raise RuntimeError(
            f"IsoExec cuBLAS workspace sizes are {workspace} bytes, expected (131072, 1024); "
            "restart with CUBLAS_WORKSPACE_CONFIG=:16:8 and CUBLASLT_WORKSPACE_SIZE=1 "
            "set before the first native GEMM"
        )
    gemm.configure_fp16_block_size_n(torch.cuda.current_device())
    library = torch.library.Library("aten", "IMPL")
    for name, function in (
        ("_log_softmax", reductions._log_softmax_batch_invariant),
        ("softmax", gemm.softmax_batch_invariant),
        ("_softmax", gemm.softmax_batch_invariant),
        ("mean.dim", reductions.mean_batch_invariant),
        ("bmm", gemm.bmm_batch_invariant),
    ):
        library.impl(name, function, "CUDA", allow_override=True)
    torch.bmm = gemm.bmm_batch_invariant
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (False, False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    torch.backends.cuda.preferred_blas_library(backend="cublaslt")
    _library = library
