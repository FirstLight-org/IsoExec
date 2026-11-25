"""Consistent process-wide IEEE FP32 policy for contracted native execution."""

import torch
from typing import Protocol, cast


class _FP32PrecisionControl(Protocol):
    """Public PyTorch descriptors omitted by the supported 2.11 type hints."""

    fp32_precision: str


class _CuDNNPrecisionControl(_FP32PrecisionControl, Protocol):
    allow_tf32: bool
    conv: _FP32PrecisionControl
    rnn: _FP32PrecisionControl


def pin_fp32_ieee() -> None:
    """Disable TF32 without changing BF16 kernels or reduction settings.

    PyTorch 2.11/2.13 export still reads cuDNN's legacy compatibility flag.
    New precision setters leave that flag stale, even when conv/RNN both use
    IEEE. Synchronize it through the public setter before applying the complete
    new policy; do not bypass export's backend checks or enable TF32.
    """
    cudnn = cast(_CuDNNPrecisionControl, torch.backends.cudnn)
    cudnn.allow_tf32 = False
    cast(_FP32PrecisionControl, torch.backends).fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    cudnn.fp32_precision = "ieee"
    cudnn.conv.fp32_precision = "ieee"
    cudnn.rnn.fp32_precision = "ieee"


__all__ = ["pin_fp32_ieee"]
