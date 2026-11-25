"""Bit-exact comparison for SkyRL's trainer and engine full-logprob rows."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import Refusal

_U32_MASK = (1 << 32) - 1
_U32_SIGN = 1 << 31
_DEFAULT_CUDA_CHUNK_ROWS = 64


class FullDistributionRefusal(Refusal):
    """The full-distribution evidence is invalid or divergent."""


@dataclass(frozen=True)
class FullDistributionMismatch:
    """Diagnostic for one unequal row; equality itself is decided only by bit views."""

    first_token: int
    trainer_value: float
    engine_value: float
    trainer_u32: int
    engine_u32: int
    mismatch_count: int
    diagnostic_ulp_distance: int


def _validate_row(row: torch.Tensor, *, field: str) -> None:
    if not isinstance(row, torch.Tensor):
        raise FullDistributionRefusal(f"{field}.type={type(row).__name__}; expected torch.Tensor")
    if row.dtype is not torch.float32:
        raise FullDistributionRefusal(f"{field}.dtype={row.dtype}; expected torch.float32 with no conversion")
    if row.ndim != 1 or row.numel() == 0:
        raise FullDistributionRefusal(f"{field}.shape={tuple(row.shape)}; expected a non-empty [V] row")
    if not row.is_contiguous():
        raise FullDistributionRefusal(f"{field}.contiguous=False; the producer must supply a contiguous [V] row")
    if row.device.type != "cpu":
        raise FullDistributionRefusal(f"{field}.device={row.device}; comparison artifacts must be on CPU")
    if not bool(torch.isfinite(row).all().item()):
        raise FullDistributionRefusal(f"{field}.finite=False; full comparison refuses NaN and infinity")


def _u32(value: torch.Tensor) -> int:
    return int(value.item()) & _U32_MASK


def _ordered_u32(bits: int) -> int:
    """Map IEEE-754 bits to an integer order used only for ULP diagnostics."""

    return (~bits & _U32_MASK) if bits & _U32_SIGN else bits | _U32_SIGN


def compare_full_distribution(
    trainer: torch.Tensor,
    engine: torch.Tensor,
) -> FullDistributionMismatch | None:
    """Compare two owned `[V]` fp32 rows by every raw bit."""

    _validate_row(trainer, field="full_distribution.trainer")
    _validate_row(engine, field="full_distribution.engine")
    if trainer.shape != engine.shape:
        raise FullDistributionRefusal(
            f"full_distribution.shape: trainer={tuple(trainer.shape)} engine={tuple(engine.shape)}; "
            "both must be the same [V]"
        )

    trainer_bits = trainer.view(torch.int32)
    engine_bits = engine.view(torch.int32)
    if torch.equal(trainer_bits, engine_bits):
        return None

    unequal = trainer_bits != engine_bits
    first = int(torch.nonzero(unequal, as_tuple=False)[0, 0].item())
    trainer_u32 = _u32(trainer_bits[first])
    engine_u32 = _u32(engine_bits[first])
    return FullDistributionMismatch(
        first_token=first,
        trainer_value=float(trainer[first].item()),
        engine_value=float(engine[first].item()),
        trainer_u32=trainer_u32,
        engine_u32=engine_u32,
        mismatch_count=int(unequal.sum().item()),
        diagnostic_ulp_distance=abs(_ordered_u32(trainer_u32) - _ordered_u32(engine_u32)),
    )


def _validate_cuda_rows(rows: torch.Tensor, *, field: str) -> None:
    if not isinstance(rows, torch.Tensor):
        raise FullDistributionRefusal(f"{field}.type={type(rows).__name__}; expected torch.Tensor")
    if rows.dtype is not torch.float32:
        raise FullDistributionRefusal(f"{field}.dtype={rows.dtype}; expected torch.float32 with no conversion")
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0:
        raise FullDistributionRefusal(f"{field}.shape={tuple(rows.shape)}; expected non-empty [N, V] rows")
    if not rows.is_contiguous():
        raise FullDistributionRefusal(f"{field}.contiguous=False; the producer must supply contiguous [N, V] rows")
    if rows.device.type != "cuda":
        raise FullDistributionRefusal(f"{field}.device={rows.device}; batched comparison requires CUDA rows")


@torch.no_grad()
def compare_full_distribution_cuda(
    trainer: torch.Tensor,
    engine: torch.Tensor,
    *,
    chunk_rows: int = _DEFAULT_CUDA_CHUNK_ROWS,
) -> torch.Tensor:
    """Find the first non-finite or bit-unequal full-logprob row on CUDA.

    Inputs are read-only contiguous float32 ``[N, V_local]`` tensors on the
    same CUDA device. Rows are processed in chunks without host reads.

    Args:
        trainer: Trainer full-logprob rows.
        engine: Matching rollout-engine full-logprob rows.
        chunk_rows: Maximum rows processed by one chunk.

    Returns:
        A 0-D int64 CUDA tensor on the input device. ``N`` means every row
        passes; otherwise the value is the first bad row index.
    """

    _validate_cuda_rows(trainer, field="full_distribution.trainer")
    _validate_cuda_rows(engine, field="full_distribution.engine")
    if trainer.shape != engine.shape:
        raise FullDistributionRefusal(
            f"full_distribution.shape: trainer={tuple(trainer.shape)} engine={tuple(engine.shape)}; "
            "both must be the same [N, V]"
        )
    if trainer.device != engine.device:
        raise FullDistributionRefusal(
            f"full_distribution.device: trainer={trainer.device} engine={engine.device}; "
            "both must be on the same CUDA device"
        )
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise FullDistributionRefusal(f"full_distribution.chunk_rows={chunk_rows!r}; expected a positive integer")

    n = trainer.shape[0]  # number of selected rows ``N``
    sentinel = torch.tensor(n, dtype=torch.int64, device=trainer.device)
    chunk_first_bad = []
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        trainer_chunk = trainer[start:stop]
        engine_chunk = engine[start:stop]
        is_finite = torch.isfinite(trainer_chunk).all(dim=-1) & torch.isfinite(engine_chunk).all(dim=-1)
        equal_bits = (trainer_chunk.view(torch.int32) == engine_chunk.view(torch.int32)).all(dim=-1)
        row_ok = is_finite & equal_bits
        row_indices = torch.arange(start, stop, dtype=torch.int64, device=trainer.device)
        chunk_first_bad.append(torch.where(row_ok, sentinel, row_indices).amin())
    return torch.stack(chunk_first_bad).amin()


__all__ = [
    "FullDistributionMismatch",
    "FullDistributionRefusal",
    "compare_full_distribution",
    "compare_full_distribution_cuda",
]
