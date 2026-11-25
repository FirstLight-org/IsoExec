"""The batched CUDA comparison preserves the CPU full-distribution result and diagnostics."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from isoexec.checks.full_distribution import (
    FullDistributionRefusal,
    compare_full_distribution,
    compare_full_distribution_cuda,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _finite_rows(rows: int, vocab: int = 8) -> torch.Tensor:
    return (torch.arange(rows * vocab, dtype=torch.float32).reshape(rows, vocab) / 7.0).contiguous()


def _flip_low_bit(rows: torch.Tensor, row: int, token: int) -> None:
    bits = rows.view(torch.int32)
    bits[row, token] = bits[row, token] ^ 1


def _cpu_outcome(trainer: torch.Tensor, engine: torch.Tensor) -> tuple[str, int, Any]:
    for index, (trainer_row, engine_row) in enumerate(zip(trainer, engine, strict=True)):
        try:
            mismatch = compare_full_distribution(trainer_row, engine_row)
        except FullDistributionRefusal as error:
            return "refusal", index, str(error)
        if mismatch is not None:
            return "mismatch", index, mismatch
    return "pass", trainer.shape[0], None


def _cuda_outcome(
    trainer: torch.Tensor,
    engine: torch.Tensor,
    *,
    chunk_rows: int | None = None,
) -> tuple[str, int, Any]:
    trainer_cuda = trainer.cuda()
    engine_cuda = engine.cuda()
    trainer_before = trainer_cuda.view(torch.int32).clone()
    engine_before = engine_cuda.view(torch.int32).clone()
    kwargs = {} if chunk_rows is None else {"chunk_rows": chunk_rows}
    verdict = compare_full_distribution_cuda(trainer_cuda, engine_cuda, **kwargs)
    assert verdict.shape == ()
    assert verdict.dtype is torch.int64
    assert verdict.device == trainer_cuda.device
    bad_row = int(verdict.item())
    assert 0 <= bad_row <= trainer.shape[0]
    assert torch.equal(trainer_cuda.view(torch.int32), trainer_before)
    assert torch.equal(engine_cuda.view(torch.int32), engine_before)
    if bad_row == trainer.shape[0]:
        return "pass", bad_row, None
    trainer_row = trainer_cuda[bad_row].to(device="cpu", copy=True).contiguous()
    engine_row = engine_cuda[bad_row].to(device="cpu", copy=True).contiguous()
    try:
        mismatch = compare_full_distribution(trainer_row, engine_row)
    except FullDistributionRefusal as error:
        return "refusal", bad_row, str(error)
    if mismatch is None:
        pytest.fail(f"CUDA comparison selected row {bad_row} as a bad row.")
    return "mismatch", bad_row, mismatch


def _assert_matches_cpu(
    trainer: torch.Tensor,
    engine: torch.Tensor,
    *,
    chunk_rows: int | None = None,
) -> None:
    assert _cuda_outcome(trainer, engine, chunk_rows=chunk_rows) == _cpu_outcome(trainer, engine)


def test_equal_finite_rows_return_the_row_count_sentinel_without_modifying_inputs():
    raw_bits = torch.tensor(
        [
            [0, -2147483648, 1, -2147483647, 1065353216, -1082130432],
            [1056964608, -1090519040, 1073741824, -1073741824, 2, -2147483646],
        ],
        dtype=torch.int32,
    )
    trainer = raw_bits.view(torch.float32)
    _assert_matches_cpu(trainer, trainer.clone())


def test_one_bit_difference_preserves_first_token_raw_bits_count_and_ulp_diagnostic():
    trainer = _finite_rows(4)
    engine = trainer.clone()
    _flip_low_bit(engine, 2, 5)
    _flip_low_bit(engine, 2, 7)
    _assert_matches_cpu(trainer, engine)


def test_signed_zero_is_a_bit_mismatch():
    trainer = torch.zeros((3, 5), dtype=torch.float32)
    engine = trainer.clone()
    engine.view(torch.int32)[1, 3] = -2147483648
    outcome = _cuda_outcome(trainer, engine)
    assert outcome == _cpu_outcome(trainer, engine)
    assert outcome[0:2] == ("mismatch", 1)
    assert outcome[2].trainer_u32 == 0x00000000
    assert outcome[2].engine_u32 == 0x80000000


@pytest.mark.parametrize("side", ["trainer", "engine"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_either_side_nonfinite_refuses(side, nonfinite):
    trainer = _finite_rows(4)
    engine = trainer.clone()
    rows = trainer if side == "trainer" else engine
    rows[1, 2] = nonfinite
    _assert_matches_cpu(trainer, engine)


def test_equal_nonfinite_bits_still_refuse():
    trainer = _finite_rows(4)
    trainer[2, 4] = float("nan")
    _assert_matches_cpu(trainer, trainer.clone())


def test_earlier_bit_mismatch_precedes_later_nonfinite_row():
    trainer = _finite_rows(5)
    engine = trainer.clone()
    _flip_low_bit(engine, 0, 6)
    trainer[3, 1] = float("nan")
    outcome = _cuda_outcome(trainer, engine, chunk_rows=2)
    assert outcome == _cpu_outcome(trainer, engine)
    assert outcome[0:2] == ("mismatch", 0)


@pytest.mark.parametrize(
    ("rows", "chunk_rows", "bad_rows"),
    [
        pytest.param(7, 1, (5, 6), id="one-row-chunks"),
        pytest.param(10, 4, (8, 9), id="partial-final-chunk"),
        pytest.param(67, None, (64, 65), id="default-64-boundary"),
    ],
)
def test_chunking_returns_first_bad_row(rows, chunk_rows, bad_rows):
    trainer = _finite_rows(rows, vocab=3)
    engine = trainer.clone()
    for token, bad_row in enumerate(bad_rows):
        _flip_low_bit(engine, bad_row, token)
    outcome = _cuda_outcome(trainer, engine, chunk_rows=chunk_rows)
    assert outcome[1] == bad_rows[0]


def test_cuda_inputs_must_be_float32_nonempty_contiguous_matrices_on_one_device():
    valid = torch.zeros((3, 4), dtype=torch.float32, device="cuda")
    with pytest.raises(FullDistributionRefusal, match=r"trainer\.dtype=torch\.float16"):
        compare_full_distribution_cuda(valid.to(torch.float16), valid.to(torch.float16))
    with pytest.raises(FullDistributionRefusal, match=r"trainer\.shape=\(12,\)"):
        compare_full_distribution_cuda(valid.reshape(-1), valid.reshape(-1))
    with pytest.raises(FullDistributionRefusal, match=r"trainer\.shape=\(0, 4\)"):
        compare_full_distribution_cuda(valid[:0], valid[:0])
    noncontiguous = torch.zeros((3, 8), dtype=torch.float32, device="cuda")[:, ::2]
    with pytest.raises(FullDistributionRefusal, match=r"trainer\.contiguous=False"):
        compare_full_distribution_cuda(noncontiguous, valid)
    with pytest.raises(FullDistributionRefusal, match=r"full_distribution\.shape"):
        compare_full_distribution_cuda(valid, torch.zeros((2, 4), dtype=torch.float32, device="cuda"))
    with pytest.raises(FullDistributionRefusal, match=r"trainer\.device=cpu"):
        compare_full_distribution_cuda(valid.cpu(), valid.cpu())


@pytest.mark.parametrize("chunk_rows", [0, -1, True, 1.5])
def test_chunk_rows_must_be_a_positive_integer(chunk_rows):
    rows = torch.zeros((2, 3), dtype=torch.float32, device="cuda")
    with pytest.raises(FullDistributionRefusal, match=r"full_distribution\.chunk_rows="):
        compare_full_distribution_cuda(rows, rows.clone(), chunk_rows=chunk_rows)
