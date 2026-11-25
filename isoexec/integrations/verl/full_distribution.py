"""Full-vocabulary row exchange and comparison for the verl integration.

The rollout server and the trainer run on the same node in verl's colocated sync mode. A complete
FP32 row per response token is about one megabyte, so rows do not travel through verl's batch
transport. The server writes each request's rows to the run directory under a key derived from
the request's own prompt and response token IDs and the weight version that produced them. The
trainer derives the same key from the token IDs it scores, so a row can only be found by the
sample that generated it. Rows are compared once and then deleted; only compact receipts remain.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isoexec.checks import Refusal

#: The run directory. It is the integration's existing evidence directory flag.
ENV_VAR = "ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR"
KEY_SCHEMA = b"isoexec-verl-full-rows-v1"
#: Plan acceptance: every compared row has |logsumexp| within this bound on each side.
LOGSUMEXP_TOLERANCE = 1e-4
#: Plan acceptance: every compared row holds at least this many distinct non-integer values.
MIN_DISTINCT_NON_INTEGER = 32


class RowExchangeRefusal(Refusal):
    """A row file is missing, malformed, or disagrees with the sample that asked for it."""


#: Full FP32 vocabulary rows per generated token (~1 MB each) are a debug qualification only. Off by
#: default: production runs compare the sampled-token logprobs bitwise (trainer.verify_sampled_logprobs).
FULL_ROWS_ENV = "ISOEXEC_VERL_FULL_ROWS"


def full_rows_enabled() -> bool:
    return os.environ.get(FULL_ROWS_ENV, "0") == "1"


#: While this file exists under the run root, the rollout server does not publish rows: the trainer
#: creates it for validation generations, which no rescoring consumes.
SUSPEND_ROWS = "rows-suspended"


def rows_suspended(root: Path) -> bool:
    return (root / SUSPEND_ROWS).exists()


def run_root() -> Path:
    root = os.environ.get(ENV_VAR)
    if not root:
        raise RowExchangeRefusal(f"{ENV_VAR} must name the run directory for full-row exchange")
    return Path(root).resolve()


def _token_bytes(token_ids: Sequence[int]) -> bytes:
    array = np.asarray(list(token_ids), dtype=np.int64)
    if array.ndim != 1 or (array.size and (int(array.min()) < 0 or int(array.max()) > np.iinfo(np.int32).max)):
        raise RowExchangeRefusal("token IDs must be a one-dimensional sequence of non-negative int32 values")
    return array.astype("<i4").tobytes()


def sample_key(prompt_ids: Sequence[int], response_ids: Sequence[int]) -> str:
    """Hash of the exact prompt and response tokens; the prompt length separates the two."""
    digest = hashlib.sha256(KEY_SCHEMA)
    prompt = _token_bytes(prompt_ids)
    digest.update(len(prompt).to_bytes(8, "little"))
    digest.update(prompt)
    digest.update(_token_bytes(response_ids))
    return digest.hexdigest()


def rows_path(root: Path, version: int, key: str) -> Path:
    if type(version) is not int or version < 0:
        raise RowExchangeRefusal(f"weight version must be a non-negative int, got {version!r}")
    return root / "rows" / f"v{version}" / f"{key}.npy"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def write_rows(
    root: Path,
    *,
    version: int,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    rows: np.ndarray,
    request_id: str,
) -> dict[str, Any]:
    """Publish one request's ``[response_len, vocab]`` FP32 rows; return a small JSON-safe reference.

    A key published twice must carry the same bytes: identical token prefixes under one weight
    version are scored identically by a batch-invariant engine, so a difference is a refusal.
    """
    if not isinstance(rows, np.ndarray) or rows.dtype != np.dtype("<f4") or rows.ndim != 2:
        raise RowExchangeRefusal("full rows must be a little-endian float32 [response, vocab] array")
    if rows.shape[0] != len(response_ids) or rows.shape[1] == 0:
        raise RowExchangeRefusal(f"full rows {rows.shape} do not cover {len(response_ids)} response tokens")
    if not np.isfinite(rows).all():
        raise RowExchangeRefusal("full rows must be finite")
    sampled = rows[np.arange(rows.shape[0]), np.asarray(list(response_ids), dtype=np.int64)]
    key = sample_key(prompt_ids, response_ids)
    path = rows_path(root, version, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{key}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    with open(tmp, "wb") as handle:
        np.save(handle, np.ascontiguousarray(rows), allow_pickle=False)
    sha = _file_sha256(tmp)
    if path.exists():
        existing = _file_sha256(path)
        tmp.unlink()
        if existing != sha:
            raise RowExchangeRefusal(
                f"full rows for key {key} at version {version} were published twice with different bytes"
            )
    else:
        os.replace(tmp, path)
    reference = {
        "key": key,
        "version": version,
        "shape": [int(rows.shape[0]), int(rows.shape[1])],
        "sha256": sha,
        "prompt_len": len(prompt_ids),
        "response_len": len(response_ids),
        "sampled_bits_sha256": hashlib.sha256(sampled.astype("<f4").tobytes()).hexdigest(),
        "request_id": str(request_id),
    }
    sidecar = path.with_suffix(".json")
    requests = []
    if sidecar.exists():
        requests = json.loads(sidecar.read_text()).get("request_ids", [])
    sidecar.write_text(json.dumps({**reference, "request_ids": sorted({*requests, str(request_id)})}, sort_keys=True))
    return reference


def read_rows(root: Path, *, version: int, prompt_ids: Sequence[int], response_ids: Sequence[int]) -> np.ndarray:
    """Return the published rows for exactly these tokens and version, memory-mapped read-only."""
    key = sample_key(prompt_ids, response_ids)
    path = rows_path(root, version, key)
    if not path.exists():
        others = sorted(p.name for p in (root / "rows").glob(f"v*/{key}.npy")) if (root / "rows").exists() else []
        raise RowExchangeRefusal(
            f"no full rows for sample key {key} at weight version {version}"
            + (f"; found other versions {others}" if others else "")
        )
    rows = np.load(path, mmap_mode="r", allow_pickle=False)
    if rows.dtype != np.dtype("<f4") or rows.ndim != 2 or rows.shape[0] != len(response_ids):
        raise RowExchangeRefusal(f"{path.name}: rows {rows.dtype} {rows.shape} do not cover the response")
    return rows


@dataclass(frozen=True)
class RowStats:
    """Independent per-side validity of complete rows."""

    rows: int
    max_abs_logsumexp: float
    min_distinct_non_integer: int


@torch.no_grad()
def row_stats(rows: torch.Tensor, *, group=None, chunk_rows: int = 64) -> RowStats:
    """Validate FP32 rows, or this rank's vocabulary slice of them when ``group`` spans the slices.

    Refuses non-FP32 or non-finite values, a row whose logsumexp leaves ``LOGSUMEXP_TOLERANCE``,
    or a row with fewer than ``MIN_DISTINCT_NON_INTEGER`` distinct non-integer values. Slices
    combine their logsumexp exactly in FP64; the distinct count is the size of the union of each
    slice's first distinct values, a lower bound on the row's count.
    """
    if rows.dtype is not torch.float32 or rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0:
        raise Refusal(f"full_distribution.rows: expected non-empty FP32 [N, V] rows, got {rows.dtype} {tuple(rows.shape)}")
    if not bool(torch.isfinite(rows).all()):
        raise Refusal("full_distribution.rows: rows must be finite")
    world = 1 if group is None else torch.distributed.get_world_size(group)
    lse_parts, distinct_parts = [], []
    for start in range(0, rows.shape[0], chunk_rows):
        chunk = rows[start:start + chunk_rows].double()
        lse_parts.append(torch.logsumexp(chunk, dim=-1))
        values = rows[start:start + chunk_rows]
        non_integer = torch.where(torch.frac(values) != 0, values, torch.full_like(values, float("nan")))
        ordered = torch.sort(non_integer, dim=-1).values
        # The first MIN_DISTINCT_NON_INTEGER distinct values per row; NaN padding sorts last.
        changes = torch.ones_like(ordered, dtype=torch.bool)
        changes[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
        changes &= ~torch.isnan(ordered)
        ranks = torch.cumsum(changes, dim=-1) - 1
        keep = changes & (ranks < MIN_DISTINCT_NON_INTEGER)
        firsts = torch.full((ordered.shape[0], MIN_DISTINCT_NON_INTEGER), float("nan"),
                            dtype=torch.float32, device=rows.device)
        row_index, column_index = keep.nonzero(as_tuple=True)
        firsts[row_index, ranks[row_index, column_index]] = ordered[row_index, column_index]
        distinct_parts.append(firsts)
    lse = torch.cat(lse_parts)
    firsts = torch.cat(distinct_parts)
    if world > 1:
        gathered = [torch.empty_like(lse) for _ in range(world)]
        torch.distributed.all_gather(gathered, lse, group=group)
        lse = torch.logsumexp(torch.stack(gathered), dim=0)
        values = [torch.empty_like(firsts) for _ in range(world)]
        torch.distributed.all_gather(values, firsts, group=group)
        firsts = torch.cat(values, dim=-1)
    merged = torch.sort(firsts, dim=-1).values
    distinct = torch.ones_like(merged, dtype=torch.bool)
    distinct[:, 1:] = merged[:, 1:] != merged[:, :-1]
    distinct &= ~torch.isnan(merged)
    max_abs = float(lse.abs().max().item())
    least = int(distinct.sum(dim=-1).min().item())
    if max_abs > LOGSUMEXP_TOLERANCE:
        raise Refusal(f"full_distribution.logsumexp: |logsumexp| reaches {max_abs:.3e} > {LOGSUMEXP_TOLERANCE}")
    if least < MIN_DISTINCT_NON_INTEGER:
        raise Refusal(
            f"full_distribution.distinct: a row has {least} distinct non-integer values < {MIN_DISTINCT_NON_INTEGER}"
        )
    return RowStats(rows=int(rows.shape[0]), max_abs_logsumexp=max_abs, min_distinct_non_integer=least)


@dataclass(frozen=True)
class CompareResult:
    rows: int
    first_bad_row: int | None
    first_token: int | None = None
    trainer_u32: int | None = None
    engine_u32: int | None = None
    mismatch_count: int | None = None
    #: Train/inference mismatch magnitude over the whole batch, recorded before refusing.
    mismatched_rows: int = 0
    max_abs_diff: float = 0.0


@torch.no_grad()
def compare_slices(trainer_rows: torch.Tensor, engine_rows: torch.Tensor, *, vocab_start: int) -> CompareResult:
    """Compare trainer rows with the matching engine vocabulary slice by every bit."""
    from isoexec.checks.full_distribution import compare_full_distribution, compare_full_distribution_cuda

    if trainer_rows.shape != engine_rows.shape:
        raise Refusal(f"full_distribution.shape: trainer {tuple(trainer_rows.shape)} engine {tuple(engine_rows.shape)}")
    n = int(trainer_rows.shape[0])
    if trainer_rows.device.type == "cuda":
        engine_device = engine_rows.to(device=trainer_rows.device, non_blocking=False).contiguous()
        bad = int(compare_full_distribution_cuda(trainer_rows.contiguous(), engine_device).item())
        if bad == n:
            return CompareResult(rows=n, first_bad_row=None)
    else:
        bad = n
        for index in range(n):
            if compare_full_distribution(trainer_rows[index].contiguous(), engine_rows[index].contiguous()) is not None:
                bad = index
                break
        if bad == n:
            return CompareResult(rows=n, first_bad_row=None)
    mismatch = compare_full_distribution(
        trainer_rows[bad].detach().cpu().contiguous(), engine_rows[bad].detach().cpu().contiguous()
    )
    if mismatch is None:
        raise Refusal(f"full_distribution: CUDA flagged row {bad} but the CPU comparator passed it")
    engine_local = engine_rows.to(device=trainer_rows.device)
    unequal = (trainer_rows.view(torch.int32) != engine_local.view(torch.int32)).any(dim=-1)
    max_abs = float((trainer_rows.double() - engine_local.double()).abs().max().item())
    return CompareResult(rows=n, first_bad_row=bad, first_token=vocab_start + mismatch.first_token,
                         trainer_u32=mismatch.trainer_u32, engine_u32=mismatch.engine_u32,
                         mismatch_count=mismatch.mismatch_count, mismatched_rows=int(unequal.sum().item()),
                         max_abs_diff=max_abs)


def append_receipt(root: Path, name: str, payload: dict[str, Any]) -> None:
    """Append one compact JSON line under ``receipts/``; never full rows."""
    directory = root / "receipts"
    directory.mkdir(parents=True, exist_ok=True)
    line = json.dumps({**payload, "wall_time": time.time(), "pid": os.getpid()}, sort_keys=True)
    with open(directory / name, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


__all__ = [
    "ENV_VAR",
    "LOGSUMEXP_TOLERANCE",
    "MIN_DISTINCT_NON_INTEGER",
    "CompareResult",
    "RowExchangeRefusal",
    "RowStats",
    "append_receipt",
    "compare_slices",
    "read_rows",
    "row_stats",
    "rows_path",
    "run_root",
    "sample_key",
    "write_rows",
]
