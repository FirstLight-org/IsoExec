"""Optional on-disk evidence for SkyRL full-distribution comparisons."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import torch

ENV_VAR = "ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR"
KEY_SCHEMA_VERSION = 1
TEMPERATURE_ONE_BITS = struct.unpack("<I", struct.pack("<f", 1.0))[0]


def prefix_digest(token_ids: Sequence[int]) -> str:
    """SHA-256 over the little-endian int32 bytes of a real-token prefix."""
    arr = np.asarray(list(token_ids), dtype="<i4")
    return hashlib.sha256(arr.tobytes()).hexdigest()


@dataclass
class RowKeys:
    """Column-oriented keys for the rows of one compared micro-batch."""

    sample_index: list[int] = field(default_factory=list)
    sample_index_scope: str = "local"
    response_token_index: list[int] = field(default_factory=list)
    token_id: list[int] = field(default_factory=list)
    prefix_length: list[int] = field(default_factory=list)
    prefix_digest: list[str] = field(default_factory=list)
    sampled_logprob_bits: list[int] = field(default_factory=list)
    engine_sampled_bits: list[int] = field(default_factory=list)
    trainer_sampled_bits: list[int] = field(default_factory=list)
    sampled_owner: list[bool] = field(default_factory=list)
    vocab_start: int = 0
    vocab_end: int = 0

    def __len__(self) -> int:
        return len(self.token_id)


def build_row_keys(
    *,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    row_indices: torch.Tensor,
    response_columns: torch.Tensor,
    target_columns: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    engine_rows: torch.Tensor,
    trainer_rows: torch.Tensor,
    sample_indices: Optional[torch.Tensor],
    vocab_start: int = 0,
    vocab_end: Optional[int] = None,
) -> RowKeys:
    """Keys for the selected rows, computed from the same tensors the comparison used."""
    seqs = sequences.detach().to("cpu")
    mask = attention_mask.detach().to("cpu").to(torch.bool)
    rows = row_indices.detach().to("cpu").tolist()
    cols = response_columns.detach().to("cpu").tolist()
    targets = target_columns.detach().to("cpu").tolist()
    sampled = rollout_logprobs.detach().to("cpu", copy=True).to(torch.float32)
    keys = RowKeys()
    keys.vocab_start = int(vocab_start)
    keys.vocab_end = int(vocab_end if vocab_end is not None else engine_rows.shape[1])
    if sample_indices is not None:
        keys.sample_index_scope = "global"
        global_index = sample_indices.detach().to("cpu").tolist()
    else:
        global_index = None
    engine_bits = engine_rows.view(torch.int32)
    trainer_bits = trainer_rows.view(torch.int32)
    for k, (row, col, target) in enumerate(zip(rows, cols, targets)):
        real = mask[row].nonzero(as_tuple=False).flatten()
        first_real = int(real[0].item()) if real.numel() else 0
        prefix = seqs[row, first_real:target].tolist()
        token = int(seqs[row, target].item())
        keys.sample_index.append(int(global_index[row]) if global_index is not None else int(row))
        keys.response_token_index.append(int(col))
        keys.token_id.append(token)
        keys.prefix_length.append(len(prefix))
        keys.prefix_digest.append(prefix_digest(prefix))
        keys.sampled_logprob_bits.append(int(sampled[row, col].view(torch.int32).item()) & 0xFFFFFFFF)
        keys.engine_sampled_bits.append(int(engine_bits[k, token].item()) & 0xFFFFFFFF)
        owner = keys.vocab_start <= token < keys.vocab_end
        keys.sampled_owner.append(bool(owner))
        keys.trainer_sampled_bits.append(
            (int(trainer_bits[k, token - keys.vocab_start].item()) & 0xFFFFFFFF) if owner else 0xFFFFFFFF
        )
    return keys


class FullDistributionEvidence:
    """Per-process writer for compared rows and their keys."""

    _instance: Optional["FullDistributionEvidence"] = None
    _checked_env: bool = False

    def __init__(self, root: str, *, rank: Optional[int] = None):
        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.rank = int(rank)
        self.pid = os.getpid()
        self.root = os.path.abspath(root)
        self.dir = os.path.join(self.root, f"rank{self.rank:04d}")
        os.makedirs(self.dir, exist_ok=True)
        self.seq = 0

    @classmethod
    def from_env(cls) -> Optional["FullDistributionEvidence"]:
        """The process-wide sink when ``ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR`` is set, else None."""
        root = os.environ.get(ENV_VAR)
        if not root:
            return None
        if cls._instance is None or cls._instance.root != os.path.abspath(root):
            cls._instance = cls(root)
        return cls._instance

    def _event(self, payload: dict[str, Any]) -> None:
        import isoexec

        payload = dict(
            payload,
            rank=self.rank,
            pid=self.pid,
            wall_time=time.time(),
            isoexec_file=os.path.abspath(isoexec.__file__),
        )
        with open(os.path.join(self.dir, "events.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")

    def record_refusal(self, message: str, *, context: Optional[dict[str, Any]] = None) -> None:
        self._event({"event": "refusal", "message": message, "context": context or {}})

    def record_compare(
        self,
        *,
        keys: RowKeys,
        trainer_rows: torch.Tensor,
        engine_rows: torch.Tensor,
        mismatch: Optional[tuple[int, Any]],
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Save the exact rows and keys of one micro-batch; return the artifact path."""
        if trainer_rows.dtype is not torch.float32 or engine_rows.dtype is not torch.float32:
            raise ValueError("evidence rows must be float32")
        if trainer_rows.shape != engine_rows.shape or trainer_rows.ndim != 2:
            raise ValueError("evidence rows must be aligned [n, V] tensors")
        if trainer_rows.shape[0] != len(keys):
            raise ValueError("evidence keys must have one entry per row")
        context = dict(context or {})
        seq = self.seq
        self.seq += 1
        path = os.path.join(self.dir, f"rows-{seq:06d}.npz")
        np.savez(
            path,
            trainer=trainer_rows.detach().cpu().contiguous().numpy(),
            engine=engine_rows.detach().cpu().contiguous().numpy(),
            sample_index=np.asarray(keys.sample_index, dtype=np.int64),
            response_token_index=np.asarray(keys.response_token_index, dtype=np.int64),
            token_id=np.asarray(keys.token_id, dtype=np.int64),
            prefix_length=np.asarray(keys.prefix_length, dtype=np.int64),
            prefix_digest=np.asarray(keys.prefix_digest, dtype="U64"),
            sampled_logprob_bits=np.asarray(keys.sampled_logprob_bits, dtype=np.uint32),
            engine_sampled_bits=np.asarray(keys.engine_sampled_bits, dtype=np.uint32),
            trainer_sampled_bits=np.asarray(keys.trainer_sampled_bits, dtype=np.uint32),
            sampled_owner=np.asarray(keys.sampled_owner, dtype=np.bool_),
            vocab_start=np.asarray(keys.vocab_start, dtype=np.int64),
            vocab_end=np.asarray(keys.vocab_end, dtype=np.int64),
        )
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        first = None
        if mismatch is not None:
            index, m = mismatch
            first = {
                "selected_index": int(index),
                "first_token": int(m.first_token),
                "trainer_u32": int(m.trainer_u32),
                "engine_u32": int(m.engine_u32),
                "mismatch_count": int(m.mismatch_count),
            }
        sampled_agree = int(sum(1 for a, b in zip(keys.sampled_logprob_bits, keys.engine_sampled_bits) if a == b))
        self._event(
            {
                "event": "compare",
                "seq": seq,
                "file": os.path.basename(path),
                "sha256": sha,
                "rows": int(trainer_rows.shape[0]),
                "vocab_size": int(trainer_rows.shape[1]),
                "vocab_slice": [keys.vocab_start, keys.vocab_end],
                "key_schema_version": KEY_SCHEMA_VERSION,
                "sample_index_scope": keys.sample_index_scope,
                "temperature_float32_bits": TEMPERATURE_ONE_BITS,
                "sampled_bits_agree": sampled_agree,
                "first_mismatch": first,
                "context": context,
            }
        )
        return path


__all__ = [
    "ENV_VAR",
    "KEY_SCHEMA_VERSION",
    "TEMPERATURE_ONE_BITS",
    "FullDistributionEvidence",
    "RowKeys",
    "build_row_keys",
    "prefix_digest",
]
