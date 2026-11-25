"""Digest terms of many tensors at once: one kernel launch and one device-to-host copy.

Per tensor, its bytes read as little-endian 16-bit words ``w_i`` (``i`` = word index in the tensor):

    s1 = sum w_i,   s2 = sum w_i**2,   s3 = sum mix(i, w_i)          (each modulo 2**64)

``mix`` is a fixed 32-bit integer hash of the position and the word, so ``s3`` changes when words move.
All terms are integer sums, which are order-free: the result is bitwise deterministic whatever the
launch geometry, and :func:`digest_terms_reference` (torch integer ops, any device) returns the same
numbers. A tensor with an odd byte count is read with one zero byte appended.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

#: 16-bit words per chunk (one CTA each): 2 MiB, enough CTAs for a few large tensors to fill the device.
CHUNK_WORDS = 1 << 20

_MASK64 = (1 << 64) - 1


def _words(t: torch.Tensor) -> torch.Tensor:
    flat = t.detach().contiguous().view(-1).view(torch.uint8)
    if flat.numel() % 2:  # an odd byte count is read with one zero byte appended
        flat = torch.cat([flat, flat.new_zeros(1)])
    return flat.view(torch.int16)


def _mix_reference(i: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``csrc/digest.cu::mix`` in int64 arithmetic masked to 32 bits."""
    m = 0xFFFFFFFF
    lo, hi = i & m, (i >> 32) & m
    h = ((lo * 0x9E3779B1) & m) ^ ((hi * 0x85EBCA77) & m) ^ ((w * 0xC2B2AE3D) & m)
    h = h ^ (h >> 15)
    h = (h * 0x27D4EB2F) & m
    return h ^ (h >> 13)


def digest_terms_reference(tensors: Sequence[torch.Tensor]) -> list[tuple[int, int, int]]:
    """The kernel's terms, computed with torch integer ops (exact; slow on large tensors)."""
    out = []
    for t in tensors:
        w = _words(t).to(torch.int64) & 0xFFFF
        i = torch.arange(w.numel(), dtype=torch.int64, device=w.device)
        s1 = int(w.sum()) & _MASK64
        s2 = int((w * w).sum()) & _MASK64
        s3 = int(_mix_reference(i, w).sum()) & _MASK64
        out.append((s1, s2, s3))
    return out


def _kernel_ok(tensors: Sequence[torch.Tensor]) -> bool:
    if not tensors or not all(t.is_cuda for t in tensors):
        return False
    dev = tensors[0].device
    if any(t.device != dev for t in tensors):
        return False
    try:
        from isoexec.ops.digest._extension import extension

        extension()
        return True
    except Exception:  # noqa: BLE001 -- no device build: the reference computes the same numbers
        return False


@torch.no_grad()
def digest_terms(tensors: Sequence[torch.Tensor]) -> list[tuple[int, int, int]]:
    """``(s1, s2, s3)`` per tensor, in order. CUDA tensors on one device: one launch + one D2H."""
    tensors = list(tensors)
    if not _kernel_ok(tensors):
        return digest_terms_reference(tensors)
    from isoexec.ops.digest._extension import extension

    dev = tensors[0].device
    rows: list[tuple[int, int, int, int]] = []
    keep = []
    for k, t in enumerate(tensors):
        w = _words(t)
        keep.append(w)  # a contiguous copy, when one was needed, must outlive the launch
        ptr, n = w.data_ptr(), w.numel()
        for base in range(0, n, CHUNK_WORDS):
            rows.append((ptr + 2 * base, min(CHUNK_WORDS, n - base), base, k))
    host = torch.tensor(rows, dtype=torch.int64).reshape(-1, 4).pin_memory()
    chunks = host.to(dev, non_blocking=True)
    out = torch.empty(len(tensors), 3, dtype=torch.int64, device=dev)
    extension().run(chunks, out)
    terms = out.cpu().tolist()  # the one device-to-host copy (it also orders the pinned upload's lifetime)
    del keep, host
    return [tuple(v & _MASK64 for v in row) for row in terms]


__all__ = ["CHUNK_WORDS", "digest_terms", "digest_terms_reference"]
