"""State-plane codec of the CPR records (reference implementation; the kda kernels implement it in
``csrc/record_state.cuh``).

The fp32 boundary state ``S_c [128 k][128 v]`` is held as ``SH`` (bf16 ``rn(S)``, ``[v][k]``: the operand image every
step reads), ``SL`` (the low 16 bits of the fp32 pattern, ``[k][v]``) and ``SC`` (one carry bit per element: ``SH`` is one
above the truncated top half), so that ``S = ((SH - SC) << 16) | SL`` bit for bit. SC is a bitmap of 8 x 8 blocks over
(r, c) = (k, v) for KDA and (d, v) for GDN: block (r / 8, c / 8) is the 64-bit word ``(c / 8) * 16 + r / 8``, bit
``(r % 8) * 8 + (c % 8)`` (little endian). Reading an element's carry is one shift of one word whatever thread holds the
element; a warp that holds an mma accumulator fragment assembles a block word with a ballot, a bit spread and (for a
fragment whose rows are v) an 8 x 8 bit transpose (csrc/cpr_planes.cuh). Both families therefore share one plane.
"""

from __future__ import annotations

import torch

HEAD_DIM = 128
SH_BYTES = SL_BYTES = HEAD_DIM * HEAD_DIM * 2
SC_BYTES = 8 * 16 * 16


def split_state(h0: torch.Tensor):
    """``h0 [H, 128 r, 128 c]`` fp32 -> ``(sh, sl, carry)``: ``sh [H, 128 c, 128 r]`` bf16 = rn(h0)^T,
    ``sl [H, 128 r, 128 c]`` int16 (the low 16 bits) and ``carry [H, 128 r, 128 c]`` int64 0/1 (see
    :func:`carry_words` for the SC byte image)."""
    if h0.dtype != torch.float32 or h0.shape[-2:] != (HEAD_DIM, HEAD_DIM):
        raise ValueError("split_state expects fp32 [H, 128, 128]")
    bits = h0.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sh_kv = h0.to(torch.bfloat16)                               # RN-even, as cvt.rn.bf16.f32
    top = bits >> 16
    sh_bits = sh_kv.view(torch.int16).to(torch.int64) & 0xFFFF
    carry = (sh_bits != top).to(torch.int64)                    # rn rounded the magnitude up
    sl = ((bits & 0xFFFF) ^ 0x8000) - 0x8000                    # int16 view of the low half
    return sh_kv.transpose(-1, -2).contiguous(), sl.to(torch.int16), carry


def carry_words(carry: torch.Tensor) -> torch.Tensor:
    """``carry [H, 128 r, 128 c]`` (0/1) -> the SC image ``[H, 512]`` int32 (little-endian pairs of 64-bit block words).
    Block (r / 8, c / 8) is word ``(c / 8) * 16 + r / 8``; bit ``(r % 8) * 8 + (c % 8)`` of it."""
    H = carry.shape[0]
    b = carry.to(torch.int64).view(H, 16, 8, 16, 8)                     # [H, rb, r8, cb, c8]
    r4 = torch.arange(4, device=carry.device, dtype=torch.int64).view(4, 1)
    c8 = torch.arange(8, device=carry.device, dtype=torch.int64).view(1, 8)
    wt = (1 << (r4 * 8 + c8)).view(1, 1, 4, 1, 8)                       # weight of (r8 % 4, c8) in its half-word
    lo = (b[:, :, 0:4] * wt).sum((2, 4))                                # [H, rb, cb]: r8 0..3 -> bits 0..31
    hi = (b[:, :, 4:8] * wt).sum((2, 4))                                # r8 4..7 -> bits 32..63
    pair = torch.stack([lo, hi], -1).permute(0, 2, 1, 3).reshape(H, 256, 2)   # word index = cb * 16 + rb
    pair = ((pair & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000
    return pair.to(torch.int32).reshape(H, 512)


def carry_from_words(words: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`carry_words`: ``[H, 512]`` int32 -> ``[H, 128 r, 128 c]`` int64 0/1."""
    H = words.shape[0]
    pair = (words.view(H, 16, 16, 2).to(torch.int64) & 0xFFFFFFFF)       # [H, cb, rb, half]
    bit = (torch.arange(4, device=words.device, dtype=torch.int64).view(4, 1) * 8
           + torch.arange(8, device=words.device, dtype=torch.int64).view(1, 8))   # (r8 % 4) * 8 + c8
    out = torch.zeros(H, 16, 8, 16, 8, dtype=torch.int64, device=words.device)   # [H, rb, r8, cb, c8]
    for half in range(2):
        x = pair[..., half].permute(0, 2, 1)                             # [H, rb, cb]
        bits = (x.view(H, 16, 1, 16, 1) >> bit.view(1, 1, 4, 1, 8)) & 1  # [H, rb, 4, cb, 8]
        out[:, :, 4 * half:4 * half + 4] = bits
    return out.reshape(H, HEAD_DIM, HEAD_DIM)


def join_state(sh: torch.Tensor, sl: torch.Tensor, carry: torch.Tensor) -> torch.Tensor:
    """The fp32 state ``[H, 128 k, 128 v]`` from its planes: ``((SH - carry) << 16) | SL``."""
    top = (sh.transpose(-1, -2).contiguous().view(torch.int16).to(torch.int64) & 0xFFFF) - carry
    bits = (top << 16) | (sl.to(torch.int64) & 0xFFFF)
    bits = ((bits & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000
    return bits.to(torch.int32).view(torch.float32)


def pack_planes(h0: torch.Tensor):
    """The (SH, SL, SC) byte images of ``h0 [H, 128, 128]``: three uint8 tensors ``[H, *_BYTES]``."""
    sh, sl, carry = split_state(h0)
    H = h0.shape[0]
    return (sh.reshape(H, -1).contiguous().view(torch.uint8),
            sl.reshape(H, -1).contiguous().view(torch.uint8),
            carry_words(carry).contiguous().view(torch.uint8))


def unpack_planes(sh_b: torch.Tensor, sl_b: torch.Tensor, sc_b: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_planes`: the fp32 state ``[H, 128, 128]`` bit for bit."""
    H = sh_b.shape[0]
    sh = sh_b.contiguous().view(torch.bfloat16).view(H, HEAD_DIM, HEAD_DIM)
    sl = sl_b.contiguous().view(torch.int16).view(H, HEAD_DIM, HEAD_DIM)
    carry = carry_from_words(sc_b.contiguous().view(torch.int32))
    return join_state(sh, sl, carry)
