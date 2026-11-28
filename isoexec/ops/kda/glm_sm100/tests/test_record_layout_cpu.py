"""Record layout and state-plane codec (no GPU, no build).

The record is the engine's per-(slot, head) state; ``record_layout.py`` mirrors ``csrc/record.cuh`` and the
codec the kernels implement (``csrc/record_state.cuh``). These tests pin the header against the mirror and the
codec against the identity it exists to provide: ``S = ((SH - carry) << 16) | SL`` for every fp32 pattern the
recipe can produce, with ``SH = rn(S)`` exactly what ``cvt.rn.bf16.f32`` gives.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from isoexec.ops.kda.glm_sm100 import record_layout as R

CSRC = Path(R.__file__).resolve().parent / "csrc"


def _header_layout(ki_stored: bool) -> dict[str, int]:
    """Evaluate ``struct Rec`` of record.cuh (constants are plain integer expressions)."""
    text = (CSRC / "record.cuh").read_text()
    env = {"X_REC_FLOATS": R.X_REC_FLOATS, "P_COLS": R.P_COLS, "KI_STORED": int(ki_stored)}
    for stmt in re.findall(r"constexpr int64_t ([^;]+);", text.split("struct Rec {")[0]):
        for name, expr in re.findall(r"(\w+) = ([^,]+)", stmt):   # `constexpr int64_t A = .., B = ..;`
            env[name] = eval(expr, {}, dict(env))
    body = text.split("struct Rec {")[1].split("};")[0]
    out = {}
    for name, expr in re.findall(r"static constexpr int64_t (\w+) = ([^;]+);", body):
        expr = expr.replace("?", " and ").replace(":", " or ") if "?" in expr else expr
        if "KI_STORED" in expr:
            expr = re.sub(r"\(KI_STORED \? (.+?) : 0\)", lambda m: f"({m.group(1)} if KI_STORED else 0)", expr)
        out[name] = env[name] = eval(expr, {}, dict(env))
    return out


@pytest.mark.parametrize("ki_stored", [True, False])
def test_python_layout_is_the_header_cpu(ki_stored):
    hdr = _header_layout(ki_stored)
    mine = R.layout(ki_stored)
    assert hdr == mine


def test_header_constants_cpu():
    text = (CSRC / "record.cuh").read_text()
    assert "constexpr int X_FLOATS = 2176;" in text and R.X_FLOATS == 2176
    assert R.X_REC_FLOATS == 2112
    assert R.xrow_off(63) == 2112 and R.xrow_off(64) == 2176


def test_profile_header_default_is_the_stored_profile_cpu():
    assert R.recompute_ki() is False
    assert R.REC_BYTES == 189_456 and R.layout(False)["BYTES"] == 177_168


def test_sections_are_16_byte_aligned_and_ordered_cpu():
    for ki in (True, False):
        lay = R.layout(ki)
        offs = [lay[k] for k in ("SH", "SL", "SC", "KB", "VB", "DB", "KGB", "KI", "GB", "XB", "PB", "BB", "POS", "BYTES")]
        assert offs == sorted(offs)
        assert all(o % 16 == 0 for o in offs)
    assert R.REC["POS"] + 16 == R.REC_BYTES


def test_the_record_is_smaller_than_the_one_it_replaces_cpu():
    legacy = 220_944                                       # S fp32 + S^ bf16 + the full open-chunk rows
    assert not hasattr(R, "LEGACY_REC_BYTES")              # no code reads the replaced record's size
    assert R.REC_BYTES < legacy
    assert legacy - R.REC_BYTES == 31_488                  # 30,720 (state planes) + 3 x 256 (rows 63)
    assert legacy / 65_536 > 3.3 > R.REC_BYTES / 65_536 > 2.8


# ------------------------------------------------------------------------------------------------------ codec
def _patterns(n: int, seed: int) -> torch.Tensor:
    """fp32 bit patterns [n]: random, exact rn ties (low half 0x8000, both parities), the largest finite binades,
    subnormals, signed zeros, infinities. No NaN (see record_state.cuh)."""
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2**32, (n,), generator=g, dtype=torch.int64)
    top = torch.randint(0, 2**16, (n,), generator=g, dtype=torch.int64)
    third = n // 3
    bits[:third] = (top[:third] << 16) | 0x8000                       # ties: SH even stays, SH odd rounds up
    bits[third:2 * third] = (top[third:2 * third] << 16) | torch.randint(0x7FF0, 0x8010, (third,), generator=g)
    special = torch.tensor([0x00000000, 0x80000000, 0x00000001, 0x007FFFFF, 0x807FFFFF, 0x7F7F8000, 0x7F7F7FFF,
                            0x7F7FFFFF, 0xFF7F8000, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x3F800000, 0x3F808000,
                            0x3F818000, 0x3F80FFFF], dtype=torch.int64)
    bits[: special.numel()] = special
    nan = ((bits >> 23) & 0xFF == 0xFF) & ((bits & 0x7FFFFF) != 0)     # drop NaN patterns
    bits[nan] = bits[nan] & ~(1 << 22) & 0xFF80_0000
    return bits


def _as_state(bits: torch.Tensor) -> torch.Tensor:
    b = ((bits & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000
    return b.to(torch.int32).view(torch.float32)


def test_planes_roundtrip_bit_for_bit_cpu():
    H = 4
    for seed in range(6):
        bits = _patterns(H * 128 * 128, seed)
        h0 = _as_state(bits).view(H, 128, 128)
        sh_b, sl_b, sc_b = R.pack_planes(h0)
        assert sh_b.shape == (H, R.SH_BYTES) and sl_b.shape == (H, R.SL_BYTES) and sc_b.shape == (H, R.SC_BYTES)
        back = R.unpack_planes(sh_b, sl_b, sc_b)
        assert torch.equal(back.view(torch.int32), h0.view(torch.int32))


def test_sh_plane_is_the_old_s_hat_image_cpu():
    h0 = _as_state(_patterns(2 * 128 * 128, 9)).view(2, 128, 128)
    sh, _, carry = R.split_state(h0)
    assert torch.equal(sh.view(torch.int16), h0.transpose(1, 2).to(torch.bfloat16).contiguous().view(torch.int16))
    top = (h0.view(torch.int32).to(torch.int64) & 0xFFFFFFFF) >> 16
    sh_bits = sh.transpose(1, 2).contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    assert set(torch.unique(sh_bits - top).tolist()) <= {0, 1}     # rn moves the magnitude by at most one
    assert torch.equal(carry, (sh_bits - top))


def test_tie_elements_need_the_carry_plane_cpu():
    """(SH, SL) alone is ambiguous exactly at the rn ties: two states share SH and SL = 0x8000."""
    even, odd = 0x3F80, 0x3F81                                        # top halves
    a = _as_state(torch.tensor([(even << 16) | 0x8000]))              # ties to even: stays at 0x3F80
    b = _as_state(torch.tensor([(odd << 16) | 0x8000]))               # ties to even: rounds up to 0x3F82
    assert a.to(torch.bfloat16).view(torch.int16).item() == even
    assert b.to(torch.bfloat16).view(torch.int16).item() == odd + 1
    c = _as_state(torch.tensor([((even + 1) << 16) | 0x8000]))        # SH 0x3F82 as well
    assert c.to(torch.bfloat16).view(torch.int16).item() == even + 2
    # two distinct states with SH 0x3F82 and SL 0x8000 exist: odd-top (rounded up) and (even+2)-top (even, kept)
    s1, s2 = _as_state(torch.tensor([(0x3F81 << 16) | 0x8000])), _as_state(torch.tensor([(0x3F82 << 16) | 0x8000]))
    assert s1.to(torch.bfloat16).view(torch.int16).item() == s2.to(torch.bfloat16).view(torch.int16).item() == 0x3F82
    assert s1.view(torch.int32).item() != s2.view(torch.int32).item()
    h = torch.stack([s1, s2]).view(1, 1, 2).expand(1, 128, 2).contiguous()
    h = torch.cat([h, torch.zeros(1, 128, 126)], -1)
    _, _, carry = R.split_state(h)
    assert carry[0, 0, 0].item() == 1 and carry[0, 0, 1].item() == 0   # the plane tells them apart


def test_largest_finite_values_round_to_infinity_and_back_cpu():
    h = _as_state(torch.tensor([0x7F7F8000, 0x7F7F8001, 0x7F7F7FFF, 0xFF7F8000, 0xFF7FFFFF] + [0] * (128 * 128 - 5))).view(1, 128, 128)
    sh_b, sl_b, sc_b = R.pack_planes(h)
    assert torch.equal(R.unpack_planes(sh_b, sl_b, sc_b).view(torch.int32), h.view(torch.int32))


M64 = (1 << 64) - 1


def _spread32(x):
    v = x & 0xFFFFFFFF
    for sh, mask in ((16, 0x0000FFFF0000FFFF), (8, 0x00FF00FF00FF00FF), (4, 0x0F0F0F0F0F0F0F0F), (2, 0x3333333333333333),
                     (1, 0x5555555555555555)):
        v = (v | (v << sh)) & mask
    return v


def _transpose8x8(x):
    for sh, mask in ((7, 0x00AA00AA00AA00AA), (14, 0x0000CCCC0000CCCC), (28, 0x00000000F0F0F0F0)):
        t = (x ^ (x >> sh)) & mask
        x = (x ^ t ^ (t << sh)) & M64
    return x


def _kernel_model_store(h0: torch.Tensor):
    """csrc/record_state.cuh ``s_store_tile`` + cpr_planes.cuh ``store_carry_frag<false>`` over every tile, lane by lane:
    (SL as [k][v], SC block words as a dict word-index -> int)."""
    sl = torch.zeros(128, 128, dtype=torch.int64)
    words = {}
    bits = h0.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rn = h0.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    for kt in range(8):
        for nt in range(16):
            M = [0, 0, 0, 0]
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                for eh in range(2):
                    k, v = 16 * kt + g + 8 * eh, 8 * nt + 2 * t
                    sl[k, v] = bits[k, v] & 0xFFFF
                    sl[k, v + 1] = bits[k, v + 1] & 0xFFFF
                    for x in range(2):
                        M[2 * eh + x] |= int(rn[k, v + x] != (bits[k, v + x] >> 16)) << lane
            w0 = _spread32(M[0]) | (_spread32(M[1]) << 1)
            w1 = _spread32(M[2]) | (_spread32(M[3]) << 1)
            words[nt * 16 + 2 * kt], words[nt * 16 + 2 * kt + 1] = w0, w1
    return sl, words


def test_fragment_store_model_is_the_codec_cpu():
    h0 = _as_state(_patterns(128 * 128, 3)).view(1, 128, 128)
    sl_model, words = _kernel_model_store(h0[0])
    _, sl, carry = R.split_state(h0)
    assert torch.equal(sl[0].to(torch.int64) & 0xFFFF, sl_model)
    img = R.carry_words(carry)[0].to(torch.int64) & 0xFFFFFFFF
    for i in range(256):
        assert int(img[2 * i]) | (int(img[2 * i + 1]) << 32) == words[i], i


def test_transposed_fragment_store_model_is_the_codec_cpu():
    """GDN decode's fragments have rows = value index, columns = d: spread then an 8x8 transpose gives the same blocks."""
    g_ = torch.Generator().manual_seed(11)
    carry = torch.randint(0, 2, (1, 128, 128), generator=g_, dtype=torch.int64)            # [d][v]
    img = R.carry_words(carry)[0].to(torch.int64) & 0xFFFFFFFF
    for vt in range(8):                  # 16 value rows per fragment
        for dt in range(16):             # 8 d columns
            M = [0, 0, 0, 0]
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                for e in range(4):
                    v, d = 16 * vt + g + 8 * (e >> 1), 8 * dt + 2 * t + (e & 1)
                    M[e] |= int(carry[0, d, v]) << lane
            for eh in range(2):
                w = _transpose8x8(_spread32(M[2 * eh]) | (_spread32(M[2 * eh + 1]) << 1))
                idx = (2 * vt + eh) * 16 + dt
                assert int(img[2 * idx]) | (int(img[2 * idx + 1]) << 32) == w, (vt, dt, eh)


def test_fragment_join_model_matches_the_codec_cpu():
    """``carry_frag`` + ``s_join_tile`` lane by lane (the kernel's read) reproduce the state from the byte planes."""
    h0 = _as_state(_patterns(128 * 128, 5)).view(1, 128, 128)
    sh_b, sl_b, sc_b = R.pack_planes(h0)
    sh = sh_b.view(torch.bfloat16).view(128, 128)                     # [v][k]
    sl = sl_b.view(torch.int16).view(128, 128).to(torch.int64) & 0xFFFF
    sc = sc_b.view(torch.int32)[0].to(torch.int64) & 0xFFFFFFFF
    word = lambda i: int(sc[2 * i]) | (int(sc[2 * i + 1]) << 32)
    out = torch.zeros(128, 128, dtype=torch.int64)
    for kt in range(8):
        for nt in range(16):
            w = [word(nt * 16 + 2 * kt), word(nt * 16 + 2 * kt + 1)]
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                for eh in range(2):
                    k, v = 16 * kt + g + 8 * eh, 8 * nt + 2 * t
                    for x in range(2):
                        c = (w[eh] >> (g * 8 + 2 * t + x)) & 1
                        top = (int(sh[v + x, k].view(torch.int16)) & 0xFFFF) - c
                        out[k, v + x] = (top << 16) | int(sl[k, v + x])
    assert torch.equal(out, h0[0].view(torch.int32).to(torch.int64) & 0xFFFFFFFF)


def test_carry_bit_addressing_cpu():
    """cpr_planes.cuh carry_bit(r, c): word (c / 8) * 16 + r / 8, bit (r % 8) * 8 + c % 8."""
    for r, c in ((0, 0), (9, 21), (127, 127), (8, 0), (0, 8), (63, 64)):
        carry = torch.zeros(1, 128, 128, dtype=torch.int64)
        carry[0, r, c] = 1
        img = (R.carry_words(carry)[0].to(torch.int64) & 0xFFFFFFFF).tolist()
        words = [img[2 * i] | (img[2 * i + 1] << 32) for i in range(256)]
        nz = [(i, w) for i, w in enumerate(words) if w]
        assert nz == [((c >> 3) * 16 + (r >> 3), 1 << ((r & 7) * 8 + (c & 7)))]


def test_zero_planes_are_the_zero_state_cpu():
    z = torch.zeros(3, R.SH_BYTES, dtype=torch.uint8), torch.zeros(3, R.SL_BYTES, dtype=torch.uint8), torch.zeros(3, R.SC_BYTES, dtype=torch.uint8)
    assert not R.unpack_planes(*z).view(torch.int32).any()               # slot reset (all-zero bytes) is a zero state


def test_transpose_helper_is_an_8x8_bit_transpose_cpu():
    for a in range(8):
        for b in range(8):
            assert _transpose8x8(1 << (8 * a + b)) == 1 << (8 * b + a)
    assert _spread32(0b1011) == 0b1000101
