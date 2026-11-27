"""The engine record's byte layout and state-plane codec, without a build (mirrors ``csrc/record.cuh``).

``tests/test_record_layout_cpu.py`` parses the header and checks every constant here against it, so this file
cannot drift from the kernels. The codec is the reference for ``csrc/record_state.cuh``: the fp32 boundary
state ``S_c [128 k][128 v]`` is held as ``SH`` (bf16 ``rn(S)``, ``[v][k]``), ``SL`` (the low 16 bits of the fp32
pattern, ``[k][v]``) and ``SC`` (one carry bit per element: ``SH`` is one above the truncated top half), in
mma-fragment order, so that ``S = ((SH - SC) << 16) | SL`` bit for bit.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

HEAD_DIM = 128
CHUNK = 64
P_COLS = 48
_CSRC = Path(__file__).resolve().parent / "csrc"


def xrow_off(i: int) -> int:
    return 4 * ((i >> 2) + 1) * (2 * (i >> 2) + (i & 3))


X_FLOATS = xrow_off(64)       # the carry's 64 packed rows
X_REC_FLOATS = xrow_off(63)   # the record keeps rows [0, 63): the closing row's X is never read
SH_BYTES = SL_BYTES = HEAD_DIM * HEAD_DIM * 2
SC_BYTES = 8 * 16 * 16


def recompute_ki() -> bool:
    """Whether csrc/record_profile.cuh defines KDA_REC_RECOMPUTE_KI (the opt-in record profile)."""
    text = (_CSRC / "record_profile.cuh").read_text()
    return re.search(r"^\s*#define\s+KDA_REC_RECOMPUTE_KI\b", text, re.M) is not None


def layout(ki_stored: bool | None = None) -> dict[str, int]:
    """Section offsets of one (slot, head) record, plus ``BYTES``."""
    if ki_stored is None:
        ki_stored = not recompute_ki()
    o: dict[str, int] = {}
    o["SH"] = 0
    o["SL"] = o["SH"] + SH_BYTES
    o["SC"] = o["SL"] + SL_BYTES
    o["KB"] = o["SC"] + SC_BYTES
    o["VB"] = o["KB"] + 64 * 128 * 2
    o["DB"] = o["VB"] + 64 * 128 * 2
    o["KGB"] = o["DB"] + 63 * 128 * 2
    o["KI"] = o["KGB"] + 63 * 128 * 2
    o["GB"] = o["KI"] + (48 * 128 * 2 if ki_stored else 0)
    o["XB"] = o["GB"] + 64 * 128 * 4
    o["PB"] = o["XB"] + X_REC_FLOATS * 4
    o["BB"] = o["PB"] + 16 * P_COLS * 4
    o["POS"] = o["BB"] + 64 * 4
    o["BYTES"] = o["POS"] + 16
    return o


REC = layout()
REC_BYTES = REC["BYTES"]
#: The record this tree replaced (S fp32 + S^ bf16 + the full open-chunk rows): kept for the capacity arithmetic.
LEGACY_REC_BYTES = 220_944


# ----------------------------------------------------------------------------------------------- state planes
# The codec (the host-side reference of csrc/cpr_planes.cuh) lives in isoexec/ops/cpr_record/planes.py.
from isoexec.ops.cpr_record.planes import (  # noqa: E402,F401
    carry_from_words, carry_words, join_state, pack_planes, split_state, unpack_planes,
)
