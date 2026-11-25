"""The K-segment policy of the sm90 mm recipe, and the execution plans that serve it.

Everything above the "execution plans" line is the PROGRAM: it may read K and the dtype, nothing else.
Everything below it is ownership and scheduling: it may read M, N, the machine -- and moves no bit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: Bumped when the policy changes what S any weight gets. Part of the recipe identity.
#: v1: S = 1 for every K -- the whole K axis is one ascending wgmma chain per output element.
POLICY_VERSION = 1

#: Interior segment boundaries are multiples of this many K elements. Every compiled configuration uses
#: a 64-element K block (one 128-byte swizzle row of bf16), so an interior segment is a whole number of K
#: blocks and the chain of k16 wgmma steps inside it is the same chain whichever configuration runs it.
#: That is what makes the tile shape free.
#:
#: The LAST block of a segment (S = 1: of K) may be partial: TMA fills the elements past the segment's
#: end with zeros, so the chain ends with k16 steps whose products are exact zeros. That is the program
#: for any K % 8 == 0 (16-byte row pitch): "the ascending k16 chain over K zero-padded to whole K
#: blocks" -- the same thing the vendored Triton kernel's masked tail loads compute, and measured
#: bit-equal to it and to the cuBLASLt leaf loop at K = 8 .. 2752 (``tests/test_leaves_gpu.py``).
SEGMENT_ALIGN = 64

#: K (or a leaf) must be a multiple of this: the operands' 16-byte TMA row pitch.
K_ALIGN = 8

#: The splits the published policy may hand a weight. v1 publishes S = 1 only. The kernel family also
#: compiles 2-, 4- and 8-segment trees (onchip fold and split + combine) and the battery gates them: they
#: are the EXECUTION splits an explicit caller asks for (pik's row-parallel leaves), not policy.
LEGAL_SPLITS = (1,)

#: Output columns per 16 bytes of bf16: the output's TMA pitch must be a multiple of 16 bytes.
PITCH_ALIGN_N = 8

#: The finest leaf an EXECUTION plan (pik leaves) may use. A leaf that is not a whole number of K blocks
#: runs as its own batch (the batched leaf launch / the split node), its tail zero-padded; the onchip fold
#: walks consecutive leaves inside one K extent and therefore needs whole blocks (:data:`SEGMENT_ALIGN`).
EXECUTION_ALIGN = K_ALIGN

_TABLE_PATH = Path(__file__).resolve().parent / "segments.json"


def _dtype_name(dtype) -> str:
    if isinstance(dtype, str):
        if dtype != "bf16":
            raise ValueError(f"the sm90 mm recipe is bf16 only, not {dtype}")
        return dtype
    import torch

    if dtype is torch.bfloat16:
        return "bf16"
    raise ValueError(f"the sm90 mm recipe is bf16 only, not {dtype}")


def segment_count_by_policy(K: int, dtype_name: str = "bf16") -> int:
    """S for a reduction of length ``K``. Takes no N on purpose: a column shard or a packed projection
    changes N and must not change any element's program. (A row-parallel layer shards K; its cross-rank
    sum is pik's fixed leaf tree over the FULL K, which this package serves as an execution split.)"""
    if dtype_name != "bf16":
        raise ValueError(f"the sm90 mm recipe is bf16 only, not {dtype_name}")
    return LEGAL_SPLITS[0]


@lru_cache(None)
def published_table() -> dict:
    """``segments.json`` as loaded, or an empty table when the file is absent."""
    if not _TABLE_PATH.is_file():
        return {"policy_version": POLICY_VERSION, "rows": []}
    return json.loads(_TABLE_PATH.read_text())


def table_digest(table: dict | None = None) -> str:
    """Digest over the policy version and every published row, in canonical order."""
    t = published_table() if table is None else table
    rows = sorted(tuple(r) for r in t.get("rows", ()))
    payload = json.dumps({"policy_version": t.get("policy_version"), "rows": [list(r) for r in rows]},
                         separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def segment_count(K: int, dtype) -> int:
    """S for a reduction of length ``K``: the published row if there is one, else the policy."""
    name = _dtype_name(dtype)
    for k, d, s in published_table().get("rows", ()):
        if k == K and d == name:
            return int(s)
    return segment_count_by_policy(K, name)


def segment_k(K: int, dtype) -> int:
    return K // segment_count(K, dtype)


def supported(K: int, N: int) -> bool:
    """Whether this package serves the shape at all (bf16, 2-D, unit strides checked by ops). An N that
    is not a multiple of :data:`PITCH_ALIGN_N` is served through a pitch-padded output (``ops.mm_sm90``):
    every output column is its own chain, so the padding moves no bit of the first N columns."""
    return K > 0 and K % K_ALIGN == 0 and N > 0


def execution_supported(K: int, N: int, segment_k: int) -> bool:
    """Whether an EXECUTION split with this segment size can run (pik leaves): whole K blocks, a
    power-of-two segment count the tree kernels compile (1..8)."""
    if segment_k <= 0 or segment_k % EXECUTION_ALIGN or K % segment_k:
        return False
    S = K // segment_k
    return S in (1, 2, 4, 8) and N > 0 and N % PITCH_ALIGN_N == 0


# --------------------------------------------------------------------------- execution plans
# Ownership and scheduling only. Every configuration below computes the program above.

ONCHIP = 0   #: one CTA walks every segment of its tile; tree in registers (or S = 1: no tree)
SPLIT = 1    #: one batch per segment writes fp32 roots, the combine kernel evaluates the tree


@dataclass(frozen=True)
class ConfigSpec:
    block_m: int
    block_n: int
    cluster_m: int
    cluster_n: int
    schedule: str
    out_fp32: int
    levels: int
    swap: int = 0     #: 1: tokens on the wgmma N side (block_m tiles weight rows, block_n tokens)


#: Mirrors csrc/mm_sm90_table.cuh; ops.config_ids() checks every field in both directions.
_CONFIGS: dict[str, ConfigSpec] = {
    "ws_64x16": ConfigSpec(64, 16, 1, 1, "ws", 0, 0),
    "ws_64x32": ConfigSpec(64, 32, 1, 1, "ws", 0, 0),
    "ws_64x64": ConfigSpec(64, 64, 1, 1, "ws", 0, 0),
    "ws_64x128": ConfigSpec(64, 128, 1, 1, "ws", 0, 0),
    "pp_64x128": ConfigSpec(64, 128, 1, 1, "pingpong", 0, 0),
    "pp_64x256": ConfigSpec(64, 256, 1, 1, "pingpong", 0, 0),
    "pp_128x128": ConfigSpec(128, 128, 1, 1, "pingpong", 0, 0),
    "co_128x64": ConfigSpec(128, 64, 1, 1, "cooperative", 0, 0),
    "co_128x128": ConfigSpec(128, 128, 1, 1, "cooperative", 0, 0),
    "co_128x256": ConfigSpec(128, 256, 1, 1, "cooperative", 0, 0),
    "co_128x256_c2": ConfigSpec(128, 256, 2, 1, "cooperative", 0, 0),
    "co_256x128_c2": ConfigSpec(256, 128, 1, 2, "cooperative", 0, 0),
    "co_128x128_c2": ConfigSpec(128, 128, 2, 1, "cooperative", 0, 0),
    "pp_128x128_c2": ConfigSpec(128, 128, 2, 1, "pingpong", 0, 0),
    "ws_64x64_f32": ConfigSpec(64, 64, 1, 1, "ws", 1, 0),
    "co_128x128_f32": ConfigSpec(128, 128, 1, 1, "cooperative", 1, 0),
    "co_128x256_f32": ConfigSpec(128, 256, 2, 1, "cooperative", 1, 0),
    "ws_64x32_t3": ConfigSpec(64, 32, 1, 1, "ws", 0, 3),
    "ws_64x64_t3": ConfigSpec(64, 64, 1, 1, "ws", 0, 3),
    "co_128x64_t3": ConfigSpec(128, 64, 1, 1, "cooperative", 0, 3),
    "ws_64x32_t3_f32": ConfigSpec(64, 32, 1, 1, "ws", 1, 3),
    "ws_64x64_t3_f32": ConfigSpec(64, 64, 1, 1, "ws", 1, 3),
    "ws_64x128_t3_f32": ConfigSpec(64, 128, 1, 1, "ws", 1, 3),
    "co_128x64_t3_f32": ConfigSpec(128, 64, 1, 1, "cooperative", 1, 3),
    # swapped operands (decode M): block_m = weight rows per tile, block_n = tokens per tile
    "sw_64x8_s6": ConfigSpec(64, 8, 1, 1, "ws", 0, 0, 1),
    "sw_64x16_s6": ConfigSpec(64, 16, 1, 1, "ws", 0, 0, 1),
    "sw_64x32_s6": ConfigSpec(64, 32, 1, 1, "ws", 0, 0, 1),
    "sw_64x64_s6": ConfigSpec(64, 64, 1, 1, "ws", 0, 0, 1),
    "sw_64x16_s10": ConfigSpec(64, 16, 1, 1, "ws", 0, 0, 1),
    "sw_64x32_s10": ConfigSpec(64, 32, 1, 1, "ws", 0, 0, 1),
    "swpp_64x16": ConfigSpec(64, 16, 1, 1, "pingpong", 0, 0, 1),
    "swpp_64x32": ConfigSpec(64, 32, 1, 1, "pingpong", 0, 0, 1),
    "swpp_64x64": ConfigSpec(64, 64, 1, 1, "pingpong", 0, 0, 1),
    "swco_128x16": ConfigSpec(128, 16, 1, 1, "cooperative", 0, 0, 1),
    "swco_128x32": ConfigSpec(128, 32, 1, 1, "cooperative", 0, 0, 1),
    "swco_128x64": ConfigSpec(128, 64, 1, 1, "cooperative", 0, 0, 1),
    "sw_64x128_s6": ConfigSpec(64, 128, 1, 1, "ws", 0, 0, 1),
    "co_256x128": ConfigSpec(256, 128, 1, 1, "cooperative", 0, 0),
    "co_128x192": ConfigSpec(128, 192, 1, 1, "cooperative", 0, 0),
    "co_256x64": ConfigSpec(256, 64, 1, 1, "cooperative", 0, 0),
    "sw_64x16_s10_t1_f32": ConfigSpec(64, 16, 1, 1, "ws", 1, 1, 1),
    "swpp_64x32_t1_f32": ConfigSpec(64, 32, 1, 1, "pingpong", 1, 1, 1),
    "ws_64x128_t1_f32": ConfigSpec(64, 128, 1, 1, "ws", 1, 1),
    "pp_64x128_t1_f32": ConfigSpec(64, 128, 1, 1, "pingpong", 1, 1),
    "co_128x128_t1_f32": ConfigSpec(128, 128, 1, 1, "cooperative", 1, 1),
    "sw_64x16_s6_t3_f32": ConfigSpec(64, 16, 1, 1, "ws", 1, 3, 1),
    "sw_64x32_s6_t3_f32": ConfigSpec(64, 32, 1, 1, "ws", 1, 3, 1),
    "swpp_64x32_t3_f32": ConfigSpec(64, 32, 1, 1, "pingpong", 1, 3, 1),
    "swpp_64x64_t3_f32": ConfigSpec(64, 64, 1, 1, "pingpong", 1, 3, 1),
    "swco_128x32_t3_f32": ConfigSpec(128, 32, 1, 1, "cooperative", 1, 3, 1),
}


def configs(*, out_fp32: int, min_levels: int = 0, exact_levels: int | None = None) -> list[str]:
    return [n for n, c in _CONFIGS.items() if c.out_fp32 == out_fp32 and c.levels >= min_levels
            and (exact_levels is None or c.levels == exact_levels)]


@dataclass(frozen=True)
class Plan:
    config: str
    split: int        #: S, segments of K
    segment_k: int
    mode: int         #: ONCHIP or SPLIT

    @property
    def spec(self) -> ConfigSpec:
        return _CONFIGS[self.config]


#: Streaming multiprocessors of H100 SXM. Only the tile chooser reads it (never the program).
H100_SM_COUNT = 132


def _tiles(M: int, N: int, bm: int, bn: int) -> int:
    return -(-M // bm) * -(-N // bn)


#: Measured fastest configuration per (shape, M bucket), written by ``tune.py``. Speed only: NOT part
#: of the recipe identity (``table_digest`` does not read it); a shape it lacks uses the rule below.
TUNED_PATH = Path(__file__).resolve().parent / "tuned.json"


@lru_cache(None)
def tuned_table() -> dict:
    if not TUNED_PATH.is_file():
        return {}
    t = json.loads(TUNED_PATH.read_text())
    out = {}
    for kind in ("mm", "leaves", "node"):
        for key, rows in t.get(kind, {}).items():
            rows = [(int(m), name) for m, name, *_ in rows
                    if parse_choice(name)[0].split(":")[-1] in _CONFIGS]  # "split:<config>": a node plan
            if rows:
                out[(kind, *(int(v) for v in key.split(",")))] = sorted(rows)
    return out


def parse_choice(choice: str) -> tuple[str, int, int]:
    """``"<config>[@s<swizzle>r<raster>]"`` -> (config, swizzle, raster). Swizzle and raster order only
    ORDER the tiles of a persistent launch; they are scheduling, like the configuration itself."""
    if "@" not in choice:
        return choice, 1, 0
    name, sched = choice.split("@", 1)
    s, r = sched[1:].split("r")
    return name, int(s), int(r)


def tuned_config(kind: str, key: tuple, M: int) -> str | None:
    """The tuned configuration at the smallest measured M >= M (the largest measured M above the grid)."""
    rows = tuned_table().get((kind, *key))
    if not rows:
        return None
    for m, name in rows:
        if M <= m:
            return name
    return rows[-1][1]


def choose_s1_config(M: int, N: int, K: int) -> str:
    """The tile for a single-chain (S = 1) bf16 launch: the tuned table, else a rule distilled from the
    census sweeps (``tune.py``). Pure speed: every answer computes the same bits."""
    hit = tuned_config("mm", (K, N), M)
    if hit is not None:
        return parse_choice(hit)[0]
    if M <= 16:
        return "sw_64x16_s10"
    if M <= 32:
        return "sw_64x32_s10"
    if M <= 64:
        return "swpp_64x32" if -(-N // 64) < H100_SM_COUNT else "ws_64x64"
    if M <= 128:
        return "ws_64x64" if _tiles(M, N, 64, 128) < H100_SM_COUNT else "co_128x128"
    if M <= 256:
        return "pp_64x128"
    if M <= 512:
        return "co_128x256" if _tiles(M, N, 128, 256) % H100_SM_COUNT > H100_SM_COUNT * 0.8 else "pp_128x128"
    if M <= 2048:
        return "co_128x256" if _tiles(M, N, 128, 256) >= 2 * H100_SM_COUNT else "co_128x128"
    return "co_128x256_c2"


def choose_s1(M: int, N: int, K: int) -> tuple[str, int, int]:
    """(config, swizzle, raster) for a single-chain bf16 launch."""
    hit = tuned_config("mm", (K, N), M)
    if hit is not None:
        return parse_choice(hit)
    return choose_s1_config(M, N, K), 1, 0


def plan_for(M: int, K: int, N: int, dtype) -> Plan:
    S = segment_count(K, dtype)
    if S != 1:
        raise NotImplementedError("policy v1 publishes S = 1 only")
    return Plan(choose_s1_config(M, N, K), 1, K, ONCHIP)


def plans_at_split(M: int, K: int, N: int, split: int, *, out_fp32: int = 0) -> list[Plan]:
    """Every compiled way to run an EXECUTION split of K into ``split`` segments (pik leaf trees, the
    battery). Same bits by construction; the battery checks it."""
    if K % split or not execution_supported(K, N, K // split):
        return []
    segk = K // split
    need = (split - 1).bit_length()
    out = []
    if split == 1:
        out += [Plan(n, 1, K, ONCHIP) for n in configs(out_fp32=out_fp32)]
    else:
        if segk % SEGMENT_ALIGN == 0:  # the fold walks consecutive segments: whole K blocks only
            out += [Plan(n, split, segk, ONCHIP) for n in configs(out_fp32=out_fp32, min_levels=need)]
        out += [Plan(n, split, segk, SPLIT) for n in configs(out_fp32=1, exact_levels=0)]
    return out
