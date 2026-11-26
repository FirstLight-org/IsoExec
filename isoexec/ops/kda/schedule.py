"""The numerical schedule of the KDA recurrence: the reducers, their blocks, and what the policy gives.

The KDA state update is a recurrence over the tokens of one sequence. Its reducer has no merge (a state cannot
be combined with another state; it can only be continued), so rule 1 of :mod:`isoexec.ops.schedule_policy`
applies: segments of the reducer's exact block, anchored at absolute position 0 of the sequence, joined by one
continued chain. Nothing else is read: no shortest segment, no occupancy, no smallest batch, no longest context.

Two reducers exist. They are different programs with different bits and different per-slot state, and a
deployment runs exactly one of them (engine and trainer alike):

``CHUNK``   ``glm_sm100``: the chunkwise kernel. The default, and the one the design's schedule names.
``TOKEN``   ``glm_rec_sm100``: the per-token recurrence, block 1. A second reducer that a deployment selects
            explicitly (``select(TOKEN)``); it is never chosen by default and never mixed with ``CHUNK``.

:func:`manifest` is what two parties of one deployment compare before they exchange state or bits
(:func:`assert_same_manifest`): it differs between the two reducers.

Source of every constant below: ``policy_validation/kda_sm100/ROUND_B.md`` (rounds D and E).
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path

from isoexec.ops import schedule_policy as policy

_HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Reducer:
    """One recurrence reducer: its name in the design's table, the package that implements it and the blocks
    (tokens folded by one step) that were built and checked bitwise on the device."""

    name: str
    impl_id: str
    package: str
    family: str                 # build_cache family of the package's extension
    blocks: tuple[int, ...]
    block_define: str | None    # the compiler definition that carries the block into the kernel (None: no parameter)

    @property
    def block(self) -> int:
        """The exact block of the schedule: the largest built block (ROUND_B.md round E: "largest built block;
        D.shape is empty"; no recipe declares a state budget or a latency objective that would publish another)."""
        return max(self.blocks)


#: The chunkwise reducer. ``blocks``: chunk lengths built from this source with ``-DKDA_CL`` and checked bitwise on
#: a B200 (ROUND_B.md round D, "blocks lists only built and checked members"; bench/roundA_check_rerun.log).
#: Step 16 is ``m16`` of ``mma.sync.m16n8k16`` (the row block of one k-step); the upper end 64 is the shared-memory
#: bound of this kernel's layout (ROUND_B.md round D table: at 96 rows ``state_kernel`` needs 238,592 B and at 128
#: rows ``intra_kernel`` needs 264,704 B, more than one CTA may have; 80 fits but is not a tile the kernel forms).
CHUNK = Reducer(name="chunk_recurrence.kda", impl_id="sm100_kda_v1", package="glm_sm100", family="kda_glm",
                blocks=(16, 32, 64), block_define="KDA_CL")

#: The per-token reducer (ROUND_B.md round E: "per-token reducer, block 1, selected with reducers=, second shipped
#: recipe; manifest differs"). Its kernel has no block parameter: one step folds one token.
TOKEN = Reducer(name="chunk_recurrence.kda_rec.kernel_delta", impl_id="sm100_kda_rec_v1", package="glm_rec_sm100",
                family="kda_glm_rec", blocks=(1,), block_define=None)

REDUCERS = (CHUNK, TOKEN)
DEFAULT = CHUNK


def segments(reducer: Reducer = DEFAULT) -> policy.Segments:
    """Rule 1 for ``reducer``: segments of its exact block, one continued chain."""
    return policy.no_merge(reducer.block)


def cuda_defines(reducer: Reducer = DEFAULT) -> tuple[str, ...]:
    """The segment length as a compiler definition: the only place the kernel gets it from."""
    if reducer.block_define is None:
        return ()
    return (f"-D{reducer.block_define}={segments(reducer).S}",)


def csrc(reducer: Reducer) -> Path:
    return _HERE / reducer.package / "csrc"


def source_sha256(reducer: Reducer) -> str:
    digest = hashlib.sha256()
    for path in sorted(csrc(reducer).rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(csrc(reducer)).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def manifest(reducer: Reducer = DEFAULT) -> dict:
    """What a party of a deployment publishes about its KDA program."""
    seg = segments(reducer)
    return {
        "op": "kda.glm",
        "policy": policy.VERSION,
        "reducer": reducer.name,
        "impl_id": reducer.impl_id,
        "schedule": {"split": f"anchored({seg.S})", "combine": seg.merge},
        "defines": list(cuda_defines(reducer)),
        "source_sha256": source_sha256(reducer),
    }


def manifest_hash(m: dict) -> str:
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()[:16]


def assert_same_manifest(mine: dict, theirs: dict) -> None:
    """Refuse two parties whose KDA programs differ (e.g. one selected ``TOKEN``): their bits and their per-slot
    state are not interchangeable."""
    differing = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
    if differing:
        raise ValueError(f"KDA manifests differ in {differing}: {mine.get('reducer')!r} "
                         f"({manifest_hash(mine)}) against {theirs.get('reducer')!r} ({manifest_hash(theirs)}); "
                         f"one deployment runs one reducer")


def select(reducer: Reducer = DEFAULT):
    """The engine module of ``reducer`` (``forward``, ``forward_tp``, ``KdaStatePool``, ``shard_layer``). A
    deployment calls this once and uses the one module for its engine and its trainer forward."""
    if reducer not in REDUCERS:
        raise ValueError(f"unknown KDA reducer {reducer!r}")
    return importlib.import_module(f"isoexec.ops.kda.{reducer.package}.engine")


def check_pool(reducer: Reducer, pool) -> None:
    """Refuse a state pool that another reducer's engine created."""
    if pool is not None and getattr(pool, "reducer", None) is not reducer:
        other = getattr(getattr(pool, "reducer", None), "name", type(pool).__name__)
        raise ValueError(f"KDA engine of reducer {reducer.name!r} was handed the state pool of {other!r}; "
                         f"the two reducers are not mixed inside one deployment")


__all__ = ["CHUNK", "DEFAULT", "REDUCERS", "Reducer", "TOKEN", "assert_same_manifest", "check_pool", "csrc",
           "cuda_defines", "manifest", "manifest_hash", "segments", "select", "source_sha256"]
