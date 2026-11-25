"""NCCL transport policy identities, without importing torch or CUDA.

The unified trainer uses NCCL's default channels on its model-parallel groups.
The vLLM engine bounds them on its original torch and PyNccl communicators. Both
remove process-wide ALGO/MIN/MAX pins: every order-sensitive forward reduction
is owned by PIK, and a process-wide algorithm mismatch can deadlock a later
trainer/engine weight-transfer group.

Legacy process-pin identities remain available for reading earlier receipts.
Deployment receipts distinguish an installed policy from actual group options;
NCCL transport settings are not cross-host numerical digest fields.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

NCCL_ALGO = "NCCL_ALGO"
NCCL_MIN_NCHANNELS = "NCCL_MIN_NCHANNELS"
NCCL_MAX_NCHANNELS = "NCCL_MAX_NCHANNELS"
NCCL_EFFECTIVE_KEYS = (NCCL_ALGO, NCCL_MIN_NCHANNELS, NCCL_MAX_NCHANNELS)

PINNED = "pinned"
ENGINE_PER_COMMUNICATOR = "engine_per_communicator"
CAP8 = "cap8"
ENGINE_CAP8 = "engine_cap8"
ENGINE_CAP16 = "engine_cap16"
UNPINNED = "unpinned"

PINNED_CONSTANTS = {
    NCCL_ALGO: "allreduce:tree",
    NCCL_MIN_NCHANNELS: "1",
    NCCL_MAX_NCHANNELS: "1",
}
CAP8_CONSTANTS = {
    NCCL_ALGO: None,
    NCCL_MIN_NCHANNELS: None,
    NCCL_MAX_NCHANNELS: "8",
}
ENGINE_CAP8_CONSTANTS = {
    NCCL_ALGO: "allreduce:tree",
    NCCL_MIN_NCHANNELS: None,
    NCCL_MAX_NCHANNELS: "8",
}
ENGINE_CAP16_CONSTANTS = {
    NCCL_ALGO: "allreduce:tree",
    NCCL_MIN_NCHANNELS: None,
    NCCL_MAX_NCHANNELS: "16",
}
UNPINNED_CONSTANTS = {
    NCCL_ALGO: None,
    NCCL_MIN_NCHANNELS: None,
    NCCL_MAX_NCHANNELS: None,
}

#: The key ``engine_per_communicator`` adds on top of the (empty) process-wide tuple: the engine's
#: per-communicator channel ceiling.
MAX_CTAS = "max_ctas"

#: Set by ``declare_engine_channels`` once, where the engine installs its channel policy.
_DECLARED_ENGINE_CAP: int | None = None


def declare_engine_channels(cap: int) -> None:
    """Record the installed engine policy; individual groups retain its options."""
    global _DECLARED_ENGINE_CAP
    if cap not in (8, 16):
        raise ValueError(f"unsupported engine per-communicator channel ceiling: {cap}")
    if _DECLARED_ENGINE_CAP is not None and _DECLARED_ENGINE_CAP != cap:
        raise ValueError(f"engine channel policy already declared as {_DECLARED_ENGINE_CAP}, got {cap}")
    _DECLARED_ENGINE_CAP = cap


_CONSTANTS_BY_IMPL = {
    PINNED: PINNED_CONSTANTS,
    CAP8: CAP8_CONSTANTS,
    ENGINE_CAP8: ENGINE_CAP8_CONSTANTS,
    ENGINE_CAP16: ENGINE_CAP16_CONSTANTS,
    UNPINNED: UNPINNED_CONSTANTS,
}


def effective_constants(env: Mapping[str, str] | None = None) -> dict[str, str | None]:
    """Read the three values NCCL itself sees, normalizing absent/blank values to ``None``."""
    src = os.environ if env is None else env
    return {key: (str(src.get(key, "")).strip() or None) for key in NCCL_EFFECTIVE_KEYS}


def identity_for_constants(constants: Mapping[str, str | None]) -> str:
    """Name an exact supported tuple; reject every partially pinned or differently capped tuple."""
    normalized = {key: constants.get(key) for key in NCCL_EFFECTIVE_KEYS}
    for impl_id, expected in _CONSTANTS_BY_IMPL.items():
        if normalized == expected:
            return impl_id
    rendered = ", ".join(f"{key}={normalized[key]!r}" for key in NCCL_EFFECTIVE_KEYS)
    raise ValueError(f"unsupported effective NCCL channel composition: {rendered}")


def effective_identity(
    env: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, str | None]]:
    """Return ``(identity, exact constants)`` for the live process, refusing unknown mixtures.

    ``engine_per_communicator`` wins when the engine has DECLARED its policy and the three
    process-wide values are all unset. If a launcher pins the process anyway, the env tuple decides:
    the declaration describes intent, the environment what NCCL will read.
    """
    constants = effective_constants(env)
    if _DECLARED_ENGINE_CAP is not None and all(v is None for v in constants.values()):
        return ENGINE_PER_COMMUNICATOR, {**UNPINNED_CONSTANTS, MAX_CTAS: str(_DECLARED_ENGINE_CAP)}
    return identity_for_constants(constants), constants


#: The ceiling the shipped recipes pin on every engine
#: (``run_qwen35_35b_dapo_isoexec.sh:459``, ``run_qwen35_9b_dapo_isoexec.sh:161``,
#: ``run_qwen35_dapo_isoexec.sh:45``, ``bench_rollout.py`` and ``bench_bridge_rollout.py``: all five
#: say ``ENGINE_NCCL_MAX_NCHANNELS=16``). It was the DEFAULT here that said eight, which no
#: deployment has ever used -- so the default named a tuple nothing runs.
ENGINE_CHANNEL_CAP_DEFAULT = "16"


def engine_channel_pin_neutralized(env: Mapping[str, str] | None = None) -> bool:
    """Whether an ENGINE process removes vLLM's one-channel NCCL floor. Default ON under pik.

    vLLM's ``override_envs_for_invariance`` re-pins ``NCCL_MIN/MAX_NCHANNELS=1`` in every worker
    because ITS reductions are channel-sensitive. Ours are not, and the module docstring's
    ``unpinned`` entry is the argument: 11 of the 12 NCCL primitives in a decode step are AllGathers
    (byte permutations, and a channel count cannot change a byte permutation), the 12th is an
    order-free bf16 VocabParallelEmbedding all-reduce with one non-zero addend per element, and the
    row-parallel reduce -- the one that WOULD be channel-sensitive -- is pik's leaf-tree plan, which
    never enters NCCL.

    That last clause is the whole condition, so it is what the default reads: ON when
    ``ISOEXEC_PIK=1``, because then ``Placement`` owns the reduce; OFF otherwise, because then
    NCCL owns it and its channel count is numerics. An explicit
    ``ISOEXEC_ENGINE_NCCL_UNPIN`` still wins in both directions -- including ``0``, which is how
    a deployment asks for vLLM's pin back.
    """
    src = os.environ if env is None else env
    configured = src.get("ISOEXEC_ENGINE_NCCL_UNPIN")
    if configured is not None and str(configured).strip() != "":
        return str(configured).strip() == "1"
    return str(src.get("ISOEXEC_PIK", "0")).strip() == "1"


def engine_channel_cap(env: Mapping[str, str] | None = None) -> str | None:
    """The ceiling applied to each original engine communicator; ``None`` is undeclared."""
    src = os.environ if env is None else env
    return str(src.get("ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS", ENGINE_CHANNEL_CAP_DEFAULT)).strip() or None


def requested_engine_identity(env: Mapping[str, str] | None = None) -> str:
    """Resolve the neutralized engine channel policy (see :func:`engine_channel_pin_neutralized`)."""
    if not engine_channel_pin_neutralized(env):
        # Runtime census still reads effective values and reports ``pinned`` if vLLM re-pins.
        return UNPINNED
    cap = engine_channel_cap(env)
    if cap not in ("8", "16"):
        raise ValueError(f"unsupported engine per-communicator channel ceiling: {cap!r}")
    return ENGINE_PER_COMMUNICATOR


def census(impl_id: str, constants: Mapping[str, str | None]) -> str:
    """Stable, grep-friendly identity plus the exact values represented by it."""
    return (
        f"impl={impl_id} ALGO={constants.get(NCCL_ALGO)!r} "
        f"MIN={constants.get(NCCL_MIN_NCHANNELS)!r} "
        f"MAX={constants.get(NCCL_MAX_NCHANNELS)!r}"
    )


# Numerical agreement compares the PIK execution digest. NCCL transport policy
# and retained communicator options are separate, side-local receipt material.
