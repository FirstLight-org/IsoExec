"""The pik reduction plan (fixed K leaves, balanced fp32 tree, one root round) and its transport selection.
Leaf GEMMs are non-split-K, the root rounds once, and NCCL/P2P move bytes without choosing reduction order."""

from __future__ import annotations

import logging
import os

import torch

from isoexec.ops.collectives.pik_bootstrap import ensure_pik

logger = logging.getLogger(__name__)

_PLAN = None


def _num_leaves() -> int:
    g = int(os.environ.get("ISOEXEC_PIK_LEAVES", "8"))
    if g < 1 or (g & (g - 1)) != 0:
        raise ValueError(
            f"ISOEXEC_PIK_LEAVES must be a power of two, got {g}. It is a CONTRACT CONSTANT: "
            "set it to the largest TP either the trainer or the engine will ever run (and no larger "
            "-- every extra leaf is pure cost), and set it identically on both sides."
        )
    return g


def _leaf_dtype() -> torch.dtype:
    # The manifest pins the resolved value, so trainer and engine cannot split on it.
    s = os.environ.get("ISOEXEC_PIK_LEAF_DTYPE", "bf16").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32", "f32"):
        return torch.float32
    raise ValueError(f"ISOEXEC_PIK_LEAF_DTYPE must be bf16 or fp32, got {s!r}")


def _disable_pik_p2p(reason: str) -> None:
    """Force pik's tree all-reduce onto NCCL, disabling the P2P symmetric-memory path.

    Bitwise neutral: NCCL moves bytes, pik still owns the tree. ``reason`` is the one logged line's why.
    """
    import pik.allreduce as _ar  # type: ignore
    import pik.linear as _lin  # type: ignore

    _false = lambda *a, **k: False  # noqa: E731
    _ar.p2p_available = _false
    _lin.p2p_available = _false  # linear.py bound its own reference at import
    print(
        f"[ISOEXEC-PIK] disabled pik P2P symmetric-memory all-reduce -> NCCL transport ({reason}; "
        "pik still owns the tree arithmetic)",
        flush=True,
    )


#: Architectures whose pik tree all-reduce is NCCL-only, with the reason. ``ISOEXEC_PIK_P2P`` selects
#: the transport only on the others. Empty: sm100 was listed after the P2P all-reduce hung at TP4 on
#: B200, but the hang was not the architecture -- symmetric pools, flag pools and admission verdicts were
#: keyed by group SIZE, so two groups of one size in a process shared allocations rendezvoused with the
#: other group's ranks (pik/allreduce.py now keys them by the group).
NCCL_ONLY_ARCHS: dict[str, str] = {}


def p2p_arch_refusal(arch: str | None = None) -> str | None:
    """Why ``arch`` (default: the live device's) may not use pik's P2P transport, or None if it may."""
    if arch is None:
        from isoexec.ops._dispatch import current_arch

        arch = current_arch()
    return NCCL_ONLY_ARCHS.get(arch)


def engine_p2p_wanted() -> bool:
    """The ENGINE keeps pik's symmetric-memory P2P transport unless ``ISOEXEC_PIK_P2P=0``.

    On an NCCL-only architecture (``NCCL_ONLY_ARCHS``) the answer is always False, and an explicit
    ``ISOEXEC_PIK_P2P=1`` there is refused rather than ignored. The engine TP group is one rank per GPU,
    so the colocation overlap that forces NCCL on the trainer does not apply.
    """
    refusal = p2p_arch_refusal()
    if refusal is None:
        return os.environ.get("ISOEXEC_PIK_P2P", "1") == "1"
    if os.environ.get("ISOEXEC_PIK_P2P") == "1":
        raise RuntimeError(
            f"ISOEXEC_PIK_P2P=1 requests pik's symmetric-memory P2P transport, but {refusal}. "
            "Unset ISOEXEC_PIK_P2P (or set it to 0)."
        )
    return False


def select_engine_transport(side: str = "ENGINE") -> str:
    """Select and log the engine transport once, returning p2p or nccl.
    Unified and patched host paths share this flag decision; hooks report actual use separately."""
    ensure_pik()
    import pik.allreduce as _ar  # type: ignore

    if engine_p2p_wanted():
        print(
            f"[ISOEXEC-{side}] pik P2P symmetric-memory all-reduce KEPT (engine TP group is 1 rank/GPU; "
            "ISOEXEC_PIK_P2P=0 forces NCCL). This states an INTENTION -- wait for the "
            "TRANSPORT RESOLVED line for what actually ran.",
            flush=True,
        )
        chosen = "p2p"
    else:
        _disable_pik_p2p(p2p_arch_refusal() or "ISOEXEC_PIK_P2P=0")
        chosen = "nccl"

    def _on_transport(state: str, impl_fn) -> None:
        counts = _ar.transport_counts()
        print(
            f"[ISOEXEC-{side}] pik tree all-reduce TRANSPORT RESOLVED: {state} via "
            f"{getattr(impl_fn, '__module__', '?')}.{getattr(impl_fn, '__qualname__', impl_fn)} "
            f"(p2p={counts['p2p_calls']} nccl={counts['nccl_calls']} fallbacks={counts['p2p_fallbacks']})",
            flush=True,
        )

    if not getattr(select_engine_transport, "_hooked", False):
        _ar.register_transport_hook(_on_transport)
        select_engine_transport._hooked = True  # type: ignore[attr-defined]
    return chosen


def trainer_p2p_wanted() -> bool:
    """``ISOEXEC_PIK_TRAINER_P2P`` (default 1): the trainer may use symmetric memory, per group, when a
    group-agreed probe finds its ranks on one host and on pairwise distinct PHYSICAL GPUs (device UUIDs).
    Never on an NCCL-only architecture."""
    if p2p_arch_refusal() is not None:
        return False
    return os.environ.get("ISOEXEC_PIK_TRAINER_P2P", "1").strip() not in ("0", "false", "no", "off")


def select_trainer_transport(side: str = "TRAINER") -> str:
    """Select the trainer's transport policy once; returns ``p2p_physical`` or ``nccl``.

    Ray trainer actors (verl, SkyRL) each see one GPU as ``cuda:0``; symmetric memory used to be refused
    for them as "overlapping devices". Peers are now admitted per group by physical identity (host +
    device UUID, ``pik/lane.py``), so a single-node TP group gets the P2P owner path the engine runs; a
    group spanning hosts or sharing a GPU keeps NCCL. Either way the trainer:
      * never hands a symmetric buffer to autograd (``set_private_outputs``): verl frees checkpoint-input
        gradients with ``untyped_storage().resize_(0)``, and a pool view would be overwritten anyway;
      * may pipeline large row-parallel sites and overlap the dgrad tree sum with wgrad on a comm lane, only
        when asked (``ISOEXEC_PIK_OVERLAP=1``, off by default) and with more than one hardware queue. Schedule
        only: same leaves, tree and root round.
    The P2P choice is a group vote: a rank with ``ISOEXEC_PIK_TRAINER_P2P=0`` still runs the group's probe and
    votes no (the probe is a collective; a rank skipping it would hang or mismatch its peers).
    """
    ensure_pik()
    if not getattr(select_trainer_transport, "_selected", None):
        import pik.allreduce as _ar  # type: ignore

        import pik.lane as _lane  # type: ignore  # canonical name (ensure_pik above)

        _ar.set_private_outputs(True)
        refusal = p2p_arch_refusal()
        if refusal is None:
            wanted = trainer_p2p_wanted()
            _lane.set_peer_policy("physical", wanted=wanted)
            chosen = "p2p_physical"
            print(
                f"[ISOEXEC-{side}] pik P2P symmetric-memory transport admitted PER GROUP by physical device "
                "identity (host + GPU UUID; Ray's rank-local cuda:0 no longer refuses it); this rank votes "
                f"{'for' if wanted else 'against'} it (ISOEXEC_PIK_TRAINER_P2P), any 'against' keeps the group on "
                "NCCL. Comm/compute overlap: "
                f"{'requested' if _lane.overlap_enabled() else 'off'} (ISOEXEC_PIK_OVERLAP).",
                flush=True,
            )
        else:
            _disable_pik_p2p(refusal)
            chosen = "nccl"
        _install_transport_hook(side)
        select_trainer_transport._selected = chosen  # type: ignore[attr-defined]
    return select_trainer_transport._selected  # type: ignore[attr-defined]


# Which transport actually ran: a silent P2P->NCCL degradation costs latency with no other symptom.
# impl_id stays `pik_tree` either way, since both transports evaluate the same tree.
def pik_transport_status() -> dict:
    """Counts of which transport pik's tree all-reduce ran on in THIS process.

    ``state == "none"`` means no all-reduce has run yet (TP=1, or pre-forward), not that P2P is off.
    """
    try:
        ensure_pik()
        import pik.allreduce as _ar  # type: ignore

        return _ar.transport_counts()
    except Exception as e:  # noqa: BLE001 -- a status read must never raise
        return {"state": "unknown", "error": repr(e)}


def _install_transport_hook(side: str) -> None:
    """Make the FIRST resolved transport (and any later degradation) a logged, recorded fact."""
    ensure_pik()
    import pik.allreduce as _ar  # type: ignore

    def _on_transport(state: str, impl_fn) -> None:
        counts = _ar.transport_counts()
        print(
            f"[ISOEXEC-{side}] pik tree all-reduce TRANSPORT RESOLVED: {state} via "
            f"{getattr(impl_fn, '__module__', '?')}.{getattr(impl_fn, '__qualname__', impl_fn)} "
            f"(p2p={counts['p2p_calls']} nccl={counts['nccl_calls']} fallbacks={counts['p2p_fallbacks']})",
            flush=True,
        )

    _ar.register_transport_hook(_on_transport)


def get_plan():
    """The single shared ReductionPlan. Both trainer and engine MUST build the same one."""
    global _PLAN
    if _PLAN is None:
        ensure_pik()
        from pik import ReductionPlan  # type: ignore

        _PLAN = ReductionPlan(num_leaves=_num_leaves(), leaf_dtype=_leaf_dtype())
    return _PLAN
