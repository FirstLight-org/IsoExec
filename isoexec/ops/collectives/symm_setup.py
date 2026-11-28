"""The one entry point for symmetric-memory rendezvous: refuse a setup whose ranks are at different setup counts.

torch numbers a rendezvous's handle exchange per PROCESS, not per group. A rank that takes part in a setup its peers
are not in (a subgroup alone, a pool growing on one rank, a setup in a branch only some ranks take) is one step
ahead of them, and the next rendezvous they share waits forever on keys the others never write. ``rendezvous``
therefore first gathers the group's process-wide setup counts and buffer sizes (one all-gather); when they differ, EVERY rank of the group raises
``SymmetricSetupSkew`` (a named refusal, never a hang) and the group is marked skewed for good, so its caller takes
its non-symmetric realization (a skewed group is remembered by its rank tuple, so a re-created group object with
the same ranks stays refused; each site keeps one with the same bits: the tree reducer's all-gather form,
MegaMoE's fixed-order exchange).

One process-wide counter: pik's (``pik.allreduce.rendezvous``, ``SymmetricSetupSkew``, a process-wide
``_RENDEZVOUS``). This module delegates every setup to it, so pik's own pools, the tree reducer and the MoE exchanges
(``ops/moe/symm_rendezvous``, a stateless delegate to this module) share ONE counter; a process whose pik lacks the
counted rendezvous is refused by name rather than counted twice. ``SHARE_PIK_COUNTER = False`` switches to this
module's own counter for the mocked tests of the check only.

``ISOEXEC_SYMM_SETUP_LOG=<dir>``: every setup appends ``<time> pid=<pid> rank=<global rank> site=<site>
group=<ranks> count=<before> ok|skew`` to ``<dir>/symm-setup-<pid>.txt`` (diagnostics).
"""
from __future__ import annotations

import os
import time

import torch

#: Delegate to pik's counted rendezvous (one counter for both; required). False only in tests of this module's own
#: check, which then counts with ``_RENDEZVOUS``.
SHARE_PIK_COUNTER = True


class SymmetricSetupSkew(RuntimeError):
    """The group's ranks are at different points of the process-wide symmetric setup sequence; no rendezvous was
    attempted. The caller takes its non-symmetric realization (same bits)."""


class SymmetricSetupInCapture(RuntimeError):
    """A symmetric setup was requested inside CUDA graph capture (setups are lifecycle operations, outside capture)."""


_RENDEZVOUS = [0]          # this module's process-wide setup counter (used when pik's is not available)
_SKEWED: set = set()


def group_key(group) -> tuple:
    """The group's global ranks in group order (the identity of a rendezvous group)."""
    import torch.distributed as dist
    g = group if group is not None else dist.group.WORLD
    return tuple(dist.get_process_group_ranks(g))


def _pik():
    if not SHARE_PIK_COUNTER:
        return None
    try:
        # THE pik instance: pik is registered as top-level ``pik`` and aliased under the dotted name
        # (``pik_bootstrap.ensure_pik``); importing the dotted child before the alias exists would execute
        # ``allreduce.py`` a second time and give it a second setup counter. ``ensure_pik`` first, then the
        # canonical module.
        import importlib

        from isoexec.ops.collectives.pik_bootstrap import ensure_pik
        ensure_pik()
        ar = importlib.import_module("pik.allreduce")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("symm_setup: pik's counted rendezvous (pik.allreduce.rendezvous) is not importable; "
                           "a second setup counter would hide skews (set SHARE_PIK_COUNTER = False only in "
                           "tests of this module's own check)") from exc
    if not (callable(getattr(ar, "rendezvous", None)) and hasattr(ar, "_RENDEZVOUS")):
        raise RuntimeError("symm_setup: the installed pik has no counted rendezvous (pik.allreduce.rendezvous)")
    return ar


def counter_owner() -> str:
    """``"pik"`` when setups are counted by pik's counter (shared with pik's own sites), else ``"symm_setup"``."""
    return "pik" if _pik() is not None else "symm_setup"


def setup_count() -> int:
    """Symmetric setups this process has performed through the counted entry point."""
    ar = _pik()
    return int(ar._RENDEZVOUS[0]) if ar is not None else _RENDEZVOUS[0]


def skewed(group) -> bool:
    ar = _pik()
    key = group_key(group)
    return key in _SKEWED or (ar is not None and key in getattr(ar, "_SKEWED", ()))


def _log(site: str, key: tuple, count: int, verdict: str) -> None:
    where = os.environ.get("ISOEXEC_SYMM_SETUP_LOG")
    if not where:
        return
    import torch.distributed as dist
    try:
        os.makedirs(where, exist_ok=True)
        with open(os.path.join(where, f"symm-setup-{os.getpid()}.txt"), "a") as f:
            f.write(f"{time.time():.3f} pid={os.getpid()} rank={dist.get_rank()} site={site} "
                    f"group={','.join(map(str, key))} count={count} {verdict}\n")
    except OSError:
        pass


def _agree(tensor, g, key, site, before, count_too: bool) -> None:
    """One all-gather of (count, bytes) over the group (bytes only when pik counts); a difference refuses the
    group for good, on every rank alike."""
    import torch.distributed as dist
    nbytes = tensor.numel() * tensor.element_size()
    mine = torch.tensor([before if count_too else 0, nbytes], dtype=torch.int64, device=tensor.device)
    flat = torch.empty(dist.get_world_size(g) * 2, dtype=torch.int64, device=tensor.device)
    dist.all_gather_into_tensor(flat, mine, group=g)          # flat output: NCCL and gloo alike
    rows = [tuple(int(v) for v in r) for r in flat.view(-1, 2).tolist()]
    counts, sizes = [r[0] for r in rows], [r[1] for r in rows]
    if len(set(counts)) > 1 or len(set(sizes)) > 1:
        _SKEWED.add(key)
        _log(site, key, before, "skew")
        what = (f"its ranks have performed {counts} symmetric-memory setups" if len(set(counts)) > 1
                else f"its ranks offer buffers of {sizes} bytes")
        raise SymmetricSetupSkew(f"{site}: group {key}: {what}; a rendezvous now would wait or map mismatched "
                                 "buffers. The group takes its non-symmetric realization (same bits).")


def rendezvous(tensor: torch.Tensor, group, *, site: str):
    """Timed (``symm.rendezvous`` in ``core/load_timers``) :func:`_rendezvous`."""
    from isoexec.core import load_timers

    with load_timers.phase("symm.rendezvous"):
        return _rendezvous(tensor, group, site=site)


def _rendezvous(tensor: torch.Tensor, group, *, site: str):
    """``symm_mem.rendezvous(tensor, group)`` after the group-agreement check (collective over ``group``: every rank
    calls it for the same setup). Raises ``SymmetricSetupInCapture`` inside CUDA graph capture (before any
    collective), and ``SymmetricSetupSkew`` on every rank of the group, without a rendezvous, when their setup
    counts or buffer sizes differ, or the group (by its rank tuple) skewed before."""
    import torch.distributed as dist
    from torch.distributed import _symmetric_memory as symm_mem

    if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
        raise SymmetricSetupInCapture(f"{site}: symmetric-memory setup inside CUDA graph capture; prepare it before")
    g = group if group is not None else dist.group.WORLD
    key = group_key(g)
    before = setup_count()
    if key in _SKEWED:
        _log(site, key, before, "skew")
        raise SymmetricSetupSkew(f"{site}: group {key} already took its non-symmetric realization after a setup skew")
    ar = _pik()
    if ar is not None:
        _agree(tensor, g, key, site, before, count_too=False)      # sizes here; pik gathers the counts
        try:
            handle = ar.rendezvous(tensor, g)
        except ar.SymmetricSetupSkew as exc:
            _SKEWED.add(key)
            _log(site, key, before, "skew")
            raise SymmetricSetupSkew(f"{site}: {exc}") from exc
        _log(site, key, before, "ok")
        return handle
    _agree(tensor, g, key, site, before, count_too=True)
    handle = symm_mem.rendezvous(tensor, g)
    _RENDEZVOUS[0] += 1
    _log(site, key, before, "ok")
    return handle


__all__ = ["SHARE_PIK_COUNTER", "SymmetricSetupInCapture", "SymmetricSetupSkew", "counter_owner", "group_key",
           "rendezvous", "setup_count", "skewed"]
