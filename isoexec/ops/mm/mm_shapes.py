"""Dense-GEMM shape vocabulary and install-time coverage reports.
The census comes from the constructed model's components (``isoexec.nn.gemm_census``) and is
instantiated at the live TP/ETP degrees; provider and tile choices remain in kernel tables."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import torch

# Short dtype spellings for banners.
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_DTYPE_NAMES = {v: k for k, v in _DTYPES.items()}


@dataclass(frozen=True)
class GemmSite:
    """One dense 2-D GEMM site, instantiated at a concrete ``(tp, etp)``.

    name:      dotted site id, e.g. ``mla.q_up``; used in the banner so a miss is legible.
    K, N:      the live per-rank shape of the ``mm`` that ``F.linear`` decomposes to, already
               divided by whatever axis shards the site.
    axis:      "tp" | "etp" | "none" -- which parallelism divided this site.
    calls:     how many times EACH TOKEN passes through this shape in one forward, normally the
               number of layers hosting the site. Per-token, not per-launch: a site whose launches
               partition the tokens rather than repeat them counts once.
    table_candidate:
               False for a site deliberately left out of the tables. Still censused, so the banner
               distinguishes "we decided not to" from "nobody looked".
    """

    name: str
    K: int
    N: int
    dtype: torch.dtype
    axis: str = "none"
    calls: int = 1
    table_candidate: bool = True

    @property
    def key(self) -> tuple:
        return (self.K, self.N, self.dtype)

    @property
    def work(self) -> int:
        """``calls * K * N`` -- MACs per forward at M=1, i.e. FLOPs with the batch factored out.

        Used to rank misses, since ranking by call count alone puts the widest (and least frequently
        called) sites last. It is a FLOP proxy, not a time proxy: at decode M these GEMMs are
        occupancy- and launch-bound, so it must never be used to predict a saving.
        """
        return int(self.calls) * int(self.K) * int(self.N)

    def __str__(self) -> str:
        return f"({self.K},{self.N},{_DTYPE_NAMES.get(self.dtype, self.dtype)}) {self.name}"


def dtype_of(name: str) -> torch.dtype:
    """``"bf16"`` -> ``torch.bfloat16``."""
    return _DTYPES[name]


@dataclass(frozen=True)
class GemmSpec:
    """One executed occurrence of a dense GEMM site at full (unsharded) width.

    ``granule`` is the head count a TP-sharded site's width is cut along when the cut must be
    whole (latent attention's ``q_up``); ``0`` means the width is floor-divided like the other
    column-parallel sites.
    """

    name: str
    K: int
    N: int
    dtype: torch.dtype
    axis: str = "none"
    granule: int = 0


@dataclass(frozen=True)
class GemmCensus:
    """The dense GEMM sites of one constructed model, independent of parallelism.

    Built from the model's components by ``isoexec.nn.gemm_census(model)``, so it is a function of
    the model package and its effective HF config alone: a trainer and an engine that construct the
    same model hold equal censuses and reach the same shape-admission decisions. ``specs`` holds one
    entry per executed occurrence; ``sites(tp, etp)`` instantiates them at a parallelism.
    """

    specs: tuple[GemmSpec, ...]

    def sites(self, tp: int, etp: int) -> tuple[GemmSite, ...]:
        """Every dense 2-D GEMM site at ``(tp, etp)``, sorted by ``(name, K, N)``.

        Column-parallel widths are floor-divided by TP (``census_union`` drops inexact cuts);
        a site with a head ``granule`` refuses a TP that does not divide it. Row-parallel sites
        and the routed experts are absent by construction (``isoexec.nn.gemm_census``).
        """
        tp = max(1, int(tp))
        _ = max(1, int(etp or tp))  # no censused site is cut by ETP: routed experts are absent
        for spec in self.specs:
            if spec.axis == "tp" and spec.granule and spec.granule % tp:
                raise ValueError(f"{spec.name}: {spec.granule} heads must be divisible by TP={tp} "
                                 "before deriving GEMM shapes")
        sites: dict[tuple, GemmSite] = {}
        for spec in self.specs:
            n = spec.N // tp if spec.axis == "tp" else spec.N
            if n <= 0:
                continue
            key = (spec.name, spec.K, n, spec.dtype, spec.axis)
            prev = sites.get(key)
            sites[key] = GemmSite(spec.name, spec.K, n, spec.dtype, spec.axis, (prev.calls if prev else 0) + 1)
        return tuple(sorted(sites.values(), key=lambda g: (g.name, g.K, g.N)))

    def digest(self) -> str:
        """A stable identity for comparing two processes' censuses in a receipt."""
        import hashlib

        rows = sorted((s.name, s.K, s.N, _DTYPE_NAMES.get(s.dtype, str(s.dtype)), s.axis, s.granule)
                      for s in self.specs)
        return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]


_LIVE: Optional[dict] = None  # explicitly-registered context, when a runtime knows better than env


def set_live_context(*, census: Optional[GemmCensus], model_path: str = "", tp: int, etp: int,
                     side: str = "?") -> None:
    """Register the model census and parallelism this process is actually running.

    ``census`` is ``isoexec.nn.gemm_census(model)`` of the model this process builds (or ``None``
    when it has none); ``model_path`` only labels the banner. Runtimes call this before the
    providers install. Last-writer-wins and never raises.
    """
    global _LIVE
    _LIVE = {"model_path": str(model_path or ""), "census": census, "tp": int(tp), "etp": int(etp),
             "side": str(side)}


def _int_env(name: str) -> Optional[int]:
    try:
        v = os.environ.get(name)
        return int(v) if v else None
    except ValueError:
        return None


def live_context() -> Optional[dict]:
    """Return model_path, census, tp, etp and side when discoverable.
    Live runtime parallelism wins over environment and registered defaults; the registered census
    and model_path remain authoritative."""
    registered = dict(_LIVE) if _LIVE is not None else None

    census = (registered or {}).get("census")
    model_path = (registered or {}).get("model_path") or os.environ.get("ISOEXEC_MODEL_PATH") or ""
    if not model_path and census is None:
        return None

    side, tp, etp = "?", None, None

    # vLLM is asked FIRST because an engine worker runs a megatron GPTModel inside vLLM and so
    # initialises a megatron parallel state too; only vLLM's state discriminates the two sides.
    try:
        vps = sys.modules["vllm.distributed.parallel_state"]

        if vps.model_parallel_is_initialized():
            tp = vps.get_tensor_model_parallel_world_size()
            side = "ENGINE"
    except Exception:
        pass

    if tp is None:
        try:
            mpu = sys.modules["megatron.core.parallel_state"]

            if mpu.model_parallel_is_initialized():
                tp = mpu.get_tensor_model_parallel_world_size()
                try:
                    etp = mpu.get_expert_tensor_parallel_world_size()
                except Exception:
                    etp = None
                side = "TRAINER"
        except Exception:
            pass

    if side == "ENGINE" or tp is None:
        tp = tp or _int_env("ISOEXEC_ENGINE_TP")
        # At EP=1 every rank tensor-shards every expert, so engine ETP == engine TP.
        if etp is None and _int_env("ISOEXEC_ENGINE_EP") == 1:
            etp = tp
        if side == "?" and tp is not None:
            side = "ENGINE"

    if tp is None and registered is not None:
        tp, etp, side = registered["tp"], registered["etp"], registered["side"]

    if tp is None:
        return None
    return {"model_path": model_path, "census": census, "tp": int(tp), "etp": int(etp or tp), "side": side}


_NO_CENSUS = ("no GEMM census is registered for this process: the runtime that constructs the model "
              "registers isoexec.nn.gemm_census(model) through mm_shapes.set_live_context(census=...)")


def census_for(census: Optional[GemmCensus], tp: int, etp: int) -> tuple:
    """The census's GEMM sites at the requested parallelism.
    A missing census raises so callers can report unavailable coverage instead of an empty model."""
    if census is None:
        raise LookupError(_NO_CENSUS)
    return census.sites(int(tp), int(etp))


def live_census() -> tuple:
    """``(sites, context, why_empty)`` for this process; ``((), None, None)`` when unresolvable.

    ``why_empty`` carries the reason an empty census is empty -- none was registered, or the
    census itself refused -- because the banner that prints it is the only place a run says so,
    and "no shapes" and "nobody registered shapes" are different diagnoses.
    """
    ctx = live_context()
    if ctx is None:
        return (), None, None
    try:
        return census_for(ctx["census"], ctx["tp"], ctx["etp"]), ctx, None
    except Exception as e:  # noqa: BLE001 -- diagnostics must never take down a run
        return (), ctx, f"{type(e).__name__}: {e}"


def census_union(census: Optional[GemmCensus], tps, etps) -> dict:
    """``{(K, N, dtype): site_name}`` over a cross-product of parallelisms.

    This is how a shape table is populated: lookup is by key, so listing every shape the supported
    TPs can produce costs nothing at a TP that does not produce it. Combinations whose division is
    not exact are dropped, as are sites marked ``table_candidate=False``.
    """
    ref = {s.name: s for s in census_for(census, 1, 1)}
    out: dict = {}
    for tp in tps:
        for etp in etps:
            try:
                sites = census_for(census, tp, etp)
            except Exception:
                continue
            for s in sites:
                if not s.table_candidate:
                    continue
                base = ref.get(s.name)
                if base is not None:
                    div = tp if s.axis == "tp" else (etp if s.axis == "etp" else 1)
                    if base.K % div or base.N % div:
                        continue  # this parallelism does not divide this site exactly
                out.setdefault(s.key, f"{s.name}@tp{tp}/etp{etp}")
    return out


def report_install_coverage(table_name: str, covered: Callable[[tuple], bool], *, flag: str) -> None:
    """Print which LIVE-MODEL shapes ``table_name`` covers, and which it does not.

    ``covered(key)`` answers membership for a ``(K, N, dtype)`` key.  Never raises.
    """
    try:
        sites, ctx, why_empty = live_census()
        if ctx is None:
            print(
                f"[ISOEXEC-MM-COVERAGE] {table_name}: could not resolve the live model/parallelism "
                f"(call mm_shapes.set_live_context with the model's census). "
                f"Coverage will be reported from OBSERVED traffic instead -- see the "
                f"[ISOEXEC-MM-COVERAGE] OBSERVED banner after the first forward.",
                flush=True,
            )
            return
        if not sites:
            print(
                f"[ISOEXEC-MM-COVERAGE] {table_name}: no census for model {ctx['model_path']!r} "
                f"({why_empty or 'the model has no dense GEMM site'}), so install-time "
                f"coverage is UNKNOWN -- a green self-check on this table proves only that the "
                f"table is internally sound, NOT that it covers this model. Register "
                f"isoexec.nn.gemm_census(model) with mm_shapes.set_live_context. "
                f"Falling back to OBSERVED traffic.",
                flush=True,
            )
            return

        hit = [s for s in sites if covered(s.key)]
        miss = [s for s in sites if not covered(s.key)]
        hit_calls = sum(s.calls for s in hit)
        all_calls = sum(s.calls for s in sites) or 1
        hit_work = sum(s.work for s in hit)
        all_work = sum(s.work for s in sites) or 1
        head = (
            f"[ISOEXEC-MM-COVERAGE] {table_name} vs the LIVE census "
            f"(model={ctx['model_path']} side={ctx['side']} tp={ctx['tp']} etp={ctx['etp']}): "
            f"{len(hit)}/{len(sites)} sites covered, {100.0 * hit_calls / all_calls:.1f}% of "
            f"forward GEMM calls, {100.0 * hit_work / all_work:.1f}% of forward GEMM work "
            f"(calls*K*N; a FLOP proxy, NOT a time proxy -- these GEMMs are occupancy-bound at "
            f"decode M)."
        )
        print(head, flush=True)
        if hit:
            print("[ISOEXEC-MM-COVERAGE]   COVERED  : " + "  ".join(str(s) for s in hit), flush=True)
        if miss:
            # Ranked by work, not calls: the widest site is often the least frequently called.
            miss = sorted(miss, key=lambda s: (-s.work, -s.calls))
            print(
                "[ISOEXEC-MM-COVERAGE]   UNCOVERED: "
                + "  ".join(f"{s} x{s.calls} work={100.0 * s.work / all_work:.1f}%" for s in miss)
                + "  <- these run vLLM's Triton matmul_persistent (2.6-4x slower than cuBLAS/GEMM),"
                + "  ranked by WORK share.",
                flush=True,
            )
        if not hit:
            print(
                f"[ISOEXEC-MM-COVERAGE]   *** {table_name} covers ZERO live shapes on this model. "
                f"{flag} is ON and is doing NOTHING. This is the silent-green failure mode: the "
                f"table's own self-check passes because it probes the TABLE, not the MODEL. Either "
                f"populate the census shapes above (and measure them) or turn {flag} off so the log "
                f"stops claiming a win that is not happening. ***",
                flush=True,
            )
    except Exception as e:  # noqa: BLE001 -- diagnostics must never take down a run
        print(f"[ISOEXEC-MM-COVERAGE] {table_name}: coverage report failed ({type(e).__name__}: {e})", flush=True)
