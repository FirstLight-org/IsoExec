"""Run one Gloo world per test module and collect per-rank claim results.
A private file rendezvous avoids the port-release race; missing rank results fail each claim."""

from __future__ import annotations

import multiprocessing as mp


def run_world(body, *, world: int, rendezvous, timeout_s: int = 240) -> dict:
    """Spawn ``world`` ranks running ``body(rank, world, out, init_method)`` and return ``{rank: res}``.

    ``body`` initialises its own process group from ``init_method`` -- the gloo shims each of these
    modules installs have to be in place before the first collective, and only the body knows
    which.
    """
    ctx = mp.get_context("spawn")
    out = ctx.Manager().dict()
    init_method = f"file://{rendezvous}"
    procs = [ctx.Process(target=body, args=(rank, world, out, init_method)) for rank in range(world)]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=timeout_s)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"worker exit codes {codes}"
        results = dict(out)
        # A rank that exited 0 having written nothing would make every claim an empty loop.
        assert sorted(results) == list(range(world)), f"ranks that reported: {sorted(results)}"
        return results
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(5)


def assert_claim(results: dict, claim: str, world: int) -> None:
    """``claim`` holds on every rank. A missing rank is a failure, not an empty loop."""
    assert sorted(results) == list(range(world)), f"ranks that reported: {sorted(results)}"
    for rank, res in results.items():
        assert res[claim], (claim, rank)
