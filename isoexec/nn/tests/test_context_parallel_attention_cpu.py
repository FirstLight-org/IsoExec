"""Two-rank gloo checks that placed CP=2 attention returns the CP=1 token rows bitwise. Context
uses global packed boundaries and local per-sequence positions; gradient checks live in the
collective tests.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import time

import pytest
import torch

from isoexec.ops.table import OpTable

WORLD = 2
#: Two packed sequences of 8 and 12 tokens. Both are multiples of ``2*cp`` = 4, which is what the
#: zigzag cut requires and what the worker's own padding guarantees
#: (``cp_layout._chunk_sizes`` refuses anything else rather than padding it here).
CU = [0, 8, 20]
SEED = 11


def _placement(cp: int, rank: int, group):
    from isoexec.schema import Axis, Placement

    one = Axis(1, 0, None)
    return Placement(
        tp=one, ep=one, dp=one, pp=one, etp=one,
        cp=Axis(cp, rank, group) if cp > 1 else one,
        side="trainer", leaves=8,
    )


def _global_positions():
    return torch.cat([torch.arange(b - a) for a, b in itertools.pairwise(CU)]).to(torch.int64)


def _row_at_a_time_mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``ops/mm/reference.ref_mm``'s recipe (fp32 accumulation over the whole K, one output round),
    evaluated one row per call, so a row's bits cannot depend on how many rows share the call."""
    w32 = w.to(torch.float32).t()
    rows = [x[i : i + 1].to(torch.float32) @ w32 for i in range(int(x.shape[0]))]
    return (torch.cat(rows) if rows else x.to(torch.float32) @ w32).to(x.dtype)


def _body(rank: int, world: int, out: dict) -> None:
    import torch.distributed as dist

    from isoexec.core.arch import NON_ACCELERATOR_ARCH
    from isoexec.ops import _dispatch
    from isoexec.ops.mm import reference as mm_reference

    # The same pin this folder's conftest applies; a spawned child runs no fixture, and without it
    # both sides would resolve the sm90 kernels on an H100 box and fail on CPU tensors.
    _dispatch.ARCH = NON_ACCELERATOR_ARCH
    # The claim below is about the CP placement (gather, unzigzag, local positions, keep the owned
    # rows), and it holds only through per-token ops that are row-invariant. The production mm
    # kernels are batch-invariant by contract; the CPU reference GEMM is not: MKL's sgemm picks its
    # blocking by M, so on some hosts (an Intel Xeon 8570, AVX-512, even at one thread) a row's bits
    # differ between the CP=1 call (M=20) and a CP=2 rank's call (M=10) before CP does anything.
    # Both sides therefore bind the same reference recipe evaluated one row at a time. The cp_gather
    # collective test checks the wrapped core itself, without any GEMM.
    mm_reference.ref_mm = _row_at_a_time_mm

    from isoexec.nn import place
    from isoexec.ops.collectives.cp_layout import local_positions, local_to_global
    from isoexec.schema import make_sequence_context

    from isoexec.nn.tests import _fixtures as F

    dist.init_process_group("gloo", rank=rank, world_size=world)
    group = dist.group.WORLD
    res: dict = {}

    row = F.spec()
    reference = F.component(row, device="cpu")
    contextual = F.component(row, device="cpu")
    F.mirror_weights(reference, contextual, seed=SEED)
    place(reference, _placement(1, 0, None), prefix=F.SITE, op_table=OpTable(mode="reference"))
    place(contextual, _placement(world, rank, group), prefix=F.SITE, op_table=OpTable(mode="reference"))

    # (1) the installation: the cp=2 module resolves the gather, the cp=1 module records no movement
    gather = contextual.slots.resolved()["cp_gather"]
    res["cp2_resolves_the_gather"] = (gather.op, gather.impl, dict(gather.inputs)) == (
        "collectives.cp_gather", "all_gather_unzigzag_fwd_sum_rs_bwd", {"cp": world}
    )
    res["cp1_resolves_no_movement"] = reference.slots.resolved()["cp_gather"].impl == "none"
    res["the_core_that_runs_is_the_wrapped_one"] = (
        contextual.slots.resolved()["core"].fn.cp_layout == "gather_full_keep_owned"
    )
    res["the_cores_name_the_same_kernel"] = (
        contextual.slots.resolved()["core"].impl == reference.slots.resolved()["core"].impl
    )
    res["both_project_through_the_row_invariant_reference"] = all(
        m.slots.resolved()["mm"].fn is _row_at_a_time_mm for m in (reference, contextual)
    )

    # (2) the numbers. One input, one set of weights; the cp rank sees only its own rows.
    total = CU[-1]
    gen = torch.Generator().manual_seed(SEED + 1)
    x = torch.randn(total, F.HIDDEN, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    mine = local_to_global(CU, world, rank)
    cu = torch.tensor(CU, dtype=torch.int32)
    max_q = max(b - a for a, b in itertools.pairwise(CU))

    with torch.no_grad():
        whole = reference(x, make_sequence_context(_global_positions(), cu, max_q=max_q))
        # The host's two halves: global boundaries, local positions.
        local_seq = make_sequence_context(
            torch.tensor(local_positions(CU, world, rank), dtype=torch.int64), cu, max_q=max_q
        )
        part = contextual(x[mine], local_seq)

    res["local_row_count"] = int(part.shape[0])
    res["expected_row_count"] = len(mine)
    res["cp2_rows_are_the_cp1_rows_bitwise"] = bool(torch.equal(F.bits(part), F.bits(whole[mine])))
    # Not vacuous: this rank must not hold every row, or "the rows it keeps" is the whole batch.
    res["the_cut_is_real"] = 0 < len(mine) < total
    # Nor may the comparison pass by both sides being zero or constant.
    res["the_output_varies"] = bool(part.to(torch.float32).std() > 0)
    with torch.no_grad():
        again = contextual(x[mine], local_seq)
    res["repeatable"] = bool(torch.equal(F.bits(part), F.bits(again)))

    out[rank] = res
    dist.barrier(group=group)
    dist.destroy_process_group()


def _run(rank, world, out, addr, port):
    os.environ.update({"MASTER_ADDR": addr, "MASTER_PORT": str(port), "RANK": str(rank), "WORLD_SIZE": str(world)})
    _body(rank, world, out)


@pytest.fixture(scope="module")
def results():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    ctx = mp.get_context("spawn")
    out = ctx.Manager().dict()
    procs = [ctx.Process(target=_run, args=(r, WORLD, out, "127.0.0.1", port)) for r in range(WORLD)]
    # Use one world deadline and terminate stragglers after a worker failure.
    try:
        for p in procs:
            p.start()
        deadline = time.monotonic() + 180
        for p in procs:
            p.join(timeout=max(0, deadline - time.monotonic()))
        assert not any(p.is_alive() for p in procs), "the cp=2 gloo world timed out"
        assert [p.exitcode for p in procs] == [0] * WORLD
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            if p.pid is not None:
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=5)
    return dict(out)


CLAIMS = (
    "cp2_resolves_the_gather",
    "cp1_resolves_no_movement",
    "the_core_that_runs_is_the_wrapped_one",
    "the_cores_name_the_same_kernel",
    "both_project_through_the_row_invariant_reference",
    "cp2_rows_are_the_cp1_rows_bitwise",
    "the_cut_is_real",
    "the_output_varies",
    "repeatable",
)


@pytest.mark.parametrize("claim", CLAIMS)
def test_a_placed_attention_row_under_context_parallelism(results, claim):
    assert sorted(results) == list(range(WORLD))
    for rank, res in results.items():
        assert res[claim], (claim, rank)


def test_the_two_ranks_between_them_hold_every_token_exactly_once(results):
    """Check complete, disjoint token ownership before comparing CP output rows."""
    assert sum(res["local_row_count"] for res in results.values()) == CU[-1]
    for res in results.values():
        assert res["local_row_count"] == res["expected_row_count"] == CU[-1] // WORLD


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
