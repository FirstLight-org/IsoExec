"""Test-harness primitives every op's colocated gate battery instantiates.

Moved out of ``core/contracts.py`` with the contract layer, because none of it was ever about a
contract: these are the three obligations a GATE inherits. Comparisons are on integer views of the
tensors, never ``torch.equal``, which reports +0.0 == -0.0. A declared hazard must be proven to have
fired, or the pass is vacuous. And backward connectivity is checked live on the first backward as
exact set equality, since a per-group presence check passes a partially severed graph.

``check_hazard_coverage`` did not come with them: it folded ``ImplSpec.hazards``, which M6.4 deleted
-- the field was an INTENT ("this kernel can hit subnormals") that one report-only test read, and an
intent is not a fact any check can compare.
"""

from __future__ import annotations

from typing import Iterable


class VacuousTestError(AssertionError):
    """Raised when a declared hazard did not actually fire, making the pass vacuous."""


class ConnectivityError(AssertionError):
    """Raised when the live first-backward grad set != the declared trainable set."""


# A float tensor is reinterpreted as the same-width signed integer so the comparison sees the raw
# bit pattern, including the sign bit of a zero.
def _int_view(t):
    import torch

    _MAP = {
        torch.float64: torch.int64,
        torch.float32: torch.int32,
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
    }
    if t.dtype in _MAP:
        return t.contiguous().view(_MAP[t.dtype])
    # already an integer / bool dtype: its bits are its value
    return t.contiguous()


def bitwise_equal(a, b) -> bool:
    """True iff ``a`` and ``b`` are bit-for-bit identical.

    Compares integer reinterpretations, not ``torch.equal``, which returns True for +0.0 == -0.0
    even though the sign bit differs. A dtype or shape mismatch is not bitwise-equal.
    """
    import torch

    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("bitwise_equal compares torch.Tensors")
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(_int_view(a), _int_view(b)))


def assert_hazard_exercised(name: str, evidence) -> None:
    """Assert a declared hazard actually fired.

    ``evidence`` is a count, bool or other truthy witness the test computes from the actual data --
    subnormals in the reference, a NULL lane being present. A falsey witness raises
    ``VacuousTestError``, since any assertion downstream of a hazard that never happened is vacuous.
    """
    ok: bool
    if isinstance(evidence, bool):
        ok = evidence
    elif isinstance(evidence, (int, float)):
        ok = evidence > 0
    else:
        # tensors, sized containers, etc.: truthiness / non-empty
        try:
            ok = bool(len(evidence) > 0)
        except TypeError:
            ok = bool(evidence)
    if not ok:
        raise VacuousTestError(
            f"hazard {name!r} declared but NOT exercised (evidence={evidence!r}); the test is "
            f"vacuous. Supply inputs that make the hazard actually fire (Section 5)."
        )


def collect_grad_set(module) -> set:
    """The set of parameter names whose ``.grad is not None`` -- the live-connectivity witness."""
    return {name for name, p in module.named_parameters() if getattr(p, "grad", None) is not None}


def assert_grad_set_equality(module, expected_names: Iterable[str]) -> None:
    """Assert the live first-backward grad set equals the declared trainable set, exactly.

    A per-group presence check passes a partially severed graph, so only set equality catches it.
    ``ConnectivityError`` names both sides of the symmetric difference. Run this on the first
    backward of a real run: a process-global rebind is invisible to an offline toy test.
    """
    have = collect_grad_set(module)
    want = set(expected_names)
    severed = want - have  # declared trainable but received no grad -> the severed-backward bug
    surprise = have - want  # got a grad but not declared trainable -> a wiring surprise
    if severed or surprise:
        raise ConnectivityError(
            "first-backward grad set != declared trainable set:\n"
            f"  SEVERED (trainable, no grad): {sorted(severed)}\n"
            f"  SURPRISE (grad, not declared): {sorted(surprise)}\n"
            "count-vs-baseline is the degraded-mode minimum; exact set equality is the contract."
        )


def _gloo_entry(target, rank: int, world: int, port: int, results, args) -> None:
    import traceback

    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    # NCCL accepts any output of world x input elements for all_gather_into_tensor; gloo wants it split into
    # chunks of the input's shape. Flatten both so the code under test runs unchanged.
    agit = dist.all_gather_into_tensor

    def all_gather_into_tensor(output, input, group=None, async_op=False):
        return agit(output.view(-1), input.reshape(-1), group=group, async_op=async_op)

    dist.all_gather_into_tensor = all_gather_into_tensor
    try:
        results[rank] = ("ok", target(rank, world, *args))
    except BaseException:  # reported to the parent, which fails the test
        results[rank] = ("error", traceback.format_exc())
    finally:
        dist.destroy_process_group()


def run_gloo(target, world: int, *args, timeout: float = 180.0) -> dict:
    """Run ``target(rank, world, *args)`` in ``world`` spawned CPU processes joined in one gloo group.

    ``target`` must be importable (module level). Returns ``{rank: target's return value}``; raises
    AssertionError on a rank's exception, a non-zero exit or a timeout (a hang in a split configuration is a
    failure, not a stuck test).
    """
    import socket
    import time

    import torch.multiprocessing as mp

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    ctx = mp.get_context("spawn")
    with ctx.Manager() as mgr:
        results = mgr.dict()
        procs = [ctx.Process(target=_gloo_entry, args=(target, r, world, port, results, args)) for r in range(world)]
        try:
            for p in procs:
                p.start()
            deadline = time.monotonic() + timeout
            for p in procs:
                p.join(timeout=max(0.0, deadline - time.monotonic()))
            hung = [r for r, p in enumerate(procs) if p.is_alive()]
            assert not hung, f"ranks {hung} did not finish within {timeout}s (hang)"
            out = {}
            for r, p in enumerate(procs):
                assert r in results, f"rank {r} produced no result (exit code {p.exitcode})"
                status, value = results[r]
                assert status == "ok", f"rank {r} raised:\n{value}"
                out[r] = value
            return out
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
