"""CPU tests for sequence reduce-scatter geometry, dtypes, and transport selection."""

from __future__ import annotations

import importlib
import os
import pathlib
import socket
import sys
import time
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))  # repo root

_WORLD = 4
_N_COLS = 8


def _load_pik_cpu():
    """Import the vendored pik's allreduce+plan WITHOUT importing the pik package (pik.gemm's
    triton autotune decorators query the GPU at import). A stub package whose __path__ points at
    the real directory lets the relative `.codegen` import resolve; codegen generates lazily."""
    if "_pik_cpu.allreduce" in sys.modules:
        return sys.modules["_pik_cpu.allreduce"], sys.modules["_pik_cpu.plan"]
    pikdir = _HERE.parents[1] / "pik"
    pkg = types.ModuleType("_pik_cpu")
    pkg.__path__ = [str(pikdir)]
    sys.modules["_pik_cpu"] = pkg
    ar = importlib.import_module("_pik_cpu.allreduce")
    plan = importlib.import_module("_pik_cpu.plan")
    return ar, plan


def _reference_tree(stacked: torch.Tensor, combine_order) -> torch.Tensor:
    """The balanced tree over the stacked rank axis, elementwise -- the CPU twin of the triton
    tree_reduce_kernel (same combine_order, fp32 adds)."""
    c = stacked.shape[0]
    if c == 1:
        return stacked[0].clone()
    slots = [stacked[i].float().clone() for i in range(c)]
    for dst, lhs, rhs in combine_order(c):
        slots[dst] = slots[lhs] + slots[rhs]
    return slots[0]


def _adversarial(shape, seed) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * torch.exp(torch.randn(*shape, generator=g) * 12)).float()


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("wire", [torch.bfloat16, torch.float32])
def test_p2p_scatter_reads_owned_peer_slices(monkeypatch, rank, wire):
    """The GPU kernel is stubbed; real tensor views verify offsets and staging/barrier order."""
    ar, _ = _load_pik_cpu()
    peers = [(torch.arange(48) + 100 * peer).to(wire) for peer in range(4)]
    events = []

    class Handle:
        def barrier(self):
            events.append("barrier")

        def get_buffer(self, peer, shape, dtype):
            assert shape == (48,) and dtype == wire
            return peers[peer]

    class Pool:
        def stage(self):
            return peers[rank], Handle()

        def flip(self):
            events.append("flip")

    class Kernel:
        def __getitem__(self, grid):
            assert grid == (1,)

            def run(*args, **kwargs):
                events.append("kernel")
                inputs, out, base, count = args[:4], args[4], args[5], args[6]
                assert base == 0 and count == 12
                assert kwargs == {"BLOCK": 1024, "num_warps": 4}
                for peer, tensor in enumerate(inputs):
                    assert torch.equal(tensor, peers[peer][rank * 12 : (rank + 1) * 12])
                a, b, c, d = [tensor.float() for tensor in inputs]
                out.copy_((a + b) + (c + d))

            return run

    def kernel(world, **kwargs):
        assert world == 4
        assert kwargs == {"push": False, "in_bf16": wire == torch.bfloat16, "out_bf16": False, "vec": 1}
        return Kernel()

    monkeypatch.setitem(sys.modules, "triton", types.SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y))
    monkeypatch.setattr(ar.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(ar.dist, "get_rank", lambda group: rank)
    monkeypatch.setattr(ar, "_sym_in_pool", lambda *args: Pool())
    monkeypatch.setattr(ar, "p2p_allreduce_kernel", kernel)
    partial = (peers[rank] + 3).view(12, 4)
    got = ar._tree_reduce_scatter_p2p(partial, object())
    a, b, c, d = [tensor.float() for tensor in peers]
    want = ((a + b) + (c + d))[rank * 12 : (rank + 1) * 12]
    assert got.dtype == torch.float32
    assert torch.equal(got, want)
    assert events == ["barrier", "kernel", "flip"]


def _worker(rank: int, results, port: int):
    # The parent chooses the rendezvous port to avoid joining another concurrent test's process group.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD)
    try:
        ar, plan = _load_pik_cpu()

        # Stub the triton kernel with the torch twin -- in BOTH paths, so AR vs RS compares
        # transports and slicing, with identical tree arithmetic.
        def _tree_reduce_cpu(stacked, out):
            out.copy_(_reference_tree(stacked, plan.combine_order))
            return out

        ar._tree_reduce = _tree_reduce_cpu

        # gloo insists output chunks have the INPUT's shape; NCCL (production) treats the output
        # as flat. Shim to the flat view so pik's one-shot [world, M, N] gather buffer works here.
        _orig_ag = dist.all_gather_into_tensor

        def _ag_flat(out, inp, group=None, **kw):
            world = dist.get_world_size(group)
            inp = inp.contiguous()
            return _orig_ag(out.view(world * inp.shape[0], *inp.shape[1:]), inp, group=group, **kw)

        ar.dist.all_gather_into_tensor = _ag_flat

        failures = []

        def check(name, cond):
            if not cond:
                failures.append(name)

        for rows, oneshot_mb, seed in (
            (8, 1 << 30, 0),  # tiny payload, AR takes ONE-SHOT (forced huge crossover)
            (8, 0, 1),  # tiny payload, AR forced TWO-SHOT
            (64, 1 << 30, 2),
            (64, 0, 3),
        ):
            ar.ONESHOT_MAX_BYTES = oneshot_mb
            # every rank builds every rank's partial deterministically; keeps its own
            partials = [_adversarial((rows, _N_COLS), 1000 * seed + r) for r in range(_WORLD)]
            mine = partials[rank].clone()

            full = ar.tree_all_reduce(mine.clone(), group=None, backend="nccl")
            sliced = ar.tree_reduce_scatter(mine.clone(), group=None)

            # (2) Megatron convention: first dim, equal chunks, rank order
            lo = rank * (rows // _WORLD)
            hi = lo + rows // _WORLD
            check(
                f"rs==ar_slice rows={rows} oneshot={oneshot_mb} ",
                torch.equal(sliced, full[lo:hi]),
            )

            # (1) all-gather of the RS slices reproduces AR bitwise
            gathered = torch.empty_like(full)
            dist.all_gather_into_tensor(gathered, sliced.contiguous())
            check(f"ag(rs)==ar rows={rows} oneshot={oneshot_mb}", torch.equal(gathered, full))

            # cross-check against the single-process reference tree
            ref = _reference_tree(torch.stack(partials), plan.combine_order)
            check(f"ar==ref rows={rows} oneshot={oneshot_mb}", torch.equal(full, ref))

        # Record the actual all-to-all dtype: widening before transport doubles
        # the payload even though each bf16 leaf can be widened exactly at load.
        wire_dtypes = []
        original_a2a = dist.all_to_all_single

        def capture_a2a(out, inp, **kwargs):
            wire_dtypes.append((inp.dtype, out.dtype))
            return original_a2a(out, inp, **kwargs)

        dist.all_to_all_single = capture_a2a

        # (4) bf16 leaf wire: fp32 adds, bf16 root, equal to AR's bf16 root sliced
        parts_bf = [_adversarial((8, _N_COLS), 77 + r).to(torch.bfloat16) for r in range(_WORLD)]
        full_bf = ar.tree_all_reduce(parts_bf[rank].clone(), group=None, backend="nccl")
        wire_dtypes.clear()
        sliced_bf = ar.tree_reduce_scatter(parts_bf[rank].clone(), group=None)
        check("bf16 stays bf16 on wire", wire_dtypes == [(torch.bfloat16, torch.bfloat16)])
        check("bf16 dtype", sliced_bf.dtype == torch.bfloat16)
        check("bf16 rs==ar_slice", torch.equal(sliced_bf, full_bf[2 * rank : 2 * rank + 2]))
        # decoupled root: bf16 wire + fp32 root
        full_f = ar.tree_all_reduce(parts_bf[rank].clone(), group=None, backend="nccl", root_dtype=torch.float32)
        wire_dtypes.clear()
        sliced_f = ar.tree_reduce_scatter(parts_bf[rank].clone(), group=None, root_dtype=torch.float32)
        check("fp32 root keeps bf16 wire", wire_dtypes == [(torch.bfloat16, torch.bfloat16)])
        dist.all_to_all_single = original_a2a
        check("bf16->fp32 root dtype", sliced_f.dtype == torch.float32)
        check("bf16->fp32 rs==ar_slice", torch.equal(sliced_f, full_f[2 * rank : 2 * rank + 2]))

        # (3) refusal: first dim 6 not divisible by world 4
        try:
            ar.tree_reduce_scatter(torch.randn(6, _N_COLS), group=None)
            failures.append("divisibility not refused")
        except AssertionError:
            pass

        results[rank] = failures
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_tree_reduce_scatter_algebra_gloo():
    ctx = mp.get_context("spawn")
    port = _free_port()
    with ctx.Manager() as mgr:
        results = mgr.dict()
        procs = [ctx.Process(target=_worker, args=(r, results, port)) for r in range(_WORLD)]
        try:
            for p in procs:
                p.start()
            deadline = time.monotonic() + 180
            for p in procs:
                p.join(timeout=max(0, deadline - time.monotonic()))
            assert not any(p.is_alive() for p in procs), "the gloo world timed out"
            for r in range(_WORLD):
                assert not procs[r].exitcode, f"rank {r} exited {procs[r].exitcode}"
                assert r in results, f"rank {r} produced no result"
                assert results[r] == [], f"rank {r} failures: {results[r]}"
        finally:
            # A timed-out rank must not outlive the assertion that reported it.
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                if p.pid is not None:
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=5)


if __name__ == "__main__":
    test_tree_reduce_scatter_algebra_gloo()
    print("OK")


def test_tree_kernel_widens_bf16_leaves_before_adding(monkeypatch):
    ar, _ = _load_pik_cpu()
    calls = []
    monkeypatch.setattr(ar, "tree_reduce_kernel", lambda size, **kw: calls.append((size, kw)))
    monkeypatch.setattr(ar, "fastlaunch", lambda *args, **kw: None)
    for wire in (torch.float32, torch.bfloat16):
        out = torch.empty(8, dtype=torch.float32)
        assert ar._tree_reduce(torch.empty(4, 8, dtype=wire), out) is out
        assert calls[-1] == (4, {"in_bf16": wire == torch.bfloat16})
