"""Node-shared host-resident buffers: one segment per host created by its lowest rank, mapped read-only by the
others, unlinked once mapped, identity-checked (fail closed), loaded by the leader alone. Ranks run as threads
with an in-process gather; CUDA registration is GPU-gated (the engine gates)."""
import os
import threading

import pytest
import torch
from torch import nn

from isoexec.runtimes.vllm import host_tables
from isoexec.runtimes.vllm.host_tables import NodeSharedHostBuffers


class Exchange:
    """``all_gather_object`` of ``world`` thread-ranks."""

    def __init__(self, world):
        self.world, self.barrier, self.slots = world, threading.Barrier(world), [None] * world

    def gather(self, rank, value):
        self.barrier.wait()
        self.slots[rank] = value
        self.barrier.wait()
        out = list(self.slots)
        self.barrier.wait()
        return out


def ranked(exchange, rank):
    class Ranked(NodeSharedHostBuffers):
        def _rank(self):
            return rank

        def _gather(self, value):
            return exchange.gather(rank, value)
    return Ranked


class Table(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.empty((3, 4096 + 5), dtype=torch.float8_e4m3fn, device="meta"))
        self.register_buffer("scale", torch.empty((3, 2), dtype=torch.float8_e8m0fnu, device="meta"))


def taken():
    t = Table()
    return t, [(t, "weight", t.weight), (t, "scale", t.scale)]


def run(world, fn):
    exchange, results, errors = Exchange(world), [None] * world, [None] * world

    def body(rank):
        try:
            results[rank] = fn(rank, ranked(exchange, rank))
        except BaseException as exc:          # surfaced by the test (every failure here is collective)
            errors[rank] = exc
    threads = [threading.Thread(target=body, args=(r,), daemon=True) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not any(t.is_alive() for t in threads), "a rank left the collective"
    return results, errors


@pytest.fixture
def shm(tmp_path, monkeypatch):
    monkeypatch.setattr(host_tables, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(host_tables, "SHM_MARGIN", 0)
    return tmp_path


def test_one_copy_per_node_loaded_by_the_leader_and_mapped_read_only(shm, tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    for i in range(200):                   # a sharded checkpoint: its fingerprint outgrows the header page
        (ckpt / f"model-{i:05d}-of-00200.safetensors").write_text("x")

    def rank_fn(rank, cls):
        table, items = taken()
        shared = cls(items, checkpoint=str(ckpt), stage="pp0/1-tp2", read_only_register=True)
        assert [p for p in os.listdir(shm) if p.startswith(host_tables.PREFIX)] == []   # unlinked once mapped
        skip = shared.skip_names(lambda module, name: f"t.{name}")
        if shared.leader:
            table.weight.view(torch.uint8).copy_(torch.arange(table.weight.numel(), dtype=torch.int64)
                                                 .remainder(251).to(torch.uint8).view(table.weight.shape))
            table.scale.view(torch.uint8).fill_(127)
        shared.finish("cpu")
        assert table.weight.data_ptr() % 4096 == 0 and table.scale.data_ptr() % 4096 == 0
        return shared.leader, skip, bytes(table.weight.view(torch.uint8).flatten()[:300].tolist()), \
            int(table.scale.view(torch.uint8).sum()), shared.writable, shared.skip_names(lambda m, n: n)

    results, errors = run(2, rank_fn)
    assert errors == [None, None]
    (lead0, skip0, w0, s0, wr0, after0), (lead1, skip1, w1, s1, wr1, after1) = results
    assert (lead0, lead1) == (True, False)
    assert skip0 == set() and skip1 == {"t.weight", "t.scale"}       # only the leader loads the bytes
    assert w0 == w1 and w0[:3] == bytes([0, 1, 2]) and s0 == s1 == 127 * 6
    assert not wr0 and not wr1                                       # host pages read-only on every rank
    assert after0 == after1 == {"weight", "scale"}                   # a later load writes nothing


def test_a_rank_that_maps_another_identity_fails_closed(shm, monkeypatch):
    original = NodeSharedHostBuffers._create

    def forged(self):
        path = original(self)
        head = host_tables._header({**self.identity, "stage": "another"}, host_tables.FILLING)
        self.mapping[:len(head)] = head
        return path
    monkeypatch.setattr(NodeSharedHostBuffers, "_create", forged)
    results, errors = run(2, lambda rank, cls: cls(taken()[1], checkpoint=None, stage="s"))
    assert all(isinstance(e, RuntimeError) and "another identity" in str(e) for e in errors)
    assert [p for p in os.listdir(shm) if p.startswith(host_tables.PREFIX)] == []


def test_ranks_disagreeing_on_the_checkpoint_refuse(shm):
    results, errors = run(2, lambda rank, cls: cls(taken()[1], checkpoint=f"/nowhere/{rank}", stage="s"))
    assert all(isinstance(e, RuntimeError) and "disagree" in str(e) for e in errors)


def test_stale_segments_of_dead_creators_are_removed(shm):
    digest = "d" * 24
    dead, alive = f"{host_tables.PREFIX}{digest}-999999999", f"{host_tables.PREFIX}{digest}-{os.getpid()}"
    other = f"{host_tables.PREFIX}{'e' * 24}-999999999"
    for name in (dead, alive, other):
        (shm / name).write_bytes(b"x")
    assert host_tables.remove_stale(digest) == [dead]
    assert sorted(os.listdir(shm)) == sorted([alive, other])


def test_finish_requires_a_ready_segment_and_runs_once(shm):
    table, items = taken()
    shared = NodeSharedHostBuffers(items, checkpoint=None, stage="s")
    assert shared.leader and shared.skip_names(lambda m, n: n) == set()
    assert shared.finish("cpu") == 2 and shared.ready
    with pytest.raises(RuntimeError, match="already finished"):
        shared.finish("cpu")
    shared.release()                                                 # no registration on host-only runs


def test_a_failed_create_leaves_no_segment(shm, monkeypatch):
    def broken(identity, state):
        raise ValueError("header")
    monkeypatch.setattr(host_tables, "_header", broken)
    with pytest.raises(RuntimeError, match="header"):
        NodeSharedHostBuffers(taken()[1], checkpoint=None, stage="s")
    assert [p for p in os.listdir(shm) if p.startswith(host_tables.PREFIX)] == []


def test_registration_is_chunked_into_one_contiguous_device_range(monkeypatch):
    """No single ``cudaHostRegister`` call spans more than ``REGISTER_CHUNK`` bytes (one ~200 GB call stalls
    every other driver client); the pieces must form one device range, and release unregisters each piece."""
    import sys
    import types
    calls, unregistered = [], []
    ok = types.SimpleNamespace(cudaSuccess=0)
    rt = types.SimpleNamespace(
        cudaHostRegisterMapped=1, cudaHostRegisterPortable=2, cudaHostRegisterReadOnly=4, cudaError_t=ok,
        cudaHostRegister=lambda base, n, flags: (calls.append((base, n, flags)), (0,))[1],
        cudaHostGetDevicePointer=lambda base, flags: (0, base),
        cudaHostUnregister=lambda base: unregistered.append(base))
    bindings = types.ModuleType("cuda.bindings")
    bindings.runtime = rt
    monkeypatch.setitem(sys.modules, "cuda", types.ModuleType("cuda"))
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", rt)
    monkeypatch.setattr(host_tables, "REGISTER_CHUNK", 1 << 20)
    monkeypatch.setattr(torch, "as_tensor", lambda *a, **k: torch.zeros(0, dtype=torch.uint8))
    class Fake(NodeSharedHostBuffers):
        base = 1 << 40
    buf = Fake.__new__(Fake)
    buf.total, buf.writable, buf.taken, buf.entries = (5 << 20) + 3, False, [], []
    buf._register("cpu")
    assert [(b - buf.base, n) for b, n, _ in calls] == [(i << 20, 1 << 20) for i in range(5)] + [(5 << 20, 3)]
    assert all(flags == 7 for _, _, flags in calls) and buf.device_base == buf.base
    buf.release()
    assert unregistered == [b for b, _, _ in calls] and buf.registered is None
    # pieces that do not map to one contiguous device range are refused (and unregistered)
    calls.clear(), unregistered.clear()
    rt.cudaHostGetDevicePointer = lambda base, flags: (0, base + (1 << 12 if base > buf.base else 0))
    with pytest.raises(RuntimeError, match="contiguous"):
        buf._register("cpu")
    assert unregistered == [b for b, _, _ in calls][:2] and buf.registered is None
