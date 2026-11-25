"""The trainer's host-resident buffers live in ONE shared host copy per node (``node_shared_host_buffers``), as on
the engine: the stage's lowest rank on the host creates and loads it, the others map it read-only and skip its
names at load, the segment is unlinked once mapped, and release drops each rank's registration. Ranks are
threads behind fake process groups; CUDA registration is GPU-gated."""
import os
import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from isoexec.runtimes.vllm import host_tables


class Exchange:
    def __init__(self, world):
        self.world, self.barrier, self.slots = world, threading.Barrier(world), [None] * world

    def gather(self, rank, value):
        self.barrier.wait()
        self.slots[rank] = value
        self.barrier.wait()
        out = list(self.slots)
        self.barrier.wait()
        return out


class Group:
    def __init__(self, exchange, rank):
        self.exchange, self.r = exchange, rank

    def size(self):
        return self.exchange.world

    def rank(self):
        return self.r


class Table(nn.Module):
    HOST_RESIDENT_BUFFERS = ("weight", "scale")

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.empty((5, 4096 + 3), dtype=torch.float8_e4m3fn, device="meta"))
        self.register_buffer("scale", torch.empty((5, 2), dtype=torch.float8_e8m0fnu, device="meta"))


class Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2, device="meta"), nn.Module()])
        self.blocks[1].embed = Table()


@pytest.fixture
def ranks(tmp_path, monkeypatch):
    monkeypatch.setattr(host_tables, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(host_tables, "SHM_MARGIN", 0)
    monkeypatch.setattr(host_tables, "read_only_registration_supported", lambda device: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: group.r)

    def all_gather_object(out, value, group):
        out[:] = group.exchange.gather(group.r, value)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    return tmp_path


def run(world, fn):
    exchange, results, errors = Exchange(world), [None] * world, [None] * world

    def body(rank):
        try:
            results[rank] = fn(rank, Group(exchange, rank))
        except BaseException as exc:
            errors[rank] = exc
    threads = [threading.Thread(target=body, args=(r,), daemon=True) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not any(t.is_alive() for t in threads)
    return results, errors


def test_trainer_ranks_share_one_host_copy_per_node(ranks, tmp_path):
    from isoexec.compiler.realization.storage import take_host_resident_buffers
    from isoexec.runtimes.megatron.model import node_shared_host_buffers
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model-00001-of-00001.safetensors").write_text("x")

    def rank_fn(rank, group):
        stage = Stage()
        taken = take_host_resident_buffers(stage)
        pg = SimpleNamespace(tp_dp_cp=group, pp=SimpleNamespace(rank=lambda: 0, size=lambda: 1))
        shared, names = node_shared_host_buffers(stage, taken, checkpoint=str(ckpt), pg_collection=pg)
        assert sorted(names.values()) == ["blocks.1.embed.scale", "blocks.1.embed.weight"]
        assert [p for p in os.listdir(ranks) if p.startswith(host_tables.PREFIX)] == []    # unlinked once mapped
        skip = shared.skip_names(lambda module, name: names[id(module), name])
        table = stage.blocks[1].embed
        if shared.leader:                                   # the loader writes the leader's view only
            table.weight.view(torch.uint8).copy_(torch.arange(table.weight.numel()).remainder(253)
                                                 .to(torch.uint8).view(table.weight.shape))
            table.scale.view(torch.uint8).fill_(126)
        shared.finish("cpu")
        assert table.weight.data_ptr() % 4096 == 0
        return (shared.leader, skip, bytes(table.weight.view(torch.uint8).flatten()[:300].tolist()),
                int(table.scale.view(torch.uint8).sum()), shared.writable, shared.identity["stage"])

    results, errors = run(2, rank_fn)
    assert errors == [None, None]
    (lead0, skip0, w0, s0, wr0, st0), (lead1, skip1, w1, s1, wr1, _) = results
    assert (lead0, lead1) == (True, False)
    assert skip0 == set() and skip1 == {"blocks.1.embed.weight", "blocks.1.embed.scale"}
    assert w0 == w1 and w0[:3] == bytes([0, 1, 2]) and s0 == s1 == 126 * 10      # same bytes on every rank
    assert not wr0 and not wr1 and st0 == "trainer-pp0/1"


def test_models_without_host_tables_take_no_collective():
    from isoexec.runtimes.megatron.model import node_shared_host_buffers
    assert node_shared_host_buffers(nn.Linear(2, 2), [], checkpoint=None, pg_collection=None) == (None, {})


def test_release_drops_each_ranks_shared_registration_once(monkeypatch):
    from isoexec.models.deepseek_v4_1_flash import engine_state
    from isoexec.models.deepseek_v4_1_flash.holders import EngramTable
    released = []
    shared = SimpleNamespace(registered=[1, 2], release=lambda: released.append(1))
    tables = [EngramTable(4, 32, "meta"), EngramTable(4, 32, "meta")]
    for t in tables:
        t.__dict__["_host_shared"] = {"weight": shared, "scale": shared}
        t.device_view = ("w", "s")
    rows = [SimpleNamespace(engram=None), *(SimpleNamespace(engram=SimpleNamespace(embed=t)) for t in tables)]
    monkeypatch.setattr(engine_state, "_rows", lambda model: rows)
    assert engine_state.release_host_tables(object()) == 2
    assert released == [1] and all(t.device_view is None for t in tables)
