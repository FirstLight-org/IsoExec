"""CPU contracts of CPR prefix caching (``runtimes/vllm/cpr_apc.py``): the boundary store's retirement, the
workers' shared index, the scheduler's admission clamp, and a worker step's capture / adoption bookkeeping.

The GPU batteries (``cpr_apc_engine_gpu.py``) prove the bits; these pin the rules that make a hit resume only
a checkpoint of exactly its prefix, and recompute otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isoexec.ops.gdn.shared import gdn_cpr_state as cs
from isoexec.runtimes.vllm import cpr_apc

# A small record layout standing in for the extension's (csrc/state.cuh): boundary bytes [0, 24), pos at 40.
LAYOUT = {"sh": 0, "lo": 8, "tie": 16, "kb": 24, "vb": 28, "db": 30, "xb": 32, "pb": 34, "gb": 36, "bb": 38,
          "pos": 40, "bytes": 72, "format": 3}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    from isoexec.ops.gdn.gdn_cpr_sm90 import ops

    monkeypatch.setattr(cpr_apc, "_DIR", str(tmp_path))
    monkeypatch.setattr(ops, "record_layout", lambda: LAYOUT)
    for hooks in (cs.CPR_APC_ON_STORE, cs.CPR_APC_ON_RETRACT, cs.CPR_APC_ON_CLEAR):
        hooks.clear()
    monkeypatch.setattr(cs, "CPR_APC_STORE", None)
    cpr_apc.REQUEST_TOKENS.clear()
    yield
    for hooks in (cs.CPR_APC_ON_STORE, cs.CPR_APC_ON_RETRACT, cs.CPR_APC_ON_CLEAR):
        hooks.clear()
    cpr_apc.REQUEST_TOKENS.clear()


def _ckpt(value: int, nbytes: int = 1000):
    return torch.full((nbytes - 8,), value, dtype=torch.uint8), torch.zeros(4, dtype=torch.bfloat16)


# ------------------------------------------------------------------------------------------- the store


def test_key_names_exactly_the_prefix():
    toks = list(range(100, 300))
    k = cs.cpr_apc_key(toks, 128)
    assert k == cs.cpr_apc_key(np.asarray(toks), 128) == cs.cpr_apc_key(toks[:128] + [7, 8], 128)
    assert k != cs.cpr_apc_key(toks, 192)
    assert k != cs.cpr_apc_key([toks[0] + 1] + toks[1:], 128)
    assert len(k) == 16
    with pytest.raises(ValueError):
        cs.cpr_apc_key(toks[:64], 128)


def test_evicted_checkpoints_leave_the_index_at_once_and_stay_readable_through_their_grace():
    retracted, stored = [], []
    cs.CPR_APC_ON_RETRACT.append(retracted.append)
    cs.CPR_APC_ON_STORE.append(stored.append)
    store = cs.CprBoundaryStore(2500, evict=True, retire_steps=3)
    assert store.put(b"a", 64, *_ckpt(1)) and store.put(b"b", 64, *_ckpt(2))
    assert store.put(b"c", 64, *_ckpt(3))            # evicts the LRU checkpoint "a"
    assert stored == [b"a", b"b", b"c"] and retracted == [b"a"]
    assert b"a" not in store and store.nbytes == 3000  # two live within the budget, "a" retiring on top
    for _ in range(2):
        store.tick()
        got = store.get(b"a")                        # an admission from just before the retraction still resumes
        assert got is not None and int(got[1][0]) == 1
    store.tick()
    assert store.get(b"a") is None and store.nbytes == 2000
    assert not store.put(b"d", 64, *_ckpt(4, nbytes=4000))  # larger than the whole budget


def test_a_new_epoch_retires_everything_and_a_recapture_never_resurrects_a_retired_twin():
    retracted = []
    cs.CPR_APC_ON_RETRACT.append(retracted.append)
    store = cs.CprBoundaryStore(10_000, evict=True, retire_steps=2)
    store.put(b"a", 64, *_ckpt(1))
    store.retire_all()
    assert retracted == [b"a"] and len(store) == 0 and store.get(b"a") is not None
    assert store.put(b"a", 64, *_ckpt(9))           # recaptured at the new epoch
    assert int(store.get(b"a")[1][0]) == 9 and store.nbytes == 1000
    cleared = []
    cs.CPR_APC_ON_CLEAR.append(lambda: cleared.append(True))
    store.clear()
    assert cleared and store.get(b"a") is None and store.nbytes == 0


def test_append_only_store_refuses_when_full():
    store = cs.CprBoundaryStore(1500, evict=False)
    assert store.put(b"a", 64, *_ckpt(1))
    assert not store.put(b"b", 64, *_ckpt(2)) and store.full_refusals == 1 and b"a" in store


# ------------------------------------------------------------------------------------------- the index


def test_index_admits_only_keys_every_rank_holds_at_the_current_epoch():
    reader = cpr_apc.IndexReader("t", ranks=2)
    assert not reader.holds(b"k" * 16, 1)            # no worker published yet
    r0, r1 = cpr_apc.RankIndex("t", 0, slots=8), cpr_apc.RankIndex("t", 1, slots=8)
    r0.reset(1)
    r1.reset(1)
    key = b"k" * 16
    r0.add(key)
    assert not reader.holds(key, 1)                  # rank 1 lacks it
    r1.add(key)
    assert reader.holds(key, 1)
    assert not reader.holds(key, 2)                  # a worker still at an older epoch admits nothing
    r1.remove(key)
    assert not reader.holds(key, 1)
    r1.add(key)
    r0.reset(2)
    assert not reader.holds(key, 1) and not reader.holds(key, 2)
    for i in range(8):
        assert r0.add(bytes([i + 1]) * 16)
    assert r0.full() and not r0.add(b"z" * 16)


def test_epoch_advances_monotonically():
    a, b = cpr_apc.Epoch("t", create=True), cpr_apc.Epoch("t", create=True)
    assert a.read() == 0 and a.advance() == 1 and b.read() == 1 and b.advance() == 2 and a.read() == 2


def test_engines_started_from_the_same_engine_args_get_separate_indexes():
    # Each engine's VllmConfig gets its own instance_id; no tag rides the engine args.
    first, second = (SimpleNamespace(instance_id=i, parallel_config=SimpleNamespace(data_parallel_rank=0))
                     for i in ("1763769600000000001", "1763769600000000002"))
    assert cpr_apc.engine_tag(first) != cpr_apc.engine_tag(second)


# ----------------------------------------------------------------------------------------- the scheduler


class _Request:
    def __init__(self, tokens, **kw):
        self.all_token_ids = tokens
        self.prompt_embeds = kw.get("prompt_embeds")
        self.mm_features = kw.get("mm_features")
        self.lora_request = kw.get("lora_request")


def _admission(monkeypatch, ranks=1):
    config = SimpleNamespace(instance_id="t", parallel_config=SimpleNamespace(
        tensor_parallel_size=ranks, pipeline_parallel_size=1, data_parallel_rank=0))
    adm = cpr_apc.SchedulerAdmission(config, cpr_apc.engine_tag(config))
    indexes = [cpr_apc.RankIndex(adm.tag, r, slots=64) for r in range(ranks)]
    for ix in indexes:
        ix.reset(adm.epoch.read())
    return adm, indexes


def test_scheduler_keeps_a_hit_only_where_every_worker_holds_its_checkpoint(monkeypatch):
    adm, (ix,) = _admission(monkeypatch)
    toks = list(range(1000, 1700))
    ix.add(cs.cpr_apc_key(toks, 256))
    # vLLM's hits for this request, longest first; ``original`` returns the longest hit <= its bound.
    hits = [640, 512, 448, 256, 128]

    def original(manager, request, bound):
        h = next((h for h in hits if h <= bound), 0)
        return (f"blocks{h}", h, 0)

    req = _Request(toks)
    assert adm.filter(None, req, None, original(None, req, 699), original) == ("blocks256", 256, 0)
    ix.add(cs.cpr_apc_key(toks, 640))
    assert adm.filter(None, req, None, original(None, req, 699), original)[1] == 640
    # A hit off the 64 grid is never a checkpoint position.
    hits[:] = [600]
    assert adm.filter(None, req, None, original(None, req, 699), original)[1] == 0
    # Prompts not identified by their token ids alone admit nothing.
    hits[:] = [640]
    for kw in ({"prompt_embeds": object()}, {"mm_features": [object()]}, {"lora_request": object()}):
        assert adm.filter(None, _Request(toks, **kw), None, original(None, req, 699), original)[1] == 0
    # A reset advances the epoch: nothing a worker advertised before it is admitted.
    adm.advance_epoch()
    assert adm.filter(None, req, None, original(None, req, 699), original)[1] == 0


def test_scheduler_needs_every_tp_rank(monkeypatch):
    adm, (ix0, ix1) = _admission(monkeypatch, ranks=2)
    toks = list(range(300))
    key = cs.cpr_apc_key(toks, 128)
    ix0.add(key)
    req = _Request(toks)
    original = lambda manager, request, bound: ("b", 128 if bound >= 128 else 0, 0)
    assert adm.filter(None, req, None, ("b", 128, 0), original)[1] == 0
    ix1.add(key)
    assert adm.filter(None, req, None, ("b", 128, 0), original)[1] == 128


# --------------------------------------------------------------------------------------------- a worker


class _Pool:
    """The slice of ``CprSm90GDN`` a worker step reads: record bytes, conv rows, the slot map."""

    def __init__(self, rows=6, hv=2):
        from isoexec.ops.gdn.gdn_cpr_sm90 import ops

        self.ssm_state = torch.zeros(rows, hv, LAYOUT["bytes"], dtype=torch.uint8)
        self.records = ops.CprState(self.ssm_state)
        self.conv_state = torch.zeros(rows, 3, 2, dtype=torch.bfloat16)
        self.num_v_heads = hv
        self._row_pos: dict[int, int] = {}
        self._slot2row: dict[int, int] = {}
        self._free = list(range(rows - 1, 0, -1))

    chunk_size = 64

    @staticmethod
    def capture_boundaries(*args):
        from isoexec.ops.gdn.gdn_cpr_sm90.engine_state import capture_boundaries

        return capture_boundaries(*args)

    @staticmethod
    def install_boundaries(*args):
        from isoexec.ops.gdn.gdn_cpr_sm90.engine_state import install_boundaries

        return install_boundaries(*args)

    def _assign_many(self, slots, publish=True):
        rows = []
        for s in slots:
            r = self._slot2row.get(int(s))
            if r is None:
                r = self._free.pop()
                self._slot2row[int(s)] = r
            rows.append(r)
        return rows


def _worker(grid=64):
    w = object.__new__(cpr_apc.WorkerApc)
    w.rank, w.grid, w.epoch, w._last_report = 0, grid, 0, 0.0
    w.index = cpr_apc.RankIndex("w-dp0", 0, slots=64)
    w.epoch_file = cpr_apc.Epoch("w-dp0", create=True)
    w.epoch_file.advance()
    w.store = cs.CprBoundaryStore(1 << 20, evict=True, retire_steps=4)
    w.tokens = cpr_apc.REQUEST_TOKENS
    cs.CPR_APC_ON_STORE.append(w.index.add)
    cs.CPR_APC_ON_RETRACT.append(w.index.remove)
    return w


def _new(rid, toks):
    return SimpleNamespace(req_id=rid, prefill_token_ids=toks, mm_features=None, lora_request=None, prompt_embeds=None)


def test_worker_captures_at_chunk_ends_and_an_admitted_hit_adopts_the_same_state(monkeypatch):
    w = _worker()
    reader = cpr_apc.IndexReader("w-dp0", 1)
    layers = [_Pool(), _Pool()]
    toks = list(range(5, 400))
    cpr_apc.track_requests(SimpleNamespace(scheduled_new_reqs=[_new("a", toks)], finished_req_ids=(),
                                           preempted_req_ids=()))
    for j, ly in enumerate(layers):  # request "a" holds row 1 after a chunk that ended at 128
        ly._slot2row[11] = ly._free.pop()
        ly.ssm_state[1] = torch.arange(LAYOUT["bytes"], dtype=torch.uint8).view(1, -1) + j
        ly.conv_state[1] = 3.0 + j
    w.step(["a"], np.array([128]), np.array([11]), layers)
    key = cs.cpr_apc_key(toks, 128)
    assert key in w.store and reader.holds(key, w.epoch_file.read())
    w.step(["a"], np.array([130]), np.array([11]), layers)   # mid-chunk: nothing to capture
    assert len(w.store) == 1
    # Request "b" shares the first 128 tokens and the scheduler admitted that hit: it arrives with 128
    # computed tokens and no row; the step installs the checkpoint on a fresh row in every pool.
    other = toks[:128] + [9] * 50
    cpr_apc.track_requests(SimpleNamespace(scheduled_new_reqs=[_new("b", other)], finished_req_ids=(),
                                           preempted_req_ids=()))
    w.step(["a", "b"], np.array([131, 128]), np.array([11, 22]), layers)
    for j, ly in enumerate(layers):
        rb = ly._slot2row[22]
        assert torch.equal(ly.ssm_state[rb, :, :LAYOUT["kb"]], ly.ssm_state[1, :, :LAYOUT["kb"]])
        assert torch.equal(ly.conv_state[rb], ly.conv_state[1])
        assert ly.records.pos[rb].eq(128).all() and ly._row_pos[rb] == 128


def test_worker_refuses_a_hit_it_holds_no_checkpoint_for(monkeypatch):
    w = _worker()
    layers = [_Pool()]
    cpr_apc.track_requests(SimpleNamespace(scheduled_new_reqs=[_new("b", list(range(300)))], finished_req_ids=(),
                                           preempted_req_ids=()))
    with pytest.raises(RuntimeError, match="holds no checkpoint"):
        w.step(["b"], np.array([128]), np.array([5]), layers)
    with pytest.raises(RuntimeError, match="no prefill tokens"):
        w.step(["c"], np.array([64]), np.array([6]), layers)


def test_worker_retires_its_store_when_the_scheduler_resets_the_prefix_cache(monkeypatch):
    w = _worker()
    reader = cpr_apc.IndexReader("w-dp0", 1)
    layers = [_Pool()]
    toks = list(range(200))
    cpr_apc.track_requests(SimpleNamespace(scheduled_new_reqs=[_new("a", toks)], finished_req_ids=(),
                                           preempted_req_ids=()))
    layers[0]._slot2row[11] = layers[0]._free.pop()
    w.step(["a"], np.array([64]), np.array([11]), layers)
    key = cs.cpr_apc_key(toks, 64)
    assert reader.holds(key, w.epoch_file.read())
    new_epoch = w.epoch_file.advance()                # Scheduler.reset_prefix_cache
    assert not reader.holds(key, new_epoch)
    w.step(["a"], np.array([65]), np.array([11]), layers)
    assert key not in w.store and w.store.get(key) is not None  # retired: readable through its grace only
    assert not reader.holds(key, new_epoch)
    cpr_apc.track_requests(SimpleNamespace(scheduled_new_reqs=(), finished_req_ids=("a",), preempted_req_ids=()))
    assert "a" not in cpr_apc.REQUEST_TOKENS
