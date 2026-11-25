"""Shared protocol tests for packed_meta_cache and sequence_metadata.
Keys combine tensor identity and version with a strong reference; writes through aliases must invalidate entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest

torch = pytest.importorskip("torch")


@dataclass(frozen=True)
class Memo:
    """One memo, described by what this protocol needs to poke at."""

    name: str
    #: The module holding the cache; reset between tests.
    module: object
    cache_attr: str
    cache_max_attr: str
    env_var: str
    #: ``cu -> the memoized value``. Called twice on one tensor, it must hand back the same object.
    read: Callable
    #: ``cu -> the value the expression the memo replaces would give``.
    rebuild: Callable
    #: ``(a, b) -> None``; raises if the two values differ.
    same: Callable
    #: ``() -> {"served": n, "built": n, "declined": n}``
    census: Callable
    #: Zero those counters. They are module globals, so a test that did not reset them would read
    #: the previous test's arithmetic.
    reset_census: Callable
    #: ``() -> bool``: is the memo on, per the environment?
    enabled: Callable
    #: A cu vector too large for this memo to hold, which it must still ANSWER correctly.
    oversized: Callable


def _cu(lens, dtype=torch.int32):
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=dtype)


def _packed() -> Memo:
    from isoexec.ops.gdn.shared import packed_meta_cache as pmc

    def same(a, b):
        assert a == b

    return Memo(
        name="packed_meta_cache",
        module=pmc,
        cache_attr="_CACHE",
        cache_max_attr="_CACHE_MAX",
        env_var="ISOEXEC_PACKED_META_CACHE",
        read=pmc.cu_list,
        rebuild=lambda cu: cu.tolist(),
        same=same,
        census=pmc.packed_meta_census,
        reset_census=lambda: pmc._STATS.update(served=0, built=0, declined=0),
        enabled=pmc.packed_meta_cache_enabled,
        oversized=lambda: torch.arange(0, pmc._MAX_NUMEL + 2, dtype=torch.int32),
    )


def _sequence() -> Memo:
    from isoexec.ops.gdn.shared import gdn_cpr as gcs

    CHUNK = 64

    def rebuild(cu):
        lens = gcs.packed_lens(cu)
        cu_chunked, state_indices, n_chunks = gcs.chunked_cu_seqlens_and_indices(cu, CHUNK, cu.device, lens=lens)
        return lens, cu.clone(), cu_chunked, state_indices, n_chunks

    def same(a, b):
        assert a[0] == b[0]
        assert torch.equal(a[1], b[1])
        assert torch.equal(a[2], b[2])
        assert torch.equal(a[3], b[3])
        assert a[4] == b[4]

    return Memo(
        name="sequence_metadata",
        module=gcs,
        cache_attr="_SEQ_CACHE",
        cache_max_attr="_SEQ_CACHE_MAX",
        env_var="ISOEXEC_GDN_SEQ_META_CACHE",
        read=lambda cu: gcs.sequence_metadata(cu, CHUNK, cu.device),
        rebuild=rebuild,
        same=same,
        census=gcs.seq_meta_census,
        reset_census=lambda: gcs._SEQ_STATS.update({k: 0 for k in gcs._SEQ_STATS}),
        enabled=gcs.seq_meta_cache_enabled,
        # numel > 8192 is not cacheable
        oversized=lambda: torch.arange(0, 9000, dtype=torch.int32),
    )


MEMOS = {"packed_meta_cache": _packed, "sequence_metadata": _sequence}


@pytest.fixture(params=sorted(MEMOS), ids=sorted(MEMOS))
def memo(request, monkeypatch):
    built = MEMOS[request.param]()
    monkeypatch.delenv(built.env_var, raising=False)
    cache = getattr(built.module, built.cache_attr)
    cache.clear()
    built.reset_census()
    yield built
    cache.clear()
    built.reset_census()


def test_an_in_place_write_invalidates(memo):
    cu = _cu([128, 128])
    first = memo.read(cu)
    cu[1] = 64  # an in-place write bumps the version counter
    second = memo.read(cu)
    assert second is not first
    memo.same(second, memo.rebuild(cu))


def test_a_write_through_an_alias_invalidates(memo):
    """torch's version counter is shared across views and bases, so a write through EITHER side
    has to invalidate an entry keyed on the other."""
    base = _cu([64, 64, 64])
    view = base[:]
    first = memo.read(base)
    view[2] = 99
    assert memo.read(base) is not first
    memo.same(memo.read(base), memo.rebuild(base))


def test_a_distinct_tensor_with_equal_values_is_never_served_the_other_entry(memo):
    """The key must be identity, never value: two tensors at version 0 are not each other."""
    a, b = _cu([256, 256]), _cu([256, 256])
    assert torch.equal(a, b)
    first, second = memo.read(a), memo.read(b)
    assert first is not second or not torch.is_tensor(first)
    memo.same(second, memo.rebuild(b))


def test_the_entry_pins_the_tensor_it_is_keyed_on(memo):
    """A STRONG reference, so CPython cannot hand the ``id`` half of the key to another tensor."""
    cu = _cu([512])
    memo.read(cu)
    cache = getattr(memo.module, memo.cache_attr)
    (key, entry), = [(k, v) for k, v in cache.items() if v[0] is cu] or [(None, None)]
    assert key is not None, "the memo stored no entry pinning this tensor"
    assert key[0] == id(cu)
    assert entry[0] is cu


def test_the_cache_is_bounded(memo):
    limit = getattr(memo.module, memo.cache_max_attr)
    for i in range(limit * 3):
        memo.read(_cu([64, 64 + i]))
    assert len(getattr(memo.module, memo.cache_attr)) <= limit


def test_an_oversized_tensor_is_declined_but_still_answered_correctly(memo):
    """A decline is a CORRECT answer computed the slow way, not a refusal and not a wrong one."""
    cu = memo.oversized()
    memo.same(memo.read(cu), memo.rebuild(cu))
    assert memo.census()["declined"] > 0
    assert memo.census()["built"] == 0 and memo.census()["served"] == 0


def test_the_flag_off_restores_the_per_call_work_and_the_values_are_unchanged(memo, monkeypatch):
    monkeypatch.setenv(memo.env_var, "0")
    assert not memo.enabled()
    cu = _cu([1024, 1024])
    a, b = memo.read(cu), memo.read(cu)
    memo.same(a, memo.rebuild(cu))
    memo.same(b, memo.rebuild(cu))
    if torch.is_tensor(a) or isinstance(a, tuple):
        assert a is not b, "flag off must restore the fresh work per call"
    census = memo.census()
    assert census["served"] == 0 and census["built"] == 0 and census["declined"] > 0


@pytest.mark.parametrize(
    "value,expect",
    [("0", False), ("false", False), ("no", False), ("FALSE", False), ("", False), ("1", True)],
)
def test_the_flag_spellings(memo, monkeypatch, value, expect):
    monkeypatch.setenv(memo.env_var, value)
    assert memo.enabled() is expect


def test_the_default_is_on(memo):
    assert memo.enabled()


def test_the_flag_is_numerically_neutral(memo, monkeypatch):
    """On or off, the same tensors: the memo is a host-work change and nothing else."""
    cu = _cu([3000, 1500, 700])
    monkeypatch.setenv(memo.env_var, "0")
    off = memo.read(cu)
    monkeypatch.setenv(memo.env_var, "1")
    memo.same(memo.read(cu), off)
