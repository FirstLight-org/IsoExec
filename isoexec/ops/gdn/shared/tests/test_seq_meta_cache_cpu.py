"""Check sequence-metadata memo values and per-forward reuse.
Shared key and lifetime checks live in test_meta_memo_protocol_cpu.py."""

import pytest

torch = pytest.importorskip("torch")

from isoexec.ops.gdn.shared import (  # noqa: E402
    gdn_cpr as gcs,  # noqa: E402
)

CHUNK = 64


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    gcs._SEQ_CACHE.clear()
    for k in gcs._SEQ_STATS:
        gcs._SEQ_STATS[k] = 0
    monkeypatch.delenv("ISOEXEC_GDN_SEQ_META_CACHE", raising=False)
    yield
    gcs._SEQ_CACHE.clear()


def _cu(lens):
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32)


def _rebuild(cu, chunk_size=CHUNK):
    """Exactly what a layer did before the memo existed."""
    lens = gcs.packed_lens(cu)
    cu_chunked, state_indices, n_chunks = gcs.chunked_cu_seqlens_and_indices(cu, chunk_size, cu.device, lens=lens)
    return lens, cu.clone(), cu_chunked, state_indices, n_chunks


def _assert_same(a, b):
    assert a[0] == b[0]
    assert torch.equal(a[1], b[1])
    assert torch.equal(a[2], b[2])
    assert torch.equal(a[3], b[3])
    assert a[4] == b[4]


# 1. exactness against the pre-memo rebuild
@pytest.mark.parametrize(
    "lens",
    [
        [64],  # one exact chunk
        [63],  # one short chunk
        [65],  # a straddling second chunk
        [128, 64],
        [7648],  # the re-trace's microbatch shape
        [5473, 2175],  # the measured mean sequence, packed in pairs
        [1, 1, 1, 1],  # degenerate short sequences
        [8617, 5304, 4096, 2560],  # the measured max + typical bin
        [],  # empty batch
    ],
)
def test_memo_equals_rebuild(lens):
    cu = _cu(lens)
    want = _rebuild(cu)
    got = gcs.sequence_metadata(cu, CHUNK, cu.device)
    _assert_same(got, want)
    # a second call (a later GDN layer of the same forward) must agree too
    _assert_same(gcs.sequence_metadata(cu, CHUNK, cu.device), want)


def test_hit_returns_the_same_objects():
    """A hit hands back the identical objects, so no device op or host loop re-runs."""
    cu = _cu([4096, 2048])
    first = gcs.sequence_metadata(cu, CHUNK, cu.device)
    second = gcs.sequence_metadata(cu, CHUNK, cu.device)
    assert first is second
    for x, y in zip(first, second):
        if torch.is_tensor(x):
            assert x is y


def test_thirty_layers_build_once():
    """One build per forward, not one per GDN layer."""
    cu = _cu([5473, 2175])
    for _ in range(30):
        gcs.sequence_metadata(cu, CHUNK, cu.device)
    census = gcs.seq_meta_census()
    assert census["built"] == 1
    assert census["served"] == 29
    assert census["declined"] == 0


def test_different_chunk_size_is_a_different_entry():
    cu = _cu([256])
    g64 = gcs.sequence_metadata(cu, 64, cu.device)
    g32 = gcs.sequence_metadata(cu, 32, cu.device)
    assert g64[4] == 4 and g32[4] == 8
    _assert_same(g32, _rebuild(cu, 32))


