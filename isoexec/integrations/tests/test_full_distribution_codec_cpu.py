"""Full-row codec tests independent of the training and serving frameworks."""

import base64
from dataclasses import dataclass

import numpy as np
import pytest

from isoexec.integrations import full_distribution as wire
from isoexec.integrations.full_distribution import decode_packed_full_logprobs, pack_full_logprobs


@dataclass
class _Logprob:
    logprob: float


def test_packed_full_logprobs_round_trip_float32():
    rows = [
        {token_id: _Logprob(-0.25 * (token_id + 1)) for token_id in range(3)},
        {token_id: _Logprob(-1.5 - token_id) for token_id in range(3)},
    ]

    payload = pack_full_logprobs([1, 2], rows, vocab_size=3)
    decoded = decode_packed_full_logprobs(payload)

    assert payload["dtype"] == "float32"
    assert decoded.dtype == np.dtype("<f4")
    assert decoded.flags.c_contiguous
    np.testing.assert_array_equal(
        decoded,
        np.array([[-0.25, -0.5, -0.75], [-1.5, -2.5, -3.5]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"data": "", "shape": [0, 3]},
        {"data": "", "shape": [0, 3], "dtype": "float64"},
        {"data": "!", "shape": [1, 1], "dtype": "float32"},
        {"data": "AAAAAAAA", "shape": [1, 1], "dtype": "float32"},
        {"data": "AAAAAA==", "shape": [True, 1], "dtype": "float32"},
        {"data": "AAAAAA==", "shape": [1, 0], "dtype": "float32"},
    ],
)
def test_decode_rejects_malformed_full_logprobs(payload):
    with pytest.raises(ValueError):
        decode_packed_full_logprobs(payload)


def test_pack_rejects_incomplete_or_nonfinite_full_logprobs():
    with pytest.raises(ValueError, match="every vocabulary token"):
        pack_full_logprobs([1], [{0: _Logprob(-1.0), 1: _Logprob(-2.0)}], vocab_size=3)

    row = {0: _Logprob(-1.0), 1: _Logprob(float("nan")), 2: _Logprob(-3.0)}
    with pytest.raises(ValueError, match="finite"):
        pack_full_logprobs([1], [row], vocab_size=3)


@pytest.mark.parametrize("stdlib", [False, True])
def test_codec_preserves_bits_and_returns_writable_rows(monkeypatch, stdlib):
    if stdlib:
        monkeypatch.setattr(wire, "_base64", base64)
    original = np.array([0x80000000, 0, 1, 0xBF800001], dtype=np.uint32).view(np.float32)
    rows = [{index: _Logprob(float(value)) for index, value in enumerate(original)}]
    payload = pack_full_logprobs([1], rows, vocab_size=len(original))
    decoded = decode_packed_full_logprobs(payload)
    np.testing.assert_array_equal(decoded.view(np.uint32)[0], original.view(np.uint32))
    assert decoded.flags.writeable
    decoded.view(np.uint32)[0, 2] ^= np.uint32(1)
    assert decoded.view(np.uint32)[0, 2] == 0


@pytest.mark.parametrize("stdlib", [False, True])
def test_empty_response_rows_keep_the_declared_vocabulary(monkeypatch, stdlib):
    if stdlib:
        monkeypatch.setattr(wire, "_base64", base64)
    payload = pack_full_logprobs([], [], vocab_size=3)
    assert payload == {"data": "", "shape": [0, 3], "dtype": "float32"}
    assert decode_packed_full_logprobs(payload).shape == (0, 3)


def _engine_row(values, sampled):
    """The engine's layout: the sampled token first, then every token in descending logprob order."""
    order = np.argsort(-values, kind="stable")
    token_ids = np.concatenate(([sampled], order)).astype(np.int32)
    return token_ids, np.concatenate((values[sampled : sampled + 1], values[order])).astype(np.float32)


def test_compact_rows_pack_to_the_same_bytes_as_per_token_objects():
    rng = np.random.default_rng(7)
    values = rng.standard_normal((3, 11)).astype(np.float32)
    # Negative zero, a denormal and a one-ULP neighbour must survive the reorder untouched.
    values[1, :4] = np.array([0x80000000, 1, 0xBF800001, 0xBF800000], dtype=np.uint32).view(np.float32)
    sampled = [4, 0, 10]
    compact = [wire.full_vocabulary_row(*_engine_row(row, tid)) for row, tid in zip(values, sampled)]
    objects = [{tid: _Logprob(float(value)) for tid, value in enumerate(row)} for row in values]

    payload = pack_full_logprobs(sampled, compact, vocab_size=11)

    assert payload == pack_full_logprobs(sampled, objects, vocab_size=11)
    np.testing.assert_array_equal(decode_packed_full_logprobs(payload).view(np.uint32), values.view(np.uint32))
    # Rows of both kinds may meet in one response.
    assert pack_full_logprobs(sampled, [compact[0], objects[1], compact[2]], vocab_size=11) == payload


def test_compact_row_serves_the_mapping_reads_the_wire_builders_use():
    values = np.array([-3.0, -0.5, -2.0], dtype=np.float32)
    row = wire.full_vocabulary_row(*_engine_row(values, 2))

    # Like the engine's own container, iteration starts at the sampled token.
    assert row and len(row) == 3 and list(row) == [2, 0, 1]
    assert row.sampled_token_id == 2 and row.sampled_logprob == -2.0
    assert row.get(1).logprob == -0.5 and row[np.int64(0)].logprob == -3.0
    assert row[2].rank == 2 and row[1].rank is None
    assert [entry.logprob for entry in row.values()] == [-2.0, -3.0, -0.5]
    assert row.get(3) is None and 3 not in row and True not in row
    assert row.__contains__(np.int64(0)) is True
    with pytest.raises(KeyError):
        row[-1]


def test_compact_row_reports_the_sampled_rank_the_engine_container_ends_with():
    # All entries tie: the count-greater rank of every token is 1, but the engine's container keeps
    # the sampled token's position in rank order, here the second.
    tied = np.full(4, -1.5, dtype=np.float32)
    row = wire.full_vocabulary_row(np.array([1, 0, 1, 2, 3], np.int32), np.concatenate((tied[:1], tied)))
    assert row[1].rank == 2 and row.sampled_logprob == -1.5


@pytest.mark.parametrize(
    ("token_ids", "logprobs", "reason"),
    [
        (np.array([0, 1, 1, 2], np.int32), np.array([-1, -2, -2, -3], np.float32), "exactly once"),
        (np.array([0, 0, 1, 3], np.int32), np.array([-1, -1, -2, -3], np.float32), "outside its vocabulary"),
        (np.array([3, 0, 1, 2], np.int32), np.array([-1, -1, -2, -3], np.float32), "samples a token outside"),
        (np.array([0, 0, 1, 2], np.int32), np.array([-9, -1, -2, -3], np.float32), "bits differ"),
        (np.array([0.0, 0.0, 1.0, 2.0]), np.array([-1, -1, -2, -3], np.float32), "integer token IDs"),
        (np.array([0, 0, 1], np.int32), np.array([-1, -1, -2, -3], np.float32), "one entry per vocabulary token"),
    ],
)
def test_compact_row_refuses_anything_but_a_complete_sampled_first_row(token_ids, logprobs, reason):
    with pytest.raises(ValueError, match=reason):
        wire.full_vocabulary_row(token_ids, logprobs)


def test_pack_still_refuses_a_compact_row_of_the_wrong_vocabulary_or_a_nonfinite_one():
    row = wire.full_vocabulary_row(*_engine_row(np.array([-1.0, -2.0, -3.0], np.float32), 0))
    with pytest.raises(ValueError, match="every vocabulary token"):
        pack_full_logprobs([0], [row], vocab_size=4)
    bad = wire.full_vocabulary_row(*_engine_row(np.array([-1.0, -np.inf, -3.0], np.float32), 0))
    with pytest.raises(ValueError, match="finite"):
        pack_full_logprobs([0], [bad], vocab_size=3)
