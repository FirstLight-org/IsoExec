"""Engine GDN glue removal: the q|k|v window is a view of the in_proj output, and alpha/beta compaction is
deferred to the split launch only for states whose decode does it (CPU, layout logic only)."""

from types import SimpleNamespace

import torch

from isoexec.ops.gdn.shared.gdn_fused_split import defer_ab
from isoexec.runtimes.vllm._gdn import _adjacent_qkv, _reads_strided_qkv

KD, VD, HV = 64, 192, 6
W = 2 * KD + 2 * VD + 2 * HV


def _views(rows=5):
    y = torch.randn(rows, W).to(torch.bfloat16)
    q = y[:, :KD].unflatten(-1, (2, 32))
    k = y[:, KD:2 * KD].unflatten(-1, (2, 32))
    v = y[:, 2 * KD:2 * KD + VD].unflatten(-1, (HV, 32))
    return y, q, k, v


def test_adjacent_window_is_the_cat_without_a_copy():
    for rows in (1, 5):
        y, q, k, v = _views(rows)
        got = _adjacent_qkv(q, k, v)
        assert got is not None and got.data_ptr() == y.data_ptr() and got.stride() == (W if rows > 1 else 2 * KD + VD, 1)
        cat = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
        assert torch.equal(got.view(torch.int16), cat.view(torch.int16))


def test_non_adjacent_operands_keep_the_cat():
    y, q, k, v = _views()
    assert _adjacent_qkv(k, q, v) is None  # wrong order
    assert _adjacent_qkv(q, k, v.clone()) is None  # different storage
    assert _adjacent_qkv(q.contiguous(), k, v) is None


def test_strided_read_only_for_native_recurrent_decode():
    from isoexec.ops.gdn.shared.gdn_recurrent_state import RecurrentGDN

    pool = object.__new__(RecurrentGDN)
    pool._native_kernels = True
    assert _reads_strided_qkv(pool, SimpleNamespace(num_prefills=0))
    assert not _reads_strided_qkv(pool, SimpleNamespace(num_prefills=2))
    pool._native_kernels = False
    assert not _reads_strided_qkv(pool, SimpleNamespace(num_prefills=0))
    assert not _reads_strided_qkv(SimpleNamespace(_native_kernels=True), SimpleNamespace(num_prefills=0))


def test_defer_ab_covers_the_native_recurrent_state(monkeypatch):
    monkeypatch.delenv("ISOEXEC_GDN_FUSED_SPLIT", raising=False)
    assert defer_ab(SimpleNamespace(_native_kernels=True))
    assert defer_ab(SimpleNamespace(_native_core=True))
    assert not defer_ab(SimpleNamespace(_native_kernels=False))
    assert not defer_ab(SimpleNamespace())
    monkeypatch.setenv("ISOEXEC_GDN_FUSED_SPLIT", "0")
    assert not defer_ab(SimpleNamespace(_native_kernels=True))
