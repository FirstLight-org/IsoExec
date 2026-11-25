"""Check packed-sequence memo values and one-build-per-forward reuse.
Shared identity/version, lifetime, invalidation and decline tests live in test_meta_memo_protocol_cpu.py."""

import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

from isoexec.ops.gdn.shared import (  # noqa: E402
    packed_meta_cache as pmc,  # noqa: E402
)


def _reset_packed_meta_census():
    # inlined: the public module keeps the counters but not this test accessor
    pmc._STATS.update(served=0, built=0, declined=0)


BATCHES = [
    [96, 64, 137, 201],  # the census harness's packed row: ragged, none chunk-aligned
    [64],  # single sequence
    [1, 1, 1],  # degenerate
    [512, 512],  # uniform
    [7, 129, 3, 64, 1000],  # very ragged
]


def _cu(lens, dtype=torch.int32):
    return torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=dtype)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("ISOEXEC_PACKED_META_CACHE", raising=False)
    pmc._CACHE.clear()
    _reset_packed_meta_census()
    yield
    pmc._CACHE.clear()
    _reset_packed_meta_census()


# 1. exactness -- the memo returns what the expression it replaces returns
@pytest.mark.parametrize("lens", BATCHES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_cu_list_equals_tolist(lens, dtype):
    cu = _cu(lens, dtype)
    assert pmc.cu_list(cu) == cu.tolist()


@pytest.mark.parametrize("lens", BATCHES)
def test_cu_last_equals_the_select_read(lens):
    cu = _cu(lens)
    assert pmc.cu_last(cu) == int(cu[-1].cpu().item())


@pytest.mark.parametrize("lens", BATCHES)
@pytest.mark.parametrize("div", [1, 2, 4])
def test_seq_lens_equals_the_device_expression(lens, div):
    """The host-side arithmetic must agree with `((cu[1:] - cu[:-1]) // div).tolist()` elementwise."""
    cu = _cu([n * div for n in lens])
    want = ((cu[1:] - cu[:-1]) // div).tolist()
    assert pmc.seq_lens(cu, div) == want


def test_seq_lens_rejects_a_nonsense_divisor():
    cu = _cu([4, 4])
    with pytest.raises(ValueError, match="div must be >= 1"):
        pmc.seq_lens(cu, 0)


@pytest.mark.parametrize("lens", BATCHES)
def test_fla_stable_clone_is_a_real_equal_clone(lens):
    cu = _cu(lens)
    clone = pmc.fla_stable_clone(cu)
    assert clone is not cu
    assert torch.equal(clone, cu)
    assert clone.dtype == cu.dtype and clone.device == cu.device


# 2. the point of the thing: one read per forward, not one per layer
def test_thirty_layers_pay_one_read_each_and_get_identical_objects():
    """A 30-layer forward: 1 build + 29 hits per derived key, the same object every time."""
    cu = _cu([96, 64, 137, 201])
    firsts = (pmc.cu_list(cu), pmc.cu_last(cu), pmc.seq_lens(cu), pmc.fla_stable_clone(cu))
    for _ in range(29):
        assert pmc.cu_list(cu) is firsts[0]
        assert pmc.cu_last(cu) == firsts[1]
        assert pmc.seq_lens(cu) is firsts[2]
        assert pmc.fla_stable_clone(cu) is firsts[3]
    c = pmc.packed_meta_census()
    # The +2 on the first round is `cu_last` and `seq_lens` consuming the already-built "list":
    # one device read backs all four derived values.
    assert c["built"] == 4, c
    assert c["served"] == 29 * 4 + 2, c
    assert c["declined"] == 0, c


def test_causal_conv_metadata_is_built_once_per_packed_forward(monkeypatch):
    """The conv metadata is built once per packed forward, not once per layer."""
    module_name = "isoexec.ops.gdn.shared.conv_metadata"
    fake_module = ModuleType(module_name)
    builds = []

    def fake_compute(cu_cpu, *, device):
        builds.append((cu_cpu.clone(), device))
        return {8: {"tot": 3}}, torch.tensor([1]), torch.tensor([2])

    fake_module.compute_causal_conv1d_metadata = fake_compute
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    cu = _cu([96, 64, 137, 201])

    first = pmc.causal_conv1d_metadata(cu)
    for _ in range(29):
        assert pmc.causal_conv1d_metadata(cu) is first

    assert len(builds) == 1
    assert torch.equal(builds[0][0], cu)
    assert first.nums_dict == {8: {"tot": 3}}
    assert torch.equal(first.batch_ptr, torch.tensor([1]))
    assert torch.equal(first.token_chunk_offset_ptr, torch.tensor([2]))
    assert torch.equal(first.cache_indices, torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    assert torch.equal(first.has_initial_state, torch.zeros(4, dtype=torch.bool))
    assert pmc.causal_conv1d_metadata(cu).cache_indices is first.cache_indices
    assert pmc.causal_conv1d_metadata(cu).has_initial_state is first.has_initial_state


def test_causal_conv_metadata_accepts_scheduler_host_offsets_without_readback(monkeypatch):
    module_name = "isoexec.ops.gdn.shared.conv_metadata"
    fake_module = ModuleType(module_name)
    seen = []

    def fake_compute(cu_cpu, *, device):
        seen.append(cu_cpu.tolist())
        return {}, torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32)

    fake_module.compute_causal_conv1d_metadata = fake_compute
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    monkeypatch.setattr(pmc, "cu_list", lambda _cu: (_ for _ in ()).throw(AssertionError("D2H")))
    cu = _cu([3, 5, 7])
    assert pmc.causal_conv1d_metadata(cu, cu_host=[0, 3, 8, 15]) is not None
    assert seen == [[0, 3, 8, 15]]


@pytest.mark.parametrize("bad", [[], [1, 4], [0, 4]])
def test_causal_conv_scheduler_offsets_decline_invalid_shape(bad):
    assert pmc.causal_conv1d_metadata(_cu([4, 5]), cu_host=bad) is None


def test_causal_conv_metadata_cache_off_restores_vllm_internal_build(monkeypatch):
    monkeypatch.setenv("ISOEXEC_PACKED_META_CACHE", "0")
    assert pmc.causal_conv1d_metadata(_cu([4, 9])) is None


def test_the_stable_clone_is_what_makes_an_identity_keyed_cache_hit():
    """A fresh clone per layer is a guaranteed miss in FLA's identity-keyed cache."""
    cu = _cu([96, 64, 137, 201])
    # hold the references: CPython recycles the id of a clone dropped on the same line, which
    # would make the control look like a hit.
    fresh = [cu.clone() for _ in range(8)]
    stable = [pmc.fla_stable_clone(cu) for _ in range(8)]
    assert len({id(t) for t in stable}) == 1, "the memo must hand every layer the SAME object"
    assert len({id(t) for t in fresh}) == 8, "sanity: fresh clones really are distinct objects"


# 3. key soundness -- an in-place write through ANY alias invalidates
def test_a_different_tensor_with_equal_values_is_not_served_the_first_entry():
    a = _cu([64, 64])
    b = _cu([64, 64])
    assert torch.equal(a, b)
    ca, cb = pmc.fla_stable_clone(a), pmc.fla_stable_clone(b)
    assert ca is not cb, "the key must be identity, never value"
    assert torch.equal(ca, cb)


def test_inference_tensor_declines_versioned_memo_without_raising(monkeypatch):
    """vLLM builds query-start offsets under inference_mode; those tensors have no ``_version``."""

    with torch.inference_mode():
        cu = _cu([3, 5, 7])
    assert torch.is_inference(cu)
    assert pmc.cu_list(cu) == [0, 3, 8, 15]
    assert pmc.is_memoized(cu) is False
    assert pmc.packed_meta_census() == {"served": 0, "built": 0, "declined": 1}

    module_name = "isoexec.ops.gdn.shared.conv_metadata"
    fake_module = ModuleType(module_name)
    seen = []

    def fake_compute(cu_cpu, *, device):
        seen.append(cu_cpu.tolist())
        return {}, torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32)

    fake_module.compute_causal_conv1d_metadata = fake_compute
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    got = pmc.causal_conv1d_metadata(cu, cu_host=[0, 3, 8, 15], include_launch_args=False)
    assert got is not None and seen == [[0, 3, 8, 15]]


# 4. declines -- and a decline is still a CORRECT answer
def test_the_flag_off_restores_the_per_call_read_and_the_values_are_unchanged(monkeypatch):
    monkeypatch.setenv("ISOEXEC_PACKED_META_CACHE", "0")
    cu = _cu([96, 64, 137])
    assert not pmc.packed_meta_cache_enabled()
    assert pmc.cu_list(cu) == cu.tolist()
    assert pmc.seq_lens(cu, 2) == ((cu[1:] - cu[:-1]) // 2).tolist()
    a, b = pmc.fla_stable_clone(cu), pmc.fla_stable_clone(cu)
    assert a is not b, "flag off must restore the fresh clone per call"
    assert torch.equal(a, b)
    c = pmc.packed_meta_census()
    assert c["served"] == 0 and c["built"] == 0 and c["declined"] > 0


# 5. peeks and split sizes
def test_is_memoized_never_creates_an_entry():
    cu = _cu([64, 64])
    assert not pmc.is_memoized(cu)
    assert len(pmc._CACHE) == 0, "a peek must not insert"
    pmc.cu_list(cu)
    assert pmc.is_memoized(cu)
    assert not pmc.is_memoized(cu, "nosuchkey")


def test_seq_lens_splits_exactly_as_a_thd_rope_would():
    """`torch.split` must receive the identical size list, or every downstream kernel shifts."""
    cu = _cu([96, 64, 137, 201])
    for cp in (1, 2, 4):
        cu_cp = _cu([n * cp for n in [96, 64, 136, 200]])
        want = ((cu_cp[1:] - cu_cp[:-1]) // cp).tolist()
        pmc._CACHE.clear()
        assert pmc.seq_lens(cu_cp, cp) == want
    assert pmc.cu_last(cu) == int(cu[-1])
