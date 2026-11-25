"""CPU envelopes: inputs are refused before the extension builds; the record-image index maps and the
decode plan heuristic are well formed."""

import pytest
import torch

from isoexec.ops.gdn.gdn_cpr_sm90 import ops as G
from isoexec.ops.gdn.gdn_cpr_sm90._extension import SOURCES


def _inputs(T=4, hk=2, hv=4, device="cpu"):
    r = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=device)
    return r(T, hk, 128), r(T, hk, 128), r(T, hv, 128), r(T, hv), r(T, hv)


def test_cpu_inputs_are_refused_before_building(monkeypatch):
    def boom():
        raise AssertionError("the extension must not be built for rejected inputs")

    monkeypatch.setattr(G, "extension", boom)
    q, k, v, a, b = _inputs()
    A = torch.zeros(4)
    with pytest.raises(ValueError, match="CUDA"):
        G.prefill(q, k, v, a, b, A, A, torch.tensor([0, 4], dtype=torch.int32))
    with pytest.raises(ValueError, match="CUDA"):
        G.decode(q, k, v, a, b, A, A, None, torch.tensor([1, 2, 3, 4], dtype=torch.int32))


def test_sources_are_listed():
    """The extension compiles every translation unit in csrc/ and nothing else."""
    from pathlib import Path

    csrc = Path(G.__file__).resolve().parent / "csrc"
    on_disk = {p.name for p in csrc.iterdir() if p.suffix in (".cu", ".cpp")}
    assert set(SOURCES) == on_disk and len(SOURCES) == len(set(SOURCES))


@pytest.mark.parametrize("fn,numel", [(G._kb_index, 64 * 128), (G._cols_index, 64 * 128), (G._s_index, 128 * 128)])
def test_record_images_are_permutations(fn, numel):
    """Every logical element of a stored tile image has exactly one storage slot."""
    idx = fn()
    assert idx.numel() == numel
    assert torch.equal(torch.sort(idx).values, torch.arange(numel))


@pytest.mark.parametrize("n,hv,want", [(1, 4, 4), (16, 4, 4), (17, 4, 2), (32, 4, 2), (33, 4, 1), (1, 32, 4),
                                       (2, 32, 4), (3, 32, 2), (4, 32, 2), (5, 32, 1), (1000, 6, 1)])
def test_decode_split_heuristic(n, hv, want):
    """The decode policy: 4 CTAs per head up to 64 heads in flight, 2 up to 128, then 1."""
    assert G.choose_decode_split(n, hv) == want


def test_fla_backward_names_what_moved(monkeypatch):
    """The backward's FLA loader refuses a checkout without the internals it launches, naming them."""
    from isoexec.ops.gdn.gdn_cpr_sm90 import fla_backward

    def chunk_gated_delta_rule():   # a stand-in whose module lacks every stage
        pass

    monkeypatch.setattr(fla_backward, "get_fla_chunk_backward", lambda: chunk_gated_delta_rule)
    fla_backward.load.cache_clear()
    try:
        with pytest.raises(fla_backward.FlaMoved, match="chunk_bwd_dqkwg"):
            fla_backward.load()
    finally:
        fla_backward.load.cache_clear()


def test_state_and_pack_validation():
    """Wrong pools, empty packs with tokens and bad sweep widths are refused."""
    with pytest.raises(ValueError, match="16, 32 or None"):
        G._sweep_rows(0)
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="no sequences"):
            G._check_cu(torch.zeros(1, dtype=torch.int32, device="cuda"), 4)
    bad = G.CprState(torch.zeros(2, 4, 16, dtype=torch.uint8))
    with pytest.raises(ValueError, match="CprState"):
        G._check_state(bad, 4, torch.device("cpu"))


@pytest.mark.parametrize("n,length,hv,want", [
    (4, 2048, 8, 0), (16, 512, 3, 0), (16, 512, 4, 2), (16, 512, 6, 2),
    (64, 320, 8, 2), (128, 160, 8, 2), (256, 80, 8, 2),
    (128, 64, 8, 0), (512, 40, 8, 0), (16, 2048, 8, 0),
])
def test_prefill_crossover(n, length, hv, want):
    cu = torch.arange(n + 1, dtype=torch.int32) * length
    assert G.choose_prefill_mode(n, n * length, hv, cu) == want


def test_prefill_dispatch_needs_host_lengths_and_rejects_skew():
    assert G.choose_prefill_mode(64, 20480, 8) == 0
    lengths = torch.full((64,), 80, dtype=torch.int32)
    lengths[0] = 15360
    cu = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0)))
    assert G.choose_prefill_mode(64, int(cu[-1]), 8, cu) == 0
    for mode, want in (("two_pass", 0), ("fused", 1), ("prepared", 2)):
        with G._prefill_mode(mode):
            assert G.choose_prefill_mode(1, 1, 1) == want


def test_declarations_cover_only_the_fused_core_stages():
    from isoexec.core.registry import Registry
    from isoexec.ops.gdn.gdn_cpr_sm90.integration import IMPL, _OWNED
    from isoexec.ops.gdn.gdn_cpr_sm90.spec import IMPLS

    registry = Registry.from_impls(IMPLS)
    assert set(registry.ops) == {"gdn.core", "gdn.l2norm", "gdn.gating"}
    assert set(_OWNED) == {"gdn.l2norm", "gdn.gating"}
    assert IMPL == "cuda_cpr_v1"
    for op in registry.ops.values():
        assert set(op.impls) == {IMPL}
        impl = op.impls[IMPL]
        assert impl.version == 1 and impl.supported_archs == frozenset({"sm90"})
        assert not impl.twin_class and not impl.inputs


@pytest.fixture
def admitted_engine_flags(monkeypatch):
    from isoexec.ops.gdn.gdn_cpr_sm90 import engine_state

    monkeypatch.setenv("ISOEXEC_GDN_KERNEL", "cpr")
    monkeypatch.setenv("ISOEXEC_GDN_NATIVE_KERNELS", "1")
    monkeypatch.delenv("ISOEXEC_GDN_NATIVE_STATE", raising=False)
    monkeypatch.delenv("ISOEXEC_GDN_CPR_APC", raising=False)
    monkeypatch.delenv("ISOEXEC_GDN_CHUNKED_PREFILL", raising=False)  # the store needs it (tested below)
    monkeypatch.setattr(engine_state, "_NOTED", True)
    engine_state.refuse_incompatible_flags()
    return engine_state


@pytest.mark.parametrize("name,value", [
    ("ISOEXEC_GDN_KERNEL", "recurrent"),
    ("ISOEXEC_GDN_NATIVE_KERNELS", "0"),
    ("ISOEXEC_GDN_NATIVE_STATE", "1"),
    ("ISOEXEC_GDN_CPR_APC", "1"),
    ("ISOEXEC_GDN_CPR_APC", "shared"),
])
def test_engine_flags_refuse_incompatible_state_before_cuda(admitted_engine_flags, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name) as exc:
        admitted_engine_flags.refuse_incompatible_flags()
    assert "ops/gdn/gdn_cpr_sm90" in str(exc.value)


def test_engine_flags_report_all_incompatibilities(admitted_engine_flags, monkeypatch):
    invalid = {"ISOEXEC_GDN_KERNEL": "recurrent", "ISOEXEC_GDN_NATIVE_KERNELS": "0",
               "ISOEXEC_GDN_NATIVE_STATE": "1", "ISOEXEC_GDN_CPR_APC": "1"}
    for name, value in invalid.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError) as exc:
        admitted_engine_flags.refuse_incompatible_flags()
    assert all(name in str(exc.value) for name in invalid)


@pytest.mark.parametrize("value", ["1", "shm"])
def test_engine_flags_admit_the_boundary_store_with_the_chunked_prefill_resume(admitted_engine_flags, monkeypatch,
                                                                               value):
    """Prefix caching resumes an adopted boundary record as a chunked-prefill continuation: the store is admitted
    with the resume flag and refused without it."""
    monkeypatch.setenv("ISOEXEC_GDN_CPR_APC", value)
    monkeypatch.setenv("ISOEXEC_GDN_CHUNKED_PREFILL", "1")
    admitted_engine_flags.refuse_incompatible_flags()
    monkeypatch.setenv("ISOEXEC_GDN_CHUNKED_PREFILL", "0")
    with pytest.raises(RuntimeError, match="ISOEXEC_GDN_CHUNKED_PREFILL"):
        admitted_engine_flags.refuse_incompatible_flags()
