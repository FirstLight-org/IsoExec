"""CPU contract of the BF16 actor's canonical MegaMoE forward (``deepseek_v41_sm100/mega.py``): forward-only
refusal of grad-enabled calls, fail-closed expert placement and geometry, zero-copy master views, the 5D master-layout L1 map (its box
order IS MegaMoE's granularity-8 gate/up interleave), chunking, and the canonical kernel edits (combine order,
first contribution, per-launch tag, precise arithmetic). The bits are GPU-gated (``test_actor_mega_gpu.py``,
``_actor_mega_ep_child.py``)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from isoexec.ops.moe import ep_fixed_order as epf
from isoexec.ops.moe.deepseek_v41_sm100 import mega, ops, training

VENDOR = Path(__file__).resolve().parents[3] / "_vendor" / "deep_gemm"


def test_grad_enabled_megamoe_is_refused():
    from isoexec.ops.moe.deepseek_v41_sm100 import ep
    w = torch.zeros(2, requires_grad=True)
    x = torch.zeros(1, 256, dtype=torch.bfloat16, requires_grad=True)
    for call in (lambda: mega.check_no_grad_pass(None, w),
                 lambda: ops.moe_forward_mega(x, None, None, torch.zeros(2, 256, 256, dtype=torch.bfloat16),
                                              torch.zeros(2, 256, 128, dtype=torch.bfloat16),
                                              top_k=1, routed_scale=1.0, limit=10.0),
                 lambda: ep.mega_ep_forward(x, None, None, torch.zeros(2, 256, 256, dtype=torch.bfloat16),
                                            torch.zeros(2, 256, 128, dtype=torch.bfloat16), top_k=1,
                                            routed_scale=1.0, limit=10.0, emap=epf.ExpertMap.contiguous(1, 1),
                                            ws=type("W", (), {"world": 1, "rank": 0})(), width=1)):
        with pytest.raises(RuntimeError, match="no-grad forward"):
            call()
    mega.check_no_grad_pass(None, w.detach())                        # no-grad tensors: allowed
    with torch.no_grad():
        mega.check_no_grad_pass(None, w)


def test_training_keeps_the_grouped_forward():
    """The grad-enabled form is the grouped-kernel evaluation (its backward reads the stage tensors)."""
    assert "mega" not in training.moe_train.__code__.co_names
    assert "mega" not in ops.moe_forward.__code__.co_names


def test_chunks_equal_and_covering():
    assert mega.chunks(0, 64) == [1]
    assert mega.chunks(64, 64) == [64]
    for width, cap in ((65, 64), (8193, 8192), (4096, 1000)):
        c = mega.chunks(width, cap)
        assert len(set(c)) == 1 and c[0] <= cap and sum(c) >= width and sum(c) - c[0] < width


def test_expert_map_contract():
    for W in (1, 2, 4, 8):
        emap = epf.ExpertMap.contiguous(384, W, shared="home")
        for r in range(W):
            mega.check_expert_map(emap, W, r)
    with pytest.raises(ValueError):
        mega.check_expert_map(epf.ExpertMap.permuted(384, 4, seed=1, shared="home"), 4, 0)
    with pytest.raises(ValueError):
        mega.check_expert_map(epf.ExpertMap.contiguous(384, 4, shared=3), 4, 0)


def test_geometry_contract():
    mega.check_geometry(384, 5120, 2304)
    mega.check_geometry(256, 512, 256)
    for h, i in ((5120 + 64, 2304), (5120, 2304 + 64)):
        with pytest.raises(ValueError):
            mega.check_geometry(384, h, i)


def test_master_views_are_zero_copy():
    gu = torch.zeros(5, 16, 8, dtype=torch.bfloat16)
    dn = torch.zeros(5, 8, 8, dtype=torch.bfloat16)
    l1, l2, s1, s2 = mega.masters(gu, dn)
    assert l1.data_ptr() == gu.data_ptr() and l2.data_ptr() == dn.data_ptr()
    assert s1.data_ptr() == gu[4].data_ptr() and s2.data_ptr() == dn[4].data_ptr()
    assert all(t.is_contiguous() for t in (l1, l2, s1, s2)) and tuple(l1.shape) == (4, 16, 8)
    with pytest.raises(ValueError):
        mega.masters(gu.transpose(1, 2), dn)


def _interleave(t, gran=8):
    """DeepGEMM ``mega._interleave_weights`` (the upstream L1 transform), restated."""
    g, n, k = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, k)
    up = t[:, half:].reshape(g, half // gran, gran, k)
    return torch.stack([gate, up], dim=2).reshape(g, n, k)


@pytest.mark.parametrize("E,I,K", [(3, 64, 16), (2, 2304, 8)])
def test_5d_master_map_box_order_is_the_upstream_interleave(E, I, K):
    """make_isoexec_split_l1_desc: dims (K, 8, 2, I/8, E), strides (K, I*K, 8*K, 2I*K) elements. TMA writes a box
    with dim 0 fastest, so row r of an N tile starting at group g0 is (group g0 + r // 16, half (r // 8) % 2, row
    r % 8): exactly upstream's interleaved rows n_idx .. n_idx + BLOCK_N with n_idx = 16 * g0."""
    t = torch.arange(E * 2 * I * K, dtype=torch.float64).reshape(E, 2 * I, K)
    view = t.as_strided((E, I // 8, 2, 8, K), (2 * I * K, 8 * K, I * K, K, 1))   # (e, group, half, row, k)
    assert torch.equal(view.reshape(E, 2 * I, K), _interleave(t))
    block_n = 128 if I % 64 == 0 else 16
    for n_idx in range(0, 2 * I, block_n):
        tile = view[:, n_idx // 16: n_idx // 16 + block_n // 16].reshape(E, block_n, K)
        assert torch.equal(tile, _interleave(t)[:, n_idx:n_idx + block_n])


def _src(rel):
    return (VENDOR / rel).read_text()


def test_canonical_bf16_kernel_edits():
    src = _src("include/deep_gemm/impls/sm100_bf16_mega_moe.cuh")
    assert "bool kIsoexecCanonical," in src
    # canonical combine: ascending global expert id, shared last, first contribution assigned, __fadd_rn
    assert "__reduce_min_sync(0xffffffffu, expert)" in src and "first_contribution ? input" in src
    assert "__fadd_rn(value.x, input.x)" in src
    # graph-replay-safe readiness tag
    assert src.count("launch_tag") >= 4 and "get_isoexec_launch_epoch_ptr() = launch_tag" in src
    assert "const auto grid_idx = launch_tag;" in src
    # master-layout L1 through the 5D map, both routed and shared phases
    assert "b_dtype_t, 5>(" in src and "k_idx, 0u, 0u, n_idx / 16u, expert_idx" in src
    assert "sched::BlockPhase::SharedLinear1" in src.split("canonical_l1 = kIsoexecCanonical")[1][:300]
    # canonical activation = swiglu_rows: clamp, IEEE silu, (silu * up) * w, one RNE
    body = src.split("namespace isoexec_canonical_bf16 {")[1].split("}  // namespace")[0]
    assert "__fdiv_rn(gate, __fadd_rn(1.0f, expf(-gate)))" in body
    assert "__fmul_rn(__fmul_rn(silu, up), weight)" in body
    assert "__expf" not in body and "fast_rcp" not in body


def test_canonical_bf16_host_and_binding():
    hpp = _src("csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp")
    assert "make_isoexec_split_l1_desc" in hpp and hpp.count("isoexec_canonical ?") >= 3
    assert '"--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"' in hpp
    api = _src("csrc/apis/mega_moe.hpp")
    assert 'm.def("bf16_mega_moe", &bf16_mega_moe<false>);' in api
    assert 'm.def("bf16_mega_moe_isoexec", &bf16_mega_moe<true>);' in api
    assert "kIsoexecCanonical ? false : fast_math, kIsoexecCanonical" in api
    py = _src("mega/__init__.py")
    assert "_C.bf16_mega_moe_isoexec if isoexec_canonical else _C.bf16_mega_moe" in py


def test_mega_paths_refuse_capture_without_workspace(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    mega.Workspace._POOLS.pop(mega.Workspace.pool_key(None, "cuda:7", 1, 1, 256, 128), None)
    with pytest.raises(RuntimeError, match="outside CUDA graph capture"):
        mega.Workspace.prepare(None, "cuda:7", 1, 1, 256, 128)


def test_split_masters_are_passed_without_concatenation():
    gu, dn = torch.zeros(3, 8, 4), torch.zeros(3, 4, 4)
    sgu, sdn = torch.zeros(8, 4), torch.zeros(4, 4)
    l1, l2, s1, s2 = mega.masters(gu, dn, (sgu, sdn))
    assert l1 is gu and l2 is dn and s1 is sgu and s2 is sdn       # routed-only + shared: no copy


def test_fused_io_kernels_and_protocol():
    src = (Path(__file__).resolve().parents[2] / "deepgemm_sm100" / "csrc" / "mega_ep_io.cu").read_text()
    assert "prep_bf16_kernel" in src and 'm.def("prep_bf16"' in src
    body = src[src.index("void prep_bf16_kernel"):src.index("// flags: this rank's")]
    assert "*counter += 1" in body and "atomic" not in body           # one writer per word, the gather tag
    assert "-1" in body and "make_uint4(0, 0, 0, 0)" in body          # padding rows: id -1, zero rows / weights


def test_replicated_gather_needs_fused_io_at_ep_gt_1(monkeypatch):
    ws = object.__new__(mega.Workspace)
    ws.world, ws.rank, ws.hidden, ws.top_k, ws.E, ws.intermediate = 1, 0, 256, 6, 4, 128
    ws.device = torch.device("cpu")
    gu, dn = torch.zeros(5, 256, 256, dtype=torch.bfloat16), torch.zeros(5, 256, 128, dtype=torch.bfloat16)
    x = torch.zeros(2, 256, dtype=torch.bfloat16)
    ids = torch.zeros(2, 7, dtype=torch.int32)
    with pytest.raises(ValueError, match="gather"):
        mega.experts(ws, x, ids, ids.float(), gu, dn, limit=10.0, out_rows=4)
