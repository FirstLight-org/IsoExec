"""CPU proof obligations for the owner Lamport all-reduce and its fused add+RMSNorm epilogue
(pik/owner.py ``lamport_*`` / ``RootHandoff``, csrc/owner_lamport.cu).

Decidable without a GPU:
  * the kernel's tree is the same recursion as the barrier owner kernel (pik.plan.combine_order);
  * the epilogue is norms_sm90 ``add_rms_fwd``'s program, statement for statement (lane chain, butterfly,
    rstd, (1 + w) weight, output product, RNE packing) -- and nothing else in the kernel does float arithmetic;
  * nothing touches memory before ``griddepcontrol.wait`` (PDL is schedule-only);
  * the sentinel is the all-ones pattern and a wire element equal to it is sent as the canonical NaN;
  * every launch decision is a pure function of rank-invariant facts; the handoff refuses a foreign consumer.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import re

import pytest

torch = pytest.importorskip("torch")

PIK = importlib.import_module("isoexec.ops.collectives.pik_bootstrap").ensure_pik()
OWNER = importlib.import_module("pik.owner")
CSRC = pathlib.Path(OWNER.__file__).parent / "csrc"
SRC = (CSRC / "owner_lamport.cu").read_text()
CODE = re.sub(r"//.*", "", SRC)
BARRIER = (CSRC / "owner_allreduce.cu").read_text()
NORM_FWD = (pathlib.Path(importlib.import_module("isoexec.ops.norms.norms_sm90").__file__).parent / "csrc"
            / "fwd.cuh").read_text()
NORM_COMMON = (pathlib.Path(importlib.import_module("isoexec.ops.norms.norms_sm90").__file__).parent / "csrc"
               / "common.cuh").read_text()


def _fn_body(src: str, signature: str) -> str:
    body = src[src.index(signature):]
    return body[: body.index("\n}\n")]


def test_tree_is_the_barrier_kernels_tree():
    sig = "__device__ __forceinline__ float tree("
    norm = lambda s: re.sub(r"\s+", " ", _fn_body(s, sig))  # noqa: E731
    assert norm(SRC) == norm(BARRIER)


def test_float_arithmetic_is_confined_to_the_tree_join_and_norm():
    assert "fma" not in CODE and "__fsub" not in CODE and "__hadd" not in CODE
    # adds: tree (1), residual join (1), lane chain (1), butterfly (1), rstd eps (1), 1 + w (1)
    assert CODE.count("__fadd_rn") == 6
    # products: v*v in the chain, (v*rs)*g in the output
    assert CODE.count("__fmul_rn") == 3
    assert CODE.count("__fdiv_rn") == 1 and CODE.count("__frsqrt_rn") == 1
    # narrowing only through cvt.rn.bf16x2.f32 (the norms kernels' instruction)
    assert CODE.count("cvt.rn.bf16x2.f32") == 1 and "__float2bfloat16" not in CODE
    # the only floating '+' / '*' tokens on float values are the intrinsics above: no operator arithmetic on
    # floats in the reduction, join or norm code (integer index arithmetic is not checked here)
    for fn in ("reduce8", "norm_row"):
        body = CODE[CODE.index(f"static void {fn}("):]
        body = body[: body.index("\n  }\n")]
        for stmt in re.findall(r"(?:acc|x\[l\]|o\[j\]|g\[j\]|root\[l\])\s*[+*]?=[^;]*;", body):
            assert re.search(r"__f(add|mul)_rn|tree<", stmt) or "=" in stmt and "+" not in stmt.split("=", 1)[1], stmt


def test_epilogue_is_add_rms_fwd_program():
    ref = _fn_body(NORM_FWD, "add_rms_fwd(")
    # residual join: one fp32 add of the widened bf16 operands, one RNE pack
    assert "a[j] = __fadd_rn(a[j], b[j]);" in ref
    assert "x[l] = __fadd_rn(x[l], r8[l]);" in CODE
    # lane chain over k ascending, j ascending, on the widened bf16 h
    assert "acc = __fadd_rn(acc, __fmul_rn(v[j], v[j]));" in ref
    assert "acc = __fadd_rn(acc, __fmul_rn(v[j], v[j]));" in CODE
    # butterfly over 32 lanes, stage offsets L/2 .. 1
    fold = _fn_body(NORM_COMMON, "__device__ __forceinline__ float warp_fold(")
    assert "acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, s));" in fold
    assert "for (int s = kLanes >> 1; s > 0; s >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, s));" \
        in CODE
    # rstd
    assert "__frsqrt_rn(__fadd_rn(__fdiv_rn(ssq, float(n)), eps))" in NORM_COMMON
    assert "__frsqrt_rn(__fadd_rn(__fdiv_rn(acc, float(a.N)), a.eps))" in CODE
    # weight: (1 + w) in mode 2, plain w otherwise; output (v * rs) * g
    assert "g[j] = __fadd_rn(1.f, g[j]);" in NORM_FWD and "g[j] = __fadd_rn(1.f, g[j]);" in CODE
    assert "y[j] = __fmul_rn(__fmul_rn(v[j], rs), g[j]);" in ref
    assert "o[j] = __fmul_rn(__fmul_rn(v[j], rs), g[j]);" in CODE
    # lane map: lane l owns columns k*8L + 8l + j (L = 32), the recipe-v1 swizzle
    assert "(k0 + i) * 8 * kLanes + 8 * lane" in CODE and "constexpr int kLanes = 32;" in CODE
    assert OWNER._NORM_IMPLS == frozenset({importlib.import_module(
        "isoexec.ops.norms.norms_sm90.integration").ADD_RMS_IMPL})


def test_nothing_touches_memory_before_griddepcontrol_wait():
    body = CODE[CODE.index("lamport_kernel(const __grid_constant__ Args a) {"):]
    first = [ln.strip() for ln in body.splitlines()[1:4] if ln.strip()]
    assert first[0].startswith("using K =") and first[1] == "pdl_wait();"
    assert "griddepcontrol.wait;" in CODE and "griddepcontrol.launch_dependents;" in CODE


def test_sentinel_and_canonical_nan():
    assert "make_uint4(~0u, ~0u, ~0u, ~0u)" in CODE  # re-fill pattern
    assert "lo == 0xFFFFu ? 0x7FFFu : lo" in CODE and "hi == 0xFFFFu ? 0x7FFFu : hi" in CODE
    assert "0xFFFFFFFFu ? 0x7FFFFFFFu" in CODE
    # peers' data is read with volatile 16-byte loads and written with volatile stores
    assert "ld.volatile.global.v4.u32" in CODE and "st.volatile.global.v4.u32" in CODE
    # three generations; the generation advance happens once, by the last block to retire
    assert "constexpr int kGens = 3;" in CODE and "a.ctl[0] = call + 1ull;" in CODE


def test_policy_functions_take_only_rank_invariant_facts():
    for fn, params in (
        (OWNER._lamport_mode, ["cfg", "world", "M", "N", "planes", "elt", "root_elt"]),
        (OWNER._lamport_grid, ["cfg", "mode", "M", "N", "planes", "world"]),
    ):
        assert list(inspect.signature(fn).parameters) == params, fn.__name__


class _Bufs:
    def __init__(self, gen):
        self.gen1 = self.gen2 = gen
        self.world = 4


class _Cfg:
    def __init__(self, gen=16 << 20, oneshot=128 << 10, blocks=132):
        self.bufs, self.oneshot_bytes, self.blocks = _Bufs(gen), oneshot, blocks


@pytest.mark.parametrize("world", [2, 4, 8, 16])
def test_mode_and_capacity(world):
    cfg = _Cfg()
    cfg.bufs.world = world
    N = 5120
    for M in (1, 3, 26, 175, 256, 409, 410, 4096):
        for planes, elt in ((2, 2), (1, 4), (1, 2)):
            mode = OWNER._lamport_mode(cfg, world, M, N, planes, elt, 2)
            wire = M * planes * N * elt
            if mode == 1:
                assert wire <= cfg.oneshot_bytes and world * wire <= cfg.bufs.gen1
            elif mode == 2:
                slot = -(-M // world) * planes * N * elt
                assert world * slot <= cfg.bufs.gen1 and M * N * 2 <= cfg.bufs.gen2
            else:
                assert -(-M // world) * planes * N * elt * world > cfg.bufs.gen1 or M * N * 2 > cfg.bufs.gen2
            g = OWNER._lamport_grid(cfg, mode or 2, M, N, planes, world)
            assert 1 <= g <= cfg.blocks


def test_decode_batches_fit_the_default_generation():
    cfg = _Cfg(gen=OWNER._DEFAULT_LAMPORT_KB << 10, oneshot=OWNER._DEFAULT_LAMPORT_ONESHOT_BYTES[4])
    for M in (1, 26, 175, 256, 400):
        assert OWNER._lamport_mode(cfg, 4, M, 5120, 2, 2, 2) in (1, 2), M


def test_handoff_refuses_a_foreign_consumer():
    h = OWNER.RootHandoff("site")
    assert h.take(object()) is None
    mine = object()
    h.pending = {"out": mine}
    assert h.take(mine)["out"] is mine and h.pending is None
    h.pending = {"out": mine}
    with pytest.raises(RuntimeError):
        h.take(object())
    assert h.pending is None


def test_deferral_drops_a_leftover_root_even_when_it_does_not_defer():
    h = OWNER.RootHandoff("site")
    h.pending = {"out": object()}
    x2, w = torch.zeros(2, 256, dtype=torch.bfloat16), torch.zeros(256, 64, dtype=torch.bfloat16)
    assert OWNER.row_parallel_deferred(x2, w, None, plan=None, tp_size=1, tp_rank=0, k_full=64, group=None,
                                       out_dtype=torch.bfloat16, handoff=h) is None
    assert h.pending is None


def test_fused_decision_reads_no_rank_local_alignment():
    """Fusing is a collective decision: the predicate names shapes, dtypes, program and grad mode only; an
    operand's address or stride is normalized (copied), never part of the decision."""
    src = inspect.getsource(OWNER.add_norm_deferred)
    decision = src[src.index("fusable = ("):src.index("    )\n", src.index("fusable = ("))]
    assert "data_ptr" not in decision and "stride" not in decision and "is_contiguous" not in decision


def test_default_grid_leaves_half_the_sms(monkeypatch):
    monkeypatch.setattr(OWNER, "_sm_count", lambda device: 132)
    assert OWNER._default_lamport_blocks(None) == 66


def _lamport_state_rank(rank, world, fail):
    import types

    import torch.distributed as dist

    importlib.import_module("isoexec.ops.collectives.pik_bootstrap").ensure_pik()
    owner = importlib.import_module("pik.owner")
    owner._ON = True
    owner._sm_count = lambda device: 8
    calls = []

    def lext():
        calls.append("build")
        if fail == ("build", rank):
            raise RuntimeError("toolchain differs on this rank")

    class Bufs:
        def __init__(self, group, device, gen_bytes):
            calls.append("alloc")
            if fail == ("alloc", rank):
                raise RuntimeError("out of memory")

        def connect(self):
            calls.append("connect")
            if fail == ("connect", rank):
                raise RuntimeError("rendezvous refused")

    owner._lext = lext
    owner._LamportBuffers = Bufs
    st = types.SimpleNamespace(transport="p2p")
    cfg = owner.lamport_state(st, dist.group.WORLD, torch.device("cpu"))
    return cfg is not None, calls


@pytest.mark.parametrize("fail,calls", [
    (None, ["build", "alloc", "connect"]),
    (("build", 1), ["build"]),  # the other rank must not go on to allocate / rendezvous
    (("alloc", 0), ["build", "alloc"]),
    (("connect", 1), ["build", "alloc", "connect"]),
])
def test_lamport_setup_failure_on_one_rank_falls_back_on_every_rank(fail, calls):
    from isoexec.ops._testing import run_gloo

    out = run_gloo(_lamport_state_rank, 2, fail)
    on = fail is None
    assert out == {0: (on, calls), 1: (on, calls)}
