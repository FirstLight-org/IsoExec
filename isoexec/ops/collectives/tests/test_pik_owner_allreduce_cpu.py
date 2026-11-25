"""CPU proof obligations for pik's owner tree all-reduce (pik/owner.py + csrc/owner_allreduce.cu).

Decidable without a GPU:
  * the kernel's tree recursion is ``pik.plan.combine_order`` / ``codegen.tree_source`` (same association);
  * the only float arithmetic in the kernel source is the one ``reduce_vec`` + ``tree`` pair, every add is
    ``__fadd_rn`` and the only narrowing is ``__float2bfloat16_rn`` at the root;
  * leaf wire and node wire are the same expression for every world size dividing G;
  * every launch-structure decision is a pure function of rank-invariant facts.
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
PLAN = importlib.import_module("pik.plan")
CG = importlib.import_module("pik.codegen")

SRC = (pathlib.Path(OWNER.__file__).parent / "csrc" / "owner_allreduce.cu").read_text()


def _tree_expr(lo, hi):
    """The kernel's recursion (`tree<LO,HI>`), as a string over operand indices."""
    if hi - lo == 1:
        return f"x{lo}"
    mid = (lo + hi) // 2
    return f"({_tree_expr(lo, mid)}+{_tree_expr(mid, hi)})"


def _combine_order_expr(n):
    slots = [f"x{i}" for i in range(n)]
    for dst, lhs, rhs in PLAN.combine_order(n):
        slots[dst] = f"({slots[lhs]}+{slots[rhs]})"
    return slots[0]


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
def test_kernel_tree_is_combine_order(n):
    assert _tree_expr(0, n) == _combine_order_expr(n)


@pytest.mark.parametrize("g", [2, 4, 8, 16])
def test_every_wire_and_world_is_the_same_expression(g):
    """Rank r's planes are operands r*C..r*C+C-1. Leaf wire (C=m leaves) flattens to the G-leaf tree directly;
    node wire (C=1) feeds rank r's subtree root, which must be the node over leaves [r*m, (r+1)*m)."""
    full = _tree_expr(0, g)
    w = 1
    while w <= g:
        m = g // w
        nodes = [_tree_expr(r * m, (r + 1) * m) for r in range(w)]
        # node wire: the tree over W node operands, with each operand substituted by its subtree
        expr = _tree_expr(0, w)
        for r in reversed(range(w)):
            expr = re.sub(rf"\bx{r}\b", nodes[r].replace("x", "L"), expr)
        assert expr.replace("L", "x") == full, (g, w)
        w *= 2


def test_kernel_tree_recursion_text():
    """The C++ recursion splits at MID=(LO+HI)/2 and adds left (lower indices) first, with __fadd_rn."""
    body = SRC[SRC.index("__device__ __forceinline__ float tree("):]
    body = body[: body.index("\n}\n")]
    assert "constexpr int MID = (LO + HI) / 2;" in body
    assert "tree<LO, MID, N>(v)" in body and "tree<MID, HI, N>(v)" in body
    assert "return __fadd_rn(a, b);" in body


def test_kernel_float_arithmetic_is_confined():
    """No float op outside tree/reduce_vec/conversions: no '*', no fma, no float '+' in the barrier code."""
    code = re.sub(r"//.*", "", SRC)
    assert "__fadd_rn" in code and code.count("__fadd_rn") == 1
    assert "__float2bfloat16_rn" in code and code.count("__float2bfloat16_rn") == 1
    for bad in ("__fmaf", "fmaf(", "__fmul", "__fsub", "__fdiv", "atomicAdd(", "__hadd", "__hfma"):
        assert bad not in code, bad
    # the only callers of the float core
    assert code.count("reduce_vec<W, C, Tin, Tout, VEC>(") == 4  # local, one-shot, two-shot push, two-shot pull


def test_policy_functions_take_only_rank_invariant_facts():
    for fn, params in (
        (OWNER.branch, ["n", "world", "planes", "elt", "oneshot_bytes"]),
        (OWNER.chunk_for, ["n", "world", "vec"]),
        (OWNER.grid_for, ["work_vecs", "max_blocks"]),
        (OWNER.vec_for, ["n"]),
    ):
        assert list(inspect.signature(fn).parameters) == params, fn.__name__


@pytest.mark.parametrize("world", [2, 4, 8, 16])
def test_chunks_cover_and_align(world):
    for n in (1, 7, 8, 5120, 5120 * 3, 256 * 5120, 1000003):
        vec = OWNER.vec_for(n)
        c = OWNER.chunk_for(n, world, vec)
        assert c % vec == 0 and c * world >= n and (c - vec) * world < n + world * vec
        g = OWNER.grid_for(c // vec, 32)
        assert 1 <= g <= 32


@pytest.mark.parametrize("tp,expect", [(2, (1, "fp32")), (4, (2, "bf16")), (8, (1, "bf16"))])
def test_wire_policy_g8_bf16(tp, expect):
    plan = PLAN.ReductionPlan(num_leaves=8, leaf_dtype=torch.bfloat16)
    planes, dt = OWNER.wire(plan, tp)
    assert (planes, {torch.bfloat16: "bf16", torch.float32: "fp32"}[dt]) == expect


def test_wire_policy_fp32_leaves_is_node_wire():
    plan = PLAN.ReductionPlan(num_leaves=8)
    for tp in (2, 4, 8):
        assert OWNER.wire(plan, tp) == (1, torch.float32)


def test_root_round_rule_matches_legacy_site():
    bf = PLAN.ReductionPlan(num_leaves=8, leaf_dtype=torch.bfloat16)
    f32 = PLAN.ReductionPlan(num_leaves=8)
    bias = object()
    assert OWNER.root_dtype_for(bf, None, torch.bfloat16) == torch.bfloat16
    assert OWNER.root_dtype_for(bf, bias, torch.bfloat16) == torch.bfloat16  # legacy rounds the root before bias
    assert OWNER.root_dtype_for(f32, None, torch.bfloat16) == torch.bfloat16  # legacy FUSED_ROOT_CAST case
    assert OWNER.root_dtype_for(f32, bias, torch.bfloat16) == torch.float32  # bias joins the fp32 root
    assert OWNER.root_dtype_for(f32, None, torch.float32) == torch.float32


def test_barrier_uses_plain_acquire_loads_and_fixed_grid():
    """The fused-barrier regression: one handshake per 1024 elements with RMW polling. Here the poll is a
    plain ld.acquire and the grid is capped by grid_for (default 132 blocks), independent of n."""
    assert "ld.acquire.sys.global.u32" in SRC and "st.release.sys.global.u32" in SRC
    assert "atom" not in re.sub(r"//.*", "", SRC)
    assert OWNER.grid_for(10**9, OWNER._DEFAULT_MAX_BLOCKS[4]) == OWNER._DEFAULT_MAX_BLOCKS[4]
