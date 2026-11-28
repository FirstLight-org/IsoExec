"""CPU: the sparse attention's schedule parameters are the schedule policy's output (rule 1).

The tile reducer has no merge, so the policy returns segments of the reducer's block folded by ONE continued chain
(``schedule_policy.no_merge``); nothing else is a schedule decision of this operator, and the policy is not called on
any forward call. This module fails when a leaf's block, the group table or the list layout departs from that.

Inside one block the dual-rail leaf's QK product keeps TWO tensor-core accumulator chains (the rails), so its lane
count is ``schedule_policy.lanes(2)``: rail r takes the 64-dim chunks r, r + 2, r + 4, r + 6 of the 512 query dims.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from isoexec.ops import schedule_policy
from isoexec.ops.attention.sparse_sm100 import domain, plan, spec

PKG = Path(spec.__file__).parent
ATT = PKG.parent


def test_rule_1_is_blocks_of_64_positions_in_one_continued_chain():
    seg = schedule_policy.no_merge(spec.TILE_SLOTS)
    assert schedule_policy.no_merge.__module__ == schedule_policy.__name__ and schedule_policy.VERSION
    assert (seg.S, seg.G, seg.merge) == (64, None, "continued")
    assert domain.BLOCK == seg.S


def test_both_leaves_fold_blocks_of_the_policys_length():
    dual = (PKG / "csrc" / "dualrail" / "leaf.cuh").read_text()
    assert int(re.search(r"static constexpr int B_TOPK = (\d+);", dual).group(1)) == schedule_policy.no_merge(spec.TILE_SLOTS).S
    for plan_file in ("persist.cuh", "split.cuh"):
        assert "B_TOPK" in (PKG / "csrc" / "dualrail" / plan_file).read_text()
    seg = (PKG / "csrc" / "attn" / "leaf.cuh").read_text()
    assert re.search(r"TILE = (\d+)", seg) and int(re.search(r"TILE = (\d+)", seg).group(1)) == 64


#: Independent accumulators the dual-rail QK product keeps for one score: two MMA chains (rail 0, rail 1).
QK_ACCUMULATORS = 2
#: Query dims one K step of a rail consumes together (one SW128 tile column block).
QK_CHUNK = 64


def test_the_dual_rail_qk_product_runs_the_policys_two_lanes():
    """``lanes(2)``: the kernel's rail count (the ``/2`` of ``D_Q/64/2``, the ``D_Q/2`` query fragment and the
    ``B_TOPK*2`` rows of the dual-GEMM key layout) is the policy's lane count for two accumulators, and rail r owns
    the chunks r, r + L, ... of the query dims. Fails if a source's constant stops agreeing."""
    lanes = schedule_policy.lanes(QK_ACCUMULATORS)
    assert lanes == 2
    dual = (PKG / "csrc" / "dualrail" / "leaf.cuh").read_text()
    d_q = int(re.search(r"static constexpr int D_Q = (\d+)", dual).group(1))
    chunks = d_q // QK_CHUNK
    assert d_q % QK_CHUNK == 0 and chunks % lanes == 0
    # the key layout of the dual GEMM stacks one copy of the tile's rows per rail
    layout = re.search(r"using SmemLayoutKTiles_DualGemm_SW128 = .*?Shape<Int<B_TOPK\*(\d+)>, Int<(\d+)\*NUM_TILES>>", dual)
    assert layout and (int(layout.group(1)), int(layout.group(2))) == (lanes, QK_CHUNK)
    for name in ("leaf.cuh", "split.cuh"):
        src = (PKG / "csrc" / "dualrail" / name).read_text()
        rails = re.findall(r"SmemLayoutKTiles_DualGemm_SW128<D_Q/(\d+)/(\d+)>", src)
        assert rails and {(int(c), int(r)) for c, r in rails} == {(QK_CHUNK, lanes)}, name
        halves = re.findall(r"make_fragment_A\(partition_shape_A\(tiled_mma_P, Shape<Int<MMA_M>, Int<D_Q/(\d+)>>", src)
        assert halves and {int(h) for h in halves} == {lanes}, name
    # the stated membership: rail r = one chain over the chunks r, r + 2, r + 4, r + 6, ascending
    stated = re.search(r"64-dim chunks \{r, r\+(\d+), r\+(\d+), r\+(\d+)\}", dual)
    members = [[c for c in range(chunks) if c % lanes == r] for r in range(lanes)]
    assert members == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert stated and [int(g) for g in stated.groups()] == [m - members[0][0] for m in members[0][1:]]
    assert "QK, dual rail: |0|2|4|6| over |1|3|5|7| (64-dim chunks), one .ws chain per rail" in dual
    persist = (PKG / "csrc" / "dualrail" / "persist.cuh").read_text()
    assert "rail r over 64-dim chunks {r, r+2} then {r+4, r+6}, one K16 chain" in persist
    assert {int(r) for r in re.findall(r"Tile<Int<B_TOPK\*(\d+)>, Int<D_V/4>>", persist)} == {lanes}


def test_the_chain_is_never_cut_into_groups():
    """One continued chain: every declared geometry has G = 1, and a flat chain over G > 1 groups is refused."""
    assert set(spec.LIST_GROUPS.values()) == {1}
    for (h, k) in spec.LIST_GROUPS:
        assert plan.groups_for(h, k) == 1 and k % spec.TILE_SLOTS == 0


def test_the_engine_paths_pass_the_selected_length_and_no_call_path_reads_the_policy():
    cache = ATT / "deepseek_v41_cache_sm100"
    for name in ("paged.py", "ced.py"):
        tree = ast.parse((cache / name).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "dualrail_attn"]
        assert calls, name
        for c in calls:
            if any(k.arg == "extra_lists" for k in c.keywords):
                assert any(k.arg == "extra_topk_length" for k in c.keywords), (name, c.lineno)
    for path in [*PKG.glob("*.py"), cache / "paged.py", cache / "ced.py", cache / "ops.py"]:
        assert "schedule_policy" not in path.read_text(), path
