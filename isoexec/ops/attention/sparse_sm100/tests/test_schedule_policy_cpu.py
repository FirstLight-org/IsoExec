"""CPU: the sparse attention's schedule parameters are the schedule policy's output (design 7.1, rule 1).

The tile reducer has no merge, so the policy returns segments of the reducer's block folded by ONE continued chain
(``schedule_policy.no_merge``); nothing else is a schedule decision of this operator, and the policy is not called on
any forward call. This module fails when a leaf's block, the group table or the list layout departs from that.
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
    assert schedule_policy.VERSION == "7.1"
    assert (seg.S, seg.G, seg.merge) == (64, None, "continued")
    assert domain.BLOCK == seg.S


def test_both_leaves_fold_blocks_of_the_policys_length():
    dual = (PKG / "csrc" / "dualrail" / "leaf.cuh").read_text()
    assert int(re.search(r"static constexpr int B_TOPK = (\d+);", dual).group(1)) == schedule_policy.no_merge(spec.TILE_SLOTS).S
    for plan_file in ("persist.cuh", "split.cuh"):
        assert "B_TOPK" in (PKG / "csrc" / "dualrail" / plan_file).read_text()
    seg = (PKG / "csrc" / "attn" / "leaf.cuh").read_text()
    assert re.search(r"TILE = (\d+)", seg) and int(re.search(r"TILE = (\d+)", seg).group(1)) == 64


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
