"""CPU: the kernel's schedule parameters are the schedule policy's output on the deployed shapes.

``isoexec.ops.schedule_policy`` (design version 7.1) is the one place a segment count or length comes
from. This module fails when

* ``ops.segment_span`` (the span every call site hands the kernel) differs from rule 3 recomputed here
  from the reducer's published constants and spec.py's declarations, at both head dims and for the KV
  head counts of every release model;
* a call site passes the kernel a span that is not ``segment_span``;
* a segment table or a literal span comes back.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from isoexec.ops import schedule_policy
from isoexec.ops.attention.gqa_sm100 import ops, spec

PACKAGE = Path(ops.__file__).parent

#: (model, query heads, KV heads, head dim): the release models' attention layers (tests/test_seam_gpu.py).
MODELS = [
    ("qwen3.5-35b-a3b", 16, 2, 256),
    ("qwen3.5-9b", 16, 4, 256),
    ("qwen3.6-27b", 24, 4, 256),
    ("qwen3.5-0.8b", 8, 2, 256),
    ("qwen3-8b", 32, 8, 128),
    ("qwen1.5-moe-a2.7b", 16, 16, 128),
]


def _rule3(n: int, b: int, L: int, p: int, cap: int) -> tuple[int, int]:
    """Rule 3 written out independently of schedule_policy: (G, S)."""
    g = max(1, min(p, cap, n // L))
    while g & (g - 1):
        g &= g - 1
    return g, b * -(-n // (b * g))


def test_the_policy_module_is_version_7_1():
    assert schedule_policy.VERSION == "7.1"


def test_published_constants():
    assert ops.KEY_TILE == 128
    assert ops.SOFTMAX_SHORTEST_SEGMENT == 512 and ops.SOFTMAX_SHORTEST_SEGMENT % ops.KEY_TILE == 0
    params = (PACKAGE / "csrc" / "params.h").read_text()
    assert ops.SOFTMAX_SEGMENT_CAP == int(re.search(r"FOLD_MAX_RUN = (\d+);", params).group(1)) == 16
    assert ops.SMS_PER_ITEM == {128: 1, 256: 2} and set(ops.SMS_PER_ITEM) == set(spec.HEAD_DIMS)
    assert ops.DEVICE_SMS == 148
    assert ops.occupancy(256, 4, 128) == 128 * 4 * 2 and ops.occupancy(128, 1, 1) == 1


def test_declarations():
    assert (spec.LONGEST_CONTEXT, spec.SMALLEST_BATCH, spec.MODEL_KV_HEADS) == (10240, 128, 2)


@pytest.mark.parametrize(("label", "hq", "hk", "d"), MODELS, ids=[m[0] for m in MODELS])
def test_the_deployed_span_is_the_policys(label, hq, hk, d):
    p = max(1, ops.DEVICE_SMS // (spec.SMALLEST_BATCH * hk * ops.SMS_PER_ITEM[d]))
    g, s = _rule3(spec.LONGEST_CONTEXT, ops.KEY_TILE, ops.SOFTMAX_SHORTEST_SEGMENT, p, ops.SOFTMAX_SEGMENT_CAP)
    got = ops.schedule_for(d, spec.LONGEST_CONTEXT, kv_heads=hk)
    assert (got.G, got.S, got.merge) == (g, s, "tree") == (1, 10240, "tree")
    want = schedule_policy.reduce(spec.LONGEST_CONTEXT, b=ops.KEY_TILE, L=ops.SOFTMAX_SHORTEST_SEGMENT,
                                  p=schedule_policy.workers(ops.DEVICE_SMS, ops.occupancy(d, hk, spec.SMALLEST_BATCH)),
                                  cap=ops.SOFTMAX_SEGMENT_CAP)
    assert got == want
    # the span the call sites hand the kernel does not depend on which release model declared its KV heads
    assert ops.segment_span(d) == ops.seg_for(d, spec.LONGEST_CONTEXT) == s


@pytest.mark.parametrize("d", (128, 256))
@pytest.mark.parametrize("hk", (1, 2, 4, 8, 16))
def test_every_declaration_follows_rule_3(d, hk):
    for batch in (1, 2, 4, 8, 16, 64, 128, 256):
        for n in (1024, 2048, 4096, 6000, 8192, 16384, 32768, 131072, 262144):
            p = max(1, ops.DEVICE_SMS // (batch * hk * ops.SMS_PER_ITEM[d]))
            g, s = _rule3(n, 128, 512, p, 16)
            got = ops.schedule_for(d, n, min_batch=batch, kv_heads=hk)
            assert (got.G, got.S) == (g, s), (d, hk, batch, n)
            assert s % ops.KEY_TILE == 0 and g * s >= n and ops.seg_for(d, n, min_batch=batch, kv_heads=hk) == s


def test_the_policy_at_batch_one_is_the_span_the_table_had():
    """The removed max-context table was this rule at a smallest batch of 1 (cap 16 binding)."""
    for d in (128, 256):
        assert [ops.seg_for(d, n, min_batch=1) for n in (8192, 32768, 131072)] == [512, 2048, 8192]
        assert ops.seg_for(d, 2048, min_batch=1) == 512   # the table had 128: below L_r, dropped


def test_the_declared_span_fits_the_decode_leafs_tile_cache():
    """``csrc/decode_d256.cu`` refuses a span of more than MAX_SLOT_CACHE tiles (one CTA walks one segment)."""
    source = (PACKAGE / "csrc" / "decode_d256.cu").read_text()
    cache = int(re.search(r"MAX_SLOT_CACHE = (\d+);", source).group(1))
    assert "seg_tiles <= MAX_SLOT_CACHE" in source
    assert ops.segment_span(256) // ops.KEY_TILE <= cache == 128


def test_refusals():
    with pytest.raises(ValueError):
        ops.seg_for(64, 8192)
    with pytest.raises(ValueError):
        ops.seg_for(128, 0)
    with pytest.raises(ValueError):
        ops.occupancy(128, 1, 0)


def _kernel_calls(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) == "gqa_sm100_fwd":
            yield node


@pytest.mark.parametrize("module", ("ops.py", "context_parallel.py", "integration.py"))
def test_every_call_site_hands_the_kernel_segment_span(module):
    calls = list(_kernel_calls(PACKAGE / module))
    if module != "integration.py":
        assert calls, module
    for call in calls:
        seg = next(k.value for k in call.keywords if k.arg == "seg_tokens")
        assert isinstance(seg, ast.Call) and getattr(seg.func, "attr", getattr(seg.func, "id", "")) == "segment_span", \
            ast.dump(seg)


def test_no_table_and_no_literal_span():
    assert not hasattr(ops, "SEG_TOKENS") and not hasattr(spec, "SEG_CONTEXT")
    tree = ast.parse((PACKAGE / "ops.py").read_text())
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ("schedule_for", "seg_for", "segment_span")):
        literals = [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, int)
                    and not isinstance(n.value, bool)]
        assert not literals, (fn.name, literals)
