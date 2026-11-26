"""CPU: the kernel's schedule parameters are the schedule policy's output on the deployed shapes.

``isoexec.ops.schedule_policy`` (design version 7.2) is the one place a segment count or length comes
from. This module fails when

* ``ops.segment_span`` (the span every call site hands the kernel) differs from rule 3 recomputed here
  from the policy's cost parameters for this reducer and spec.py's declarations, at both head dims and
  for the KV head counts of every release model, with no target batch declared (the default) and with a
  target batch of 1, 16, 64 and 128 declared;
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


#: The target batches this module declares; ``None`` is the default (none declared).
TARGETS = (None, 1, 16, 64, 128)

#: (target batch, head dim) -> (G, S) at the longest context of 10240 with the model's 2 KV heads, written out.
DEPLOYED = {
    (None, 256): (1, 10240), (None, 128): (1, 10240),
    (1, 256): (16, 640), (1, 128): (16, 640),
    (16, 256): (2, 5120), (16, 128): (4, 2560),
    (64, 256): (1, 10240), (64, 128): (1, 10240),
    (128, 256): (1, 10240), (128, 128): (1, 10240),
}


def _workers(batch: int | None, hk: int, d: int) -> int:
    """p written out independently of schedule_policy: 1 with no target batch, else the SMs free per output."""
    return 1 if batch is None else max(1, ops.DEVICE_SMS // (batch * hk * ops.SMS_PER_ITEM[d]))


def _rule3(n: int, b: int, L: int, p: int, cap: int) -> tuple[int, int]:
    """Rule 3 written out independently of schedule_policy: (G, S)."""
    g = max(1, min(p, cap, n // L))
    while g & (g - 1):
        g &= g - 1
    return g, b * -(-n // (b * g))


@pytest.fixture
def declare(monkeypatch):
    """Declare a target batch for one test and bind the schedule; the default binding comes back afterwards."""
    def _declare(batch):
        monkeypatch.setattr(spec, "TARGET_BATCH", batch)
        return ops.bind_schedule()

    monkeypatch.setattr(spec, "TARGET_BATCH", None)
    ops.bind_schedule()
    yield _declare
    monkeypatch.undo()
    ops.bind_schedule()


def test_the_policy_module_is_version_7_2():
    assert schedule_policy.VERSION == "7.2"
    assert schedule_policy.workers(ops.DEVICE_SMS) == schedule_policy.workers(ops.DEVICE_SMS, None) == 1


def test_published_constants():
    assert ops.KEY_TILE == 128
    assert ops.SOFTMAX_SHORTEST_SEGMENT == 512 and ops.SOFTMAX_SHORTEST_SEGMENT % ops.KEY_TILE == 0
    params = (PACKAGE / "csrc" / "params.h").read_text()
    assert ops.SOFTMAX_SEGMENT_CAP == int(re.search(r"FOLD_MAX_RUN = (\d+);", params).group(1)) == 16
    assert ops.SMS_PER_ITEM == {128: 1, 256: 2} and set(ops.SMS_PER_ITEM) == set(spec.HEAD_DIMS)
    assert ops.DEVICE_SMS == 148
    assert ops.occupancy(256, 4, 128) == 128 * 4 * 2 and ops.occupancy(128, 1, 1) == 1


def test_declarations_and_the_default(monkeypatch):
    from isoexec.ops.attention import target_batch

    assert (spec.LONGEST_CONTEXT, spec.MODEL_KV_HEADS) == (10240, 2)
    # the target batch is optional and absent by default: unset and empty both declare none
    monkeypatch.delenv(target_batch.FLAG, raising=False)
    assert target_batch.declared() is None
    monkeypatch.setenv(target_batch.FLAG, "")
    assert target_batch.declared() is None
    monkeypatch.setenv(target_batch.FLAG, "16")
    assert target_batch.declared() == 16
    for bad in ("0", "-4", "many", "1.5"):
        monkeypatch.setenv(target_batch.FLAG, bad)
        with pytest.raises(ValueError, match=target_batch.FLAG):
            target_batch.declared()


def test_the_default_declares_no_target_batch_and_gives_one_segment(declare):
    """No target batch: p = workers(sms) = 1, G = 1, one segment of the longest context, at every KV head count."""
    assert spec.TARGET_BATCH is None
    for d in spec.HEAD_DIMS:
        for hk in (1, 2, 4, 8, 16):
            got = ops.schedule_for(d, spec.LONGEST_CONTEXT, kv_heads=hk)
            assert (got.G, got.S, got.merge) == (1, spec.LONGEST_CONTEXT, "tree")
        assert ops.schedule_for(d, spec.LONGEST_CONTEXT) == ops.schedule_for(d, spec.LONGEST_CONTEXT, target_batch=None)
        assert ops.segment_span(d) == 10240


@pytest.mark.parametrize("batch", TARGETS, ids=lambda b: f"target_{b}")
@pytest.mark.parametrize(("label", "hq", "hk", "d"), MODELS, ids=[m[0] for m in MODELS])
def test_the_deployed_span_is_the_policys(label, hq, hk, d, batch, declare):
    """Default and declared target batches, every release model: the kernel's span == the policy's == rule 3."""
    bound = declare(batch)
    p = _workers(batch, hk, d)
    g, s = _rule3(spec.LONGEST_CONTEXT, ops.KEY_TILE, ops.SOFTMAX_SHORTEST_SEGMENT, p, ops.SOFTMAX_SEGMENT_CAP)
    got = ops.schedule_for(d, spec.LONGEST_CONTEXT, kv_heads=hk)
    assert (got.G, got.S, got.merge) == (g, s, "tree")
    occupied = None if batch is None else ops.occupancy(d, hk, batch)
    want = schedule_policy.reduce(spec.LONGEST_CONTEXT, b=ops.KEY_TILE, L=ops.SOFTMAX_SHORTEST_SEGMENT,
                                  p=schedule_policy.workers(ops.DEVICE_SMS, occupied), cap=ops.SOFTMAX_SEGMENT_CAP)
    assert got == want
    # the span the call sites hand the kernel is the one at the declared model KV heads (spec.MODEL_KV_HEADS)
    deployed = ops.schedule_for(d, spec.LONGEST_CONTEXT)
    assert (deployed.G, deployed.S) == DEPLOYED[(batch, d)]
    assert ops.segment_span(d) == bound[d] == ops.seg_for(d, spec.LONGEST_CONTEXT) == deployed.S
    if hk == spec.MODEL_KV_HEADS:
        assert got == deployed


@pytest.mark.parametrize("d", (128, 256))
@pytest.mark.parametrize("hk", (1, 2, 4, 8, 16))
def test_every_declaration_follows_rule_3(d, hk):
    for batch in (None, 1, 2, 4, 8, 16, 64, 128, 256):
        for n in (1024, 2048, 4096, 6000, 8192, 10240, 16384, 32768, 131072, 262144):
            g, s = _rule3(n, 128, 512, _workers(batch, hk, d), 16)
            got = ops.schedule_for(d, n, target_batch=batch, kv_heads=hk)
            assert (got.G, got.S) == (g, s), (d, hk, batch, n)
            assert s % ops.KEY_TILE == 0 and g * s >= n and ops.seg_for(d, n, target_batch=batch, kv_heads=hk) == s
            if batch is None:
                assert (g, s) == (1, 128 * -(-n // 128))


def test_the_policy_at_a_target_batch_of_one_is_the_span_the_table_had():
    """The removed max-context table was this rule at a target batch of 1 (cap 16 binding)."""
    for d in (128, 256):
        assert [ops.seg_for(d, n, target_batch=1) for n in (8192, 32768, 131072)] == [512, 2048, 8192]
        assert ops.seg_for(d, 2048, target_batch=1) == 512   # the table had 128: below L, dropped


def test_the_span_fits_the_decode_leafs_tile_cache(declare):
    """``csrc/decode_d256.cu`` refuses a span of more than MAX_SLOT_CACHE tiles (one CTA walks one segment). The
    default (no target batch) gives the longest span a declaration can: it fits, so every target batch does."""
    source = (PACKAGE / "csrc" / "decode_d256.cu").read_text()
    cache = int(re.search(r"MAX_SLOT_CACHE = (\d+);", source).group(1))
    assert "seg_tiles <= MAX_SLOT_CACHE" in source
    for batch in TARGETS:
        assert declare(batch)[256] // ops.KEY_TILE <= cache == 128
        assert ops.segment_span(256) <= ops.seg_for(256, spec.LONGEST_CONTEXT, target_batch=None)


def test_refusals():
    with pytest.raises(ValueError):
        ops.seg_for(64, 8192)
    with pytest.raises(ValueError):
        ops.seg_for(128, 0)
    with pytest.raises(ValueError):
        ops.occupancy(128, 1, 0)
    with pytest.raises(ValueError):
        ops.seg_for(128, 8192, target_batch=0)
    with pytest.raises(ValueError):
        ops.seg_for(128, 8192, target_batch="smallest")


def test_two_parties_with_different_target_batches_refuse_to_start(declare):
    """The record a process bound, compared the way the first weight sync compares it (integrations/verl)."""
    import json

    from isoexec.checks import Refusal
    from isoexec.ops.attention import target_batch

    declare(None)
    trainer = json.loads(json.dumps(target_batch.schedule_record()))     # as it crosses the acknowledgement file
    assert trainer["policy"] == "7.2" and trainer["gqa_sm100"]["target_batch"] is None
    assert trainer["gqa_sm100"]["seg_tokens"] == {"128": 10240, "256": 10240}
    target_batch.require_same_schedule(trainer, json.loads(json.dumps(target_batch.schedule_record())), who="engine")
    for batch, field in ((16, "gqa_sm100.seg_tokens.128"), (1, "gqa_sm100.seg_tokens.128"),
                         (128, "gqa_sm100.target_batch")):
        declare(batch)
        engine = json.loads(json.dumps(target_batch.schedule_record()))
        with pytest.raises(Refusal, match=f"attention_schedule.{field}"):
            target_batch.require_same_schedule(trainer, engine, who="engine")
    # 128 and none give the same span today and are still refused: the declaration itself is compared
    with pytest.raises(Refusal, match="attention_schedule.missing"):
        target_batch.require_same_schedule(trainer, None, who="engine")


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


def test_the_policy_is_evaluated_when_the_schedule_is_bound_and_never_on_a_call(monkeypatch, declare):
    """``segment_span`` (what every call site reads) is a dict lookup; ``schedule_policy`` runs in ``bind_schedule``."""
    calls = []
    real = schedule_policy.reduce
    monkeypatch.setattr(schedule_policy, "reduce", lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    for d in spec.HEAD_DIMS:
        for _ in range(3):
            assert ops.segment_span(d) == 10240
    assert not calls, "segment_span called the policy"
    assert ops.bind_schedule() == {128: 10240, 256: 10240} and len(calls) == len(spec.HEAD_DIMS)
    tree = ast.parse((PACKAGE / "ops.py").read_text())
    span = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "segment_span")
    names = {n.attr if isinstance(n, ast.Attribute) else getattr(n, "id", "") for n in ast.walk(span)}
    assert not names & {"schedule_policy", "schedule_for", "seg_for", "reduce", "workers"}, names
    with pytest.raises(ValueError):
        ops.segment_span(64)


def test_no_table_and_no_literal_span():
    assert not hasattr(ops, "SEG_TOKENS") and not hasattr(spec, "SEG_CONTEXT")
    tree = ast.parse((PACKAGE / "ops.py").read_text())
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ("schedule_for", "seg_for", "segment_span", "bind_schedule")):
        literals = [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, int)
                    and not isinstance(n.value, bool)]
        assert not literals, (fn.name, literals)
