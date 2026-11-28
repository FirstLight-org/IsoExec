"""CPU: the kernel's schedule parameters are the schedule policy's output on the deployed shapes.

``isoexec.ops.schedule_policy`` is the one place a segment count or length comes
from. This module fails when

* ``fwd.segment_span`` (the span the trainer adapter and the SM100 resolution admit) differs from rule 3
  recomputed here from the hardware specifications of this reducer and spec.py's declarations, with no
  target batch declared (the default) and with a target batch of 1, 16, 64 and 128 declared;
* the trainer adapter or the resolution accepts another span;
* a segment table, its digest or a literal span comes back.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from isoexec.ops import schedule_policy
from isoexec.ops.attention.mla_sm100 import fwd, spec

PACKAGE = Path(fwd.__file__).parent


#: target batch (``None``: none declared, the default) -> (G, S) at the longest context of 10240, written out.
DEPLOYED = {None: (1, 10240), 1: (32, 320), 16: (8, 1280), 64: (2, 5120), 128: (1, 10240)}


def _workers(batch: int | None, heads: int) -> int:
    """p written out independently of schedule_policy: 1 with no target batch, else the SMs free per output."""
    return 1 if batch is None else max(1, fwd.DEVICE_SMS // (batch * -(-heads // 64)))


def _rule3(n: int, b: int, min_segment: int, p: int) -> tuple[int, int]:
    """Rule 3 written out independently of schedule_policy (no max_segments): (G, S)."""
    g = max(1, min(p, n // min_segment))
    while g & (g - 1):
        g &= g - 1
    return g, b * -(-n // (b * g))


@pytest.fixture
def declare(monkeypatch):
    """Declare a target batch for one test and bind the schedule; the default binding comes back afterwards."""
    def _declare(batch):
        monkeypatch.setattr(spec, "TARGET_BATCH", batch)
        return fwd.bind_schedule()

    monkeypatch.setattr(spec, "TARGET_BATCH", None)
    fwd.bind_schedule()
    yield _declare
    monkeypatch.undo()
    fwd.bind_schedule()


def test_the_recorded_policy_version_and_no_target_batch_gives_one_worker():
    from isoexec.ops.attention import target_batch

    assert target_batch.schedule_record()["policy"] == schedule_policy.VERSION
    assert schedule_policy.workers(fwd.DEVICE_SMS) == schedule_policy.workers(fwd.DEVICE_SMS, None) == 1


def test_hardware_specifications_and_declarations():
    assert spec.PAGE == fwd.M_BLOCK == 64
    assert fwd.SOFTMAX_SHORTEST_SEGMENT == 256 and fwd.SOFTMAX_SHORTEST_SEGMENT % spec.PAGE == 0
    assert fwd.DEVICE_SMS == 148
    assert fwd.occupancy(20, 64) == 64 and fwd.occupancy(64, 3) == 3 and fwd.occupancy(65, 3) == 6
    assert (spec.LONGEST_CONTEXT, spec.MODEL_HEADS) == (10240, 20)


def test_the_default_declares_no_target_batch_and_gives_one_segment(declare):
    """No target batch: p = workers(sms) = 1, G = 1, one segment of the longest context, at every head count."""
    assert spec.TARGET_BATCH is None
    for heads in (5, 10, 20, 64, 65, 128):
        got = fwd.schedule_for(spec.LONGEST_CONTEXT, model_heads=heads)
        assert (got.G, got.S, got.merge) == (1, spec.LONGEST_CONTEXT, "tree")
    assert fwd.schedule_for(spec.LONGEST_CONTEXT) == fwd.schedule_for(spec.LONGEST_CONTEXT, target_batch=None)
    assert fwd.segment_span() == 10240


@pytest.mark.parametrize("batch", list(DEPLOYED), ids=lambda b: f"target_{b}")
def test_the_deployed_span_is_the_policys(batch, declare):
    """Default and declared target batches: the span the kernel is handed == the policy's == rule 3."""
    bound = declare(batch)
    g, s = _rule3(spec.LONGEST_CONTEXT, spec.PAGE, fwd.SOFTMAX_SHORTEST_SEGMENT, _workers(batch, spec.MODEL_HEADS))
    got = fwd.schedule_for(spec.LONGEST_CONTEXT)
    assert (got.G, got.S, got.merge) == (g, s, "tree") == (*DEPLOYED[batch], "tree")
    occupied = None if batch is None else fwd.occupancy(spec.MODEL_HEADS, batch)
    want = schedule_policy.reduce(spec.LONGEST_CONTEXT, b=spec.PAGE, min_segment=fwd.SOFTMAX_SHORTEST_SEGMENT,
                                  p=schedule_policy.workers(fwd.DEVICE_SMS, occupied))
    assert got == want
    assert fwd.segment_span() == bound == s
    # a rank's head count (20, 10, 5 at TP 1, 2, 4) does not reach the span
    assert {fwd.seg_tokens_for(h, spec.LONGEST_CONTEXT) for h in (20, 10, 5, 1)} == {s}
    assert s // fwd.SOFTMAX_SHORTEST_SEGMENT >= 1 and spec.LONGEST_CONTEXT / s <= fwd.MAX_TREE_LEAVES


@pytest.mark.parametrize("heads", (5, 10, 20, 64, 65, 128))
def test_every_declaration_follows_rule_3(heads):
    for batch in (None, 1, 2, 4, 16, 64, 128, 512):
        for n in (1024, 4096, 6000, 8192, 10240, 20000, 32768, 131072, 202752):
            g, s = _rule3(n, 64, 256, _workers(batch, heads))
            got = fwd.schedule_for(n, target_batch=batch, model_heads=heads)
            assert (got.G, got.S) == (g, s), (heads, batch, n)
            assert s % spec.PAGE == 0 and g * s >= n
            assert fwd.seg_tokens_for(3, n, target_batch=batch, model_heads=heads) == s
            if batch is None:
                assert (g, s) == (1, 64 * -(-n // 64))


def test_refusals():
    with pytest.raises(ValueError):
        fwd.seg_tokens_for(0, 8192)
    with pytest.raises(ValueError):
        fwd.seg_tokens_for(20, 0)
    with pytest.raises(ValueError):
        fwd.occupancy(20, 0)
    with pytest.raises(ValueError):
        fwd.seg_tokens_for(20, 8192, target_batch=0)
    with pytest.raises(ValueError):
        fwd.seg_tokens_for(20, 8192, target_batch="smallest")


def test_two_parties_with_different_target_batches_refuse_to_start(declare, monkeypatch):
    """Two start-up comparisons: the schedule record of the first weight sync, and the SM100 resolution of a model
    whose ``seg_tokens`` was written under another declaration."""
    import json

    from isoexec.checks import Refusal
    from isoexec.ops import _dispatch
    from isoexec.ops._resolve import ResolutionError
    from isoexec.ops.attention import resolve as attention_resolve
    from isoexec.ops.attention import target_batch

    declare(None)
    trainer = json.loads(json.dumps(target_batch.schedule_record()))
    assert trainer["mla_sm100"] == {"target_batch": None, "longest_context": 10240, "model_heads": 20,
                                    "seg_tokens": 10240}
    for batch, field in ((64, "seg_tokens"), (16, "seg_tokens"), (1, "seg_tokens"), (128, "target_batch")):
        declare(batch)
        engine = json.loads(json.dumps(target_batch.schedule_record()))
        with pytest.raises(Refusal, match=f"attention_schedule.mla_sm100.{field}"):
            target_batch.require_same_schedule(trainer, engine, who="engine")
    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    declare(None)
    model_seg = fwd.segment_span()                        # the model config both parties load, written at the default
    assert attention_resolve.resolve("attention.core.mla", site="t", seg_tokens=model_seg).inputs["seg_tokens"] == 10240
    for batch in (64, 16, 1):
        declare(batch)                                    # a party launched with another target batch
        with pytest.raises(ResolutionError):
            attention_resolve.resolve("attention.core.mla", site="t", seg_tokens=model_seg)


def test_the_trainer_adapter_refuses_another_span(declare):
    import torch

    from isoexec.ops.attention.mla_sm100.ops import mla_sm100_attention

    if not torch.cuda.is_available():
        pytest.skip("the adapter checks its operands' device before the span")
    h, t = 2, 3
    mk = lambda *shape: torch.zeros(*shape, device="cuda", dtype=torch.bfloat16)
    args = (mk(t, h, 8), mk(t, h, 64), mk(t, 512), mk(t, 64), mk(h, 8, 512), mk(h, 16, 512))
    for seg in (512, 1024, 5120, fwd.segment_span() + 64):
        with pytest.raises(ValueError, match="schedule policy"):
            mla_sm100_attention(*args, seq_lens=(t,), scale=0.1, seg_tokens=seg)


def test_the_sm100_resolution_refuses_another_span(monkeypatch, declare):
    from isoexec.ops import _dispatch
    from isoexec.ops._resolve import ResolutionError
    from isoexec.ops.attention import resolve as attention_resolve

    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    bound = attention_resolve.resolve("attention.core.mla", site="t", seg_tokens=fwd.segment_span())
    assert bound.impl == "sm100_absorbed_segtree" and dict(bound.inputs) == {"seg_tokens": fwd.segment_span()}
    assert bound.fn.seg_tokens == fwd.segment_span()
    for seg in (512, 1024, 5120, None):
        with pytest.raises(ResolutionError):
            attention_resolve.resolve("attention.core.mla", site="t", seg_tokens=seg)


def test_the_policy_is_evaluated_when_the_schedule_is_bound_and_never_on_a_call(monkeypatch, declare):
    calls = []
    real = schedule_policy.reduce
    monkeypatch.setattr(schedule_policy, "reduce", lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    assert [fwd.segment_span() for _ in range(3)] == [10240] * 3 and not calls, "segment_span called the policy"
    assert fwd.bind_schedule() == 10240 and len(calls) == 1
    tree = ast.parse((PACKAGE / "fwd.py").read_text())
    for name in ("segment_span", "mla_sm100_fwd", "mla_sm100_plan", "_run"):
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        names = {n.attr if isinstance(n, ast.Attribute) else getattr(n, "id", "") for n in ast.walk(fn)}
        assert not names & {"schedule_policy", "schedule_for", "seg_tokens_for", "bind_schedule"}, (name, names)


def test_no_literal_span():
    tree = ast.parse((PACKAGE / "fwd.py").read_text())
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name in ("schedule_for", "seg_tokens_for", "segment_span", "bind_schedule")):
        literals = [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, int)
                    and not isinstance(n.value, bool) and n.value > 1]
        assert not literals, (fn.name, literals)
