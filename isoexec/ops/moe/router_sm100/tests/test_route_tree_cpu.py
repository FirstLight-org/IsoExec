"""CPU contract of the segment-root router path: its segment count is the declared logits program's, and
``route`` keeps the fused kernel wherever the tree form does not apply."""
import torch

from isoexec.ops.moe.router_sm100 import ops, spec


def test_declared_segments_are_the_schedule_policys():
    """The logits program's segment count is rule 3 of ``schedule_policy`` (block 16, no published shortest
    segment): one segment for every hidden size, with no table keyed on H."""
    from isoexec.ops import schedule_policy as P

    assert ops.LOGITS_REDUCER_BLOCK == spec.LOGIT_REDUCER_BLOCK == 16
    assert ops.LOGITS_REDUCER_SHORTEST is None and spec.LOGIT_REDUCER_SHORTEST is None
    for h in (2048, 4096, 5120, 7168, 64 * 6, 64 * 3):
        want = P.reduce(h, b=16)
        assert (want.G, want.S, want.merge) == (1, h, "tree")
        assert ops.declared_segments(h) == spec.logit_segments(h) == want.G
    assert not hasattr(spec, "LOGIT_SEGMENTS")
    assert ops.RENORM_SCHEDULE == P.selected() == P.Segments(1, None, "chain")
    assert spec.RENORM_FOLD == "ascending_expert_id_chain_from_zero_plus_1e-20"


def test_route_keeps_the_fused_kernel_outside_decode_rows(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "route_fused", lambda x, w, **kw: calls.append(x.shape[0]) or "fused")
    monkeypatch.setattr(ops, "route_select_tree", lambda *a, **k: (_ for _ in ()).throw(AssertionError("tree")))
    w = torch.zeros(256, 4096, dtype=torch.bfloat16)
    kw = dict(routing=ops.SQRTSOFTPLUS, top_k=6, num_shared=1)
    assert ops.route(torch.zeros(4, 4096, dtype=torch.bfloat16), w, **kw) == "fused"          # host rows
    assert ops.route(torch.zeros(4, 4096, dtype=torch.bfloat16), w, return_logits=True, **kw) == "fused"
    assert calls == [4, 4]
