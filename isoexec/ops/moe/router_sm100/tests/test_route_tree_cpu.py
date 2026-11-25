"""CPU contract of the segment-root router path: its segment count is the declared logits program's, and
``route`` keeps the fused kernel wherever the tree form does not apply."""
import torch

from isoexec.ops.moe.router_sm100 import ops, spec


def test_declared_segments_are_the_spec():
    for h, s in spec.LOGIT_SEGMENTS.items():
        assert ops.declared_segments(h) == s
    assert ops.declared_segments(64 * 6) == 2 and ops.declared_segments(64 * 3) == 1


def test_route_keeps_the_fused_kernel_outside_decode_rows(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "route_fused", lambda x, w, **kw: calls.append(x.shape[0]) or "fused")
    monkeypatch.setattr(ops, "route_select_tree", lambda *a, **k: (_ for _ in ()).throw(AssertionError("tree")))
    w = torch.zeros(256, 4096, dtype=torch.bfloat16)
    kw = dict(routing=ops.SQRTSOFTPLUS, top_k=6, num_shared=1)
    assert ops.route(torch.zeros(4, 4096, dtype=torch.bfloat16), w, **kw) == "fused"          # host rows
    assert ops.route(torch.zeros(4, 4096, dtype=torch.bfloat16), w, return_logits=True, **kw) == "fused"
    assert calls == [4, 4]
