"""Production declines never execute a reference arithmetic recipe."""

from types import SimpleNamespace

import pytest
import torch

from isoexec.ops.attention.rope_sm90 import ops


def test_rope_decline_refuses_in_forward_and_backward(monkeypatch):
    monkeypatch.setattr(ops, "extension", lambda: SimpleNamespace(rope=lambda *args: []))
    q = torch.ones(2, 1, 64)
    positions = torch.arange(2)
    table = ops.RopeTable(torch.ones(2, 64), 64, 10000.0, 0, 0)
    for transpose in (False, True):
        with pytest.raises(ValueError, match="SM90 rope declined"):
            ops._rope_raw(q, None, positions, table, False, False, transpose)


def test_qk_forward_decline_refuses(monkeypatch):
    monkeypatch.setattr(ops, "extension", lambda: SimpleNamespace(qknorm_rope_forward=lambda *args: []))
    q = torch.ones(2, 1, 64)
    w = torch.ones(64)
    with pytest.raises(ValueError, match="forward declined"):
        ops.qk_norm_rope(q, q, w, w, None, None, 1e-6)


def test_qk_backward_decline_refuses(monkeypatch):
    def forward(q, k, *args):
        return q.clone(), k.clone(), torch.ones(q.shape[0], q.shape[1] + k.shape[1])

    monkeypatch.setattr(ops, "extension", lambda: SimpleNamespace(
        qknorm_rope_forward=forward, qknorm_rope_backward=lambda *args: []))
    q = torch.ones(2, 1, 64, requires_grad=True)
    w = torch.ones(64)
    yq, yk = ops.qk_norm_rope(q, q, w, w, None, None, 1e-6)
    with pytest.raises(ValueError, match="backward declined"):
        (yq.sum() + yk.sum()).backward()
