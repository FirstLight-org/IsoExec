"""Packed host boundaries reach GDN once per batch, including checkpoint and autograd seams."""

from dataclasses import replace

import pytest
import torch

from isoexec.nn.tests import _gdn_fixtures as F
from isoexec.ops.gdn.gdn_cpr_sm90 import integration
from isoexec.ops.table import OpTable
from isoexec.schema import make_sequence_context, register_pytrees


def test_context_pytree_retains_host_boundaries():
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    seq = make_sequence_context(torch.arange(7), cu, max_q=4, cu_seqlens_cpu=cu)
    register_pytrees()
    from torch.utils._pytree import tree_flatten, tree_unflatten

    values, spec = tree_flatten(seq)
    restored = tree_unflatten(values, spec)
    assert restored.cu_seqlens_cpu is cu
    with pytest.raises(ValueError, match="same number"):
        make_sequence_context(torch.arange(7), cu, max_q=4, cu_seqlens_cpu=cu[:2])
    with pytest.raises(ValueError, match="CPU tensor"):
        make_sequence_context(torch.arange(7), cu, max_q=4, cu_seqlens_cpu=cu.to("meta"))


def test_gdn_forward_passes_host_boundaries_to_bound_core():
    model = F.component(F.spec(), device="cpu")
    model.apply_placement(F.placement(), site=F.SITE, op_table=OpTable(mode="reference"))
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator).to(parameter.dtype))
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    seq = make_sequence_context(torch.arange(7), cu, max_q=4)
    x = torch.randn(7, F.HIDDEN).to(torch.bfloat16)
    with torch.no_grad():
        expected = model(x, seq)
    original = model.slots.resolved()["core"]
    seen = []

    def core(*args, cu_seqlens_cpu=None):
        seen.append(cu_seqlens_cpu)
        return original.fn(*args, cu_seqlens_cpu=cu_seqlens_cpu)

    model.slots.core = replace(original, fn=core)
    seq.cu_seqlens_cpu = cu
    with torch.no_grad():
        actual = model(x, seq)
    assert seen == [cu]
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_trainer_family_wrapper_passes_host_boundaries(monkeypatch):
    import isoexec.ops.gdn as facade
    from isoexec.ops.gdn.trainer import gdn_core_native_cpr

    cu_cpu = torch.tensor([0, 2], dtype=torch.int32)
    seen = []

    def call(*args, **kwargs):
        seen.append(kwargs["cu_seqlens_cpu"])
        return args[0]

    monkeypatch.setattr(facade, "gdn_core", call)
    q = torch.ones(2, 1, 1)
    assert gdn_core_native_cpr(q, q, q, q, q, q, q, cu_cpu, cu_seqlens_cpu=cu_cpu) is q
    assert seen == [cu_cpu]


@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("save_states", [None, False, True], ids=["default", "recompute", "save"])
def test_autograd_core_preserves_host_metadata_in_forward_and_backward(monkeypatch, grad, save_states):
    cu_cpu = torch.tensor([0, 2], dtype=torch.int32)
    leaves = [torch.ones(shape, requires_grad=grad) for shape in
              [(2, 1, 1)] * 3 + [(2, 1)] * 2 + [(1,)] * 2]
    seen = []

    def prefill(*args, cu_seqlens_cpu=None):
        seen.append(("prefill", cu_seqlens_cpu))
        return args[0].clone()

    def trainer(*args, save_states, cu_seqlens_cpu=None):
        assert save_states is expected_save_states
        seen.append(("trainer", cu_seqlens_cpu))
        return args[0].clone(), {"saved": args[0]}

    def backward(saved, a, b, A_log, dt_bias, cu, do, cu_seqlens_cpu=None):
        seen.append(("backward", cu_seqlens_cpu))
        return tuple(torch.ones_like(t) for t in leaves)

    monkeypatch.setattr(integration.ops, "prefill", prefill)
    monkeypatch.setattr(integration.ops, "trainer_forward", trainer)
    monkeypatch.setattr(integration, "_fla_grads", backward)
    expected_save_states = False if save_states is None else save_states
    kwargs = {} if save_states is None else {"save_states": save_states}
    out = integration.gdn_cpr_sm90_core(*leaves, cu_cpu, cu_seqlens_cpu=cu_cpu, **kwargs)
    if grad:
        assert len(torch.autograd.grad(out.sum(), leaves)) == 7
    assert [phase for phase, _ in seen] == (["trainer", "backward"] if grad else ["prefill"])
    assert all(host is cu_cpu for _, host in seen)
