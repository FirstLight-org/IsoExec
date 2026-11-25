"""The trainer's host metadata enables production plans without changing forward or gradient results."""

import pytest
import torch

from isoexec.ops._testing import bitwise_equal
from isoexec.ops.gdn.gdn_cpr_sm90 import integration, ops
from isoexec.ops.gdn.gdn_cpr_sm90.tests._fixtures import SM90, make

pytestmark = SM90
NAMES = ("q", "k", "v", "a", "b", "A_log", "dt_bias")


@pytest.mark.parametrize("save_states", [False, True])
def test_host_metadata_selects_prepared_and_preserves_gradients(monkeypatch, save_states):
    data = make([128] * 16, 2, 4, seed=83, scale=0.25)
    host = torch.tensor(data["cu"], dtype=torch.int32)
    selected = []
    choose = ops.choose_prefill_mode

    def select(n, T, hv, cu_seqlens_cpu=None):
        mode = choose(n, T, hv, cu_seqlens_cpu)
        selected.append((cu_seqlens_cpu, mode))
        return mode

    monkeypatch.setattr(ops, "choose_prefill_mode", select)
    leaves = [data[name].detach().requires_grad_(True) for name in NAMES]
    expected = integration.gdn_cpr_sm90_core(*leaves, data["cut"], save_states=save_states)
    do = torch.randn_like(expected) * 0.01
    expected_grads = torch.autograd.grad(expected, leaves, do)
    actual = integration.gdn_cpr_sm90_core(*leaves, data["cut"], save_states=save_states, cu_seqlens_cpu=host)
    grads = torch.autograd.grad(actual, leaves, do)
    assert selected[0][0] is None and selected[0][1] == 0
    assert selected[1][0] is host and selected[1][1] == 2
    assert bitwise_equal(actual, expected)
    for got, want in zip(grads, expected_grads):
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    with torch.no_grad():
        output_only = integration.gdn_cpr_sm90_core(*leaves, data["cut"], cu_seqlens_cpu=host)
    assert selected[-1][0] is host and selected[-1][1] == 2
    assert bitwise_equal(output_only, expected)
