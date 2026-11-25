"""Independent recurrence evidence for the query scale and packed request reset."""

import torch

from isoexec.ops.gdn.reference import ref_gdn_core
from isoexec.ops.gdn.gdn_cpr_sm90.reference import gdn_reference


def test_single_token_query_scale_and_gradient_match_the_closed_form():
    q = torch.tensor([[[1., 2., 3., 4.]]], requires_grad=True)
    k = torch.tensor([[[4., 3., 2., 1.]]], requires_grad=True)
    v = torch.tensor([[[2., -3.]]], requires_grad=True)
    zero = torch.zeros(1, 1)
    actual = ref_gdn_core(q, k, v, zero, zero, zero.flatten(), zero.flatten(), torch.tensor([0, 1]))
    qn = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    kn = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    expected = 0.5 * v * (qn * kn).sum(-1, keepdim=True) / 2
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), (q, k, v), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (q, k, v))
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want)


def test_packed_recurrence_matches_independent_float64_reference():
    torch.manual_seed(43)
    q, k = (torch.randn(5, 1, 4) for _ in range(2))
    v = torch.randn(5, 2, 4)
    alpha, beta = (torch.randn(5, 2) for _ in range(2))
    A_log, bias = torch.randn(2), torch.randn(2)
    cu = torch.tensor([0, 2, 5])
    got = ref_gdn_core(q, k, v, alpha, beta, A_log, bias, cu)
    expected, _ = gdn_reference(q, k, v, alpha, beta, A_log, bias, cu)
    torch.testing.assert_close(got.double(), expected, atol=2e-7, rtol=2e-6)
