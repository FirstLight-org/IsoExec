"""CPU contract: declarations, refusals that need no GPU, and the reference backward's shape."""

import torch

from isoexec.ops.moe.router_sm100 import reference as R, spec


def test_spec_constants():
    assert spec.ARCH == "sm100"
    assert spec.IMPL_GROUPED == "sm100_route_grouped_v1"
    assert spec.IMPL_SQRTSOFTPLUS == "sm100_route_sqrtsoftplus_v1"
    assert spec.TABLE_SLOT_ORDER == "ascending_global_expert_id"
    assert all(h // 64 % s == 0 for h, s in spec.LOGIT_SEGMENTS.items())
    assert spec.SEG_T % 16 == 0


def test_ops_import_builds_nothing():
    import isoexec.ops.moe.router_sm100.ops as ops  # noqa: F401
    from isoexec.ops.moe.router_sm100 import _extension

    assert _extension.extension.cache_info().currsize == 0


def test_reference_backward_cpu_matches_autograd_in_float64():
    torch.manual_seed(0)
    T, E, H, k = 64, 64, 128, 4
    x = (torch.randn(T, H) / 4).bfloat16()
    w = (torch.randn(E, H) / 8).bfloat16()
    b = torch.randn(E) / 8
    logits = x.float() @ w.float().t()
    dp = torch.randn(T, k)
    dx, dw, dl, idx = R.route_backward_reference(dp, x, w, logits, routing=R.GROUPED, top_k=k,
                                                 expert_bias=b, routed_scale=1.8, seg_t=32)
    # float64 autograd of the same selection (selection held fixed)
    ids = idx.long()
    l64 = logits.double().requires_grad_(True)
    s = torch.sigmoid(torch.gather(l64, 1, ids))
    wts = s / (s.sum(1, keepdim=True) + 1e-20) * 1.8
    (wts * dp.double()).sum().backward()
    g = torch.zeros(T, E, dtype=torch.float64).scatter(1, ids, torch.gather(l64.grad, 1, ids))
    torch.testing.assert_close(torch.gather(g, 1, ids), dl.double(), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(dw.double(), g.t() @ x.double(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(dx.double(), g @ w.double(), rtol=2e-2, atol=1e-3)


def test_routes_are_registered_and_resolution_is_fail_closed(monkeypatch):
    import pytest

    from isoexec.core.registry_build import build_registry
    from isoexec.ops import _dispatch
    from isoexec.ops._resolve import ResolutionError, resolve

    reg = build_registry()
    assert set(reg.get_op("moe.route").impls) == {spec.IMPL_GROUPED, spec.IMPL_SQRTSOFTPLUS}
    assert set(reg.get_op("moe.route.backward").impls) == {spec.IMPL_BACKWARD}
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    with pytest.raises(ResolutionError, match="sm100 only"):
        resolve("moe.route", site="t", routing=spec.SQRTSOFTPLUS, top_k=6)
    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    for bad in ({"routing": "softmax", "top_k": 6}, {"routing": spec.SQRTSOFTPLUS, "top_k": 0},
                {"routing": spec.GROUPED, "top_k": 8, "n_group": 2, "topk_group": 3},
                {"routing": spec.SQRTSOFTPLUS, "top_k": 6, "segments": 2}):
        with pytest.raises(ResolutionError):
            resolve("moe.route", site="t", **bad)
    bound = resolve("moe.route", site="t", routing=spec.SQRTSOFTPLUS, top_k=6, num_shared=1, routed_scale=1.5)
    assert bound.impl == spec.IMPL_SQRTSOFTPLUS and dict(bound.inputs)["top_k"] == 6
    with pytest.raises(ValueError, match="fixed at placement"):
        bound.fn(None, None, top_k=8)
