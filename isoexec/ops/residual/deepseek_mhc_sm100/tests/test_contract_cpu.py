"""CPU contract for the DeepSeek package: declarations, float64 references vs the eager module, refusals."""

from __future__ import annotations

import torch

from isoexec.ops.residual.deepseek_mhc_sm100 import reference as R
from isoexec.ops.residual.deepseek_mhc_sm100 import spec


def test_impl_records_cpu():
    ids = {i.impl_id for i in spec.IMPLS}
    assert ids == {spec.IMPL_ID, spec.IMPL_ID_BWD, spec.IMPL_ID_COLLAPSE, spec.IMPL_ID_COLLAPSE_BWD,
                   spec.IMPL_ID_EXPAND, spec.IMPL_ID_EXPAND_BWD}
    assert all(i.supported_archs == frozenset({"sm100"}) for i in spec.IMPLS)
    assert spec.HIDDEN == 5120 and spec.HC_MULT == 4


def test_ops_import_builds_nothing_cpu():
    import subprocess
    import sys

    code = ("import isoexec.ops.residual.deepseek_mhc_sm100.ops;"
            "from isoexec.ops.residual.mhc_sm100 import _extension;"
            "assert _extension.extension.cache_info().currsize == 0")
    subprocess.run([sys.executable, "-c", code], check=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})


def test_float64_reference_matches_module_cpu():
    torch.manual_seed(0)
    H = 64
    x = (torch.randn(5, 4, H) * 2).bfloat16()
    w, base, scale = torch.randn(24, 4 * H) * 0.05, torch.randn(24) * 0.3, torch.tensor([0.7, 0.4, 0.9])
    got = R.eager_params(x, w, scale, base, rms_eps=1e-20, sinkhorn_eps=1e-6, iterations=20)
    want = R.params_reference(x, w, scale, base, rms_eps=1e-20, sinkhorn_eps=1e-6, iterations=20)
    for a, b in zip(got, want):
        torch.testing.assert_close(a.double(), b, atol=2e-6, rtol=0)
    post, comb = want[1].float(), want[2].float()
    out = torch.randn(5, H).bfloat16()
    e = R.eager_expand(out, x, post, comb)
    torch.testing.assert_close(e.double(), R.expand_reference(out, x, post, comb).to(torch.bfloat16).double(), atol=0.05, rtol=2 ** -7)
    torch.testing.assert_close(R.expand_declared(out, x, post, comb).double(), e.double(), atol=0.05, rtol=2 ** -7)
    pre = want[0].float()
    torch.testing.assert_close(R.collapse_declared(x, pre).double(), R.eager_collapse(x, pre).double(), atol=0.05, rtol=2 ** -7)


def test_the_residual_family_declares_the_recipe_cpu():
    from isoexec.core.registry_build import build_registry

    reg = build_registry()
    for op, impl in ((spec.OP_ID, spec.IMPL_ID), (spec.OP_ID_EXPAND, spec.IMPL_ID_EXPAND)):
        assert reg.get_op(op).impls[impl].version == spec.RECIPE_VERSION
