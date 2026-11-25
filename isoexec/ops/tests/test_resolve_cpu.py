"""Check all family resolver bindings and refusals with architecture pinned to sm90.
Resolution builds declarations and callables without launching kernels."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from isoexec.core.registry_build import build_registry
from isoexec.ops import _resolve
from isoexec.ops._resolve import FAMILY_PACKAGES, BoundOp, ResolutionError

_ROOT = Path(__file__).resolve().parents[3]

# Explicit operation/fact cases detect any resolver entry silently removed from the supported surface.
SERVED: tuple[tuple[str, dict], ...] = (
    ("mm", {}),
    ("norms.rms.pre_attn", {"convention": "rms_zero_centered", "width": 256}),
    ("norms.rms.pre_mlp", {"convention": "rms", "width": 256}),
    ("norms.rms.final", {"convention": "rms_zero_centered", "width": 384}),
    ("norms.mla", {"convention": "rms", "width": 512}),
    ("norms.qk", {"convention": "rms_zero_centered"}),
    ("norms.gated_out", {"convention": "gated_rms_zero_centered", "width": 128}),
    ("rope.apply", {"style": "neox", "rotary_dim": 16, "theta": 1e7, "qk_norm": None}),
    ("attention.core.gqa", {}),
    ("attention.core.mla", {}),
    ("attention.gate", {"gate": "sigmoid"}),
    ("attention.gate", {"gate": "swish"}),
    ("conv.short", {}),
    ("gdn.core", {}),
    ("mlp.act", {}),
    ("moe.route_score", {"routing": "softmax_renorm"}),
    ("moe.route_mechanics", {"routing": "softmax_renorm"}),
    ("moe.route_mechanics", {"routing": "sigmoid_noaux_bias_select"}),
    ("moe.expert_mlp", {}),
    ("moe.combine", {}),
    ("moe.shared_expert", {}),
    ("residual.add.attn", {}),
    ("residual.add.mlp", {}),
    ("embed.lookup", {}),
    ("logprobs.head", {}),
    ("logprobs.lm_head_slice", {}),
    ("logprobs.log_softmax", {}),
)

SITE = "lm.layers.0.mixer"


@pytest.fixture(autouse=True)
def _on_sm90(monkeypatch):
    """Resolve as the fleet's node resolves, whatever this machine is. See the module docstring."""
    from isoexec.ops import _dispatch

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")


@pytest.mark.parametrize(("op", "facts"), SERVED, ids=[f"{op}:{'-'.join(str(v) for v in f.values())}" for op, f in SERVED])
def test_every_op_the_routing_table_claims_resolves_to_a_declared_kernel(op, facts):
    """The four properties, on every op: a callable, a declared impl id, its declared version, and a
    provider naming the ops package that served it."""
    resolved = _resolve.resolve(op, site=SITE, **facts)
    assert isinstance(resolved, BoundOp)
    assert resolved.op == op
    assert callable(resolved.fn)
    assert resolved.provider.startswith("ops.")
    spec = build_registry().get_op(op).impls[resolved.impl]
    assert resolved.version == spec.version, "the version is the declaration's, never the resolver's"


def test_the_routing_table_covers_every_family_that_declares_an_op():
    """Require resolver coverage for every family, with explicitly named placement-owned collective exceptions."""
    families = {op.split(".", 1)[0] for op in build_registry().ops}
    uncovered = sorted(families - set(FAMILY_PACKAGES) - {"collectives", "moe_sm90"})
    assert not uncovered, f"declared families with no resolver: {uncovered}"


def test_the_ops_a_family_serves_are_exactly_the_ops_it_is_asked_for():
    """Each resolver refuses an op id its family does not state, naming the ones it does."""
    for op in ("norms.nonesuch", "moe.nonesuch", "gdn.nonesuch", "attention.nonesuch", "logprobs.nonesuch"):
        with pytest.raises(ResolutionError) as exc:
            _resolve.resolve(op, site=SITE)
        assert op in str(exc.value)


def test_an_op_id_no_family_owns_refuses_and_points_at_the_seam_that_does():
    with pytest.raises(ResolutionError) as exc:
        _resolve.resolve("collectives.row_parallel", site=SITE)
    assert "nn/_seams.py" in str(exc.value)


def test_a_fact_a_family_does_not_take_refuses_rather_than_being_dropped():
    """A fact nobody reads is a decision nobody made."""
    with pytest.raises(ResolutionError, match="rounding"):
        _resolve.resolve("residual.add.attn", site=SITE, rounding="fp32")


@pytest.mark.parametrize(
    ("op", "facts", "needle"),
    (
        ("norms.rms.pre_attn", {"convention": "rms_zero_centered", "width": 96}, "no forward for width 96"),
        ("norms.gated_out", {"convention": "gated_rms", "width": 128}, "gated_rms_zero_centered"),
        ("norms.mla", {"convention": "rms_zero_centered", "width": 512}, "'rms'"),
        ("attention.gate", {"gate": "gelu"}, "gelu"),
        ("moe.route_mechanics", {"routing": "nonesuch"}, "nonesuch"),
        ("mlp.act", {"activation": "swiglu_clamp"}, "swiglu_clamp"),
        ("rope.apply", {"style": "interleaved", "rotary_dim": 16, "theta": 1e7, "qk_norm": None}, "interleaved"),
    ),
)
def test_facts_nothing_serves_refuse_by_name(op, facts, needle):
    """The refusal names the op, what was asked and what is on offer -- never a neighbouring impl."""
    with pytest.raises(ResolutionError) as exc:
        _resolve.resolve(op, site=SITE, **facts)
    message = str(exc.value)
    assert SITE in message and op in message and needle in message


def test_a_row_with_a_qk_norm_has_no_standalone_rotation_to_resolve():
    """``rope_sm90`` ships ONE kernel for the q/k norm and the rotation, so the rotation has no
    binding of its own -- the same shape ``integration.bind`` gives a host table by installing the
    fused entry and popping ``rope.apply``. The resolver says so instead of building a second
    kernel that would compute the rotation a second time."""
    with pytest.raises(ResolutionError) as exc:
        _resolve.resolve("rope.apply", site=SITE, style="neox", rotary_dim=16, theta=1e7,
                         qk_norm="rms_zero_centered")
    assert "norms.qk" in str(exc.value) and "owns" in str(exc.value)
    fused = _resolve.resolve("norms.qk", site=SITE, convention="rms_zero_centered")
    assert dict(fused.owns) == {"norms.qk": fused.impl, "rope.apply": fused.impl}


def test_the_gdn_core_owns_the_ops_its_kernel_folds_in():
    """Ownership is the resolution's statement, read off the package's own list rather than a copy."""
    from isoexec.ops.gdn import integration

    core = _resolve.resolve("gdn.core", site=SITE)
    assert dict(core.owns) == {"gdn.core": core.impl, **{op: core.impl for op in integration._OWNED}}


def test_the_expert_region_states_the_activation_its_own_kernel_runs():
    """``act`` is a declared input whose VALUE is the kernel's, so the kernel says it -- and it is
    the same constant ``integration.bind`` pins on the host table entry."""
    from isoexec.ops.moe.moe_sm90 import integration as moe_sm90

    experts = _resolve.resolve("moe.expert_mlp", site=SITE)
    assert dict(experts.inputs) == {"act": moe_sm90.ACT}
    assert experts.impl == moe_sm90.IMPL


def test_the_mm_impl_id_is_read_back_off_the_installed_object():
    """The pattern the whole tier is built on. Nothing is installed in this process, so the readback
    answers the default provider -- and it answers it by LOOKING, not by remembering."""
    from isoexec.ops.mm.trainer import mm_impl_id, mm_trainer

    resolved = _resolve.resolve("mm", site=SITE)
    assert resolved.impl == mm_impl_id()
    assert resolved.fn is mm_trainer


def test_a_gemm_whose_provider_changed_under_the_record_refuses_by_name():
    """Allow plan-only resolution, then refuse the first GEMM if its pinned provider changed."""
    import torch

    from isoexec.ops.mm import mm_batch_invariant as vendored
    from isoexec.ops.mm.trainer import mm_impl_id, mm_trainer

    # Start from the stock provider: on a real H100 an earlier test may have installed the sm90 wrapper.
    installed = vendored.matmul_persistent
    vendored.matmul_persistent = getattr(vendored, "_isoexec_stock_matmul_persistent", installed)
    resolved = _resolve.resolve("mm", site=SITE)
    assert (resolved.impl, resolved.fn) == (mm_impl_id(), mm_trainer)  # the record, and the OBJECT

    def segtree(a, b):  # what the installed sm90 segment-tree provider looks like to the readback
        raise AssertionError("the refusal must come before the GEMM")

    segtree._isoexec_sm90_segtree = True
    try:
        vendored.matmul_persistent = segtree
        assert mm_impl_id() == "sm90_wgmma_segtree"
        with pytest.raises(RuntimeError, match=r"mm was recorded as 'triton_batch_invariant'"):
            mm_trainer(torch.zeros(2, 2), torch.zeros(2, 2))
        # Re-resolving after the change is the other way out, and it does not refuse.
        assert _resolve.resolve("mm", site=SITE).impl == "sm90_wgmma_segtree"
    finally:
        vendored.matmul_persistent = installed
        _resolve.resolve("mm", site=SITE)  # leave the pin naming what this process actually has


def test_the_conv_wrapper_refuses_the_call_when_the_kernel_behind_it_is_not_the_one_recorded():
    """Resolve the conv wrapper before install, but refuse execution when its live kernel differs from the recorded impl."""
    import torch

    from isoexec.ops.gdn.trainer import conv_short_native

    assert _resolve.resolve("conv.short", site=SITE).fn is conv_short_native
    with pytest.raises(RuntimeError, match="has not run"):
        conv_short_native(torch.zeros(4, 2), torch.zeros(2, 1, 4), torch.tensor([0, 4]))


def test_a_version_for_a_pair_nothing_declares_refuses_rather_than_being_invented():
    with pytest.raises(ResolutionError, match="no twin class and no battery"):
        _resolve.declared_version("mm", "invented_here", site=SITE)


# -- import lightness ------------------------------------------------------------------------------


def test_importing_every_resolver_loads_no_kernel_and_initializes_no_cuda(tmp_path):
    """Check imports in a clean subprocess with GPUs hidden so CUDA initialization cannot pass unnoticed."""
    modules = sorted({f"isoexec.ops.{package}.resolve" for package in FAMILY_PACKAGES.values()})
    code = f"""
import importlib, sys
import isoexec.ops._resolve
for name in {modules!r}:
    importlib.import_module(name)
leaked = sorted(n for n in sys.modules if n.startswith("isoexec.ops.") and "_sm90" in n)
assert not leaked, leaked
import torch
assert not torch.cuda.is_initialized()
print("OK")
"""
    out = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(_ROOT)},
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "OK" in out.stdout


def test_a_resolver_module_names_no_kernel_at_module_scope():
    """The same property, read off the source rather than off ``sys.modules``: every import a
    resolver performs of a kernel package is inside a function, so merely importing the module --
    which a linter, a doc build or an editor's autocomplete does -- selects nothing."""
    import ast

    for package in sorted(set(FAMILY_PACKAGES.values())):
        path = Path(importlib.import_module(f"isoexec.ops.{package}.resolve").__file__)
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                assert not any("_sm90" in n for n in names), f"{path.name}: module-scope kernel import"


@pytest.mark.parametrize("arch", ("cpu", "sm100"))
def test_binding_refuses_a_declared_recipe_not_proven_on_active_arch(monkeypatch, arch):
    from isoexec.core.registry import ImplSpec, Registry
    from isoexec.ops import _dispatch

    registry = Registry.from_impls((
        ImplSpec(op="mm", impl_id="sm90_only", version=7, supported_archs=frozenset({"sm90"})),
    ))
    monkeypatch.setattr(_resolve, "registry", lambda: registry)
    monkeypatch.setattr(_dispatch, "ARCH", arch)
    with pytest.raises(ResolutionError, match=f"mm:sm90_only is not proven on architecture '{arch}'"):
        _resolve.resolution("mm", "sm90_only", lambda x: x, provider="ops.mm", site=SITE)

    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    binding = _resolve.resolution("mm", "sm90_only", lambda x: x, provider="ops.mm", site=SITE)
    assert binding.version == 7


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
