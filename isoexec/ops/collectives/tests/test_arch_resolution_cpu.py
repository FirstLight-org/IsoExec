"""Every collectives impl resolves on each declared accelerator arch and refuses an undeclared one."""

from __future__ import annotations

import pytest

from isoexec.ops import _dispatch, _resolve
from isoexec.ops.collectives import spec


#: Impls proven on one arch only: the gqa_sm100 trainer core's CP form rides FA4's sm100 backward.
SM100_ONLY = {("collectives.cp_gather", "kv_all_gather_fwd_owner_tiles_bwd")}


@pytest.mark.parametrize("arch", ["sm90", "sm100"])
def test_every_impl_resolves_on_a_declared_arch(monkeypatch, arch):
    monkeypatch.setattr(_dispatch, "ARCH", arch)
    for impl in spec.IMPLS:
        if arch == "sm90" and (impl.op, impl.impl_id) in SM100_ONLY:
            assert impl.supported_archs == {"sm100"}, (impl.op, impl.impl_id)
            continue
        assert arch in impl.supported_archs, (impl.op, impl.impl_id)
        assert _resolve.declared_version(impl.op, impl.impl_id, site="t") == impl.version


def test_an_undeclared_arch_is_refused(monkeypatch):
    monkeypatch.setattr(_dispatch, "ARCH", "sm120")
    impl = spec.IMPLS[0]
    with pytest.raises((_resolve.ResolutionError, _dispatch.ArchSelectionError)):
        _resolve.declared_version(impl.op, impl.impl_id, site="t")

