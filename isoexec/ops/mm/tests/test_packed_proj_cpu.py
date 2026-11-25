"""``ops/mm/packed_proj.py``: the alias holds, the invalidation fires, and the fallback is exact.

The sibling of ``ops/moe/moe_sm90/tests/test_packed_experts_cpu.py``, for the column-parallel projection
sites (GDN in_proj, attention q|k|v|gate, dense fc1). Duck-typed on a stand-in module, so it tests
the MECHANISM without building a model: the real kinds are covered end to end by ``gate_eager_oracle``.

The five claims, and why each is here rather than left to a gate:

  1. THE ALIAS IS THE PARAMETERS. After a pack, every member's ``.data`` points INTO the buffer, the
     buffer is a REGISTERED non-persistent buffer on the module (design v4.1 section 2.2: every
     tensor the model owns is a Parameter or a registered buffer), and the packed operand's VALUES
     equal the ``torch.cat`` it replaces, byte for byte.
  2. IT IS AN ALIAS AND NOT A CACHE. A write THROUGH a parameter -- the weight-sync spelling
     ``param.data.copy_(fresh)``, which moves neither the data pointer nor ``_version`` -- is visible
     in the packed operand IMMEDIATELY, with no invalidation and no re-pack. This is the claim that
     rules out the copy-cache design; see ``version_counter_probe.py`` under the perf2 evidence dir.
  3. A BROKEN ALIAS IS DETECTED, NEVER SERVED. Re-pointing a member's ``.data`` off the buffer --
     what Megatron's DDP does when it buckets -- makes the hook return ``None`` from then on, so the
     caller takes the exact unfused path, and the stale bytes in the buffer are unreachable.
  4. FOREIGN STORAGE IS REFUSED AND THE BUFFER IS RELEASED. A member that is a view into a larger
     storage (a DDP bucket) is never repointed, the site gives up, and no dead buffer is left
     registered -- ``named_buffers()`` consumers (the FSDP2 host broadcasts every buffer to the DP
     replicas; SGLang's release/resume clones every buffer) must not be handed multi-GiB of bytes
     nothing reads.
  5. THE VJP IS ``cat``'s. Each member's weight gradient is exactly the row block ``CatBackward0``
     would have narrowed out.

CPU-only. Nothing here is device-dependent, and the suite runs in the dev venv.
"""

from __future__ import annotations

import weakref

import pytest
import torch
import torch.nn as nn

from isoexec.ops.mm import packed_proj as PP
from isoexec.ops.moe.shared import packed_experts as PE
from isoexec.ops.moe.shared import param_storage as PS


@pytest.fixture(autouse=True)
def _no_foreign_moe_layers(monkeypatch):
    """``bump_sync_epoch`` refreshes every packed MoE layer in the process. These tests build none,
    so an MoE layer another test file left alive must not be refreshed (and refused) here."""
    monkeypatch.setattr(PE, "_PACKED_LAYERS", weakref.WeakSet())

ROLES = ("q", "k", "v", "z", "beta", "alpha")
WIDTHS = (12, 12, 24, 24, 3, 3)
H = 8


class _Site(nn.Module):
    """The duck type ``packed_proj`` names, and nothing else."""

    packed_proj_name = "in_proj_packed"

    def __init__(self, widths=WIDTHS, hidden=H, ns="lm.layers.0.mixer"):
        super().__init__()
        self.ns = ns
        g = torch.Generator().manual_seed(7)
        for role, n in zip(ROLES, widths):
            self.register_parameter(role, nn.Parameter(torch.randn(n, hidden, generator=g)))
        self.packed_proj = None

    def packed_proj_roles(self):
        return ROLES

    def members(self):
        return tuple(getattr(self, r) for r in ROLES)

    def cat(self):
        return torch.cat([m for m in self.members()], dim=0)


class _Model(nn.Module):
    def __init__(self, site):
        super().__init__()
        self.site = site


def _attached():
    site = _Site()
    want = site.cat().clone()
    assert PP.attach(_Model(site)) == 1
    return site, want


def test_pack_registers_a_buffer_and_aliases_every_member():
    site, want = _attached()
    w = site.packed_proj()
    assert w is not None
    assert torch.equal(w, want), "the packed operand is not the cat it replaces"

    names = [n for n, _ in site.named_buffers()]
    assert names == ["in_proj_packed"], names
    assert "in_proj_packed" not in site.state_dict(), "the buffer must be NON-PERSISTENT"

    lo = site.in_proj_packed.data_ptr()
    hi = lo + site.in_proj_packed.untyped_storage().nbytes()
    for r in ROLES:
        p = getattr(site, r)
        assert lo <= p.data_ptr() < hi, f"{r} is not aliased onto the buffer"
    # The parameter objects and their names survive.
    assert sorted(n for n, _ in site.named_parameters()) == sorted(ROLES)
    ok, why = PP.check_module(site)
    assert ok, why


def test_a_write_through_a_parameter_is_seen_with_no_repack():
    """Claim 2: an ALIAS, not a cache. ``param.data.copy_`` bumps neither data_ptr nor ``_version``."""
    site, _ = _attached()
    site.packed_proj()
    repacks = PP.stats()["repacks"]

    fresh = torch.full_like(site.v.data, 3.5)
    ptr, ver = site.v.data_ptr(), site.v._version
    with torch.no_grad():
        site.v.data.copy_(fresh)  # the sync/weights.py spelling
    assert site.v.data_ptr() == ptr and site.v._version == ver, "the probe premise no longer holds"

    w = site.packed_proj()
    assert torch.equal(w, site.cat()), "the packed operand did not see the weight sync"
    assert (w[24:48] == 3.5).all()
    assert PP.stats()["repacks"] == repacks, "an alias must not need a re-pack to see a write"


def test_a_rebound_member_invalidates_and_falls_back():
    """Claim 3: what Megatron's DDP does to a packed site."""
    site, _ = _attached()
    assert site.packed_proj() is not None
    site.q.data = site.q.data.clone()  # off the buffer, DataWatch bumps the epoch
    ok, why = PP.check_module(site)
    assert not ok and "rebound off the packed buffer" in why, why
    # Same sync epoch -> no re-pack is attempted, so the caller must take the unfused path.
    assert site.packed_proj() is None
    # A sync boundary licenses one attempt, and it re-packs from the CURRENT values.
    PS.bump_sync_epoch()
    w = site.packed_proj()
    assert w is not None and torch.equal(w, site.cat())


def test_foreign_storage_is_refused_and_no_buffer_is_left_behind():
    """Claim 4: the DDP-bucket give-up path, and that it releases."""
    site = _Site()
    bucket = torch.zeros(sum(WIDTHS) * H + 16)
    off = 0
    for r, n in zip(ROLES, WIDTHS):
        p = getattr(site, r)
        with torch.no_grad():
            p.data = bucket[off : off + n * H].view(n, H)  # a view into a LARGER storage
        off += n * H
    assert PP.attach(_Model(site)) == 1
    assert site.packed_proj() is None
    assert site.__dict__.get("_ix_projpack_giveup") is True
    assert [n for n, _ in site.named_buffers()] == [], "a dead buffer was left registered"
    assert site.packed_proj() is None  # and it stays given up


def test_the_vjp_is_cats():
    """Claim 5: member gradients are the row blocks ``CatBackward0`` narrows out."""
    site, _ = _attached()
    x = torch.randn(5, H)

    ref = _Site()
    with torch.no_grad():
        for r in ROLES:
            getattr(ref, r).data.copy_(getattr(site, r).data)

    cot = torch.randn(5, sum(WIDTHS))
    (x @ site.packed_proj().t()).backward(cot)
    (x @ ref.cat().t()).backward(cot)
    for r in ROLES:
        a, b = getattr(site, r).grad, getattr(ref, r).grad
        assert torch.equal(a.view(torch.int32), b.view(torch.int32)), f"{r}: gradient is not cat's"


def test_unpack_restores_private_storage():
    site, want = _attached()
    site.packed_proj()
    assert PP.unpack(site)
    assert [n for n, _ in site.named_buffers()] == []
    assert torch.equal(site.cat(), want)
    for r in ROLES:
        p = getattr(site, r)
        assert p.data.storage_offset() == 0
        assert p.data.untyped_storage().nbytes() == p.numel() * p.element_size()


def test_sync_refuses_when_a_captured_buffer_cannot_be_refreshed():
    import pytest

    site, _ = _attached()
    site.packed_proj()
    bucket = torch.zeros((24, H), dtype=site.q.dtype, device=site.q.device)
    site.q.data = bucket[:12]
    try:
        with pytest.raises(RuntimeError, match="refresh refused.*in_proj_packed"):
            PS.bump_sync_epoch()
    finally:
        PP.unpack(site)


if __name__ == "__main__":  # pragma: no cover
    for fn in (
        test_pack_registers_a_buffer_and_aliases_every_member,
        test_a_write_through_a_parameter_is_seen_with_no_repack,
        test_a_rebound_member_invalidates_and_falls_back,
        test_foreign_storage_is_refused_and_no_buffer_is_left_behind,
        test_the_vjp_is_cats,
        test_unpack_restores_private_storage,
        test_sync_refuses_when_a_captured_buffer_cannot_be_refreshed,
    ):
        fn()
        print("PASS ", fn.__name__)
