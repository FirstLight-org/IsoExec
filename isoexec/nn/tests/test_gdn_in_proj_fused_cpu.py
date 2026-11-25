"""Count GDN in-projection launches through core entry. Bitwise parity alone cannot detect
replacement of one packed GEMM with six separate GEMMs.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from isoexec.nn._slots import BoundOp, Unbound
from isoexec.ops.mm.reference import ref_mm
from isoexec.ops.table import OpTable
from isoexec.schema import make_sequence_context

from isoexec.nn.tests import _gdn_fixtures as F

#: The order ``nn/gdn.py::packed_proj_roles`` concatenates the six projections in.
ROLES = ("q", "k", "v", "z", "beta", "alpha")

#: What the shipped trainer core computes in-kernel, beside its own op. The same tuple
#: ``test_gdn_ownership_cpu.py`` states; a core claiming less is refused by ``_refuse_unowned_stages``.
TRAINER_CORE_OPS = ("gdn.core", "gdn.l2norm", "gdn.gating")

#: The aten ops a GEMM can show up as. Counting the set rather than ``aten.mm`` alone is what makes
#: the assertion survive a reference that reaches matmul or addmm instead.
GEMMS = ("aten.mm", "aten.matmul", "aten.addmm")

#: Ops that MOVE bytes. Between the in_proj GEMM and the core there must be none of them when the
#: core owns the convolution: the six operands are column views of one buffer, never copies.
MOVES = ("aten.cat", "aten.copy_", "aten.clone", "aten.contiguous")

T = 7


class _AtenLog(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func.overloadpacket))
        return func(*args, **(kwargs or {}))


class _Spy:
    """Stands in for ``mm`` and ``gdn.core``: records what the mixer hands the core, and where the
    aten log stood when the in_proj GEMM returned and when the core was called.

    The ``mm`` substitution wraps the real resolution rather than replacing it, so the arithmetic
    under the window is the one the tier resolved and only the bracket is the test's.
    """

    def __init__(self, log: _AtenLog, mm) -> None:
        self.log, self._mm = log, mm
        self.args = None
        self.after_in_proj: int | None = None
        self.at_core: int | None = None

    def mm(self, x, w):
        out = self._mm(x, w)
        if self.after_in_proj is None:
            self.after_in_proj = len(self.log.ops)
        return out

    def core(self, q, k, v, alpha, beta, A_log, dt_bias, cu_seqlens):
        self.args = (q, k, v, alpha, beta)
        self.at_core = len(self.log.ops)
        return torch.zeros(v.shape, dtype=v.dtype)

    @property
    def in_proj_to_core(self) -> list[str]:
        """The aten ops between the in_proj GEMM returning and the core being called."""
        return self.log.ops[self.after_in_proj : self.at_core]

    @property
    def up_to_core(self) -> list[str]:
        return self.log.ops[: self.at_core]


def _seq():
    return make_sequence_context(torch.arange(T, dtype=torch.int64), torch.tensor([0, T], dtype=torch.int32), max_q=T)


def _fill(module, *, seed: int = 3) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_((torch.randn(p.shape, generator=gen) * 0.05).to(p.dtype))
            p.requires_grad_(False)


def _x(seed: int = 0) -> torch.Tensor:
    return (torch.randn(T, F.HIDDEN, generator=torch.Generator().manual_seed(seed)) * 0.7).to(torch.bfloat16)


def _mixer(log: _AtenLog, *, conv_subsumed: bool):
    """A placed ``nn.GDN`` with ``mm`` and the core spied on.

    ``conv_subsumed`` picks the side, because on this tier that is what decides the question: a
    trainer resolves its own core (which leaves the convolution a stage of its own) and an engine
    waits for ``attach_core``, whose ownership statement is what settles the ``conv`` slot. Neither
    arm declares the answer -- ``nn/gdn.py::_settle_conv`` reads it off the resolution.
    """
    gdn = F.component(F.spec(), device="cpu")
    gdn.apply_placement(F.placement(side="engine" if conv_subsumed else "trainer"), site=F.SITE, op_table=OpTable(mode="reference"))
    spy = _Spy(log, gdn.slots.mm)
    if conv_subsumed:
        # An adapter-shaped core: a plain function carrying the ownership statement, arriving as a
        # BoundOp. ``owns`` cannot go on the bound method, so the spy is wrapped once here.
        def engine_core(q, k, v, alpha, beta, A_log, dt_bias, cu_seqlens):
            return spy.core(q, k, v, alpha, beta, A_log, dt_bias, cu_seqlens)

        engine_core.owns = {op: "fake_paged_recurrence" for op in (*TRAINER_CORE_OPS, "conv.short")}
        gdn.attach_core(BoundOp("gdn.core", "fake_paged_recurrence", 1, "test.adapter", {}, engine_core, owns=engine_core.owns))
    else:
        gdn.slots.core = spy.core
    gdn.slots.mm = spy.mm
    _fill(gdn)
    return gdn, spy




@pytest.mark.parametrize("conv_subsumed", [True, False])
def test_in_proj_is_one_mm_over_the_packed_weight_view(conv_subsumed):
    """Require one GEMM over the packed projection weights and adjacent column views of its output."""
    log = _AtenLog()
    gdn, spy = _mixer(log, conv_subsumed=conv_subsumed)
    x, seq = _x(), _seq()  # built outside the trace: the log holds the forward and nothing else
    with log, torch.no_grad():
        gdn(x, seq)
    gemms = [o for o in spy.up_to_core if o in GEMMS]
    assert len(gemms) == 1, spy.up_to_core

    # ... and every column is the same function of x and its weight rows as a separate GEMM computes,
    # so the one GEMM is the six projections and not merely one launch that resembles them.
    off = 0
    out = ref_mm(x, torch.cat([getattr(gdn, r) for r in ROLES], dim=0))
    for r in ROLES:
        w = getattr(gdn, r)
        sep = ref_mm(x, w)
        assert torch.equal(out[:, off : off + w.shape[0]].contiguous().view(torch.int16), sep.view(torch.int16)), r
        off += w.shape[0]


def test_subsumed_conv_hands_the_core_adjacent_views_of_one_buffer():
    """A convolution-owning core receives adjacent q/k/v views without an intervening copy."""
    log = _AtenLog()
    gdn, spy = _mixer(log, conv_subsumed=True)
    assert isinstance(gdn.slots.conv, Unbound), "the attached core claims conv.short, so the slot is out"
    x, seq = _x(1), _seq()
    with log, torch.no_grad():
        gdn(x, seq)
    moved = [o for o in spy.in_proj_to_core if o in MOVES]
    assert moved == [], spy.in_proj_to_core  # pure views between the GEMM and the core

    q, k, v, alpha, beta = spy.args
    base = q.untyped_storage().data_ptr()
    assert all(t.untyped_storage().data_ptr() == base for t in (k, v, alpha, beta))
    widths = {r: getattr(gdn, r).shape[0] for r in ROLES}
    off, offsets = 0, {}
    for r in ROLES:
        offsets[r] = off
        off += widths[r]
    hk, hv = gdn.key_heads_local, gdn.value_heads_local
    dk, dv = gdn.key_head_dim, gdn.value_head_dim
    assert q.shape == (T, hk, dk) and q.storage_offset() == offsets["q"]
    assert k.shape == (T, hk, dk) and k.storage_offset() == offsets["k"]
    assert v.shape == (T, hv, dv) and v.storage_offset() == offsets["v"]
    assert beta.shape == (T, hv) and beta.storage_offset() == offsets["beta"]
    assert alpha.shape == (T, hv) and alpha.storage_offset() == offsets["alpha"]
    assert q.stride(0) == k.stride(0) == v.stride(0) == off  # one row of the packed GEMM output


def test_a_bound_conv_short_reads_the_qkv_window_then_splits():
    """A separate convolution receives the q|k|v channels as ONE column window of the in_proj output (no
    cat, no copy), then returns the same segment layout."""
    log = _AtenLog()
    gdn, spy = _mixer(log, conv_subsumed=False)
    assert not isinstance(gdn.slots.conv, Unbound)
    x, seq = _x(2), _seq()
    with log, torch.no_grad():
        gdn(x, seq)
    assert "aten.cat" not in spy.in_proj_to_core  # the conv reads the adjacent q|k|v columns in place
    assert len([o for o in spy.up_to_core if o in GEMMS]) == 1
    q, _, v, _, _ = spy.args
    assert q.shape == (T, gdn.key_heads_local, gdn.key_head_dim)
    assert v.shape == (T, gdn.value_heads_local, gdn.value_head_dim)




