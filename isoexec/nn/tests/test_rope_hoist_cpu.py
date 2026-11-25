"""Count one rotary-angle table per forward across component rows. Count inv_freq to cover both 1-D
and MRoPE paths; numerical parity alone cannot detect redundant recomputation.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest
import torch
from torch import nn

from isoexec.core import rope as rope_mod  # where the angle table is DEFINED, so a patch here is seen
from isoexec.nn import MLP, Attention, DecoderBlock, place
from isoexec.ops.table import OpTable
from isoexec.schema import make_sequence_context

from isoexec.nn.tests import _fixtures as F

#: Two packed sequences of 9 and 14 tokens. Packed, because that is what both hosts hand a forward.
CU_SEQLENS = (0, 9, 23)

#: The rope schedule keys one cached angle table assumes every attention row agrees on. The
#: component tier has no build-time refusal for a disagreement, so this tuple is what a shipped
#: composition is checked to agree about.
SCHEDULE = ("rope_theta", "rotary_dim", "rope_style", "mrope_section", "mrope_interleaved")


class _Counter:
    """Counts every angle-table build, whoever makes it.

    ``inv_freq`` is looked up as a module global by both arms of ``resolve_rope``'s closure, so
    rebinding it here is seen by a table already constructed -- which every placed ``Attention``
    holds, built in its constructor.
    """

    def __init__(self) -> None:
        self.real, self.n = rope_mod.inv_freq, 0

    def __enter__(self):
        def counted(theta, rotary_dim, device=None):
            self.n += 1
            return self.real(theta, rotary_dim, device)

        rope_mod.inv_freq = counted
        return self

    def __exit__(self, *exc):
        rope_mod.inv_freq = self.real
        return False




class _Rows(nn.Module):
    """Two decoder rows over two softmax mixers, iterated. Two, because one attention row can never
    show a per-row recompute -- it computes once either way -- which is exactly why a single-row
    fixture is blind to this."""

    def __init__(self, n: int = 2, *, device: str = "cpu") -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderBlock(
                mixer=F.component(F.spec(), device=device),
                mlp=MLP(hidden=F.HIDDEN, intermediate=512, device=device),
                hidden=F.HIDDEN,
                eps=F.EPS,
                device=device,
            )
            for _ in range(n)
        )

    def forward(self, h, seq, aux=None):
        for block in self.layers:
            h = block(h, seq, aux)
        return h


def _fill(module: nn.Module, *, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in module.named_parameters():
            p.copy_((torch.randn(p.shape, generator=gen) * 0.05).to(p.dtype))


def _sequence(*, mrope: bool = False):
    """A fresh packed-batch context, fields empty -- which is how a host builds one."""
    cu = torch.tensor(CU_SEQLENS, dtype=torch.int32)
    pos = torch.cat([torch.arange(b - a) for a, b in itertools.pairwise(CU_SEQLENS)]).to(torch.int64)
    max_q = max(b - a for a, b in itertools.pairwise(CU_SEQLENS))
    return make_sequence_context(pos.repeat(3, 1) if mrope else pos, cu, max_q=max_q)


def _rows_and_input(seed: int = 0):
    rows = _Rows()
    place(rows, F.placement(), prefix="lm", op_table=OpTable(mode="reference"))
    _fill(rows, seed=seed)
    gen = torch.Generator().manual_seed(seed + 100)
    h = (torch.randn(CU_SEQLENS[-1], F.HIDDEN, generator=gen) * 0.5).to(torch.bfloat16)
    return rows, h


def test_two_component_rows_build_the_angle_table_exactly_once():
    """Both rows share one sequence-context angle table per forward."""
    rows, h = _rows_and_input()
    seq = _sequence()
    assert seq.rope_freqs is None and seq.rope_cos_sin is None, "a host builds a context with the fields empty"
    with _Counter() as c, torch.no_grad():
        out = rows(h, seq, {})
    assert c.n == 1, f"two attention rows made {c.n} angle tables; the hoist is gone"

    # ... and what they were handed is bit-for-bit the table each row would have built for itself:
    # the recompute this saves was redundant, not merely cheap.
    mixer = rows.layers[0].mixer
    want = rope_mod.rotary_freqs(seq.positions, mixer.rope_theta, mixer.facts()["rotary_dim"])
    assert seq.rope_freqs is not None and seq.rope_freqs.dtype == torch.float32
    assert tuple(seq.rope_freqs.shape) == (CU_SEQLENS[-1], mixer.facts()["rotary_dim"] // 2)
    assert torch.equal(seq.rope_freqs.view(torch.int32), want.view(torch.int32))
    assert torch.isfinite(out.float()).all()


def test_carrying_the_table_moves_no_bits():
    """The same forward with the table carried and with it rebuilt per row must agree bit for bit.
    The per-row arm is produced by clearing the field between rows -- the pre-hoist behaviour."""
    rows, h = _rows_and_input(seed=2)

    def run(per_row: bool):
        seq = _sequence()
        with torch.no_grad():
            x = h
            for block in rows.layers:
                if per_row:
                    seq.rope_freqs = None
                x = block(x, seq, {})
            return x

    assert torch.equal(F.bits(run(False)), F.bits(run(True)))


def test_the_forward_writes_rope_freqs_and_nothing_else_of_the_hosts():
    """The one mutation this tier performs on a context the host owns. ``rope_cos_sin`` has no
    writer -- no bound ``rope.apply`` derives a second table -- so a forward that filled it would be
    paying for a field nothing reads."""
    rows, h = _rows_and_input(seed=3)
    seq = _sequence()
    before = (seq.positions, seq.cu_seqlens, seq.max_q, seq.max_kv, seq.input_ids)
    with torch.no_grad():
        rows(h, seq, {})
    after = (seq.positions, seq.cu_seqlens, seq.max_q, seq.max_kv, seq.input_ids)
    assert all(a is b for a, b in zip(before, after)), "the forward rewrote a field that is the host's"
    assert seq.rope_freqs is not None
    assert seq.rope_cos_sin is None




@pytest.mark.parametrize("name", ["qwen3_5_moe/released_35b_a3b", "qwen3_5/released_9b", "qwen3_5/released_27b"])
def test_every_attention_row_of_a_shipped_model_wants_the_same_angles(name):
    """Check every shipped attention row uses the same rotary schedule before relying on one cached
    table.
    """
    from isoexec.nn.tests import _packages

    rows = [m for m in _packages.build_meta(name).modules() if isinstance(m, Attention)]
    assert len(rows) >= 2
    schedules = {tuple(row.facts()[k] for k in SCHEDULE) for row in rows}
    assert len(schedules) == 1, f"attention rows disagree on {SCHEDULE}: {sorted(schedules, key=repr)}"


@pytest.mark.parametrize("reference_angles", [False, True])
def test_the_rope_binding_owns_angle_preparation(reference_angles, monkeypatch):
    """Both bindings receive one contract; only the reference prepares forward angles."""
    from isoexec.ops import table as _resolve

    seen: list[object] = []
    real = _resolve.OpTable.bind

    def probe(self, op, **kw):
        got = real(self, op, **kw)
        if op != "rope.apply":
            return got

        def rope(q, k, freqs, *, style, rotary_dim, seq=None, rotary=None, input_pairs=None):
            seen.append((freqs, rotary))
            if reference_angles:
                return got.fn(q, k, freqs, style=style, rotary_dim=rotary_dim,
                              seq=seq, rotary=rotary, input_pairs=input_pairs)
            return q, k

        return dataclasses.replace(got, fn=rope)

    monkeypatch.setattr(_resolve.OpTable, "bind", probe)
    rows, h = _rows_and_input(seed=5)
    seq = _sequence()
    with _Counter() as c, torch.no_grad():
        rows(h, seq, {})

    assert len(seen) == 2
    assert all(freqs is None and rotary is not None for freqs, rotary in seen)
    assert (seq.rope_freqs is not None) is reference_angles
    assert c.n == int(reference_angles)


def test_a_fused_owner_is_handed_this_rows_own_mrope_layout(monkeypatch):
    """Pass the component's MRoPE section and interleaving data into its fused binding."""
    from isoexec.ops import table as _resolve

    calls: list[dict] = []
    real = _resolve.OpTable.bind

    def probe(self, op, **kw):
        got = real(self, op, **kw)
        if op != "norms.qk":
            return got

        def fused(q, k, wq, wk, eps, *, seq, rotary, spec):
            calls.append({"eps": eps, "rotary": rotary, "spec": spec})
            return q, k

        fused.owns = {"norms.qk": got.impl, "rope.apply": got.impl}
        return dataclasses.replace(got, fn=fused, owns={"norms.qk": got.impl, "rope.apply": got.impl})

    monkeypatch.setattr(_resolve.OpTable, "bind", probe)
    row = F.component(F.spec(mrope=True))
    row.apply_placement(F.placement(), site=F.SITE, op_table=OpTable(mode="reference"))
    assert row.slots.unfilled()["qk_norm"].fill_verb.startswith("nothing does:")

    _fill(row, seed=9)
    seq = _sequence(mrope=True)
    gen = torch.Generator().manual_seed(21)
    h = (torch.randn(CU_SEQLENS[-1], F.HIDDEN, generator=gen) * 0.5).to(torch.bfloat16)
    with torch.no_grad():
        row(h, seq)

    assert len(calls) == 1, f"the fused owner ran {len(calls)} times for one row"
    got = calls[0]
    assert got["eps"] == F.EPS
    assert got["spec"].mrope_section == row.mrope_section == (3, 3, 2)
    assert got["spec"].mrope_interleaved is row.mrope_interleaved is True
    assert got["rotary"] is row._rotary, "the stage must hand the kernel this row's own rotary tables"


@pytest.mark.parametrize("input_pairs", ["neox", "interleaved"])
def test_mla_passes_raw_lanes_to_the_production_rope_factory(input_pairs, monkeypatch):
    """Exercise the shipped factory contract on CPU without building tables or launching kernels."""
    from isoexec.core.bindings import BoundOp
    from isoexec.nn import LatentAttention
    from isoexec.ops.attention.rope_sm90 import integration, ops

    calls = []
    table = object()
    monkeypatch.setattr(integration._TableSet, "get", lambda *args: table)

    def rope(q, k, positions, tab, **kwargs):
        calls.append((q, k, positions, tab, kwargs))
        return q, k

    monkeypatch.setattr(ops, "rope", rope)
    mixer = LatentAttention(hidden=16, heads=2, q_lora_rank=8, kv_lora_rank=8,
                            qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
                            rope_theta=10000, rope_style=input_pairs)
    fn = integration.rope_apply_for((10000.0, 4, None, False))
    mixer.slots.rope = BoundOp("rope.apply", integration.ROPE_IMPL, 1, "ops.attention", {}, fn)
    q = torch.randn(23, 2, 4, requires_grad=True)
    k = torch.randn(23, 1, 4, requires_grad=True)
    seq = _sequence()
    with _Counter() as counter:
        got_q, got_k = mixer._rotate(q, k, seq)
    assert got_q is q and got_k is k
    assert len(calls) == 1
    seen_q, seen_k, positions, tab, kwargs = calls[0]
    assert seen_q is q and seen_k is k and positions is seq.positions and tab is table
    assert kwargs == {"style": input_pairs, "out_style": "neox"}
    assert counter.n == 0 and seq.rope_freqs is None


def test_reference_pair_conversion_preserves_unrotated_lanes():
    from isoexec.ops.attention.reference import ref_rope_apply

    q, k = torch.randn(23, 2, 8), torch.randn(23, 1, 8)
    freqs = rope_mod.rotary_freqs(_sequence().positions, 10000.0, 4)

    def gathered(x):
        return torch.cat((x[..., :4:2], x[..., 1:4:2], x[..., 4:]), dim=-1)

    got = ref_rope_apply(q, k, freqs, style="neox", rotary_dim=4, input_pairs="interleaved")
    expected = ref_rope_apply(gathered(q), gathered(k), freqs, style="neox", rotary_dim=4)
    for actual, wanted, source in zip(got, expected, (q, k)):
        assert torch.equal(actual, wanted)
        assert torch.equal(actual[..., 4:], source[..., 4:])
