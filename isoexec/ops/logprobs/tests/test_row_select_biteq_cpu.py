"""Check sampled_rows(H[idx]) == identity(H)[idx] on raw bit patterns, including signed zero.
Engine selection is along token dim 0; downstream head row invariance is checked by the GPU battery."""

from __future__ import annotations

import pytest
import torch

from isoexec.ops._testing import assert_hazard_exercised, bitwise_equal
from isoexec.ops.logprobs.eager import lm_head_slice_identity
from isoexec.ops.logprobs.reference import ref_head
from isoexec.runtimes.engine_ops import SLICE_IMPL, lm_head_slice_sampled_rows

HIDDEN = 24
TOKENS = 17

#: The dim-0 selections a real host hands the stage. The int-tensor forms are vLLM's
#: ``logits_indices`` / SGLang's ``last_index``; ``slice`` is vLLM's prompt-logprob window and
#: SGLang's ``[pt + start_len : pt + extend_len]``.
SELECTIONS: tuple[tuple[str, object], ...] = (
    ("every row (the degenerate selection)", torch.arange(TOKENS)),
    ("last row only (greedy decode)", torch.tensor([TOKENS - 1])),
    ("scattered ascending", torch.tensor([0, 3, 4, 9, 16])),
    ("out of order", torch.tensor([9, 0, 16, 3, 4])),
    ("strictly descending", torch.arange(TOKENS - 1, -1, -1)),
    ("repeated indices", torch.tensor([5, 5, 5, 2, 5, 2])),
    ("repeated and out of order", torch.tensor([16, 16, 0, 16, 0, 8])),
    ("empty selection", torch.empty(0, dtype=torch.long)),
    ("a dim-0 slice view (vLLM prompt logprobs)", slice(3, 11)),
    ("a dim-0 slice view starting at 0", slice(0, 5)),
    ("an empty dim-0 slice", slice(4, 4)),
)

DTYPES = (torch.float32, torch.bfloat16)


def _hidden(dtype: torch.dtype, seed: int = 0, rows: int = TOKENS) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, HIDDEN, generator=gen, dtype=torch.float32) * 6.0).to(dtype)


def _plant_specials(h: torch.Tensor) -> torch.Tensor:
    """Signed zeros, both infinities and two DISTINCT NaN payloads, on rows a selection reaches.

    The payloads matter: a stage that "passed the NaNs through" by recomputing them (``x + 0.0``,
    ``x * 1.0`` on some backends, any cast round-trip) would hand back a canonical quiet NaN and
    lose the payload bits, which ``bitwise_equal`` sees and ``torch.isnan`` does not.
    """
    h = h.clone()
    h[0, 0] = -0.0
    h[0, 1] = 0.0
    h[3, :4] = torch.tensor([-0.0, 0.0, -0.0, 0.0], dtype=h.dtype)
    h[4, 0] = float("inf")
    h[4, 1] = float("-inf")
    h[9, 2] = float("inf")
    h[16, 0] = float("nan")
    if h.dtype is torch.float32:
        h[16, 1] = torch.tensor([0x7FC0_00AB], dtype=torch.int32).view(torch.float32)[0]
        h[16, 2] = torch.tensor([-1], dtype=torch.int32).view(torch.float32)[0]  # 0xFFFFFFFF, a NaN
    else:
        h[16, 1] = torch.tensor([0x7FAB], dtype=torch.int16).view(torch.bfloat16)[0]
        h[16, 2] = torch.tensor([-1], dtype=torch.int16).view(torch.bfloat16)[0]  # 0xFFFF, a NaN
    return h


def _selected_rows(sel) -> list[int]:
    """The row numbers ``sel`` names, for both the int-tensor and the slice spellings."""
    if isinstance(sel, slice):
        return list(range(*sel.indices(TOKENS)))
    return [int(i) for i in sel.tolist()]


def _int_bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32 if t.dtype is torch.float32 else torch.int16)


def _trainer_rows(h: torch.Tensor, sel) -> torch.Tensor:
    """The rows the TRAINER produces at ``sel``: score every row, then look at the shared ones."""
    return lm_head_slice_identity(h)[sel]


def _engine_rows(h: torch.Tensor, sel) -> torch.Tensor:
    """The rows the ENGINE produces: the host selects first, the stage sees only those."""
    return lm_head_slice_sampled_rows(h[sel])


# the mechanism: neither member touches its argument at all


@pytest.mark.parametrize("dtype", DTYPES, ids=("fp32", "bf16"))
def test_the_fixture_really_carries_the_bit_patterns_the_cases_claim(dtype):
    """Guard the hazards themselves: if the planting were canonicalized on assignment, every
    "NaN payload" case below would be quietly testing one quiet NaN."""
    h = _plant_specials(_hidden(dtype))
    bits = _int_bits(h)
    assert int((bits == torch.iinfo(bits.dtype).min).sum()) == 3  # -0.0, and only where planted
    assert int(torch.isinf(h).sum()) == 3
    assert len({int(b) for b in bits[torch.isnan(h)].tolist()}) == 3  # three DISTINCT payloads
    assert int((h == 0).sum()) == 6  # three +0.0 and three -0.0, indistinguishable to ``==``


def test_both_members_return_the_very_object_they_were_handed():
    """``is``, not ``bitwise_equal``: the reason the square below holds is that there is no copy.

    A stage that cloned, cast, called ``.contiguous()`` or added a zero would still pass many bit
    comparisons; it would not pass this. This is the strongest statement available about the pair
    and it is what makes the twin declaration honest rather than lucky.
    """
    for dtype in DTYPES:
        h = _plant_specials(_hidden(dtype))
        assert lm_head_slice_identity(h) is h
        assert lm_head_slice_sampled_rows(h) is h
    # ... and on the awkward layouts too, where a hidden ``.contiguous()`` would be tempting.
    view = _hidden(torch.float32).t()
    assert lm_head_slice_identity(view) is view
    assert lm_head_slice_sampled_rows(view) is view
    empty = torch.empty(0, HIDDEN)
    assert lm_head_slice_identity(empty) is empty
    assert lm_head_slice_sampled_rows(empty) is empty


def test_the_two_members_are_declared_a_twin_class_and_are_the_only_two():
    """The claim under test is a DECLARATION; pin the test to it so a third member cannot appear
    silently and inherit this evidence."""
    from isoexec.core.registry_build import build_registry

    impls = build_registry().get_op("logprobs.lm_head_slice").impls
    members = {i: s for i, s in impls.items() if getattr(s, "twin_class", None) == "logprobs.row_select"}
    assert set(members) == {"identity", SLICE_IMPL} == {"identity", "sampled_rows"}
    assert {s.version for s in members.values()} == {1}


# the claim: selection commutes with the stage, bit for bit


@pytest.mark.parametrize("dtype", DTYPES, ids=("fp32", "bf16"))
@pytest.mark.parametrize("label,sel", SELECTIONS, ids=[s[0] for s in SELECTIONS])
def test_the_selection_commutes_with_the_stage_bitwise(dtype, label, sel):
    h = _hidden(dtype, seed=11)
    trainer, engine = _trainer_rows(h, sel), _engine_rows(h, sel)
    assert trainer.dtype is engine.dtype is dtype
    assert bitwise_equal(trainer, engine), label


@pytest.mark.parametrize("dtype", DTYPES, ids=("fp32", "bf16"))
@pytest.mark.parametrize("label,sel", SELECTIONS, ids=[s[0] for s in SELECTIONS])
def test_the_square_holds_over_signed_zeros_infinities_and_nan_payloads(dtype, label, sel):
    """The same square on a hidden state carrying every bit pattern the stream can carry.

    The hazards are asserted to have FIRED in the selected rows, not merely to exist in ``h``: a
    selection that missed them would make the case vacuous.
    """
    h = _plant_specials(_hidden(dtype, seed=11))
    trainer, engine = _trainer_rows(h, sel), _engine_rows(h, sel)
    assert bitwise_equal(trainer, engine), label

    rows = _selected_rows(sel)
    if not rows:  # the empty selections carry no hazard by construction
        assert trainer.shape == engine.shape == (0, HIDDEN)
        return
    bits = _int_bits(trainer)
    neg_zero = int((bits == torch.iinfo(bits.dtype).min).sum())  # 0x80000000 / 0x8000 exactly
    nan_patterns = {int(b) for b in bits[torch.isnan(trainer)].tolist()}
    infs = int(torch.isinf(trainer).sum())
    # Each hazard is asserted where the PLANTED rows say it must be, so a selection that happened
    # to miss it cannot pass this case quietly.
    if {0, 3} & set(rows):
        assert_hazard_exercised(f"signed zero in the selected rows ({label})", neg_zero)
    if 16 in rows:
        # Three DISTINCT NaN bit patterns were planted in row 16; a stage that recomputed them
        # would hand back one canonical quiet NaN and this count would collapse.
        assert_hazard_exercised(f"distinct NaN payloads in the selected rows ({label})", len(nan_patterns) >= 3)
    if {4, 9} & set(rows):
        assert_hazard_exercised(f"infinities in the selected rows ({label})", infs)


@pytest.mark.parametrize("dtype", DTYPES, ids=("fp32", "bf16"))
def test_the_square_holds_on_non_contiguous_hidden_states(dtype):
    """Check offset, transposed and strided hidden-state views without changing their bytes."""
    base = _plant_specials(_hidden(dtype, seed=3, rows=TOKENS * 2))
    layouts = {
        "storage-offset row view": base[5 : 5 + TOKENS],
        "transpose-of-transpose (stride-swapped)": base[:TOKENS].t().t(),
        "every other row": base[::2][:TOKENS],
        "column-strided view": base[:TOKENS, ::2],
        "expanded (stride-0) rows": base[:1].expand(TOKENS, HIDDEN),
    }
    for name, h in layouts.items():
        assert h.shape[0] == TOKENS, name
        for label, sel in SELECTIONS:
            assert bitwise_equal(_trainer_rows(h, sel), _engine_rows(h, sel)), f"{name} / {label}"
    assert not layouts["column-strided view"].is_contiguous()
    assert not layouts["every other row"].is_contiguous()


@pytest.mark.parametrize("dtype", DTYPES, ids=("fp32", "bf16"))
def test_an_empty_selection_keeps_the_shape_and_the_dtype_on_both_sides(dtype):
    """An engine step can select nothing (a chunked prefill that produced no logit row). The two
    sides must agree that the answer is ``[0, hidden]`` of the stream dtype, not a scalar, not a
    zero-width row, and not an error."""
    h = _plant_specials(_hidden(dtype))
    for sel in (torch.empty(0, dtype=torch.long), slice(4, 4), slice(TOKENS, TOKENS)):
        trainer, engine = _trainer_rows(h, sel), _engine_rows(h, sel)
        assert trainer.shape == engine.shape == (0, HIDDEN)
        assert bitwise_equal(trainer, engine)


# the index convention: which axis, and what "row i" means on the far side of the head


def test_the_row_axis_is_dim_zero_and_dim_one_is_a_different_answer():
    """Teeth on the axis. ``hidden_states[idx]`` is dim-0; if either side ever read it as dim-1 the
    square would break, so show that the dim-1 reading really is a different tensor -- both at a
    rectangular shape (where it differs in SHAPE) and at a square one (where it differs only in
    VALUES, which is the case a shape assertion would miss)."""
    rect = _hidden(torch.float32, seed=7)  # [17, 24]
    idx = torch.tensor([0, 3, 4, 9, 16])
    assert rect.shape[0] != rect.shape[1]
    assert not bitwise_equal(rect[idx], rect[:, idx[idx < rect.shape[1]]])

    gen = torch.Generator().manual_seed(8)
    sq = torch.randn(HIDDEN, HIDDEN, generator=gen)
    sq_idx = torch.tensor([0, 3, 4, 9, 16])
    assert sq[sq_idx].shape == sq[:, sq_idx].t().shape  # same shape: only the values differ
    assert not bitwise_equal(sq[sq_idx], sq[:, sq_idx].t())
    # and the square itself still holds on the square tensor, dim-0
    assert bitwise_equal(_trainer_rows(sq, sq_idx), _engine_rows(sq, sq_idx))


def test_the_head_chain_maps_row_i_of_hidden_to_row_i_of_logits():
    """Check token-row identity through the reference head using exact one-hot weights.
    This tests layout without conflating it with the GPU kernel's rounding schedule."""
    n_vocab = 2 * HIDDEN + 5
    w = torch.zeros(n_vocab, HIDDEN)
    w[torch.arange(n_vocab), torch.arange(n_vocab) % HIDDEN] = 1.0  # one-hot rows
    h = _hidden(torch.float32, seed=21)
    assert bool(torch.isfinite(h).all()) and int((h == 0).sum()) == 0  # keep the GEMM exact

    logits = ref_head(lm_head_slice_identity(h), w)
    for i in range(TOKENS):  # row i of hidden -> row i of logits, element for element
        assert bitwise_equal(logits[i], h[i, torch.arange(n_vocab) % HIDDEN])

    for label, sel in SELECTIONS:
        trainer = ref_head(lm_head_slice_identity(h), w)[sel]
        engine = ref_head(lm_head_slice_sampled_rows(h[sel]), w)
        assert bitwise_equal(trainer, engine), label


# non-vacuity: what this file would catch


def test_the_comparator_would_see_a_sign_of_zero_flip():
    """``bitwise_equal`` over ``torch.equal`` is load-bearing HERE, not in the abstract: the
    selected rows really do carry -0.0, and ``x + 0.0`` -- the most innocent thing a stage could
    ever grow -- normalizes it away while leaving every other element alone."""
    for dtype in DTYPES:
        h = _plant_specials(_hidden(dtype))
        sel = torch.tensor([0, 3, 4, 9, 16])
        trainer = _trainer_rows(h, sel)
        mutant = lm_head_slice_sampled_rows(h[sel]) + 0.0
        assert not bitwise_equal(trainer, mutant), dtype
        # ... and the finite part is torch.equal, so torch.equal alone would have passed the mutant.
        finite = torch.isfinite(trainer)
        assert torch.equal(trainer[finite], mutant[finite])
        differing = int((_int_bits(trainer[finite]) != _int_bits(mutant[finite])).sum())
        assert differing == int((_int_bits(trainer[finite]) == torch.iinfo(_int_bits(trainer).dtype).min).sum())
        assert differing > 0, dtype


@pytest.mark.parametrize(
    "name,mutant",
    [
        ("adds a zero (kills -0.0)", lambda h: h + 0.0),
        ("round-trips through bf16", lambda h: h.to(torch.bfloat16).to(h.dtype)),
        ("canonicalizes NaN payloads", lambda h: torch.where(torch.isnan(h), torch.full_like(h, float("nan")), h)),
        ("scales by one ulp", lambda h: torch.where(torch.isfinite(h), torch.nextafter(h, torch.full_like(h, 1e30)), h)),
        ("reverses the rows", lambda h: h.flip(0)),
    ],
)
def test_a_stage_that_touched_the_bits_or_the_order_is_caught(name, mutant):
    """Each mutant is a thing ``sampled_rows`` could plausibly become. None of them may pass."""
    h = _plant_specials(_hidden(torch.float32, seed=11))
    sel = torch.tensor([0, 3, 4, 9, 16])
    assert not bitwise_equal(_trainer_rows(h, sel), mutant(h[sel])), name


@pytest.mark.parametrize("shift", [1, -1])
def test_an_off_by_one_selection_is_caught(shift):
    """The convention is an index, so the failure mode with no numeric signature is an index that
    is off by one. It must not survive the square."""
    h = _hidden(torch.float32, seed=11)
    sel = torch.tensor([2, 5, 9, 13])
    assert not bitwise_equal(_trainer_rows(h, sel), _engine_rows(h, sel + shift))
