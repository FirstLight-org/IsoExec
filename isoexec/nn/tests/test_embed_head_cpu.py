"""Embedding and head components: the trainer's resolved stages, fp32 output boundaries and padding
trim, ties, and the engine stages that require adapter attachment.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from isoexec.nn._slots import UnfilledSlotError
from isoexec.ops.logprobs.reference import ref_lm_head_slice, ref_log_softmax
from isoexec.ops.table import OpTable

from isoexec.nn.tests import _moe_fixtures as F

TOKENS = 12


def _pair(*, tied: bool, seed: int):
    """A placed embedding and head on the reference table, carrying seeded bytes."""
    embed = F.embedding(device="cpu")
    head = F.head(tied_to=embed.weight if tied else None, device="cpu")
    embed.apply_placement(F.placement(), site="lm.embed", op_table=OpTable(mode="reference"))
    head.apply_placement(F.placement(), site="lm.head", op_table=OpTable(mode="reference"))
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        embed.weight.copy_(torch.randn(F.PADDED_VOCAB, F.HIDDEN, generator=gen).to(torch.bfloat16))
        if not tied:
            head.weight.copy_(torch.randn(F.PADDED_VOCAB, F.HIDDEN, generator=gen).to(torch.bfloat16))
    return embed, head


def _hidden(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(TOKENS, F.HIDDEN, generator=gen, dtype=torch.float32).to(torch.bfloat16)


def _ids(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, F.VOCAB, (TOKENS,), generator=gen, dtype=torch.int64)


@pytest.mark.parametrize("tied", [False, True])
def test_the_embedding_lookup_is_a_row_gather(tied):
    embed, _ = _pair(tied=tied, seed=5)
    ids = _ids(9)
    with torch.no_grad():
        assert torch.equal(F.bits(embed(ids)), F.bits(embed.weight[ids]))


@pytest.mark.parametrize("tied", [False, True])
def test_compute_logits_uses_the_row_selection_the_trainer_resolved(tied):
    """A trainer's slice is the identity over every row, resolved like any other stateless stage --
    so the identity report shows the reference provider and not a ``test`` substitution."""
    _, head = _pair(tied=tied, seed=5)
    slice_stage = head.slots.resolved()["lm_head_slice"]
    assert slice_stage.provider == "ops.reference" and slice_stage.fn is ref_lm_head_slice
    assert head.slots.resolved()["log_softmax"].fn is ref_log_softmax
    with torch.no_grad():
        logits = head(_hidden(13))
    assert logits.dtype is torch.float32
    assert logits.shape == (TOKENS, F.VOCAB)  # padding columns trimmed


def test_an_engine_head_whose_stages_nothing_filled_refuses_as_an_installation_mistake():
    """The two slots an engine genuinely owes: the sampler's rows and the engine's own group."""
    head = F.head(device="cpu")
    head.apply_placement(F.placement(side="engine"), site="lm.head", op_table=OpTable(mode="reference"))
    with pytest.raises(UnfilledSlotError, match=r"lm\.head\.lm_head_slice: .*engine adapter"):
        head(_hidden(13))
    with pytest.raises(UnfilledSlotError, match=r"lm\.head\.log_softmax: .*vocab-parallel"):
        head.log_probs(torch.zeros(2, 4), None, 1.0)


def test_a_tied_head_reads_the_embedding_bytes():
    """The tied head holds the embedding's ``nn.Parameter`` object, so changing the embedding moves
    the head's output."""
    embed, head = _pair(tied=True, seed=5)
    assert head.weight is embed.weight
    h = _hidden(13)
    with torch.no_grad():
        before = head.head_output(h).clone()
        embed.weight.mul_(2)
        after = head.head_output(h)
    assert not torch.equal(F.bits(before), F.bits(after))


def test_the_embedding_records_the_vocab_parallel_movement_at_tp_above_one():
    """The registered impl id is preserved -- the arithmetic is identical at every degree -- and the
    movement is recorded as an input instead."""
    embed = F.embedding()
    embed.apply_placement(F.placement(tp=4), site="lm.embed", op_table=OpTable(mode="reference"))
    lookup = embed.slots.resolved()["lookup"]
    assert lookup.op == "embed.lookup" and lookup.impl == "ref.embedding"
    assert lookup.inputs["vocab_parallel"] == "select_masked_then_tree_all_reduce"
    assert embed.slots.lookup.__name__ == "vocab_parallel_embed_tp4"

    plain = F.embedding()
    plain.apply_placement(F.placement(tp=1), site="lm.embed", op_table=OpTable(mode="reference"))
    assert "vocab_parallel" not in plain.slots.resolved()["lookup"].inputs


def test_the_two_ends_hold_no_stray_modules():
    """Both are leaves: the walk refuses a parameter with no axis, and neither end has submodules."""
    for piece in (F.embedding(), F.head()):
        assert list(piece.named_children()) == []
        assert set(dict(piece.named_parameters())) == {"weight"}
    assert isinstance(F.embedding(), nn.Module)
