"""The trainer binding of the CED program on the CPU reference: rollout turns, SkyRL's padded batches, per-
segment scoring equal to the engine's logprobs, and a deterministic backward through the segmented decoder
into decoder.entry and the encoder."""
import pytest
import torch

from isoexec.models.deepseek_v4_1_flash import reference as R
from isoexec.models.deepseek_v4_1_flash.config import BOUNDED_REPLAY
from isoexec.models.deepseek_v4_1_flash.program import DecoderSegments
from isoexec.models.deepseek_v4_1_flash.training_ced import (
    RolloutTurns,
    pack_padded,
    scatter_to_response,
    score,
    single_turn,
)


@pytest.fixture(scope="module")
def ref():
    return R.tiny(seed=3, window=8)


def tokens(n, seed):
    return torch.randint(0, 64, (n,), generator=torch.Generator().manual_seed(seed))


def _padded(prompts, responses, P, A, pad=2):
    ids = torch.full((len(prompts), P + A), pad)
    mask = torch.zeros_like(ids)
    for b, (p, r) in enumerate(zip(prompts, responses)):
        ids[b, P - len(p):P], mask[b, P - len(p):P] = p, 1
        ids[b, P:P + len(r)], mask[b, P:P + len(r)] = r, 1
    return ids, mask


def _trainer_logprobs(ref, batch, turns, A):
    segs = DecoderSegments.for_program(BOUNDED_REPLAY, batch.cu_seqlens, turns.turns(), ref.W)
    seqs = [batch.ids[a:b] for a, b in zip(batch.cu_seqlens[:-1], batch.cu_seqlens[1:])]
    logits, _ = R.ced_forward(ref, seqs, segs)
    values = score(logits, segs, batch.ids, R.Reference.logprobs)
    return scatter_to_response(values, segs, batch, A)


def test_padded_rollout_scores_like_the_engine(ref):
    prompts = [tokens(13, 1), tokens(5, 2), tokens(20, 3)]
    responses = [tokens(6, 4), tokens(3, 5), tokens(7, 6)]
    P, A = 22, 8
    ids, mask = _padded(prompts, responses, P, A)
    batch = pack_padded(ids, mask, A)
    loss_mask = mask[:, P:]
    turns = single_turn(batch.prompt_lens, batch.response_lens, loss_mask)
    got = _trainer_logprobs(ref, batch, turns, A)
    engine = R.ReferenceEngine(ref, BOUNDED_REPLAY)
    for b, (p, r) in enumerate(zip(prompts, responses)):
        logits, _ = R.rollout(engine, p, r, [len(p)])
        want = R.Reference.logprobs(logits, r)
        assert torch.equal(got[b, :len(r)], want) and not got[b, len(r):].any()


def test_multi_turn_needs_explicit_turns(ref):
    conv = tokens(11 + 4 + 6 + 5, 9)
    loss_mask = torch.tensor([[1] * 4 + [0] * 6 + [1] * 5])
    with pytest.raises(ValueError, match="CED_TURNS_REQUIRED"):
        single_turn([11], [15], loss_mask)
    turns = RolloutTurns((((11, 4), (21, 5)),))
    batch = pack_padded(conv.view(1, -1), torch.ones(1, conv.numel(), dtype=torch.long), 15)
    got = _trainer_logprobs(ref, batch, turns, 15)
    engine = R.ReferenceEngine(ref, BOUNDED_REPLAY)
    l1, _ = R.rollout(engine, conv[:11], conv[11:15], [11])
    l2, _ = R.rollout(engine, conv[:21], conv[21:], [21])
    assert torch.equal(got[0, :4], R.Reference.logprobs(l1, conv[11:15]))
    assert torch.equal(got[0, 10:], R.Reference.logprobs(l2, conv[21:]))
    assert not got[0, 4:10].any()                    # observation tokens are not scored
    with pytest.raises(ValueError):
        RolloutTurns((((11, 4), (13, 5)),)).turns()  # a turn starting inside the previous one


def test_backward_through_the_segmented_decoder_is_deterministic_and_reaches_the_encoder(ref):
    b = ref.cfg.boundary
    names = ["layers.2.attn.compressor.wkv.weight",            # encoder global KV source
             "layers.0.attn.wq_a.weight",                       # encoder layer below the replay window
             f"layers.{b}.attn.compressor.wkv.weight",          # decoder.entry: the decoder global KV
             f"layers.{b + 2}.attn.wq_a.weight"]                # a decoder block
    seq = tokens(30, 11)
    grads = []
    for _ in range(2):
        for n in names:
            ref.p[n] = ref.p[n].detach().requires_grad_(True)
        turns = RolloutTurns((((24, 6),),))
        batch = pack_padded(seq.view(1, -1), torch.ones(1, 30, dtype=torch.long), 6)
        loss = -_trainer_logprobs(ref, batch, turns, 6).sum()
        loss.backward()
        grads.append([ref.p[n].grad.clone() for n in names])
        for n in names:
            ref.p[n] = ref.p[n].detach()
    for g in grads[0]:
        assert torch.isfinite(g.float()).all() and g.abs().sum() > 0
    assert all(torch.equal(a, c) for a, c in zip(*grads))
