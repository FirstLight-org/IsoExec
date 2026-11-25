"""The engine requests behind each trained sequence, derived from SkyRL's batch and carried into V4.1's trainer
forward: the decoder segments equal the engine planner's for the same requests, by construction."""

import random
from types import SimpleNamespace as NS

import pytest
import torch

from isoexec.integrations.skyrl.rollout import microbatch_requests, requests_from_microbatch
from isoexec.models.deepseek_v4_1_flash.config import BOUNDED_REPLAY, FULL_DECODER
from isoexec.models.deepseek_v4_1_flash.engine_ced import plan_step
from isoexec.models.deepseek_v4_1_flash.training_ced import decoder_segments, segments_for_forward
from isoexec.schema import rollout

W = 8


def skyrl_batch(prompts, responses, loss_masks=None):
    """SkyRL's own layout (``convert_prompts_responses_to_batch_tensors``: one left-padded block per row,
    right-aligned response tensors)."""
    from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors

    loss_masks = loss_masks or [[1] * len(r) for r in responses]
    out = convert_prompts_responses_to_batch_tensors(0, prompts, responses, [[0.0] * len(r) for r in responses],
                                                     loss_masks)
    sequences, attention_mask, response_mask, _rewards, loss_mask = out[:5]
    return {"sequences": sequences, "attention_mask": attention_mask, "response_mask": response_mask,
            "loss_mask": loss_mask, "num_actions": response_mask.shape[1]}


def packed_cu(attention_mask, align=1):
    """``preprocess_packed_seqs``' padded boundaries (each sequence padded to ``align``)."""
    cu = [0]
    for n in attention_mask.sum(1).tolist():
        cu.append(cu[-1] + n + (-n) % align)
    return cu


def engine_rows(prompt_len, generated, rng):
    """Decoder positions the engine runs for one request under a random chunked prefill followed by decode
    (``engine_ced.plan_step``), and the segment start it uses."""
    total = prompt_len + generated - 1          # the last sampled token is never fed back
    computed, positions, start = 0, [], None
    while computed < total:
        q = rng.randint(1, max(1, prompt_len // 2)) if computed < prompt_len else 1
        q = min(q, (prompt_len if computed < prompt_len else total) - computed)
        plan = plan_step([prompt_len], [computed], [q], window=W, mode=BOUNDED_REPLAY)
        start = int(plan.segment_start[0])
        positions += [computed + int(r) for r in plan.rows]
        computed += q
    return start, positions


def test_segments_equal_the_engine_planners():
    rng = random.Random(0)
    prompts = [[1] * p for p in (3, 8, 9, 20, 1, 31)]
    responses = [[2] * g for g in (5, 1, 12, 7, 4, 2)]
    batch = skyrl_batch(prompts, responses)
    cu = packed_cu(batch["attention_mask"], align=4)
    requests = requests_from_microbatch(batch, cu)
    assert requests.requests == tuple(((len(p), len(r)),) for p, r in zip(prompts, responses))
    with rollout.carry(requests):
        segments = segments_for_forward(BOUNDED_REPLAY, cu, W)
    for s, g in enumerate(segments.segments):
        P, G = len(prompts[s]), len(responses[s])
        start, rows = engine_rows(P, G, rng)
        assert g.sequence == s and g.start == start and g.end == P + G
        assert sorted(rows) == list(range(start, P + G - 1))            # every engine row once
        assert [p for p in range(g.start, g.end) if g.scores(p)] == list(range(P - 1, P + G - 1))


def test_explicit_segments_win_and_full_decoder_needs_nothing():
    batch = skyrl_batch([[1] * 4], [[2] * 3])
    cu = packed_cu(batch["attention_mask"])
    assert len(segments_for_forward(FULL_DECODER, cu, W)) == 7
    with pytest.raises(ValueError, match="CED_SEGMENTS_MISSING"):
        segments_for_forward(BOUNDED_REPLAY, cu, W)
    with rollout.carry(requests_from_microbatch(batch, cu)):
        carried = segments_for_forward(BOUNDED_REPLAY, cu, W)
        with decoder_segments(carried) as explicit:
            assert segments_for_forward(BOUNDED_REPLAY, cu, W) is explicit
        with pytest.raises(ValueError, match="CED_ROLLOUT_PACKING"):
            segments_for_forward(BOUNDED_REPLAY, [0, 8], W)


def test_decoder_entry_reads_the_carried_requests():
    from isoexec.models.deepseek_v4_1_flash.block import DecoderEntry, _decoder_rows

    assert DecoderEntry.READS_ROLLOUT_REQUESTS
    batch = skyrl_batch([[1] * 12, [1] * 3], [[2] * 4, [2] * 2])
    cu = packed_cu(batch["attention_mask"])
    T = cu[-1]
    positions = torch.cat([torch.arange(cu[i + 1] - cu[i]) for i in range(len(cu) - 1)])
    seq = NS(positions=positions, cu_seqlens_cpu=torch.tensor(cu), cu_seqlens=torch.tensor(cu))
    cfg = NS(replay_window=W)
    with pytest.raises(ValueError, match="CED_SEGMENTS_MISSING"):
        _decoder_rows(cfg, torch.zeros(1, T, 1), seq, None, BOUNDED_REPLAY, "trainer")
    with rollout.carry(requests_from_microbatch(batch, cu)):
        rows, all_rows = _decoder_rows(cfg, torch.zeros(1, T, 1), seq, None, BOUNDED_REPLAY, "trainer")
    assert not all_rows
    assert rows.index.tolist() == list(range(12 - W, 16)) + list(range(16, 21))
    assert rows.window_floor.tolist() == [12 - W] * (W + 4) + [0] * 5


def test_padding_rows_and_padding_batches_score_nothing():
    batch = skyrl_batch([[1] * 5, [1] * 2], [[2] * 3, [2] * 4])
    # SkyRL's _pad_microbatch_to_size dummy row: one attended token, no response, no loss.
    batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.zeros(1, 8, dtype=torch.long)])
    batch["attention_mask"][-1, 0] = 1
    batch["response_mask"] = torch.cat([batch["response_mask"], torch.zeros(1, 4, dtype=torch.long)])
    batch["loss_mask"] = torch.cat([batch["loss_mask"], torch.zeros(1, 4)])
    cu = packed_cu(batch["attention_mask"])
    assert requests_from_microbatch(batch, cu).requests[-1] == ((1, 0),)
    # worker_utils._create_padding_microbatch: one attended token, response_mask all ones, loss all zero.
    pad = {"attention_mask": torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]), "response_mask": torch.ones(1, 4),
           "loss_mask": torch.zeros(1, 4), "num_actions": 4, "is_padding_batch": True}
    req = requests_from_microbatch(pad, [0, 1])
    assert req.requests == ((((1, 0),)),)
    with rollout.carry(req):
        seg = segments_for_forward(BOUNDED_REPLAY, [0, 1], W)
    assert len(seg) == 1 and not seg.score.any()


def test_named_refusals():
    batch = skyrl_batch([[1] * 4, [1] * 3], [[2] * 6, [2] * 2], [[1, 1, 0, 0, 1, 1], [1, 1]])
    cu = packed_cu(batch["attention_mask"])
    with pytest.raises(ValueError, match="CED_TURNS_REQUIRED"):
        requests_from_microbatch(batch, cu)
    forward_only = {k: v for k, v in batch.items() if k not in ("response_mask", "loss_mask")}
    with pytest.raises(ValueError, match="ROLLOUT_RESPONSE_MASK_MISSING"):
        requests_from_microbatch(forward_only, cu)
    with pytest.raises(ValueError, match="ROLLOUT_SUBSEQ_PACKING"):
        requests_from_microbatch({**batch, "sub_seq_lengths": [[3, 4]]}, cu)
    with pytest.raises(ValueError, match="ROLLOUT_PACKING"):
        requests_from_microbatch({**batch, "loss_mask": None}, cu[:-1])
    # DAPO overlong filtering zeroes a whole loss mask: still one request.
    batch["loss_mask"][0] = 0
    assert requests_from_microbatch(batch, cu).requests[0] == ((4, 6),)


def test_microbatches_pair_with_forwards_in_schedule_order():
    batches = [skyrl_batch([[1] * p], [[2] * g]) for p, g in ((4, 2), (9, 3), (2, 5))]
    cus = [packed_cu(b["attention_mask"], align=2) for b in batches]
    stage = NS(reads_rollout_requests=True, forward_context=None)
    seen = []
    with microbatch_requests(stage, batches):
        for cu in cus:
            psp = NS(cu_seqlens_q_padded=torch.tensor(cu, dtype=torch.int32), cu_seqlens_q=None)
            with stage.forward_context(psp):
                seen.append(rollout.current().requests)
        with pytest.raises(RuntimeError, match="ROLLOUT_PAIRING"):
            stage.forward_context(psp)
    assert stage.forward_context is None and rollout.current() is None
    assert seen == [(((4, 2),),), (((9, 3),),), (((2, 5),),)]
    with pytest.raises(RuntimeError, match="ROLLOUT_PAIRING"):
        with microbatch_requests(stage, batches):
            pass
    with pytest.raises(ValueError, match="ROLLOUT_VIRTUAL_PIPELINE"):
        with microbatch_requests(stage, batches, chunks=2):
            pass
    other = NS(reads_rollout_requests=False, forward_context=None)
    with microbatch_requests(other, batches):
        assert other.forward_context is None


def test_generator_admission_is_single_turn_only():
    from isoexec.integrations.skyrl.rollout import admit_generator

    stage = NS(reads_rollout_requests=True)
    ok = [NS(batched=True, max_turns=5, step_wise_trajectories=False),       # the V4.1 recipe: batched=true
          NS(batched=False, max_turns=1, step_wise_trajectories=False)]
    for generator in ok:
        admit_generator(NS(generator=generator), stage)
    for generator in (NS(batched=False, max_turns=3, step_wise_trajectories=False),
                      NS(batched=False, max_turns=1, step_wise_trajectories=True)):
        with pytest.raises(ValueError, match="CED_TURNS_REQUIRED"):
            admit_generator(NS(generator=generator), stage)
        admit_generator(NS(generator=generator), NS(reads_rollout_requests=False))    # Qwen: unchanged


def test_generator_admission_in_the_trainer_worker_reads_the_resolved_options():
    """SkyRL's trainer worker holds TrainerConfig (no ``generator``): the turns come from the resolved options."""
    from isoexec.integrations.skyrl.rollout import admit_generator

    stage = NS(reads_rollout_requests=True)
    worker_cfg = NS(_isoexec_config=NS(generator_turns={"batched": True, "max_turns": 1,
                                                        "step_wise_trajectories": False}))
    admit_generator(worker_cfg, stage)
    worker_cfg._isoexec_config.generator_turns = {"batched": False, "max_turns": 3, "step_wise_trajectories": False}
    with pytest.raises(ValueError, match="CED_TURNS_REQUIRED"):
        admit_generator(worker_cfg, stage)
    with pytest.raises(ValueError, match="did not reach the trainer worker"):
        admit_generator(NS(_isoexec_config=NS()), stage)
    admit_generator(NS(), NS(reads_rollout_requests=False))    # Qwen: no requirement
