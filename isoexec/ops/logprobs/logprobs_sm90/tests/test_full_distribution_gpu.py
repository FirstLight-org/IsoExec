"""The full-distribution observer preserves every production output bit across TP layouts."""

import pytest
import torch

from isoexec.ops.logprobs.logprobs_sm90 import integration, ops


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("vocab", [64, 151680])
def test_observer_matches_engine_and_sampled_bits(monkeypatch, world, vocab):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    g = torch.Generator(device="cuda").manual_seed(15)
    logits = torch.randn(9, vocab, device="cuda", generator=g)
    logits[0, :vocab // 8] = -float("inf")
    logits[1] = -float("inf")
    logits[2, 0] = float("nan")
    logits[3] = -0.0
    temperature = torch.tensor(0.7, device="cuda")
    tempered = (logits / temperature).contiguous()
    engine = integration.full_logprobs(logits, temperature=temperature)
    before = integration.stats()
    width = vocab // world
    shards = [tempered[:, rank * width:(rank + 1) * width].contiguous() for rank in range(world)]
    states = [ops.leaf_states(shard, 8 // world) for shard in shards]
    group = object() if world > 1 else None
    monkeypatch.setattr(ops.dist, "get_world_size", lambda **kw: world)
    for rank, shard in enumerate(shards):
        monkeypatch.setattr(ops.dist, "get_rank", lambda **kw: rank)
        calls = []

        def gather(parts, value, *, group):
            column = len(calls)
            calls.append(column)
            for part, state in zip(parts, states):
                part.copy_(state[column])

        monkeypatch.setattr(ops.dist, "all_gather", gather)
        got = integration.full_logprobs_for_check_sharded(
            shard, vocab_start_index=rank * width, vocab_end_index=(rank + 1) * width, group=group
        )
        expected = engine[:, rank * width:(rank + 1) * width]
        assert torch.equal(got.view(torch.int32), expected.view(torch.int32))
        assert len(calls) == (2 if world > 1 else 0)
    assert integration.stats() == before
    targets = torch.arange(9, device="cuda") % vocab
    sampled = ops.sampled_logprobs(tempered, targets)
    assert torch.equal(sampled.view(torch.int32), engine.gather(1, targets[:, None]).squeeze(1).view(torch.int32))
