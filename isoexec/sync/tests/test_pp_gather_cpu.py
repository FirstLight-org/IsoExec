"""Stage schedule and stage broadcast that gather pipeline stages into one weight stream."""

import pytest
import torch
import torch.distributed as dist

from isoexec.schema import ShardError
from isoexec.sync.pp_gather import broadcast_stage_stream, stage_schedule


def test_schedule_walks_stages_in_order_and_keeps_each_stream_order():
    s0 = [("lm.embed.weight", "bfloat16", [8, 4]), ("layers.0.w", "bfloat16", [4, 4])]
    s1 = [("layers.1.w", "bfloat16", [4, 4]), ("lm.head.weight", "bfloat16", [8, 4])]
    assert stage_schedule([s0, s1]) == [
        (0, "lm.embed.weight", "bfloat16", [8, 4]),
        (0, "layers.0.w", "bfloat16", [4, 4]),
        (1, "layers.1.w", "bfloat16", [4, 4]),
        (1, "lm.head.weight", "bfloat16", [8, 4]),
    ]


def test_schedule_refuses_a_name_advertised_by_two_stages():
    with pytest.raises(ShardError, match="advertised by stages 0 and 1"):
        stage_schedule([[("w", "bfloat16", [1])], [("w", "bfloat16", [1])]])


@pytest.mark.parametrize("stream,match", [
    ([("b", torch.ones(2))], "yielded 'b' where 'a' was advertised"),
    ([("a", torch.ones(3))], "advertised \\[2\\]/torch.float32"),
    ([], "advertised 'a' but its stream ended"),
    ([("a", torch.ones(2)), ("extra", torch.ones(2))], "yielded 'extra' beyond its advertisement"),
])
def test_stage_stream_refuses_tensors_its_stage_did_not_advertise(tmp_path, stream, match):
    # One rank owns every scheduled tensor, so a local refusal cannot strand a waiting peer.
    dist.init_process_group("gloo", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1)
    try:
        schedule = stage_schedule([[("a", "torch.float32", [2])]])
        with pytest.raises(ShardError, match=match):
            list(broadcast_stage_stream(schedule, iter(stream), group=dist.group.WORLD, device="cpu"))
        streamed = list(broadcast_stage_stream(schedule, iter([("a", torch.ones(2))]), group=dist.group.WORLD,
                                               device="cpu"))
        assert [(stage, name) for stage, name, _ in streamed] == [(0, "a")] and torch.equal(streamed[0][2],
                                                                                            torch.ones(2))
    finally:
        dist.destroy_process_group()


def test_schedule_normalises_shapes_to_ints():
    (row,) = stage_schedule([[("w", "torch.float32", (2, 3))]])
    assert row == (0, "w", "torch.float32", [2, 3]) and all(isinstance(d, int) for d in row[3])
