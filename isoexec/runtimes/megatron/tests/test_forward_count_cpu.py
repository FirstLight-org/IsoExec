"""EP_FORWARD_COUNT_MISMATCH: a group-agreed refusal, with real gloo processes (CPU)."""
from datetime import timedelta
from types import SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist

from isoexec.checks import Refusal
from isoexec.runtimes.megatron.forward_count import require_equal_forward_count


def _world(rank, rendezvous, world, counts, out):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(seconds=60))
    try:
        group = dist.new_group(list(range(world)))
        try:
            got = ("ok", require_equal_forward_count(counts[rank], group, site="test"))
        except Refusal as exc:
            got = ("refused", str(exc))
        # Every rank reached the end: nobody waits in a collective the others skipped.
        dist.barrier(group)
        torch.save(got, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def _run(tmp_path, counts):
    world = len(counts)
    out = tmp_path / "result"
    torch.multiprocessing.spawn(_world, args=((tmp_path / "rdv").as_uri(), world, counts, str(out)),
                                nprocs=world, join=True)
    return [torch.load(f"{out}.{rank}") for rank in range(world)]


@pytest.mark.parametrize("counts", [(3, 3), (1, 1, 1, 1), (0, 0)])
def test_equal_counts_pass_on_every_rank(tmp_path, counts):
    assert _run(tmp_path, counts) == [("ok", counts[0])] * len(counts)


@pytest.mark.parametrize("counts", [(2, 3), (4, 4, 5, 4), (1, 0, 1)])
def test_unequal_counts_are_refused_by_every_rank_with_the_same_message(tmp_path, counts):
    results = _run(tmp_path, counts)
    assert {kind for kind, _ in results} == {"refused"}
    messages = {message for _, message in results}
    assert len(messages) == 1                       # group-agreed: one decision, not a per-rank one
    message = messages.pop()
    assert message.startswith("EP_FORWARD_COUNT_MISMATCH: test:")
    by_count = {}
    for rank, count in enumerate(counts):
        by_count.setdefault(count, []).append(rank)
    assert str(dict(sorted(by_count.items()))) in message


def test_no_group_or_single_rank_issues_no_collective(monkeypatch):
    monkeypatch.setattr(dist, "all_gather_into_tensor", lambda *a, **k: pytest.fail("collective issued"))
    assert require_equal_forward_count(5, None, site="test") == 5
    assert not dist.is_initialized()
    assert require_equal_forward_count(5, object(), site="test") == 5


@pytest.mark.parametrize("count", [-1, 1.0, True, "2"])
def test_count_must_be_a_nonnegative_int(count):
    with pytest.raises(ValueError, match="nonnegative int"):
        require_equal_forward_count(count, None, site="test")


@pytest.mark.parametrize("ep_size,expected", [(1, []), (2, [(3, "group-ep", "forward")])])
def test_wrapper_agrees_before_each_schedule_only_under_ep(monkeypatch, ep_size, expected):
    pytest.importorskip("skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper")
    try:
        from isoexec.integrations.skyrl import model_wrapper
    except RuntimeError as exc:               # a SkyRL without the model-wrapper hook refuses the import
        pytest.skip(str(exc))
    calls = []
    monkeypatch.setattr(model_wrapper, "require_equal_forward_count",
                        lambda count, group, site: calls.append((count, group, site)))
    stage = NS(placement=NS(ep=NS(size=ep_size, group="group-ep")))
    monkeypatch.setattr(model_wrapper, "_find_isoexec_stage", lambda modules: stage)
    wrapper = model_wrapper.IsoExecModelWrapper.__new__(model_wrapper.IsoExecModelWrapper)
    wrapper.actor_module = [object()]
    wrapper._agree_forward_count([{}, {}, {}], "forward")
    assert calls == expected
