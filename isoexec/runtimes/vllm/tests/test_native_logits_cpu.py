"""Inference gathers BF16 head shards while preserving public FP32 logits."""

from types import SimpleNamespace

import pytest
import torch

from isoexec.core.bindings import BoundOp
from isoexec.nn import LMHead
from isoexec.runtimes.vllm import model as native


@pytest.mark.parametrize("world,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("vocab", [7, 8])
def test_logits_gather_before_widening_preserves_bits_slice_and_padding(monkeypatch, world, rank, vocab):
    group = torch.distributed.ProcessGroup(rank, world)
    head = LMHead(vocab=vocab, padded_vocab=8, hidden=4, device="cpu")
    head.tp_group = group
    weights = (torch.arange(32).reshape(8, 4).float() / 37).to(torch.bfloat16)
    head.weight = torch.nn.Parameter(weights.chunk(world)[rank].clone())
    hidden = (torch.arange(12).reshape(3, 4).float() / 13).to(torch.bfloat16)
    calls = []

    def select(value):
        calls.append("slice")
        return value[1:]

    def bind(op, fn):
        return BoundOp(op, "cpu_fixture", None, "test", fn=fn)

    head.slots.lm_head_slice = bind("logprobs.lm_head_slice", select)
    head.slots.tp_copy = bind("collectives.tp_copy", lambda value: value)
    head.slots.head = bind("logprobs.head", lambda value, weight: value @ weight.t())
    shards = [hidden[1:] @ weight.t() for weight in weights.chunk(world)]

    def gather(local, actual_group, out_dtype=None):
        assert actual_group is group
        assert local.dtype == torch.bfloat16
        assert torch.equal(local, shards[rank])
        calls.append("gather")
        gathered = torch.cat(shards, dim=-1)
        return gathered if out_dtype is None else gathered.to(out_dtype)

    monkeypatch.setattr(native, "gather_vocab", gather)
    actual = native.VllmModel.compute_logits(SimpleNamespace(head=head, _tp_group=group), hidden)
    # The previous native path widened each shard before gathering. Widening
    # after this byte-copy collective must preserve every public output bit.
    previous = torch.cat([shard.float() for shard in shards], dim=-1)[:, :vocab]
    assert actual.dtype == torch.float32 and actual.shape == (2, vocab)
    assert torch.equal(actual, previous)
    assert calls == (["slice"] if world == 1 else ["slice", "gather"])


def test_nonfinal_pipeline_stage_has_no_logits():
    assert native.VllmModel.compute_logits(SimpleNamespace(head=None), torch.empty(1, 4)) is None


def test_gather_vocab_interleaves_and_widens_in_one_copy(monkeypatch):
    """The rank-major collective buffer, moved to vocab order and widened, is cat(shards).float() bit for bit."""
    from isoexec.runtimes import engine_ops

    world, rows, width = 4, 3, 7
    shards = [torch.randn(rows, width).bfloat16() for _ in range(world)]
    shards[2][0, 0] = -0.0
    group = object()

    def all_gather_into_tensor(out, local, group=None):
        assert out.shape == (world, rows, width) and local.is_contiguous()
        for r, shard in enumerate(shards):
            out[r].copy_(shard)

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", all_gather_into_tensor)
    expected = torch.cat(shards, dim=-1)
    for dtype in (None, torch.float32):
        got = engine_ops.gather_vocab(shards[1], group, out_dtype=dtype)
        want = expected if dtype is None else expected.float()
        assert got.dtype == want.dtype and torch.equal(got.view(torch.int16 if dtype is None else torch.int32),
                                                       want.view(torch.int16 if dtype is None else torch.int32))


@pytest.mark.parametrize("p2p", [True, False])
def test_gather_vocab_takes_p2p_where_the_group_has_it_and_nccl_otherwise(monkeypatch, p2p):
    """A CUDA shard goes to pik's symmetric-memory gather first; only its refusal runs the NCCL collective."""
    from isoexec.runtimes import engine_ops

    world, rows, width = 2, 3, 5
    local = torch.randn(rows, width).bfloat16()
    calls = []

    def gather_p2p(shard, group, out):
        calls.append("p2p")
        assert out.shape == (rows, world * width) and out.dtype == torch.float32
        if p2p:
            out.fill_(1.0)
        return p2p

    def all_gather_into_tensor(out, shard, group=None):
        calls.append("nccl")
        out.copy_(shard.expand(world, *shard.shape))

    class Cuda(torch.Tensor):  # only the device test reads is_cuda; the math stays on the CPU
        @property
        def is_cuda(self):
            return True

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", all_gather_into_tensor)
    monkeypatch.setattr(engine_ops, "_gather_p2p", gather_p2p)
    got = engine_ops.gather_vocab(local.as_subclass(Cuda), object(), out_dtype=torch.float32)
    assert calls == (["p2p"] if p2p else ["p2p", "nccl"])
    want = torch.ones(rows, world * width) if p2p else torch.cat([local] * world, dim=-1).float()
    assert torch.equal(torch.Tensor(got), want)
