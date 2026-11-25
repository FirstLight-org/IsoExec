"""Entropy is verl's: the IsoExec engine hands its logits to verl's vocab-parallel entropy (2 and 4 gloo ranks).

Regression: IsoExec once computed the trainer entropy itself and differentiated through a row max that
``all_reduce(MAX)`` had overwritten in place; with ``entropy_coeff > 0`` at TP > 1 every rank whose vocabulary
shard did not hold the row max got a NaN gradient and the first optimizer step refused. Entropy is a loss term
the engine never reports, so the engine now delegates it to verl and only adapts the layout (IsoExec masks
trimmed vocabulary padding to ``-inf``; verl's function needs finite columns).
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROWS = 6


def _logits(padded: int, vocab: int) -> torch.Tensor:
    logits = torch.randn(ROWS, padded, generator=torch.Generator().manual_seed(7)) * 3
    logits[:, vocab:] = float("-inf")  # LMHead.trim_to_vocab's padding mask
    logits[0, 1] += 20.0  # row 0's max lives in the first shard only
    logits[1, vocab - 1] += 20.0  # row 1's max lives in the last real column
    return logits


def _reference(logits: torch.Tensor, vocab: int, weights: torch.Tensor, temperature: float):
    """Unsharded fp64 entropy of softmax(logits / T) over the real vocabulary, and its gradient."""
    x = logits[:, :vocab].double().requires_grad_()
    log_p = torch.log_softmax(x / temperature, dim=-1)
    entropy = -(log_p.exp() * log_p).sum(dim=-1)
    (entropy * weights).sum().backward()
    grad = torch.zeros(logits.shape, dtype=torch.float64)
    grad[:, :vocab] = x.grad
    return entropy.detach(), grad


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _megatron_tp_is_world(world: int, rank: int) -> None:
    """Megatron's TP group is the gloo world (verl's function and the engine read the same module)."""
    from megatron.core import parallel_state

    parallel_state.get_tensor_model_parallel_group = lambda *a, **k: dist.group.WORLD
    parallel_state.get_tensor_model_parallel_world_size = lambda *a, **k: world
    parallel_state.get_tensor_model_parallel_rank = lambda *a, **k: rank


def _host_entropy(shard, vocab, padded, temperature, chunking):
    from isoexec.integrations.verl.engine import IsoExecMegatronEngine

    engine = SimpleNamespace(
        _lm_head=lambda: SimpleNamespace(vocab=vocab, padded_vocab=padded),
        engine_config=SimpleNamespace(entropy_from_logits_with_chunking=chunking, entropy_from_logits_chunk_size=4),
    )
    return IsoExecMegatronEngine._host_entropy(engine, shard, temperature)


def _worker(rank, world, port, vocab, padded, temperature, chunking, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    _megatron_tp_is_world(world, rank)
    from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy

    width = padded // world
    weights = torch.linspace(0.5, 1.5, ROWS)
    out = []
    for _ in range(2):  # twice: the gradient is deterministic
        shard = _logits(padded, vocab)[:, rank * width:(rank + 1) * width].clone().requires_grad_()
        entropy = _host_entropy(shard, vocab, padded, temperature, chunking)
        (entropy[0] * weights).sum().backward()
        out.append((entropy.detach()[0], shard.grad.clone()))
    native = None
    if vocab == padded:
        # verl's native Megatron path on the same logits: logits.div_(T), then vocab_parallel_entropy.
        logits = _logits(padded, vocab)[:, rank * width:(rank + 1) * width].clone().unsqueeze(0)
        logits.div_(torch.tensor([temperature]).unsqueeze(-1).to(logits.dtype))
        native = vocab_parallel_entropy(logits)[0]
    results[rank] = (out, native)
    dist.destroy_process_group()


@pytest.mark.parametrize(
    ("world", "vocab", "padded", "temperature", "chunking"),
    [(2, 30, 32, 1.0, False), (4, 30, 32, 0.7, False), (4, 30, 32, 1.0, True), (2, 32, 32, 0.7, False)],
)
def test_entropy_gradient_is_finite_deterministic_and_verls(world, vocab, padded, temperature, chunking):
    results = mp.Manager().dict()
    mp.spawn(_worker, args=(world, _free_port(), vocab, padded, temperature, chunking, results),
             nprocs=world, join=True)
    weights = torch.linspace(0.5, 1.5, ROWS, dtype=torch.float64)
    ref_entropy, ref_grad = _reference(_logits(padded, vocab), vocab, weights, temperature)
    width = padded // world
    for rank in range(world):
        (first, second), native = results[rank]
        entropy, grad = first
        assert torch.isfinite(grad).all(), f"rank {rank}: non-finite entropy gradient"
        assert torch.equal(entropy, second[0]) and torch.equal(grad, second[1]), f"rank {rank}: not deterministic"
        torch.testing.assert_close(entropy.double(), ref_entropy, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(grad.double(), ref_grad[:, rank * width:(rank + 1) * width], rtol=1e-4, atol=1e-5)
        if native is not None:
            assert torch.equal(entropy, native), f"rank {rank}: differs from verl's native entropy"


def test_refuses_a_shard_of_padding_only(monkeypatch):
    from isoexec.checks import Refusal
    from isoexec.integrations.verl import engine as eng

    monkeypatch.setattr(eng.mpu, "get_tensor_model_parallel_world_size", lambda *a, **k: 4)
    monkeypatch.setattr(eng.mpu, "get_tensor_model_parallel_rank", lambda *a, **k: 3)
    with pytest.raises(Refusal, match="only vocabulary padding"):
        eng._real_vocab_columns(SimpleNamespace(vocab=24, padded_vocab=32), 8)
    monkeypatch.setattr(eng.mpu, "get_tensor_model_parallel_rank", lambda *a, **k: 2)
    assert eng._real_vocab_columns(SimpleNamespace(vocab=20, padded_vocab=32), 8) == 4
