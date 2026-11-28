"""Real Gloo regressions for unequal parameter usage across token-owning ranks."""
import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from isoexec.ops.collectives.ordered_sum import ordered_sum
from isoexec.ops.collectives.parallel_autograd import sum_parameter_gradients


def _module():
    module = nn.Module()
    module.a = nn.Parameter(torch.ones(1, dtype=torch.float64))
    module.b = nn.Parameter(torch.ones(1, dtype=torch.float64))
    return module


def _worker(rank, init):
    dist.init_process_group('gloo', rank=rank, world_size=2, init_method=init)
    try:
        # Executable old-pattern failure: both collectives have the same shape,
        # but rank 0 enters a's hook while rank 1 enters b's. They silently sum
        # different logical parameters. The unused parameter gets no gradient.
        old = _module()
        for parameter in old.parameters():
            parameter.register_hook(lambda gradient: ordered_sum(gradient))
        (old.a * 2 if rank == 0 else old.b * 3).sum().backward()
        used, unused = (old.a, old.b) if rank == 0 else (old.b, old.a)
        assert used.grad.item() == 5 and unused.grad is None

        model = _module()
        originals = tuple(model.parameters())
        identities = tuple(map(id, originals))
        for _ in range(2):
            with sum_parameter_gradients(model, ('a', 'b')):
                loss = (model.a * 2 if rank == 0 else model.b * 3).sum()
            assert tuple(map(id, model.parameters())) == identities
            gradients = torch.autograd.grad(loss, originals, retain_graph=True)
            assert tuple(g.item() for g in gradients) == (2, 3)
            loss.backward()
            assert all(torch.equal(p.grad, g) for p, g in zip(originals, gradients))
            torch.optim.SGD(originals, lr=.1).step()
            model.zero_grad(set_to_none=True)
        torch.testing.assert_close(model.a, torch.tensor([.6], dtype=torch.float64))
        torch.testing.assert_close(model.b, torch.tensor([.4], dtype=torch.float64))
    finally:
        dist.destroy_process_group()


def test_distinct_unused_parameters_are_not_cross_reduced(tmp_path):
    mp.spawn(_worker, args=('file://' + str(tmp_path / 'group'),), nprocs=2, join=True)


def test_parameter_aliases_restore_after_forward_exception():
    model = _module()
    originals = tuple(model.parameters())
    with pytest.raises(RuntimeError, match='forward failed'):
        with sum_parameter_gradients(model, ('a', 'b')):
            raise RuntimeError('forward failed')
    assert all(a is b for a, b in zip(originals, model.parameters()))
