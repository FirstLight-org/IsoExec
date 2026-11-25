"""Native layer checkpointing replays expert-parallel collectives."""

import pytest
import torch

from isoexec.nn import MoE
from isoexec.ops.table import OpTable
from isoexec.schema import make_sequence_context

HIDDEN = 64
SITE = "lm.layers.0.mlp"


def _sequence():
    """Two packed sequences of 4 and 5 tokens."""
    cu = torch.tensor((0, 4, 9), dtype=torch.int32)
    pos = torch.cat([torch.arange(4), torch.arange(5)]).to(torch.int64)
    return make_sequence_context(pos, cu, max_q=5)


@pytest.fixture
def cpu_checkpoint_rng(monkeypatch):
    # Exercise the actual MCore CheckpointFunction, replacing only GPU RNG I/O.
    from megatron.core.tensor_parallel import random

    monkeypatch.setattr(random, "_get_all_rng_states", lambda: (torch.get_rng_state(), None, None))
    monkeypatch.setattr(random, "_set_all_rng_states", lambda cpu, cuda, tracker: torch.set_rng_state(cpu))


def _ep_body(rank, world, out, init_method):
    import operator
    from dataclasses import replace

    import torch.distributed as dist
    from megatron.core.tensor_parallel import random

    from isoexec.ops.collectives.tests.test_tp_bytes_cpu import _install_cpu_shims
    from isoexec.ops.moe.carriers import ExpertOutputs
    from isoexec.ops.moe.reference import ref_expert_mlp
    from isoexec.runtimes.megatron._recompute import checkpoint_layers
    from isoexec.schema import Axis, Placement

    dist.init_process_group('gloo', init_method=init_method, rank=rank, world_size=world)
    _install_cpu_shims()
    random._get_all_rng_states = lambda: (torch.get_rng_state(), None, None)
    random._set_all_rng_states = lambda cpu, cuda, tracker: torch.set_rng_state(cpu)
    one = Axis(1, 0, None)
    ep = Axis(world, rank, dist.group.WORLD)
    placement = Placement(tp=one, ep=ep, dp=ep, pp=one, etp=one, cp=one, side='trainer', leaves=8)

    # EP's production dispatch consumes permuted expert rows. Keep reference
    # arithmetic but adapt its per-slot carrier to that wire representation.
    def experts(x, routing, gate_up, down):
        result = ref_expert_mlp(x, routing, gate_up, down)
        tokens = torch.arange(x.shape[0]).repeat_interleave(routing.indices.shape[1])
        return ExpertOutputs(rows=result.rows.flatten(0, 1), layout='permuted', routing=routing, extra=tokens)

    def combine(routed, shared):
        assert shared is None
        return torch.stack([routed.rows[routed.extra == token].sum(dim=0)
                            for token in range(routed.routing.indices.shape[0])])

    class Table(OpTable):
        def bind(self, op, **kwargs):
            binding = super().bind(op, **kwargs)
            if op == 'moe.expert_mlp':
                return replace(binding, fn=experts)
            if op == 'moe.combine':
                return replace(binding, fn=combine)
            return binding
    outcomes = []
    for empty_owner in (False, True):
        runs = []
        for checkpointed in (False, True):
            torch.manual_seed(919)
            module = MoE(hidden=HIDDEN, num_experts=8, top_k=2, intermediate=32, routing='softmax_renorm',
                         shared_intermediate=0, device='cpu')
            with torch.no_grad():
                for p in module.parameters():
                    p.copy_(torch.randn_like(p) * 0.03)
                if empty_owner:
                    module.router.weight.zero_()
                    module.router.weight[:2].fill_(1)
            module.apply_placement(placement, site=SITE, op_table=Table(mode='reference'))
            # Exercise the complete local decoder boundary, including norm and
            # residual, around the real EP routing/collective implementation.
            root = torch.nn.Module()
            root.layer = torch.nn.Module()
            root.layer.norm = torch.nn.RMSNorm(HIDDEN, eps=1e-6, dtype=torch.bfloat16)
            root.layer.moe = module
            graph = torch.fx.Graph()
            hidden, seq = graph.placeholder('hidden'), graph.placeholder('seq')
            norm = graph.call_module('layer.norm', (hidden,))
            pair = graph.call_module('layer.moe', (norm, seq), {'return_aux': True})
            values = graph.call_function(operator.getitem, (pair, 0))
            scores = graph.call_function(operator.getitem, (pair, 1))
            residual = graph.call_function(torch.ops.aten.add.Tensor, (hidden, values))
            for node in graph.nodes:
                if node.op != 'placeholder':
                    node.meta['isoexec_logical_layer'] = (0, 'layer')
            graph.output((residual, scores))
            module = torch.fx.GraphModule(root, graph)
            if checkpointed:
                checkpoint_layers(module)
            x = (torch.ones(9, HIDDEN, dtype=torch.bfloat16) if empty_owner else
                 torch.randn(9, HIDDEN, dtype=torch.bfloat16)).requires_grad_()
            y, scores = module(x, _sequence())
            (y.float().square().sum() + scores.float().square().sum() * 0.01).backward()
            runs.append((y.detach(), x.grad, {n: p.grad for n, p in module.named_parameters()}))
        outcomes.append(torch.equal(runs[0][0], runs[1][0]) and torch.equal(runs[0][1], runs[1][1])
                        and all(g is None and runs[1][2][n] is None or torch.equal(g, runs[1][2][n])
                                for n, g in runs[0][2].items()))
    out[rank] = outcomes
    dist.destroy_process_group()


def test_ep_checkpoint_replays_collectives_including_empty_expert_owner(tmp_path):
    from isoexec.ops.collectives.tests._gloo_harness import run_world

    results = run_world(_ep_body, world=2, rendezvous=tmp_path / 'store')
    assert all(all(outcomes) for outcomes in results.values()), results
