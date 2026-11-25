"""Inference drops unused MoE auxiliary work without changing authored training graphs."""
import operator

import pytest
import torch
from torch import fx, nn

from isoexec.nn import MoE
from isoexec.ops.moe.routing import softmax_renorm
from isoexec.runtimes.vllm._definition import _elide_unused_moe_aux


class EagerMoE(MoE):
    def forward(self, x, seq=None, aux=None, *, return_aux=False):
        routing = softmax_renorm(x, self._spec, want_scores_full=return_aux or aux is not None)
        output = routing.probs
        return (output, routing.scores_full) if return_aux else output


def graph(*, consumers=(0,), direct=False, aux=None):
    root = nn.Module()
    root.mlp = EagerMoE(hidden=8, num_experts=8, top_k=2, intermediate=4)
    g = fx.Graph()
    x = g.placeholder('x')
    call = g.call_module('mlp', (x,), {'return_aux': True, 'aux': aux})
    values = [g.call_function(operator.getitem, (call, index)) for index in consumers]
    for value in values:
        value.meta['output_marker'] = 'tensor'
    g.output(call if direct else tuple(values))
    return fx.GraphModule(root, g), call


def test_unused_aux_elision_preserves_eager_values_and_tensor_metadata():
    module, call = graph(consumers=(0, 0))
    logits = torch.tensor([[3., 3., 3., 1., 0., -1., -2., -3.], [0.] * 8])
    original = module(logits)
    _elide_unused_moe_aux(module)
    module.graph.lint()
    module.recompile()
    assert call.kwargs['return_aux'] is False
    assert call.meta['output_marker'] == 'tensor'
    assert all(torch.equal(a, b) for a, b in zip(original, module(logits)))
    assert not any(n.target is operator.getitem for n in module.graph.nodes)


@pytest.mark.parametrize('options', [{'consumers': (0, 1)}, {'direct': True}, {'aux': {'keep': 1}}])
def test_live_aux_or_tuple_consumers_are_preserved(options):
    module, call = graph(**options)
    before = str(module.graph)
    _elide_unused_moe_aux(module)
    assert call.kwargs['return_aux'] is True
    assert str(module.graph) == before
