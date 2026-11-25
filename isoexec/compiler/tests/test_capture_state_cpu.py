"""Pure-marker and clone-isolation counterexamples on real torch.export."""

from dataclasses import dataclass
from collections import defaultdict
from functools import partial
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import isoexec.nn as inn
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.capture.component_capture import register_capture_component
from isoexec.compiler.capture.export_boundary import CaptureSession, clone_for_export
from isoexec.compiler.capture.graph import PipelineError
from isoexec.compiler.tests.test_capture_cpu import Model, example_inputs
from isoexec.schema import SequenceContext
from isoexec.ops.table import OpTable
from isoexec.runtimes.parallel import trivial_placement


@pytest.mark.parametrize('operation', ['rand_like', 'randn_like', 'bernoulli', 'dropout'])
@pytest.mark.parametrize('used', [False, True], ids=['discarded', 'used'])
def test_authored_rng_effect_is_refused_before_functionalization(
    operation: str, used: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StochasticModel(Model):
        def forward(self, input_ids, seq, *, factor=1.0):
            hidden = self.lexicon(input_ids)
            for block in self.units:
                hidden = block(hidden, seq)
            if operation == 'rand_like':
                noise = torch.rand_like(hidden)
            elif operation == 'randn_like':
                noise = torch.randn_like(hidden)
            elif operation == 'bernoulli':
                noise = torch.bernoulli(torch.full_like(hidden, 0.5))
            else:
                noise = torch.nn.functional.dropout(hidden, p=0.5, training=True)
            if used:
                hidden = hidden + noise
            return self.readout(self.finish(hidden))

    def unexpected_decomposition(*args, **kwargs):
        pytest.fail('RNG effects must be rejected before decomposition can delete them')

    monkeypatch.setattr(torch.export.ExportedProgram, 'run_decompositions', unexpected_decomposition)
    state = torch.get_rng_state().clone()
    with pytest.raises(PipelineError, match='RNG-effect contract') as caught:
        capture(StochasticModel(), example_inputs())
    assert caught.value.code == 'PP_CAPTURE_UNSUPPORTED'
    assert caught.value.phase == 'capture'
    captured_operation = caught.value.context['operation']
    assert isinstance(captured_operation, str) and captured_operation.startswith('aten.')
    assert caught.value.location and caught.value.action
    assert torch.equal(torch.get_rng_state(), state)


class OwnedMutation(inn.MLP):
    gate: torch.nn.Parameter
    counter: torch.Tensor

    def __init__(self, mutation: str, *, device: str = 'cpu') -> None:
        super().__init__(hidden=16, intermediate=32, device=device)
        self.mutation = mutation
        self.register_buffer('counter', torch.zeros((), dtype=torch.int64, device=device))

    def forward(self, x: torch.Tensor, seq: SequenceContext | None = None,
                aux: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        with torch.no_grad():
            if self.mutation == 'parameter':
                self.gate.add_(1)
            elif self.mutation == 'parameter_data':
                self.gate.data.add_(1)
            elif self.mutation == 'buffer':
                self.counter.add_(1)
            elif self.mutation == 'replacement':
                self.gate = torch.nn.Parameter(self.gate + 1)
            elif self.mutation == 'storage':
                self.gate.data = self.gate.detach().clone()
        return super().forward(x, seq, aux)


register_capture_component(OwnedMutation)


class InputMutation(inn.MLP):
    def __init__(self, mutation: str) -> None:
        super().__init__(hidden=16, intermediate=32, device='cpu')
        self.mutation = mutation

    def forward(self, x: torch.Tensor, seq: SequenceContext | None = None,
                aux: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if self.mutation == 'data':
            x.data.add_(1)
        elif self.mutation == 'data_view':
            x.data.view(-1).add_(1)
        elif self.mutation == 'detached_view':
            x.detach().view(-1).add_(1)
        elif self.mutation == 'storage':
            x.data = x.detach().clone()
        elif self.mutation == 'restore_storage':
            original = x.data
            x.data = x.detach().clone()
            original.add_(1)
            x.data = original
        elif self.mutation == 'sequence_data':
            assert seq is not None
            seq.positions.data.add_(1)
        elif self.mutation == 'aux_data':
            assert aux is not None
            aux['hidden'].data.add_(1)
        return super().forward(x, seq, aux)


register_capture_component(InputMutation)


class InputModel(torch.nn.Module):
    def __init__(self, mutation: str) -> None:
        super().__init__()
        self.units = torch.nn.ModuleList([InputMutation(mutation)])
        self.head = inn.LMHead(vocab=32, hidden=16, device='cpu')

    def forward(self, hidden: torch.Tensor, seq: SequenceContext,
                aux: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        for unit in self.units:
            hidden = unit(hidden, seq, aux)
        return self.head(hidden)


class RootInputMutation(InputModel):
    def forward(self, hidden: torch.Tensor, seq: SequenceContext,
                aux: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        hidden.data.add_(1)
        return super().forward(hidden, seq, aux)


def test_authored_root_input_data_write_is_refused_before_export_processes_it() -> None:
    model = RootInputMutation('none')
    hidden = torch.ones(4, 16, dtype=torch.bfloat16)
    _, seq = example_inputs()
    with pytest.raises(PipelineError, match='writes an argument'):
        capture(model, (hidden, seq))
    assert torch.equal(hidden, torch.ones_like(hidden)) and hidden._version == 0


class RootFreshActivation(InputModel):
    def forward(self, hidden: torch.Tensor, seq: SequenceContext,
                aux: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        hidden = hidden + 1
        hidden += 1
        return super().forward(hidden, seq, aux)


def test_authored_fresh_activation_iadd_still_captures_and_matches_eager_bits() -> None:
    model = RootFreshActivation('none')
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.125)
    hidden = torch.ones(4, 16, dtype=torch.bfloat16)
    _, seq = example_inputs()
    graph = capture(model, (hidden, seq))
    inn.place(model, trivial_placement(side='trainer'), op_table=OpTable(mode='reference'))
    expected, actual = model(hidden, seq), graph.program(hidden, seq)
    assert torch.isfinite(actual).all()
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert torch.equal(hidden, torch.ones_like(hidden)) and hidden._version == 0


@pytest.mark.parametrize('mutation', ['data', 'data_view', 'detached_view', 'storage', 'restore_storage',
                                    'sequence_data', 'aux_data'])
def test_opaque_input_alias_and_storage_writes_refuse_before_pure_marker(mutation: str) -> None:
    model = InputModel(mutation)
    hidden = torch.randn(4, 16, dtype=torch.bfloat16)
    original, storage, version = hidden.clone(), hidden.untyped_storage()._cdata, hidden._version
    _, seq = example_inputs()
    positions, position_version = seq.positions.clone(), seq.positions._version
    auxiliary = hidden.clone()
    with pytest.raises(PipelineError, match='writes an argument'):
        capture(model, (hidden, seq), {'aux': {'hidden': auxiliary}})
    assert torch.equal(hidden, original)
    assert hidden.untyped_storage()._cdata == storage and hidden._version == version
    assert torch.equal(seq.positions, positions) and seq.positions._version == position_version
    assert torch.equal(auxiliary, original) and auxiliary._version == 0


@pytest.mark.parametrize('mutation', ['parameter', 'parameter_data', 'buffer', 'replacement', 'storage'])
def test_opaque_owned_state_writes_refuse_before_pure_marker_and_preserve_original(mutation: str) -> None:
    model = Model(device='cpu')
    component = OwnedMutation(mutation)
    model.units[0].ffn = component
    with torch.no_grad():
        component.gate.fill_(0.125)
    original = component.gate.detach().clone()
    version, identity = component.gate._version, id(component.gate)
    with pytest.raises(PipelineError, match='owned Parameter/Buffer|changed owned parameters'):
        capture(model, example_inputs())
    assert id(component.gate) == identity and component.gate._version == version
    assert torch.equal(component.gate, original) and component.counter.item() == 0


@dataclass
class Config:
    values: list[float]


class PythonMutation(Model):
    config: dict[str, Any]

    def __init__(self, mutation: str) -> None:
        super().__init__()
        self.config = {'scale': 1., 'nested': {'count': 0}, 'items': [1.]}
        self.alias = self.config
        self.details = Config([1.])
        self.mutation = mutation
        self.tick = self.advance
        self.closed_tick = lambda: self.advance()
        def default_tick(state: dict[str, int] = {'n': 0}) -> None:
            state['n'] += 1
        self.default_tick = default_tick
        self.user_hooks = {'counter': {'n': 0}}

    def advance(self) -> None:
        self.config['nested']['count'] += 1

    def forward(self, input_ids: torch.Tensor, seq: SequenceContext, *, factor: float = 1.) -> torch.Tensor:
        if self.mutation == 'dict':
            self.config['scale'] += 1.
        elif self.mutation == 'nested':
            self.config['nested']['count'] += 1
        elif self.mutation == 'list':
            self.config['items'][0] += 1.
        elif self.mutation == 'dataclass':
            self.details.values[0] += 1.
        elif self.mutation == 'method':
            self.tick()
        elif self.mutation == 'closure':
            self.closed_tick()
        elif self.mutation == 'default':
            self.default_tick()
        elif self.mutation == 'hooks_name':
            self.user_hooks['counter']['n'] += 1
        return super().forward(input_ids, seq, factor=factor)


@pytest.mark.parametrize('mutation', ['dict', 'nested', 'list', 'dataclass', 'method', 'closure', 'default', 'hooks_name'])
def test_nested_python_mutation_refuses_without_polluting_original(mutation: str) -> None:
    model = PythonMutation(mutation)
    with pytest.raises(PipelineError, match='mutated module state'):
        capture(model, example_inputs())
    assert model.config == {'scale': 1., 'nested': {'count': 0}, 'items': [1.]}
    assert model.alias is model.config and model.details.values == [1.]
    defaults = getattr(model.default_tick, '__defaults__')
    assert defaults is not None and defaults[0] == {'n': 0}
    assert model.user_hooks == {'counter': {'n': 0}}


def test_supported_container_and_tensor_aliases_are_isolated_and_readonly_capture_works() -> None:
    model = PythonMutation('readonly')
    clone = clone_for_export(model, CaptureSession(model, example_inputs()))
    assert isinstance(clone, PythonMutation)
    assert clone.config is clone.alias and clone.config is not model.config
    assert clone.config['nested'] is not model.config['nested']
    assert getattr(clone.tick, '__self__') is clone
    assert clone.get_parameter('lexicon.weight') is clone.get_parameter('readout.weight')
    assert clone.get_parameter('lexicon.weight') is not model.get_parameter('lexicon.weight')
    result = capture(model, example_inputs())
    assert result.inventory.ties[0].aliases == ('readout.weight',)


def test_unknown_mutable_runtime_object_is_explicitly_refused() -> None:
    class Resource:
        def __init__(self) -> None:
            self.count = 0
    model = Model()
    resource = Resource()
    setattr(model, 'resource', resource)
    with pytest.raises(PipelineError, match='Unsupported capture state Resource'):
        capture(model, example_inputs())
    assert resource.count == 0


class _CustomGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
        return value.clone()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> torch.Tensor:
        (gradient,) = grad_outputs
        return gradient * 10


@torch.no_grad()
def _frozen_scale(value: torch.Tensor) -> torch.Tensor:
    return value * 1


class BackwardEffect(Model):
    def __init__(self, effect: str) -> None:
        super().__init__('cpu')
        self.effect = effect
        self.readout = inn.LMHead(vocab=32, hidden=16, device='cpu')

    def forward(self, input_ids: torch.Tensor, seq: SequenceContext, *, factor: float = 1.) -> torch.Tensor:
        hidden = self.lexicon(input_ids)
        if self.effect == 'custom':
            hidden = _CustomGradient.apply(hidden)
        elif self.effect == 'hook':
            hidden.register_hook(lambda gradient: gradient * 10)
        elif self.effect == 'no_grad':
            with torch.no_grad():
                hidden = hidden * 1
        elif self.effect == 'decorated':
            hidden = _frozen_scale(hidden)
        elif self.effect == 'detach':
            hidden = hidden.detach()
        for unit in self.units:
            hidden = unit(hidden, seq)
        return self.readout(self.finish(hidden) * factor)


@pytest.mark.parametrize('effect', ['custom', 'hook', 'no_grad', 'decorated'])
def test_backward_only_author_effects_refuse_instead_of_silently_training_a_forward_decomposition(effect: str) -> None:
    model = BackwardEffect(effect)
    previous = sys.getprofile()
    with pytest.raises(PipelineError, match='backward|effect contract|execution context'):
        capture(model, example_inputs())
    assert sys.getprofile() is previous
    # Independent eager PyTorch execution retains the effect. This is a model
    # semantic counterexample, not a mock of an exported backward graph.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.02)
    inn.place(model, trivial_placement(), op_table=OpTable(mode='reference'))
    model(*example_inputs()).square().mean().backward()
    if effect in ('no_grad', 'decorated'):
        assert model.lexicon.weight.grad is None and model.readout.weight.grad is not None
    else:
        assert model.lexicon.weight.grad is not None and torch.count_nonzero(model.lexicon.weight.grad)


def test_explicit_detach_retains_its_eager_backward_and_tensor_parameter_hooks_refuse() -> None:
    model = BackwardEffect('detach')
    graph = capture(model, example_inputs())
    assert any(node.target is torch.ops.aten.detach.default for node in graph.program.graph.nodes)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.02)
    inn.place(model, trivial_placement(), op_table=OpTable(mode='reference'))
    expected = model(*example_inputs())
    expected.square().mean().backward()
    assert model.lexicon.weight.grad is None
    gradients = {name: None if parameter.grad is None else parameter.grad.clone()
                 for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    actual = graph.program(*example_inputs())
    actual.square().mean().backward()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    for name, parameter in model.named_parameters():
        reference = gradients[name]
        if reference is None:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None and torch.equal(parameter.grad.view(torch.uint8), reference.view(torch.uint8)), name
    handle = model.lexicon.weight.register_hook(lambda gradient: gradient * 10)
    with pytest.raises(PipelineError, match='Tensor backward hooks'):
        capture(model, example_inputs())
    handle.remove()


def test_plain_namespace_partial_defaultdict_and_range_configuration_is_isolated() -> None:
    model = Model()
    namespace = SimpleNamespace(items=[1.], counts=defaultdict(int, {'initial': 0}), indices=range(1, 3))
    scale = partial(torch.mul, other=0.5)
    setattr(model, 'author_config', namespace)
    setattr(model, 'author_alias', namespace)
    setattr(model, 'author_scale', scale)
    clone = clone_for_export(model, CaptureSession(model, example_inputs()))
    config = getattr(clone, 'author_config')
    copied_scale = getattr(clone, 'author_scale')
    assert isinstance(config, SimpleNamespace) and isinstance(copied_scale, partial)
    assert config is getattr(clone, 'author_alias') and config is not namespace
    assert config.items is not namespace.items
    assert config.counts.default_factory is int and config.indices == range(1, 3)
    assert copied_scale is not scale and copied_scale.keywords is not scale.keywords
    cast(list[float], config.items).append(2.)
    cast(defaultdict[str, int], config.counts)['added'] += 1
    assert namespace.items == [1.] and dict(namespace.counts) == {'initial': 0}
    capture(model, example_inputs())
