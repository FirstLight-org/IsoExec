"""Observable Python input specializations fail before component execution."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import pytest
from torch.utils import _pytree as pytree

from isoexec.compiler.capture.capture import capture
from isoexec.compiler.tests.test_capture_cpu import Model, example_inputs


class SignedInputModel(Model):
    def forward(self, input_ids, seq, *, factor=1.0, sign_source=0.0):
        return super().forward(input_ids, seq, factor=factor * math.copysign(1.0, sign_source))


@pytest.mark.parametrize("captured_zero,runtime_zero", [(0.0, -0.0), (-0.0, 0.0)])
def test_specialized_float_preserves_signed_zero(captured_zero: float, runtime_zero: float):
    ids, seq = example_inputs()
    graph = capture(SignedInputModel(), (ids, seq), {"sign_source": captured_zero})
    with pytest.raises(ValueError, match="specialized scalar"):
        graph.program(ids, seq, sign_source=runtime_zero)


class KeyedInputsModel(Model):
    def forward(self, input_ids, seq, *, factor=1.0, coefficients: dict[int | float, float] | None = None):
        hidden = self.lexicon(input_ids)
        for unit in self.units:
            hidden = unit(hidden, seq)
        for key, value in (coefficients or {}).items():
            scale = math.copysign(value, key)
            hidden = hidden * (-scale if type(key) is bool else scale)
        return self.readout(self.finish(hidden) * factor)


@pytest.mark.parametrize("captured_key,runtime_key", [(0.0, -0.0), (-0.0, 0.0), (1, True), (True, 1)])
def test_mapping_keys_preserve_float_bits_and_python_types(captured_key, runtime_key):
    ids, seq = example_inputs()
    graph = capture(KeyedInputsModel(), (ids, seq), {"coefficients": {captured_key: 2.0}})
    with pytest.raises(ValueError, match="mapping key order"):
        graph.program(ids, seq, coefficients={runtime_key: 2.0})


@dataclass
class ContextBox:
    coefficient: float
    sign: float


def _box_flatten(box: ContextBox) -> tuple[list[Any], float]:
    return [box.coefficient], box.sign


def _box_unflatten(values: Iterable[Any], sign: float) -> ContextBox:
    return ContextBox(next(iter(values)), sign)


def _box_keys(box: ContextBox) -> tuple[list[tuple[pytree.KeyEntry, Any]], float]:
    key = cast(pytree.KeyEntry, pytree.GetAttrKey("coefficient"))
    return [(key, box.coefficient)], box.sign


pytree.register_pytree_node(ContextBox, _box_flatten, _box_unflatten, flatten_with_keys_fn=_box_keys)


class ContextInputsModel(Model):
    def forward(self, input_ids, seq, *, factor=1.0, box: ContextBox | None = None):
        scale = 1.0 if box is None else math.copysign(box.coefficient, box.sign)
        return super().forward(input_ids, seq, factor=factor * scale)


@pytest.mark.parametrize("captured_zero,runtime_zero", [(0.0, -0.0), (-0.0, 0.0)])
def test_registered_pytree_context_preserves_specialized_float_bits(captured_zero: float, runtime_zero: float):
    ids, seq = example_inputs()
    graph = capture(ContextInputsModel(), (ids, seq), {"box": ContextBox(2.0, captured_zero)})
    with pytest.raises(ValueError, match="input structure"):
        graph.program(ids, seq, box=ContextBox(2.0, runtime_zero))
