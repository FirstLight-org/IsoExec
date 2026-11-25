"""Native text binding retains export constraints across packed batches."""

from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental import _config as symbolic_config
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils import _pytree

import isoexec.nn as inn
from isoexec.compiler.capture.capture import capture
from isoexec.compiler.tests.test_capture_cpu import Model
from isoexec.compiler.text import text_inputs, text_roots
from isoexec.compiler.capture.graph import PipelineError
from isoexec.ops.table import OpTable
from isoexec.runtimes.parallel import trivial_placement
from isoexec.schema import make_sequence_context
from isoexec.schema.sequence import register_pytrees


@pytest.mark.parametrize("profile", ["trainer", "engine"])
def test_text_capture_handles_singleton_and_variable_packed_requests(profile):
    model = Model("cpu")
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.04)
    examples = text_inputs(model, device="cpu", max_tokens=32, profile=profile)
    captured = capture(model, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes,
                       root_layouts=examples.root_layouts)
    inn.place(model, trivial_placement(), op_table=OpTable(mode="reference"))
    for lengths in ([1], [2, 4], [1, 1, 3]):
        tokens = sum(lengths)
        ids = torch.arange(tokens, dtype=torch.int32) % 32
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu_cpu = torch.tensor(offsets, dtype=torch.int32)
        seq = make_sequence_context(torch.cat([torch.arange(length) for length in lengths]),
                                    cu_cpu.clone(), max_q=max(lengths), max_kv=max(lengths) + 5,
                                    cu_seqlens_cpu=cu_cpu if profile == "trainer" else None)
        roots = text_roots(ids, seq)
        assert roots["input_ids"].dtype == torch.int64
        assert roots["seq"].input_ids is roots["input_ids"]
        assert seq.input_ids is None and ids.dtype == torch.int32
        assert torch.equal(model(**roots), captured.program(**roots))
        with pytest.raises(ValueError, match="specialized scalar"):
            captured.program(**roots, factor=0.5)
        with pytest.raises(ValueError, match="identity/alias"):
            captured.program(roots["input_ids"], replace(roots["seq"], input_ids=roots["input_ids"].clone()))
    seq = roots["seq"]
    with pytest.raises(ValueError, match="dynamic scalar Python type"):
        captured.program(roots["input_ids"], replace(seq, max_q=float(seq.max_q)))


def test_text_provider_preserves_call_convention_and_requires_explicit_extra_roots():
    class KeywordModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Model()

        def forward(self, input_ids, *, seq, factor=1.0):
            return self.model(input_ids, seq, factor=factor)

    model = KeywordModel()
    examples = text_inputs(model, device="cpu", profile="engine")
    assert len(examples.args) == 1 and set(examples.kwargs) == {"seq", "factor"}
    captured = capture(model, examples.args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes)
    seq = examples.kwargs["seq"]
    roots = text_roots(examples.args[0].to(torch.int32), seq)
    assert roots["seq"].positions is seq.positions and roots["seq"].cu_seqlens is seq.cu_seqlens
    assert roots["seq"].cu_seqlens_cpu is None and roots["seq"].max_kv == seq.max_kv
    guard = captured.program.get_submodule("_isoexec_export_input_guard")
    guard((roots["input_ids"], roots["seq"], 1.0))
    for invalid in (examples.args[0].float(), examples.args[0].bool()):
        with pytest.raises(ValueError, match="int32 or int64"):
            text_roots(invalid, seq)

    # Position rank follows a component declaration, independent of model family.
    model.model.units[0].mix = inn.Attention(hidden=16, heads=4, kv_heads=2, head_dim=4,
                                            rope_theta=10000, mrope_section=(1, 1, 0))
    mrope_inputs = text_inputs(model, device="cpu", profile="engine")
    assert mrope_inputs.kwargs["seq"].positions.shape == (3, 4)
    mrope_capture = capture(model, mrope_inputs.args, mrope_inputs.kwargs,
                            dynamic_shapes=mrope_inputs.dynamic_shapes)
    mrope_capture.program.get_submodule("_isoexec_export_input_guard")(
        (mrope_inputs.args[0], mrope_inputs.kwargs["seq"], 1.0))

    class ExtraRoots(nn.Module):
        def forward(self, input_ids, seq, residual):
            return residual

    with pytest.raises(ValueError, match="residual.*explicit CaptureInputs"):
        text_inputs(ExtraRoots(), device="cpu")


@pytest.mark.parametrize("condition", ["equal", "at_most"])
@pytest.mark.parametrize("existing_fake_mode", [False, True])
def test_singleton_shape_branch_cannot_silently_disappear(condition: str, existing_fake_mode: bool) -> None:
    class Branch(Model):
        def forward(self, input_ids, seq, *, factor=1.0):
            x = self.lexicon(input_ids)
            for unit in self.units:
                x = unit(x, seq)
            singleton = input_ids.shape[0] == 1 if condition == "equal" else input_ids.shape[0] <= 1
            return self.readout(self.finish(x) * (2.0 if singleton else 1.0) * factor)

    model = Branch()
    examples = text_inputs(model, device="cpu", max_tokens=16)
    with cast(Any, symbolic_config).patch(backed_size_oblivious=False):
        args = examples.args
        mode = FakeTensorMode(shape_env=ShapeEnv())
        if existing_fake_mode:
            register_pytrees()
            args = _pytree.tree_map_only(torch.Tensor, lambda value: mode.from_tensor(value, static_shapes=False), args)
        if condition == "equal":
            with pytest.raises(PipelineError, match="Constraints violated.*tokens"):
                capture(model, args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes)
        else:
            captured = capture(model, args, examples.kwargs, dynamic_shapes=examples.dynamic_shapes)
            ids = torch.ones(1, dtype=torch.int64)
            cu = torch.tensor([0, 1], dtype=torch.int32)
            seq = make_sequence_context(torch.zeros(1, dtype=torch.int64), cu, max_q=1, max_kv=1,
                                        input_ids=ids, cu_seqlens_cpu=cu.clone())
            runtime_args = (ids, seq)
            if existing_fake_mode:
                runtime_args = _pytree.tree_map_only(torch.Tensor,
                    lambda value: mode.from_tensor(value, static_shapes=True), runtime_args)
            # Export may retain a lower-bound predicate in the generated
            # runtime guard instead of refusing the entire capture.
            with pytest.raises(ValueError, match="Guard failed:.*> 1"):
                captured.program(*runtime_args)
        assert symbolic_config.backed_size_oblivious is False
