"""Both Qwen3 packages place every op on SM90 for the trainer and the engine, with the q/k norm in the
fused plain-gamma kernel; SM100 refuses the plain-gamma q/k norm by name (not yet verified there)."""

import json
from pathlib import Path

import pytest

from isoexec import nn
from isoexec.compiler.source import ModelSource
from isoexec.ops import _dispatch
from isoexec.ops.table import OpResolutionError
from isoexec.runtimes.parallel import trivial_placement

_MODELS = Path(__file__).resolve().parents[2]
FIXTURES = {
    "isoexec.models.qwen3": "qwen3/tests/fixtures/qwen3_8b.json",
    "isoexec.models.qwen3_moe": "qwen3_moe/tests/fixtures/qwen3_30b_a3b.json",
}


def _model(package):
    return ModelSource(package).build_meta(json.loads((_MODELS / FIXTURES[package]).read_text()))


@pytest.mark.parametrize("package", sorted(FIXTURES))
@pytest.mark.parametrize("side", ["trainer", "engine"])
def test_released_models_place_on_sm90(monkeypatch, package, side):
    monkeypatch.setattr(_dispatch, "ARCH", "sm90")
    from isoexec.ops.mm import mm_batch_invariant as V

    monkeypatch.setattr(V, "matmul_persistent", getattr(V, "_isoexec_stock_matmul_persistent", V.matmul_persistent))
    model = _model(package)
    nn.place(model, trivial_placement(side=side))
    resolved = nn.resolved_ops(model)
    stages = [slots["qk_rope"] for site, slots in resolved.items() if "qk_rope" in slots]
    assert len(stages) == len(model.layers)
    for stage in stages:
        assert stage.impl == "sm90_qk_norm_rope_v2" and stage.provider == "ops.attention.rope_sm90"
        assert stage.inputs["convention"] == "rms"
        assert dict(stage.owns) == {"norms.qk": stage.impl, "rope.apply": stage.impl}
    bound = [b for slots in resolved.values() for b in slots.values()]
    assert not [b for b in bound if b.provider == "ops.reference"]
    assert {b.inputs.get("convention") for b in bound if b.op.startswith("norms.rms")} == {"rms"}
    if package.endswith("moe"):
        assert {b.inputs.get("routing") for b in bound if b.op == "moe.route_mechanics"} <= {None, "softmax_renorm"}


@pytest.mark.parametrize("package", sorted(FIXTURES))
def test_sm100_refuses_the_plain_gamma_qk_norm_by_name(monkeypatch, package):
    monkeypatch.setattr(_dispatch, "ARCH", "sm100")
    with pytest.raises(OpResolutionError, match=r"norms\.qk.*'rms'|'rms'.*norms\.qk") as exc:
        nn.place(_model(package), trivial_placement(side="trainer"))
    assert "rope_sm100" in str(exc.value) and "layers.0.mixer" in str(exc.value)
