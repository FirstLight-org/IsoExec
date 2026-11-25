"""Check that component facts cover every component of a constructed model as plain JSON data."""

from __future__ import annotations

import json
import os

import pytest

from isoexec.checks.config import UnhashableFact, component_facts
from isoexec.compiler.source import ModelSource

PACKAGE = "isoexec.models.qwen3_5"
_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "qwen3_5", "tests", "fixtures", "qwen3_5_9b.json",
)
LAYERS = 4


def _tiny_hf() -> dict:
    """The shipped Qwen3.5 geometry cut to a few layers: one full-attention layer among GDN ones."""
    with open(_FIXTURE, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    text = raw["text_config"]
    text["num_hidden_layers"] = LAYERS
    text["layer_types"] = text["layer_types"][:LAYERS]
    assert "full_attention" in text["layer_types"] and "linear_attention" in text["layer_types"]
    return raw


@pytest.fixture(scope="module")
def hf() -> dict:
    return _tiny_hf()


def _model(hf: dict):
    return ModelSource(PACKAGE).build_meta(hf)


def test_every_component_states_its_facts_as_plain_data(hf):
    from isoexec.nn.component import Component

    model = _model(hf)
    facts = component_facts(model)
    sites = {path or "<root>" for path, module in model.named_modules() if isinstance(module, Component)}
    assert facts and set(facts) == sites
    assert json.loads(json.dumps(facts, sort_keys=True)) == facts
    kinds = {row.get("kind") for row in facts.values()}
    assert {"embed", "rms_norm", "gdn"} <= kinds


def test_the_attention_facts_carry_the_rotary_sections_capture_reads(hf):
    """``compiler/text.py`` reads ``mrope_section`` from these facts to shape its position input."""
    facts = component_facts(_model(hf))
    sections = [row["mrope_section"] for row in facts.values() if row.get("mrope_section") is not None]
    assert sections and all(isinstance(section, list) for section in sections)


def test_a_fact_with_no_canonical_json_form_refuses_naming_the_component(hf):
    from isoexec.nn.component import Component

    model = _model(hf)
    path, module = next((p, m) for p, m in model.named_modules() if isinstance(m, Component) and p)
    stated = module.facts()
    module.facts = lambda: {**stated, "eps": object()}
    with pytest.raises(UnhashableFact) as exc:
        component_facts(model)
    text = str(exc.value)
    assert text.startswith(f"config.facts: {path}.facts().eps holds")
    assert "must be plain data" in text
