"""The census comes from the registered model, never from a profile the shape module invents.
A process with no registered census is refused, not handed an empty or borrowed site list."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from isoexec.ops.mm import mm_shapes

BF16, FP32 = mm_shapes.dtype_of("bf16"), mm_shapes.dtype_of("fp32")


@pytest.fixture
def census() -> mm_shapes.GemmCensus:
    """Two decoder layers' worth of sites, the way ``isoexec.nn.gemm_census`` records them."""
    layer = (
        mm_shapes.GemmSpec("attn.qkv", 256, 768, BF16, "tp"),
        mm_shapes.GemmSpec("moe.router", 256, 8, FP32),
        mm_shapes.GemmSpec("shexp.gate", 256, 1, BF16),
        mm_shapes.GemmSpec("shexp.fc1", 256, 256, BF16, "tp"),
    )
    return mm_shapes.GemmCensus(layer + layer + (mm_shapes.GemmSpec("lm_head", 256, 2048, BF16, "tp"),))


@pytest.fixture(autouse=True)
def _isolate_live_context(monkeypatch):
    monkeypatch.setattr(mm_shapes, "_LIVE", None, raising=False)
    yield
    mm_shapes._LIVE = None


def test_no_registered_census_is_refused_not_given_another_models_sites():
    with pytest.raises(LookupError, match="set_live_context"):
        mm_shapes.census_for(None, 1, 1)


def test_the_refusal_is_not_an_empty_census():
    """An empty census reads as "this model runs no dense GEMMs", which is the same wrong answer
    with the evidence deleted. `census_union` must propagate, not swallow into `{}`."""
    with pytest.raises(LookupError):
        mm_shapes.census_union(None, (1, 2), (1, 2))


def test_occurrences_become_per_token_call_counts(census):
    sites = {s.name: s for s in mm_shapes.census_for(census, 1, 1)}
    assert [s.name for s in mm_shapes.census_for(census, 1, 1)] == sorted(sites)
    assert (sites["attn.qkv"].calls, sites["lm_head"].calls) == (2, 1)
    assert sites["moe.router"].dtype is torch.float32


def test_tp_shards_the_sharded_sites(census):
    at1 = {s.name: s for s in mm_shapes.census_for(census, 1, 1)}
    at2 = {s.name: s for s in mm_shapes.census_for(census, 2, 2)}
    assert at2["attn.qkv"].N == at1["attn.qkv"].N // 2
    assert at2["moe.router"].N == at1["moe.router"].N  # axis="none": unsharded either way


def test_union_drops_inexact_cuts_and_head_granule_refusals():
    census = mm_shapes.GemmCensus((
        mm_shapes.GemmSpec("attn.qkv", 64, 96, BF16, "tp"),
        mm_shapes.GemmSpec("mla.q_up", 32, 6 * 16, BF16, "tp", granule=6),
    ))
    with pytest.raises(ValueError, match="divisible by TP=4"):
        census.sites(4, 4)
    union = mm_shapes.census_union(census, (1, 2, 4), (1,))
    assert union == {
        (64, 96, BF16): "attn.qkv@tp1/etp1", (32, 96, BF16): "mla.q_up@tp1/etp1",
        (64, 48, BF16): "attn.qkv@tp2/etp1", (32, 48, BF16): "mla.q_up@tp2/etp1",
    }


def test_coverage_banner_names_the_missing_census_instead_of_inventing_coverage(capsys, monkeypatch):
    monkeypatch.setenv("ISOEXEC_MODEL_PATH", "/ckpt")
    mm_shapes.set_live_context(census=None, model_path="/ckpt", tp=1, etp=1, side="TRAINER")
    mm_shapes.report_install_coverage("MM_SM90_SEGTREE", lambda key: True, flag="ISOEXEC_MM_SM90_SEGTREE")
    out = capsys.readouterr().out
    assert "no census" in out and "LookupError" in out and "set_live_context" in out
    assert "sites covered" not in out  # never reports coverage against another model's sites


def test_coverage_banner_reports_the_registered_census(census, capsys):
    mm_shapes.set_live_context(census=census, model_path="/ckpt", tp=2, etp=2, side="TRAINER")
    mm_shapes.report_install_coverage("MM_SM90_SEGTREE", lambda key: key == (256, 384, BF16),
                                      flag="ISOEXEC_MM_SM90_SEGTREE")
    out = capsys.readouterr().out
    assert "1/5 sites covered" in out and "(256,384,bf16) attn.qkv" in out


def test_mm_shapes_names_no_model_package():
    """The structural half: `ops/` is handed a census, and may not know model packages."""
    src = Path(mm_shapes.__file__).read_text(encoding="utf-8")
    assert "isoexec.models" not in src and "qwen" not in src.lower()
