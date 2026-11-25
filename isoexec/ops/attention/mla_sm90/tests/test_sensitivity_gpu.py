"""Require the suite to detect stock FlashMLA/FA3 num_splits=0 nondeterminism and the emulated role split.
The role-split control uses unsplit prompt rows and segmented generated rows to expose multi-turn divergence;
FA3 num_splits=1 is the batch-invariant control."""

import pytest

from isoexec.ops.attention.mla_sm90.tests._harness import cuda_ok, load_backend
from isoexec.ops.attention.mla_sm90.tests._relations import Config, run_relation

pytestmark = pytest.mark.skipif(not cuda_ok(), reason="requires an SM90 (H100) device")


def _backend(name):
    b, why = load_backend(name)
    if b is None:
        pytest.skip(f"{name} unavailable: {why}")
    return b


@pytest.mark.parametrize("name", ["flashmla_head", "flashmla_vllm", "fa3_ns0"])
@pytest.mark.parametrize("relation", ["logprob_contract", "batch_composition", "decode_vs_multirow"])
def test_suite_detects_incumbent_nondeterminism(name, relation):
    o = run_relation(relation, _backend(name), Config())
    assert o.status == "fail" and o.failures, f"{relation} did not detect {name}: {o.summary()}"


def test_role_split_program_is_detected_breaking_multi_turn():
    o = run_relation("role_split_detected", _backend("mla_sm90"), Config())
    assert o.status == "pass", o.summary()


def test_flashmla_long_context_fails_composition_with_few_companions():
    o = run_relation("batch_composition", _backend("flashmla_head"), Config(long_ctx=32768 + 21))
    small = [d.name for d in o.failures if d.name.startswith("decode/") and "many" not in d.name]
    assert small, f"expected decode composition failures at B<=6, got {[d.name for d in o.failures]}"


@pytest.mark.parametrize("relation", ["batch_composition", "decode_vs_multirow", "chunked_prefill",
                                      "boundary_lengths", "row_placement"])
def test_fa3_ns1_passes_its_proven_relations(relation):
    o = run_relation(relation, _backend("fa3_ns1"), Config())
    assert o.status == "pass", o.summary()


def test_nonfinite_relation_detects_fa3_padding_read():
    """FA3 multiplies masked (zero) probabilities by padded V in the last tile (ops/attention/mla_absorbed.pack_latent)."""
    o = run_relation("nonfinite_unused_pages", _backend("fa3_ns1"), Config())
    assert o.status == "fail" and any("/nan/" in d.name for d in o.failures), o.summary()
