"""Native decode lengths use host metadata without reading device tensor values."""

from types import SimpleNamespace

import pytest
import torch

from isoexec.runtimes.vllm._inputs import engine_sequence_context, select_sequence_metadata


@pytest.mark.parametrize("gdn_first", [False, True])
@pytest.mark.parametrize("decode", [False, True])
def test_hybrid_context_uses_host_lengths_independent_of_mapping_order(gdn_first, decode):
    # Meta boundaries cannot be read with item(): this proves the host-length
    # path is used even when a GDN layer precedes softmax attention.
    count = 2 if decode else 5
    ids = torch.empty(count, dtype=torch.int64, device="meta")
    positions = torch.empty_like(ids)
    cu = torch.empty(3, dtype=torch.int32, device="meta")
    gdn = SimpleNamespace(non_spec_query_start_loc=cu, num_decodes=2 if decode else 1,
                          num_prefills=0 if decode else 1)
    attention = SimpleNamespace(query_start_loc=cu, max_query_len=1 if decode else 4, max_seq_len=32)
    entries = [("gdn", gdn), ("attention", attention)]
    metadata = dict(entries if gdn_first else reversed(entries))
    selected = select_sequence_metadata({"unused": None, **metadata})
    seq = engine_sequence_context(ids, positions, selected, max_model_len=128, mrope=True)
    assert selected is attention and seq.cu_seqlens is cu
    assert seq.max_q == (1 if decode else 4) and seq.max_kv == 32


def test_gdn_only_metadata_retains_mixed_batch_fallback():
    cu = torch.tensor([0, 1, 5], dtype=torch.int32)
    gdn = SimpleNamespace(non_spec_query_start_loc=cu, num_decodes=1, num_prefills=1)
    selected = select_sequence_metadata({"unused": None, "gdn": gdn})
    seq = engine_sequence_context(torch.arange(5), torch.arange(5), selected, max_model_len=128, mrope=False)
    assert selected is gdn and seq.max_q == 4 and seq.max_kv == 128
    assert select_sequence_metadata({"unused": None}) is None


def test_gdn_decode_builds_context_without_request_boundary_values():
    ids = torch.empty(2, dtype=torch.int64, device="meta")
    positions = torch.empty(2, dtype=torch.int64, device="meta")
    cu = torch.empty(3, dtype=torch.int32, device="meta")
    metadata = SimpleNamespace(non_spec_query_start_loc=cu, num_decodes=2, num_prefills=0)

    selected = select_sequence_metadata({"gdn": metadata})
    seq = engine_sequence_context(ids, positions, selected, max_model_len=128, mrope=True)

    assert seq.max_q == 1 and seq.max_kv == 128
    assert seq.cu_seqlens is cu and seq.positions.shape == (3, 2)


def test_nonfirst_stage_derives_one_request_from_positions():
    positions = torch.arange(5, dtype=torch.int64)

    seq = engine_sequence_context(None, positions, None, max_model_len=128, mrope=False)

    assert seq.input_ids is None and seq.max_q == seq.max_kv == 5
    assert seq.cu_seqlens.tolist() == [0, 5] and torch.equal(seq.positions, positions)
