"""The index-score route choice of the paged path (``paged.index_gather_route``) is a function of the launch shape
only: prefill-shaped launches take the gathered route, decode-shaped ones the paged kernel, and admission limits
(sequences per launch, int32 key offsets) always keep the paged kernel. GPU relations: test_index_gather_route_gpu.py.
"""
import pytest

from isoexec.ops.attention.deepseek_v41_cache_sm100 import paged
from isoexec.ops.attention.sparse_sm100 import index_logits as il


@pytest.mark.parametrize("rows,seqs,table_rows,width,want", [
    (4096, 1, 1, 4096, True),          # one long prefill
    (1024, 1, 1, 2048, True),
    (400, 2, 2, 1024, True),           # two prefills
    (255, 1, 1, 4096, False),          # below the row floor
    (256, 64, 64, 1024, False),        # decode-heavy: 4 rows per sequence
    (64, 64, 64, 1024, False),         # decode
    (4096, 4, 4, 1 << 19, False),      # packed buffer above the key cap
    (8192, 1, 1, 1 << 20, True),       # one sequence at the key cap
])
def test_rule_cpu(monkeypatch, rows, seqs, table_rows, width, want):
    monkeypatch.setattr(paged, "INDEX_GATHER", None)
    assert paged.index_gather_route(rows, seqs, table_rows, width) is want


def test_forcing_and_admission_cpu(monkeypatch):
    monkeypatch.setattr(paged, "INDEX_GATHER", True)
    assert paged.index_gather_route(4, 4, 4, 1024)
    assert not paged.index_gather_route(4, 4, il.MAX_GATHER_SEQS + 1, 64)           # key_ranges' sequence limit
    assert not paged.index_gather_route(4, 4, 4096, 1 << 20)                        # int32 key offsets
    assert not paged.index_gather_route(0, 1, 1, 64)
    monkeypatch.setattr(paged, "INDEX_GATHER", False)
    assert not paged.index_gather_route(4096, 1, 1, 4096)


def test_capacity_bound_cpu():
    # at most min(rows, sequences) sequences have a row, each with at most width keys
    assert il.gather_capacity(4096, 3, 1000) == 3000
    assert il.gather_capacity(2, 300, 1000) == 2000
