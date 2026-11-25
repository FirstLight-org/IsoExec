"""The engine step's one-launch window glue (``EngineStep._bind_windows`` / ``decoder_windows`` / ``_page_tables``)
gives every layer the same window rows, main lists and page tables as the per-layer torch glue
(``ISOEXEC_V41_WINDOW_GLUE=0``), on native-shaped metadata on the GPU (decode, graph and mixed steps), and the
decoder rows the same gathered and floored lists as ``rows_subset`` + ``paged.floor_window_lists``."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from isoexec.models.deepseek_v4_1_flash import engine  # noqa: E402
from isoexec.models.deepseek_v4_1_flash.tests.test_engine_cache_cpu import (  # noqa: E402,F401
    _metadata, build, sm100, vllm_config)
from isoexec.ops.attention.deepseek_v41_cache_sm100 import paged  # noqa: E402


def _cuda(md):
    out = {}
    for name, ns in md.items():
        out[name] = SimpleNamespace(**{k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in vars(ns).items()})
    return out


CASES = {"graph": ([0, 1, 2, 3], [300, 1, 129], True), "mixed": ([0, 5, 6, 40], [5, 260, 34], False),
         "decode": ([0, 1, 2, 3], [77, 300, 2], False)}


@pytest.mark.parametrize("case", sorted(CASES))
def test_step_windows_equal_per_layer_glue(vllm_config, sm100, monkeypatch, case):
    starts, seq_lens, graph = CASES[case]
    b = build(vllm_config, "debug_config.json", 40, device="cuda")
    R = len(seq_lens)
    comp = [[3, 7, 9], [11, 2, 30], [5, 6, 8]]
    tables = {"comp": comp, "state": [13, 14, 15]}
    swa_tables = {g: [[(g * 7 + s * 11 + j * 3) % 39 + 1 for j in range(10)] for s in range(R)]
                  for g in range(len(b.groups))}
    md, positions, _, _ = _metadata(b, tables, swa_tables, starts, seq_lens, graph)
    md, positions = _cuda(md), positions.cuda()
    steps = {}
    for glue in (True, False):
        monkeypatch.setattr(engine, "WINDOW_GLUE", glue)
        engine.WINDOW_GLUE_MISSES.clear()
        step = engine.EngineStep(b.cfg, b.rows[0], md, positions, graph, capturing=graph)
        for i, row in enumerate(b.rows):
            step.bind_window(i, row, md)
        assert (step._win is not None) == glue, engine.WINDOW_GLUE_MISSES
        steps[glue] = step
    on, off = steps[True].batch, steps[False].batch
    assert sorted(on.window_rows) == sorted(off.window_rows) == list(range(len(b.rows)))
    for i in off.window_rows:
        assert torch.equal(on.window_rows[i], off.window_rows[i]), i
        assert torch.equal(on.window_lists[i], off.window_lists[i]), i
        assert steps[True].pools.window[i].data_ptr() == steps[False].pools.window[i].data_ptr()
    for name in ("comp_bt_by_source", "index_bt", "index_read_bt"):
        a, c = getattr(on, name), getattr(off, name)
        assert list(a) == list(c)
        for k in c:
            assert torch.equal(a[k], c[k]), (name, k)

    # the decoder rows: every other token, the floor at the segment start of the case
    cfg = b.cfg
    bnd = max(cfg.kv_source_layer_ids)
    sel = torch.arange(0, starts[-1], 2, device="cuda")
    floor = torch.clamp(on.tok_pos.index_select(0, sel) - 60, min=0).to(torch.int32)
    full = steps[False].batch.rows_subset(sel, window_floor=floor)
    sub = steps[True].batch.rows_subset(sel, window_floor=floor, windows=False)
    assert steps[True].decoder_windows(sub, sel, bnd) and sub.lists_floored
    layers = [i for i in range(len(b.rows)) if i >= bnd]
    assert sorted(sub.window_lists) == layers
    for i in layers:
        assert torch.equal(sub.window_rows[i], full.window_rows[i]), i
        want = full.window_lists[i] if i == bnd else paged.floor_window_lists(full.window_lists[i], full.tok_pos, floor)
        assert torch.equal(sub.window_lists[i], want), i
