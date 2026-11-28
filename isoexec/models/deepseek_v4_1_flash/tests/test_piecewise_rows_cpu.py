"""Row-count contract of a padded PIECEWISE engine step (defect 17): vLLM builds the attention metadata of a breakable
PIECEWISE step for its scheduled tokens while the model inputs are padded to the graph size. The step is its real
tokens: every per-token tensor of ``EngineStep`` has the metadata's row count, equal to the unpadded step's values;
the step is not a captured decode bucket; the attention runs on the real rows and the padding rows' output is zero."""
import pytest
import torch

pytest.importorskip("vllm")

from isoexec.models.deepseek_v4_1_flash import block, engine  # noqa: E402
from isoexec.models.deepseek_v4_1_flash.tests.test_engine_cache_cpu import (  # noqa: E402,F401
    _metadata, build, sm100, vllm_config)


def _pad(md, positions, extra):
    """The same step as vLLM's PIECEWISE runner hands it: inputs and slot mappings padded by ``extra`` rows (padding
    slots -1), the per-token request map for the scheduled tokens only."""
    out = {}
    for name, ns in md.items():
        d = dict(vars(ns))
        if name.endswith(".swa_cache"):
            d["slot_mapping"] = torch.cat([ns.slot_mapping, torch.full((extra,), -1, dtype=ns.slot_mapping.dtype)])
            d["is_valid_token"] = d["slot_mapping"] >= 0
        out[name] = type(ns)(**d)
    return out, torch.cat([positions, torch.zeros(extra, dtype=positions.dtype)])


@pytest.mark.parametrize("starts,seq_lens", [([0, 5, 6, 40], [5, 260, 34]), ([0, 1, 2, 3, 4, 5], [9, 300, 40, 2, 77])])
def test_padded_piecewise_step_is_its_real_tokens(vllm_config, sm100, starts, seq_lens):
    b = build(vllm_config, "debug_config.json", 40)
    R = len(seq_lens)
    comp = [[3, 7, 9]] * R
    tables = {"comp": comp, "state": list(range(13, 13 + R))}
    swa_tables = {g: [[(g * 7 + s * 11 + j * 3) % 39 + 1 for j in range(10)] for s in range(R)]
                  for g in range(len(b.groups))}
    md, positions, tok, pos = _metadata(b, tables, swa_tables, starts, seq_lens, False)
    n = starts[-1]
    padded, ppos = _pad(md, positions, 64 - n)
    swa = padded[b.rows[0].swa.prefix]
    assert engine.real_tokens(swa, ppos.shape[-1]) == n
    assert engine.real_tokens(md[b.rows[0].swa.prefix], positions.shape[-1]) == n      # eager: no padding
    ref = engine.EngineStep(b.cfg, b.rows[0], md, positions, False)
    got = engine.EngineStep(b.cfg, b.rows[0], padded, ppos, False)
    for i, row in enumerate(b.rows):
        ref.bind_window(i, row, md)
        got.bind_window(i, row, padded)
    assert (got.tokens, got.padded, ref.tokens, ref.padded) == (n, 64, n, n)
    for name in ("tok_seq", "tok_pos", "valid", "cu_q", "p0", "state_slot"):
        a, c = getattr(got.batch, name), getattr(ref.batch, name)
        assert a.shape == c.shape and torch.equal(a, c), name
    for i in range(len(b.rows)):
        assert torch.equal(got.batch.window_rows[i], ref.batch.window_rows[i]), i
        assert torch.equal(got.batch.window_lists[i], ref.batch.window_lists[i]), i


def test_attention_rows_and_zero_padding():
    step = type("S", (), {"tokens": 5})()
    rows = torch.randn(8, 16)
    assert block._step_rows(step, rows).shape == (5, 16)
    out = torch.randn(5, 16)
    padded = block._padded_rows(out, 8)
    assert torch.equal(padded[:5], out) and not padded[5:].any()
    step.tokens = 8
    assert block._step_rows(step, rows) is rows and block._padded_rows(rows, 8) is rows
