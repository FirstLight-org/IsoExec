"""CPU contract: at trainer TP > 1, ``decoder.entry``'s token-wise projection (its outputs feed the decoder's
head-split attention, so each rank holds a partial gradient of them) sums its input gradient over TP in rank
order, like every attention row's input seam; the returned attention input ``h`` (the boundary block sums its own
part) and TP 1 are left alone."""
import copy
from types import SimpleNamespace

import pytest
import torch

from isoexec.compiler.realization.tp import shard_parameter_shapes
from isoexec.models.deepseek_v4_1_flash import block
from isoexec.models.deepseek_v4_1_flash.tests.test_deepseek_v4_1_flash_cpu import SOURCE, placement, raw


@pytest.fixture(scope="module")
def debug_model():
    return SOURCE.build_meta(raw("debug_config.json"))


def _entry(debug_model, tp):
    model = copy.deepcopy(debug_model)
    shard_parameter_shapes(model, tp_size=tp, side="trainer", etp_size=tp)
    entry = model.decoder.entry
    entry.apply_placement(placement(tp, 0, 1, 0, "trainer"), site="decoder.entry")
    return entry


@pytest.mark.parametrize("tp", [1, 2])
def test_entry_projection_input_gradient_is_summed_over_tp(debug_model, tp, monkeypatch):
    from isoexec.ops.collectives import parallel_autograd
    sums = []
    # a stand-in for the rank-ordered sum: two ranks holding the same partial
    monkeypatch.setattr(parallel_autograd, "ordered_sum", lambda g, group: sums.append(group) or 2 * g)
    entry = _entry(debug_model, tp)
    cfg = entry.cfg
    S, H, T = cfg.hc_mult, cfg.hidden_size, 4
    seen = {}

    def boundary(cfg, fn, scale, base, norm_weight, carry, *, sublayer):
        x = carry.x
        return x * 1, (carry.pre * 1, carry.pre * 1, x.new_zeros(x.shape[0], S, S), None), x.sum(1)

    def entry_projection(holder, h, cfg, layer, positions, *, linear_fn, step=None):
        seen["h"] = h
        return 3 * h, (None, None), 5 * h

    monkeypatch.setattr(block, "boundary", boundary)
    monkeypatch.setattr(block, "ced_ops", lambda: SimpleNamespace(entry_projection=entry_projection))
    monkeypatch.setattr(block, "_decoder_rows", lambda *a: (None, None))
    x = torch.randn(1, T, S, H, requires_grad=True)
    pre = torch.randn(1, T, S)
    out = entry._forward_impl(x, pre, seq=SimpleNamespace(positions=torch.arange(T)))
    h, main_kv, window_kv = out[2], out[5], out[8]

    # through the projection: (3 + 5) per element, times the TP sum at TP 2
    (grad,) = torch.autograd.grad(main_kv.sum() + window_kv.sum(), x, retain_graph=True)
    assert torch.equal(grad, torch.full_like(grad, 8.0 * (2 if tp > 1 else 1)))
    assert len(sums) == (1 if tp > 1 else 0)
    # the returned attention input is not summed here (the boundary block's own input seam sums its part)
    (grad,) = torch.autograd.grad(h.sum(), x)
    assert torch.equal(grad, torch.ones_like(grad))
    assert len(sums) == (1 if tp > 1 else 0)
    # outside autograd (scoring, the engine's no-grad forward) nothing is inserted
    sums.clear()
    with torch.no_grad():
        entry._forward_impl(x, pre, seq=SimpleNamespace(positions=torch.arange(T)))
    assert seen["h"].grad_fn is None and not sums
