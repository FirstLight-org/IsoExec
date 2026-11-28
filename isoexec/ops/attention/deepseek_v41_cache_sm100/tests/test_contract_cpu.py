"""CPU contract: served geometry, declared constants, and the CPU emulation against float64."""

import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops, reference as ref, spec
from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import _harness as h


def _published():
    from ._config import published_config
    return published_config()


def test_published_config_is_served_cpu():
    cfg = _published()
    assert ops.supported_config(cfg)
    assert spec.index_weight_scale(cfg) == 2.0 ** -6
    assert cfg.sliding_window + cfg.index_topk == 640


@pytest.mark.parametrize("change,match", [
    (dict(index_topk=600), "window"),
    (dict(candidate_blocks=8), "candidate"),
    (dict(hidden=1056), "hidden"),
])
def test_refusals_cpu(change, match):
    with pytest.raises(ValueError, match=match):
        ops.supported_config(h.native_cfg(**change))


def test_rms_emulation_is_close_to_float64_cpu():
    g = torch.Generator().manual_seed(0)
    for D in (128, 512, 1280):
        x = (torch.randn((64, D), generator=g) * 3).to(torch.bfloat16)
        w = torch.rand(D, generator=g) + 0.5
        got = ref.rms_emul(x, w, 1e-20).double()
        want = x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-20) * w.double()
        assert ((got - want).abs() <= want.abs() * 2 ** -8 + 1e-30).all()


def test_rope_table_rows_do_not_depend_on_length_cpu():
    a = ref.rope_frequencies(64, 1024, 65536, 160000.0, 16, 32, 1)
    b = ref.rope_frequencies(64, 4096, 65536, 160000.0, 16, 32, 1)
    assert torch.equal(torch.view_as_real(a), torch.view_as_real(b[:1024]))


def test_index_codes_are_the_b128_record_cpu():
    """An index key's codes times its one exponent (four equal UE8M0 bytes) == the quantize/dequantize program."""
    g = torch.Generator().manual_seed(1)
    x = (torch.randn((16, 128), generator=g) * 2).to(torch.bfloat16)
    x[3] = 0
    codes, sf = ref.index_codes(x)
    b = torch.stack([((sf >> (8 * i)) & 255) for i in range(4)], -1)
    assert bool((b == b[:, :1]).all()) and int(b[3, 0]) == 127 - 22
    vals = codes.view(torch.float8_e4m3fn).float() * torch.exp2(b[:, :1].float() - 127)
    assert torch.equal(vals, ref.quant_emul(x, 128).float())