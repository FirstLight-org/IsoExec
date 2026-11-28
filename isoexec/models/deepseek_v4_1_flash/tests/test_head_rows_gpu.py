"""The inference head stage in one launch (``boundary_rows``) equals expand + collapse + norms_sm100 RMSNorm bit for
bit (``DeepSeekV41HeadStage.forward``'s two forms)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("rows", [1, 7, 32, 64, 300, 2048])
def test_head_rows_equal_three_launches(rows):
    from isoexec.ops.norms.norms_sm100.ops import rms_norm
    from isoexec.ops.residual.deepseek_mhc_sm100.ops import boundary_rows, collapse, expand
    g = torch.Generator(device="cuda").manual_seed(rows)
    H = 5120
    x = torch.randn((1, rows, 4, H), generator=g, device="cuda").to(torch.bfloat16)
    out = torch.randn((1, rows, H), generator=g, device="cuda").to(torch.bfloat16)
    pre = torch.rand((1, rows, 4), generator=g, device="cuda")
    post = torch.rand((1, rows, 4), generator=g, device="cuda") * 2
    comb = torch.rand((1, rows, 4, 4), generator=g, device="cuda")
    w = (torch.rand(H, generator=g, device="cuda") + 0.5).to(torch.bfloat16)
    eps = 1e-20
    with torch.no_grad():
        new = expand(out, x, post, comb)
        h = collapse(new, pre)
        ref = rms_norm(h.reshape(-1, H).contiguous(), w, eps, False).view(h.shape)
        got = boundary_rows(x, pre, w, eps, out=out, post=post, comb=comb, encode=False)
    assert got is not None
    assert torch.equal(got[1], ref)
    assert torch.equal(got[0], new)
