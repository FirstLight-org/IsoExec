"""Engine window glue: the one-launch window rows / main lists of every layer (``window_rows_lists``) and of the
decoder rows (``decoder_window_lists``) equal the per-layer torch glue they replace (``engine.EngineStep.bind_window``'s
``slot_rows`` / ``torch.where`` / ``torch.cat``, ``ced._engine_rows``' gathers, ``paged.floor_window_lists``) bit for
bit, on decode-only and mixed steps, padded rows, int32 and int64 slot ids and offsets past the int32 range."""
import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import paged
from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

BLOCK, W = 32, 128


def _slot_rows(slots, per, offset, block=BLOCK):
    """engine.slot_rows (the reference formula, verbatim)."""
    return torch.where(slots >= 0, slots // block * per + slots % block + offset, -1)


def _torch_window(slot_mapping, dec, pre, nd, npf, valid, per, offset, N):
    """engine.EngineStep.bind_window's torch glue for one layer (verbatim)."""
    rows = _slot_rows(slot_mapping[:N].to(torch.int64), per, offset)
    if npf == 0:
        lists = dec.as_strided((N, dec.shape[-1]), (dec.stride(0), 1), dec.storage_offset())
    else:
        parts = [dec[:nd].reshape(nd, W), pre[:npf].reshape(npf, W)]
        if nd + npf < N:
            parts.append(torch.full((N - nd - npf, W), -1, dtype=torch.int32, device=dec.device))
        lists = torch.cat(parts)
    lists = torch.where(valid[:, None], _slot_rows(lists, per, offset), -1).to(torch.int32)
    return rows.contiguous(), lists.contiguous()


def _metadata(g, N, nd, npf, slot_dtype, blocks=4000):
    slots = torch.randint(-1, blocks * BLOCK, (N + 5,), generator=g, device="cuda").to(slot_dtype)
    # window index buffers sized beyond the step (graph buffers), -1 tails as native writes them
    dec_buf = torch.randint(-1, blocks * BLOCK, (N + 7, W), generator=g, device="cuda", dtype=torch.int32)
    dec_buf[:, 100:] = -1
    pre_buf = torch.randint(-1, blocks * BLOCK, (max(npf, 1) + 3, W), generator=g, device="cuda", dtype=torch.int32)
    valid = torch.rand(N, generator=g, device="cuda") > 0.2
    return slots, dec_buf[: max(nd, 1)], pre_buf, valid


@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("case", ["decode", "decode_padded", "mixed", "mixed_tail", "prefill"])
def test_window_rows_lists_equal_torch_glue(slot_dtype, case):
    g = torch.Generator(device="cuda").manual_seed(11)
    N, nd, npf = {"decode": (32, 32, 0), "decode_padded": (64, 37, 0), "mixed": (300, 20, 280),
                  "mixed_tail": (300, 20, 250), "prefill": (2048, 0, 2048)}[case]
    layers = []
    for l in range(5):
        slots, dec, pre, valid = _metadata(g, N, nd, npf, slot_dtype)
        # per-layer row geometry; one layer past the int32 range (int32 lists wrap exactly as torch int32 math does)
        per, offset = [17, 64, 33, 64, 9][l], [0, 5, 63, 2**31 - 3, 2**30][l]
        layers.append((slots, dec, pre, per, offset))
    valid = torch.rand(N, generator=g, device="cuda") > 0.2
    rows = torch.empty((len(layers), N), dtype=torch.int64, device="cuda")
    lists = torch.empty((len(layers), N, W), dtype=torch.int32, device="cuda")
    extension().window_rows_lists([s[:N] for s, *_ in layers], [d for _, d, *_ in layers],
                                  [p if npf else None for _, _, p, *_ in layers], [l[3] for l in layers],
                                  [l[4] for l in layers], valid, nd, npf, BLOCK, rows, lists)
    for l, (slots, dec, pre, per, offset) in enumerate(layers):
        ref_rows, ref_lists = _torch_window(slots, dec, pre, nd, npf, valid, per, offset, N)
        assert torch.equal(rows[l], ref_rows), (case, l)
        assert torch.equal(lists[l], ref_lists), (case, l)


@pytest.mark.parametrize("with_floor", [True, False])
def test_decoder_window_lists_equal_gather_and_floor(with_floor):
    g = torch.Generator(device="cuda").manual_seed(12)
    L, N, R = 6, 400, 150
    rows_in = torch.randint(-1, 10**9, (L, N), generator=g, device="cuda", dtype=torch.int64)
    # engine main-list contract: ascending valid rows in slots [0, n), then -1
    n = torch.randint(0, W + 1, (L, N), generator=g, device="cuda")
    lists_in = torch.randint(0, 10**6, (L, N, W), generator=g, device="cuda", dtype=torch.int32)
    lists_in = torch.where(torch.arange(W, device="cuda") < n[..., None], lists_in, -1).to(torch.int32).contiguous()
    sel = torch.sort(torch.randperm(N, generator=g, device="cuda")[:R]).values
    pos = torch.randint(0, 5000, (R,), generator=g, device="cuda", dtype=torch.int32)
    floor = torch.clamp(pos - torch.randint(-20, 200, (R,), generator=g, device="cuda", dtype=torch.int32), min=0)
    floor = floor.to(torch.int32).contiguous() if with_floor else None
    src = [2, 3, 4, 5]                  # decoder layers (boundary first, then the floored ones)
    flags = [0, 1, 1, 1]
    rows_out = torch.empty((len(src), R), dtype=torch.int64, device="cuda")
    lists_out = torch.empty((len(src), R, W), dtype=torch.int32, device="cuda")
    extension().decoder_window_lists(rows_in, lists_in, src, flags, sel, pos, floor, rows_out, lists_out)
    for d, (s, f) in enumerate(zip(src, flags)):
        ref_rows = rows_in[s].index_select(0, sel).contiguous()
        ref_lists = lists_in[s].index_select(0, sel).contiguous()
        if f and floor is not None:
            ref_lists = paged.floor_window_lists(ref_lists, pos, floor)
        assert torch.equal(rows_out[d], ref_rows), d
        assert torch.equal(lists_out[d], ref_lists), d


def test_page_tables_equal_expand_and_split_pages():
    from isoexec.models.deepseek_v4_1_flash.engine import expand_pages, split_pages
    g = torch.Generator(device="cuda").manual_seed(13)
    R = 9
    base = torch.randint(0, 2**20, (R + 3, 40), generator=g, device="cuda", dtype=torch.int32)
    specs = [(base[:R], 2, 389, 64, 0), (base[:R, :17], 1, 2**21 + 7, 64, 63),      # expand_pages (wrapping int32)
             (base[:R], 2, 2, 1, 0), (base[:R, 3:30], 4, 4, 1, 0)]                    # split_pages
    outs = [torch.empty((R, bt.shape[1] * p), dtype=torch.int32, device="cuda") for bt, p, *_ in specs]
    extension().page_tables([s[0] for s in specs], [s[1] for s in specs], [s[2] for s in specs],
                            [s[3] for s in specs], [s[4] for s in specs], outs)
    for (bt, pages, mult, step, offset), out in zip(specs, outs):
        ref = expand_pages(bt, mult, pages, offset) if step == 64 else split_pages(bt, pages)
        assert torch.equal(out, ref)
