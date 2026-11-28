"""The CED decoder at op level (``ced.py``), through the entry points the model calls (``entry_projection``,
``decoder_attention_gen``): the boundary layer split into the entry projection (every token) and the query side
(decoder rows), the window truncated at the segment start, query rows that are a subset of a sequence.

* R1: entry projection + query side on all rows == today's boundary layer (trainer ``attention_full``, engine
  ``attention_paged``), outputs, published state and pool bytes, bit for bit; the trainer backward is deterministic
  and its parameter gradients are today's.
* R2: ``floor`` = 0 reproduces today's lists and outputs (trainer and engine); the floored lists are the CPU
  reference's ``_window(..., floor, W)`` keys in order, also when the engine supplies its own window lists.
* R3: any subset of query rows == the same rows of the full-rows computation (trainer and engine), and the engine's
  bounded-replay program (entry projection on every chunk token, decoder on the rows at positions >= s, chunked
  prefill + decode, with this package's or the engine's window lists) == the trainer's segment program, bit for bit.

Layers of ``_harness.native_cfg``: 4 = the boundary (ratio-1 KV + index + candidate source), 5 = Reuse,
6 = Reindex (candidate consumer)."""
from types import SimpleNamespace

import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ced, ops, paged
from isoexec.ops.attention.deepseek_v41_cache_sm100._extension import extension
from isoexec.ops.attention.deepseek_v41_cache_sm100.tests import _harness as h

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
MAXLEN = 512
W = 128
DEC = (4, 5, 6)
KW = dict(linear_fn=h.native_linear, dequant_fn=h.dequant)


def _eq(a, b):
    if a.dtype in (torch.bfloat16, torch.float16):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


def _setup(B, T, seed=7, grad=False):
    # top-k 192 from a pool of 32 blocks of 8: the candidate mask acts from 257 visible entries on
    cfg = h.native_cfg(index_topk=192, candidate_blocks=32)
    layers = {i: h.make_layer(cfg, i, i) for i in DEC}
    if not grad:
        for m in layers.values():
            m.requires_grad_(False)
    g = torch.Generator(device="cuda").manual_seed(seed)
    xs = {i: torch.randn((B, T, cfg.hidden_size), generator=g, device="cuda").to(torch.bfloat16) for i in DEC}
    return cfg, layers, xs


# ---------------------------------------------------------------------------------------------- trainer helpers
def _decoder_rows(seq, pos, cu, floor=None, window_index=True):
    """``program.DecoderRows`` of the rows (sequence ``seq``, position ``pos``; host tensors) of packed sequences
    ``cu``; ``window_index`` as ``DecoderSegments.window_rows`` (segment rows consecutive)."""
    seq, pos = torch.as_tensor(seq).long(), torch.as_tensor(pos).long()
    cu_t = torch.as_tensor(cu).long()
    wi = None
    if window_index:
        lo = torch.zeros_like(pos) if floor is None else torch.as_tensor(floor).long().clamp(min=0)
        j = torch.arange(W)
        r = torch.arange(pos.numel())
        wi = torch.where(pos[:, None] - W + 1 + j >= lo[:, None], r[:, None] - (W - 1 - j), -1).cuda()
    return SimpleNamespace(index=(cu_t[seq] + pos).cuda(), positions=pos.cuda(), sequence=seq.cuda(),
                           window_floor=None if floor is None else torch.as_tensor(floor).long().cuda(),
                           window_index=wi, cu_seqlens=cu_t.cuda())


def _entry(cfg, layers, xs, lengths, grad_x=None):
    """Packed entry projection of sequences ``s`` of lengths ``lengths`` (inputs ``xs[4][s, :L]``)."""
    hp = (torch.cat([xs[4][s, :L] for s, L in enumerate(lengths)]) if grad_x is None else grad_x)
    pos = torch.cat([torch.arange(L) for L in lengths]).cuda()
    main, index_k, window = ced.entry_projection(layers[4], hp, cfg, 4, pos, linear_fn=h.native_linear)
    cu = [0]
    for L in lengths:
        cu.append(cu[-1] + L)
    return SimpleNamespace(main_kv=main, index_k=index_k, window_kv=window), cu


def _decode(cfg, layers, xs, dkv, rows, layers_run=DEC, step=None, xin=None):
    shared, out = {}, {}
    for i in layers_run:
        x = xs[i][rows.sequence, rows.positions].contiguous() if xin is None else xin[i]
        out[i] = ops.drive(ced.decoder_attention_gen(layers[i], x, cfg, i, rows, dkv, shared, step=step, **KW))
    return out, shared


# ---------------------------------------------------------------------------------------------- R2: lists
def _window_statement(pos, floor, wend):
    """The CPU reference's ``_window(rows_of_pos, pos, floor, W)`` valid keys in slot order, as rows ``wend - (p - q)``."""
    out = []
    for p, f, e in zip(pos.tolist(), floor.tolist(), wend.tolist()):
        out.append([e - (p - q) for q in range(p - W + 1, p + 1) if q >= max(f, 0)])
    return out


def test_build_lists_floor():
    K, R = W + 448, 700
    g = torch.Generator(device="cuda").manual_seed(3)
    pos = torch.randint(0, 2000, (R,), generator=g, device="cuda", dtype=torch.int32)
    topk = torch.randint(-1, 600, (R, 448), generator=g, device="cuda", dtype=torch.int32)
    topk = torch.sort(torch.where(topk >= 0, topk, 1 << 30), 1).values
    topk = torch.where(topk >= (1 << 30), -1, topk).to(torch.int32).contiguous()
    for mode in (0, 2):
        a = torch.empty((R, K), dtype=torch.int32, device="cuda")
        b, c = torch.empty_like(a), torch.empty_like(a)
        extension().build_lists(topk, pos, R, 2048, 4096, 0, 2048, W, mode, a)
        extension().build_lists(topk, pos, R, 2048, 4096, 0, 2048, W, mode, b, torch.zeros_like(pos))
        assert torch.equal(a, b), mode                                  # floor 0 == no floor
        extension().build_lists(topk, pos, R, 2048, 4096, 0, 2048, W, mode, c, None, pos.clone(),
                                torch.full_like(pos, 2048))
        assert torch.equal(a, c), mode                                  # explicit row bases == implicit ones
    floor = (pos - torch.randint(0, 300, (R,), generator=g, device="cuda", dtype=torch.int32)).clamp(min=-5)
    wend = torch.randint(5000, 9000, (R,), generator=g, device="cuda", dtype=torch.int32)
    cbase = torch.randint(10000, 20000, (R,), generator=g, device="cuda", dtype=torch.int32)
    lists = torch.empty((R, K), dtype=torch.int32, device="cuda")
    extension().build_lists(topk, pos, R, 1, 0, 0, 0, W, 2, lists, floor, wend, cbase)
    want = _window_statement(pos.cpu(), floor.cpu(), wend.cpu())
    L, tk, cb = lists.cpu(), topk.cpu(), cbase.cpu()
    for r in range(R):
        n = len(want[r])
        assert L[r, :n].tolist() == want[r] and bool((L[r, n:W] == -1).all()), r
        ids = tk[r][tk[r] >= 0]
        assert L[r, W:W + ids.numel()].tolist() == (ids + cb[r]).tolist() and bool((L[r, W + ids.numel():] == -1).all())


def test_paged_lists_floor_and_engine_lists_floor():
    R = 300
    g = torch.Generator(device="cuda").manual_seed(4)
    seq = torch.randint(0, 3, (R,), generator=g, device="cuda", dtype=torch.int32)
    pos = torch.randint(0, 1000, (R,), generator=g, device="cuda", dtype=torch.int32)
    bt = torch.randperm(3 * 16, generator=torch.Generator().manual_seed(1)).view(3, 16).int().cuda()
    topk = torch.sort(torch.randint(0, 600, (R, 448), generator=g, device="cuda", dtype=torch.int32), 1).values
    valid = torch.rand((R,), generator=g, device="cuda") > 0.1
    m0, e0 = torch.empty((R, W), dtype=torch.int32, device="cuda"), torch.empty((R, 448), dtype=torch.int32, device="cuda")
    m1, e1 = torch.empty_like(m0), torch.empty_like(e0)
    extension().paged_lists_masked(seq, pos, bt, W, topk, bt, m0, e0, valid, 64, 64)
    extension().paged_lists_masked(seq, pos, bt, W, topk, bt, m1, e1, valid, 64, 64, torch.zeros_like(pos))
    assert torch.equal(m0, m1) and torch.equal(e0, e1)
    floor = (pos - torch.randint(0, 300, (R,), generator=g, device="cuda", dtype=torch.int32)).clamp(min=0)
    extension().paged_lists_masked(seq, pos, bt, W, topk, bt, m1, e1, valid, 64, 64, floor)
    assert torch.equal(e0, e1)
    assert torch.equal(paged.floor_window_lists(m0, pos, floor), m1)    # the engine's own lists, floored
    assert torch.equal(paged.floor_window_lists(m0, pos, torch.zeros_like(pos)), m0)
    btc, M = bt.cpu(), m1.cpu()
    for r in range(R):
        if not bool(valid[r]):
            assert bool((M[r] == -1).all())
            continue
        p, f, s = int(pos[r]), int(floor[r]), int(seq[r])
        rows_ = [int(btc[s, q // 64]) * 64 + q % 64 for q in range(p - W + 1, p + 1) if q >= f]
        assert M[r, :len(rows_)].tolist() == rows_ and bool((M[r, len(rows_):] == -1).all()), r


# ---------------------------------------------------------------------------------------------- R1 / R2 / R3 trainer
def _full(cfg, layers, xs):
    state, out = {}, {}
    for i in DEC:
        out[i] = ops.attention_full(layers[i], xs[i], cfg, i, state, recipe=3, **KW)
        if i == 4:
            topk4 = state["topk_native"]
    return out, state, topk4


def test_trainer_split_equals_boundary_layer_and_floor_zero():
    B, T = 2, 300
    cfg, layers, xs = _setup(B, T)
    with torch.no_grad():
        ref, state, topk4 = _full(cfg, layers, xs)
        dkv, cu = _entry(cfg, layers, xs, [T] * B)
        assert _eq(dkv.main_kv, state["compress_kv_native"])
        kc, ksf = state["index_k_native"]
        for b in range(B):
            assert _eq(dkv.index_k[0][b * T:(b + 1) * T], kc[b, :T]) and _eq(dkv.index_k[1][b * T:(b + 1) * T], ksf[b, :T])
        seq, pos = torch.arange(B).repeat_interleave(T), torch.arange(T).repeat(B)
        for floor in (None, torch.zeros(B * T, dtype=torch.int64)):
            out, shared = _decode(cfg, layers, xs, dkv, _decoder_rows(seq, pos, cu, floor))
            for i in DEC:
                assert _eq(out[i], ref[i].reshape(B * T, -1)), (i, floor is None)
            assert _eq(shared["topk_rows"], state["topk_native"])
            for b in range(B):
                assert _eq(shared["candidates_rows"][b * T:(b + 1) * T], state["candidates_native"][b])
        _, shared = _decode(cfg, layers, xs, dkv, _decoder_rows(seq, pos, cu), layers_run=(4,))
        assert _eq(shared["topk_rows"], topk4)


def test_trainer_row_subsets_equal_full_rows():
    B, T = 2, 400
    cfg, layers, xs = _setup(B, T, seed=11)
    starts = [T - 150, 37]
    lengths = [T, T - 23]                                   # unequal packed sequences
    with torch.no_grad():
        dkv, cu = _entry(cfg, layers, xs, lengths)
        seq_all = torch.cat([torch.full((L,), s) for s, L in enumerate(lengths)])
        pos_all = torch.cat([torch.arange(L) for L in lengths])
        s_all = torch.tensor(starts)[seq_all]
        full_floor = torch.where(pos_all >= s_all, s_all, 0)
        full, _ = _decode(cfg, layers, xs, dkv, _decoder_rows(seq_all, pos_all, cu, full_floor))
        # the boundary's query side on an arbitrary subset (unsorted, interleaved sequences)
        pick = torch.randperm(seq_all.numel(), generator=torch.Generator().manual_seed(2))[:173]
        sub4, _ = _decode(cfg, layers, xs, dkv, _decoder_rows(seq_all[pick], pos_all[pick], cu, window_index=False),
                          layers_run=(4,))
        assert _eq(sub4[4], full[4][pick.cuda()])
        # the decoder segments [s_b, L_b): the decoder rows of bounded replay
        keep = pos_all >= s_all
        sub, _ = _decode(cfg, layers, xs, dkv, _decoder_rows(seq_all[keep], pos_all[keep], cu, s_all[keep]))
        idx = torch.nonzero(keep).reshape(-1).cuda()
        for i in DEC:
            assert _eq(sub[i], full[i][idx]), i
        # the floor acts: layer 5 rows within W of s differ from the unfloored program, the others do not
        nofl, _ = _decode(cfg, layers, xs, dkv, _decoder_rows(seq_all, pos_all, cu))
        near = (keep & (pos_all < s_all + W - 1)).cuda()
        far = (keep & (pos_all >= s_all + W - 1)).cuda()
        assert not _eq(full[5][near], nofl[5][near]) and _eq(full[5][far], nofl[5][far])
    with pytest.raises(ValueError):
        _decode(cfg, layers, xs, dkv, _decoder_rows(seq_all[pick], pos_all[pick], cu, window_index=False),
                layers_run=(4, 5))
    with pytest.raises(ValueError):                          # a window index that is not the segment's rows
        rows = _decoder_rows(seq_all[keep], pos_all[keep], cu, s_all[keep])
        rows.window_index = rows.window_index.roll(1, 0)
        _decode(cfg, layers, xs, dkv, rows, layers_run=(4, 5))


def test_trainer_backward_deterministic_and_equal_to_boundary_layer():
    B, T = 2, 256
    cfg, layers, xs = _setup(B, T, seed=13, grad=True)
    g = torch.Generator(device="cuda").manual_seed(5)
    dy = {i: torch.randn((B * T, cfg.hidden_size), generator=g, device="cuda").to(torch.bfloat16) for i in (4, 5)}
    seq, pos = torch.arange(B).repeat_interleave(T), torch.arange(T).repeat(B)

    def run(split):
        xg = {i: xs[i].clone().requires_grad_(True) for i in (4, 5)}
        for m in layers.values():
            m.zero_grad(set_to_none=True)
        if split:
            dkv, cu = _entry(cfg, layers, xs, [T] * B, grad_x=xg[4].reshape(B * T, -1))
            out, _ = _decode(cfg, layers, xs, dkv, _decoder_rows(seq, pos, cu), layers_run=(4, 5),
                             xin={i: xg[i].reshape(B * T, -1) for i in (4, 5)})
        else:
            state = {}
            out = {i: ops.attention_full(layers[i], xg[i], cfg, i, state, recipe=3, **KW).reshape(B * T, -1)
                   for i in (4, 5)}
        sum((out[i].float() * dy[i].float()).sum() for i in (4, 5)).backward()
        grads = [xg[4].grad, xg[5].grad] + [p.grad for i in (4, 5) for p in layers[i].parameters()]
        return [None if t is None else t.clone() for t in grads], {i: o.detach() for i, o in out.items()}

    a, oa = run(True)
    b, _ = run(True)
    c, oc = run(False)
    for i in (4, 5):
        assert _eq(oa[i], oc[i])
    assert all((x is None) == (y is None) for x, y in zip(a, b))
    assert all(x is None or _eq(x, y) for x, y in zip(a, b))           # run to run
    names = ["x4", "x5"] + [f"{i}.{n}" for i in (4, 5) for n, _ in layers[i].named_parameters()]
    diff = [n for n, x, y in zip(names, a, c) if (x is None) != (y is None) or (x is not None and not _eq(x, y))]
    # parameter gradients == today's boundary + reuse layers; an input gradient sums several BF16 contributions
    # (wq_a, wkv, compressor) in autograd's accumulation order, which the split graph may order differently
    print("gradients differing from the unsplit layers:", diff)
    assert not [n for n in diff if not n.startswith("x")], diff


# ---------------------------------------------------------------------------------------------- engine
def _pools(cfg, nseq):
    P = MAXLEN // 64
    return paged.PagedPools(cfg, window_pages=nseq * P, comp_pages={1: nseq * P}, state_slots=nseq, max_len=MAXLEN,
                            layers=list(DEC))


def _tables(nseq, seed):
    g = torch.Generator().manual_seed(seed)
    P = MAXLEN // 64
    win = torch.randperm(nseq * P, generator=g).view(nseq, P).int().cuda()
    comp = {1: torch.randperm(nseq * P, generator=g).view(nseq, P).int().cuda()}
    return win, comp


def _batch(plan, win, comp, engine_lists=False):
    sel = torch.tensor([s for s, _, _ in plan], device="cuda").long()
    batch = paged.PagedBatch.make([a for _, a, _ in plan], [c for _, _, c in plan], win[sel],
                                  {r: b[sel] for r, b in comp.items()}, [s for s, _, _ in plan], ratios=())
    if engine_lists:
        # the engine's own window addressing (vLLM's sliding-window metadata): write rows and unfloored main lists
        N = batch.tok_seq.numel()
        batch.window_rows, batch.window_lists = {}, {}
        for i in DEC:
            batch.window_rows[i] = paged._rows(batch.win_bt, batch.tok_seq, batch.tok_pos)
            m = torch.empty((N, W), dtype=torch.int32, device="cuda")
            extension().paged_lists(batch.tok_seq, batch.tok_pos, batch.win_bt, W, None, None, m, None)
            batch.window_lists[i] = m
        batch.win_bt = None
    return batch


def _rows_in(xs, plan):
    return {i: torch.cat([xs[i][s, a:a + c] for s, a, c in plan], 0).contiguous() for i in DEC}


def _pool_bytes(pools):
    return [pools.window[i].clone() for i in DEC] + [pools.comp[4].clone(), pools.index[4].clone()]


STEPS = [[(0, 0, 150), (1, 0, 60)],
         [(0, 150, 150), (1, 60, 40), (2, 0, 64)],
         [(2, 64, 136), (0, 300, 1), (1, 100, 1)]] + [[(0, 301 + k, 1), (1, 101 + k, 1), (2, 200 + k, 1)]
                                                       for k in range(4)]
PROMPTS = {0: 300, 1: 100, 2: 200}


def _engine_rows(batch, sel, floor):
    return SimpleNamespace(index=sel, positions=batch.tok_pos.index_select(0, sel).long(),
                           sequence=batch.tok_seq.index_select(0, sel).long(),
                           window_floor=None if floor is None else floor.cuda(), window_index=None, cu_seqlens=None)


def test_engine_split_equals_boundary_layer_and_floor_zero():
    cfg, layers, xs = _setup(3, MAXLEN, seed=17)
    win, comp = _tables(3, 3)
    ref_pools, pools = _pools(cfg, 3), _pools(cfg, 3)
    with torch.no_grad():
        for plan in STEPS:
            xin = _rows_in(xs, plan)
            N = xin[4].shape[0]
            rb = _batch(plan, win, comp)
            ref = {i: paged.attention_paged(layers[i], xin[i], cfg, i, ref_pools, rb, **KW) for i in DEC}
            batch = _batch(plan, win, comp)
            step = SimpleNamespace(pools=pools, batch=batch)
            assert ced.entry_projection(layers[4], xin[4], cfg, 4, batch.tok_pos, linear_fn=h.native_linear,
                                        step=step) is None
            for floor in (None, torch.zeros(N, dtype=torch.int64)):
                rows = _engine_rows(batch, torch.arange(N, device="cuda"), floor)
                out, _ = _decode(cfg, layers, xs, None, rows, step=step, xin=xin)
                for i in DEC:
                    assert _eq(out[i], ref[i]), (plan, i, floor is None)
            for a, b in zip(_pool_bytes(pools), _pool_bytes(ref_pools)):
                assert torch.equal(a, b), plan


def _replay_engine(cfg, layers, xs, engine_lists):
    win, comp = _tables(3, 7)
    pools = _pools(cfg, 3)
    start = {s: max(0, P - W) for s, P in PROMPTS.items()}
    eng = {}
    for plan in STEPS:
        xin = _rows_in(xs, plan)
        N = xin[4].shape[0]
        batch = _batch(plan, win, comp, engine_lists)
        step = SimpleNamespace(pools=pools, batch=batch)
        ced.entry_projection(layers[4], xin[4], cfg, 4, batch.tok_pos, linear_fn=h.native_linear, step=step)
        seqs = torch.cat([torch.full((c,), s) for s, a, c in plan])
        poss = torch.cat([torch.arange(a, a + c) for s, a, c in plan])
        s_row = torch.tensor([start[int(s)] for s in seqs])
        keep = torch.nonzero(poss >= s_row).reshape(-1)
        all4, _ = _decode(cfg, layers, xs, None, _engine_rows(batch, torch.arange(N, device="cuda"), None),
                          layers_run=(4,), step=step, xin=xin)
        if keep.numel() == 0:
            continue
        sel = keep.cuda()
        rows = _engine_rows(batch, sel, s_row[keep])
        out, _ = _decode(cfg, layers, xs, None, rows, step=step, xin={i: xin[i][sel].contiguous() for i in DEC})
        assert _eq(out[4], all4[4][sel]), plan                                       # R3, engine
        for j, r in enumerate(keep.tolist()):
            eng[(int(seqs[r]), int(poss[r]))] = {i: out[i][j] for i in DEC}
    return eng, start


def test_engine_replay_rows_equal_trainer_segments():
    """Bounded replay: the engine runs the entry projection on every chunk token and the decoder on the rows at
    positions >= s = max(0, P - W) (none for a chunk below s), the decoder layers after b with the floor s; the
    boundary's query side on the decoder rows == on all the step's rows at those rows; engine (this package's
    window lists or the engine's own, floored) == trainer."""
    cfg, layers, xs = _setup(3, MAXLEN, seed=19)
    with torch.no_grad():
        eng, start = _replay_engine(cfg, layers, xs, False)
        eng2, _ = _replay_engine(cfg, layers, xs, True)
        ends = {s: max(a + c for t, a, c in sum(STEPS, []) if t == s) for s in PROMPTS}
        lengths = [ends[s] for s in sorted(PROMPTS)]
        dkv, cu = _entry(cfg, layers, xs, lengths)
        seq = torch.cat([torch.full((ends[s] - start[s],), s) for s in sorted(PROMPTS)])
        pos = torch.cat([torch.arange(start[s], ends[s]) for s in sorted(PROMPTS)])
        flo = torch.tensor([start[int(s)] for s in seq])
        tr, _ = _decode(cfg, layers, xs, dkv, _decoder_rows(seq, pos, cu, flo))
    assert len(eng) == len(eng2) == seq.numel()
    for r in range(seq.numel()):
        key = (int(seq[r]), int(pos[r]))
        for i in DEC:
            assert _eq(eng[key][i], tr[i][r]) and _eq(eng2[key][i], tr[i][r]), (key, i)
