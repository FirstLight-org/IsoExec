"""Boundary and real GPU gates for the extracted MLA primitive."""

import pytest
import torch

from isoexec.ops.attention.mla_absorbed import (
    absorbed_kernel, absorbed_mla, absorb_query, pack_latent,
    split_kv_up_weight, unabsorb_output,
)


def test_head_interleaving_and_live_weight_sync():
    weight = torch.arange(2 * (3 + 4) * 5).reshape(14, 5).float()
    uk, uv = split_kv_up_weight(weight, num_heads=2, qk_nope_dim=3, v_head_dim=4, kv_lora_rank=5)
    assert torch.equal(uk[1], weight[7:10])
    assert torch.equal(uv[1], weight[10:14])
    weight.add_(100)
    assert torch.equal(uk[0], weight[:3])
    assert uk.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()


@pytest.mark.parametrize("lens", [(17, 65, 3), (1,), (64, 64)])
def test_page_boundaries_and_gradient(lens):
    c = torch.randn(sum(lens), 8, requires_grad=True)
    k = torch.randn(sum(lens), 4, requires_grad=True)
    cache, table = pack_latent(c, k, lens)
    offset = 0
    used = set()
    for i, n in enumerate(lens):
        blocks = table[i, :(n + 63) // 64].tolist()
        assert not used.intersection(blocks)
        used.update(blocks)
        rows = cache[blocks].reshape(-1, 12)
        assert torch.equal(rows[:n], torch.cat((c, k), dim=-1)[offset:offset + n])
        assert not rows[n:].count_nonzero()
        offset += n
    cache.sum().backward()
    assert torch.equal(c.grad, torch.ones_like(c))
    assert torch.equal(k.grad, torch.ones_like(k))


@pytest.mark.parametrize("lens", [(), (0, 4), (-1, 5), (3,), (2.0, 2), torch.tensor([4])])
def test_invalid_lengths_refuse(lens):
    with pytest.raises((TypeError, ValueError)):
        pack_latent(torch.zeros(4, 8), torch.zeros(4, 4), lens)


@pytest.mark.parametrize("page", [0, -16, 15, 64.0])
def test_invalid_page_refuses(page):
    with pytest.raises(ValueError):
        pack_latent(torch.zeros(4, 8), torch.zeros(4, 4), (4,), page)


def test_shared_fa3_call_pins_and_cache_views():
    observed = {}
    def flash(**kwargs):
        observed.update(kwargs)
        return kwargs["q_v"]
    c = torch.randn(4, 8)
    k = torch.randn(4, 4)
    cache, table = pack_latent(c, k, (4,))
    q = torch.randn(4, 2, 8)
    result = absorbed_kernel(q, torch.randn(4, 2, 4), cache, table,
                             torch.tensor([0, 4], dtype=torch.int32), torch.tensor([4], dtype=torch.int32),
                             4, 4, scale=.25, kv_lora_rank=8, flash_attn=flash)
    assert result is q
    assert observed["num_splits"] == 1 and "fa_version" not in observed
    assert observed["causal"] and observed["window_size"] == [-1, 0]
    assert observed["k"].untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert observed["v"].untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert torch.equal(observed["v"][:1, :4, 0], c.unsqueeze(0))


def _bits(tensor):
    return tensor.contiguous().view(torch.int16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("heads", [5, 20])
def test_glm_projection_batch_invariance_and_gradients(heads):
    torch.manual_seed(17)
    device, dtype = "cuda", torch.bfloat16
    q = torch.randn(259, heads, 192, device=device, dtype=dtype, requires_grad=True)
    uk = torch.randn(heads, 192, 512, device=device, dtype=dtype, requires_grad=True)
    uv = torch.randn(heads, 256, 512, device=device, dtype=dtype, requires_grad=True)
    full = absorb_query(q, uk)
    for start, end in [(0, 1), (2, 67), (64, 129), (258, 259)]:
        assert torch.equal(_bits(absorb_query(q[start:end], uk)), _bits(full[start:end]))
    output = unabsorb_output(full, uv)
    assert torch.equal(_bits(unabsorb_output(full[128:], uv)), _bits(output[128:]))
    loss = output.float().square().mean()
    loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.count_nonzero() for t in (q, uk, uv))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hybrid_backward_wiring_against_unabsorbed_fp64(monkeypatch):
    # Small algebra gate uses explicit reference providers, not an FA3 certificate.
    from isoexec.ops.attention.mla_absorbed import _unabsorbed
    torch.manual_seed(73)
    shapes = [(9, 2, 16), (9, 2, 16), (9, 32), (9, 16), (2, 16, 32), (2, 16, 32)]
    tensors = [torch.randn(s, device="cuda", dtype=torch.bfloat16).mul_(.15).requires_grad_() for s in shapes]
    # Installed batch-invariance scopes can replace torch.bmm with a forward-only
    # Python provider. Hybrid backward must use aten, preserving its autograd node.
    def forward_only_bmm(a, b):
        with torch.no_grad():
            return torch.ops.aten.bmm(a, b)
    monkeypatch.setattr(torch, "bmm", forward_only_bmm)
    def varlen(q, k, v, cu, cu_k, max_q, max_k, scale):
        result = []
        for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
            score = torch.einsum("thd,shd->hts", q[start:end], k[start:end]) * scale
            mask = torch.ones(end - start, end - start, device=q.device, dtype=torch.bool).triu(1)
            prob = score.masked_fill(mask, -float("inf")).softmax(-1)
            result.append(torch.einsum("hts,shd->thd", prob, v[start:end]))
        return torch.cat(result)
    def flash(**kw):
        result, offset = [], 0
        for i, n in enumerate(kw["seqused_k"].tolist()):
            blocks = kw["block_table"][i, :(n + 63) // 64].long()
            c = kw["v"][blocks].reshape(-1, 1, 32)[:n].expand(-1, 2, -1)
            k = kw["k"][blocks].reshape(-1, 1, 16)[:n].expand(-1, 2, -1)
            q = torch.cat((kw["q_v"][offset:offset+n], kw["q"][offset:offset+n]), -1)
            cu = torch.tensor([0, n], device=q.device, dtype=torch.int32)
            result.append(varlen(q, torch.cat((c, k), -1), c, cu, cu, n, n, kw["softmax_scale"]))
            offset += n
        return torch.cat(result)
    got = absorbed_mla(*tensors, seq_lens=(4, 5), scale=.25, flash_attn=flash, varlen_attn=varlen)
    grad = torch.randn_like(got)
    got.backward(grad)
    leaves64 = [t.detach().double().requires_grad_() for t in tensors]
    ref = _unabsorbed(*leaves64, torch.tensor([0, 4, 9], device="cuda", dtype=torch.int32), 5, .25, varlen)
    ref.backward(grad.double())
    assert torch.allclose(got.double(), ref, atol=.003, rtol=.03)
    for t, r in zip(tensors, leaves64):
        rel = (t.grad.double() - r.grad).norm() / r.grad.norm()
        assert rel < .025, rel.item()


def _gqa_sm90():
    from isoexec.ops.attention import gqa_sm90

    if not gqa_sm90.available():
        pytest.skip("requires an SM90 GPU and the gqa_sm90 FA3 build")
    return gqa_sm90.flash_attn_paged


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("heads", [5, 20])
def test_real_fa3_prefill_decode_chunk_batch_and_poison(heads):
    flash_attn_varlen_func = _gqa_sm90()
    torch.manual_seed(111)
    length, device, dtype = 259, "cuda", torch.bfloat16
    c = torch.randn(length, 512, device=device, dtype=dtype)
    k = torch.randn(length, 64, device=device, dtype=dtype)
    q = torch.randn(length, heads, 512, device=device, dtype=dtype)
    qp = torch.randn(length, heads, 64, device=device, dtype=dtype)
    cache, table = pack_latent(c, k, (length,))
    def call(q, qp, cache, table, qlens, klens, flash=None):
        lengths = torch.tensor(qlens, device=device, dtype=torch.int32)
        cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
        return absorbed_kernel(q, qp, cache, table, cu,
            torch.tensor(klens, device=device, dtype=torch.int32), max(qlens), max(klens),
            scale=1/16, kv_lora_rank=512, flash_attn=flash or flash_attn_varlen_func)
    full = call(q, qp, cache, table, (length,), (length,))
    for start, end in [(0, 1), (127, 128), (128, 194), (258, 259)]:
        part = call(q[start:end], qp[start:end], cache, table, (end-start,), (end,))
        assert torch.equal(_bits(part), _bits(full[start:end]))
    # Prepend an unrelated sequence and use non-contiguous page numbers.
    neighbor = 17
    c2 = torch.cat((torch.randn(neighbor, 512, device=device, dtype=dtype), c))
    k2 = torch.cat((torch.randn(neighbor, 64, device=device, dtype=dtype), k))
    cache2, table2 = pack_latent(c2, k2, (neighbor, length))
    q2 = torch.cat((torch.randn(neighbor, heads, 512, device=device, dtype=dtype), q))
    qp2 = torch.cat((torch.randn(neighbor, heads, 64, device=device, dtype=dtype), qp))
    batch = call(q2, qp2, cache2, table2, (neighbor, length), (neighbor, length))
    assert torch.equal(_bits(batch[neighbor:]), _bits(full))
    # FA3 masks scores, but may still multiply zero probabilities by padded V.
    # Padding therefore must be finite; pack_latent initializes it to zero.
    cache2[0, neighbor:] = 1e3
    cache2[-1, length % 64:] = 1e3
    poisoned = call(q2, qp2, cache2, table2, (neighbor, length), (neighbor, length))
    assert torch.equal(_bits(poisoned), _bits(batch))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize('heads', [5, 20])
def test_fa3_decode_graph_replay_with_null_lane(heads):
    """Require bit-identical graph replay with a null lane through gqa_sm90's FA3 paged entry."""
    flash_attn_paged = _gqa_sm90()
    torch.manual_seed(61)
    q = torch.randn(2, heads, 512, device='cuda', dtype=torch.bfloat16)
    qp = torch.randn(2, heads, 64, device='cuda', dtype=torch.bfloat16)
    cache, _ = pack_latent(torch.randn(17, 512, device='cuda', dtype=torch.bfloat16),
                           torch.randn(17, 64, device='cuda', dtype=torch.bfloat16), (17,))
    table = torch.zeros(2, 1, device='cuda', dtype=torch.int32)
    cu = torch.tensor([0, 1, 2], device='cuda', dtype=torch.int32)
    used = torch.tensor([17, 0], device='cuda', dtype=torch.int32)
    def call():
        return absorbed_kernel(q, qp, cache, table, cu, used, 1, 17, scale=1/16,
                                kv_lora_rank=512, flash_attn=flash_attn_paged)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert torch.isfinite(expected).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    graph.replay()
    assert torch.equal(_bits(captured), _bits(expected))
    q.mul_(.5)
    cache.mul_(.8)
    expected2 = call()
    graph.replay()
    assert torch.equal(_bits(captured), _bits(expected2))
