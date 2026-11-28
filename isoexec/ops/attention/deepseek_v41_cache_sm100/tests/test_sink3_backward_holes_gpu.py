"""Recipe-3 internal pads must not hide compressed keys from the dKV reducer."""
import pytest
import torch

from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires leased CUDA GPU')


def test_sink3_internal_window_padding_all_gradients_and_exact_repeat():
    torch.manual_seed(311)
    q=torch.randn(5,64,512,device='cuda',dtype=torch.bfloat16).requires_grad_()
    kv=torch.randn(8,512,device='cuda',dtype=torch.bfloat16).requires_grad_()
    sink=torch.randn(64,device='cuda',requires_grad=True)
    lists=torch.full((5,64),-1,device='cuda',dtype=torch.int32)
    for row in range(3):
        lists[row,:row+1]=torch.arange(row+1,device='cuda',dtype=torch.int32)
        lists[row,32:33+row]=torch.arange(4,5+row,device='cuda',dtype=torch.int32)
    # The low-level sparse-list contract counts duplicate slots independently.
    # Also exercise the last slot and an entirely padded row.
    lists[3,0]=lists[3,1]=0
    lists[3,32]=lists[3,63]=4
    original=lists.clone()
    compact=ops.extension().compact_lists(lists)
    for row in range(5):
        valid=lists[row][lists[row]>=0]
        assert torch.equal(compact[row,:len(valid)],valid)
        assert torch.all(compact[row,len(valid):]==-1)
    assert torch.equal(lists,original)
    forward_only=ops.sink_forward(q.detach(),kv.detach(),lists,sink.detach(),512**-.5,3)[0]
    dy=torch.randn_like(q)
    first=None
    for _ in range(2):
        output=ops.sink_attention(q,kv,lists,sink,512**-.5,recipe=3)
        assert torch.equal(output,forward_only)
        grads=torch.autograd.grad(output,(q,kv,sink),dy)
        if first is not None:
            assert torch.equal(output,first[0])
            assert all(torch.equal(a,b) for a,b in zip(grads,first[1]))
        first=(output.detach().clone(),[g.clone() for g in grads])
    q64,kv64,s64=[value.detach().double().requires_grad_() for value in (q,kv,sink)]
    selected=kv64[lists.clamp_min(0).long()]
    scores=torch.einsum('thd,tkd->thk',q64,selected)*(512**-.5)
    scores=scores.masked_fill(lists[:,None,:]<0,-torch.inf)
    scores=torch.cat((scores,s64[None,:,None].expand(5,-1,1)),dim=-1)
    probability=scores.softmax(-1)[...,:-1]
    reference=torch.einsum('thk,tkd->thd',probability,selected)
    wants=torch.autograd.grad(reference,(q64,kv64,s64),dy.double())
    for name,actual,want in zip(('q','kv','sink'),grads,wants):
        error=(actual.double()-want).norm()/want.norm()
        print(f'sink3 internalpads {name} relativeL2={float(error):.8g}',flush=True)
        assert error<.015
    assert torch.count_nonzero(grads[1][[3,7]])==0
