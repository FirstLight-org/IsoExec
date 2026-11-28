"""One published-width BF16 expert, not a full router/model test.

H5120/I2304, actual FC1 -> weighted SwiGLU -> FC2 and native dgrad/wgrad
chain. CPU FP64 independent equation keeps the GPU footprint bounded.
"""
import pytest
import torch
import torch.nn.functional as F

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='dispatcher-admitted SM100 required')


def bits(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    def raw(t):
        flat=torch.empty(t.numel(),dtype=t.dtype,device=t.device)
        flat.copy_(t.reshape(-1))
        return flat.view(torch.uint8)
    assert torch.equal(raw(a),raw(b))


@pytest.mark.parametrize('rows',[1,9])
def test_published_expert_all_gradients_repeat_graph(rows):
    from isoexec.ops.moe.deepseek_v41_sm100 import ep
    from isoexec.ops.moe.deepseek_v41_sm100 import rows as training
    from isoexec.ops.moe.deepseek_v41_sm100 import rows as ops
    from isoexec.ops.moe import ep_fixed_order as epf
    torch.set_num_threads(1)
    H,I=5120,2304
    g=torch.Generator(device='cuda').manual_seed(452)
    x=torch.randn(rows,H,device='cuda',generator=g).bfloat16()
    gu=(torch.randn(1,2*I,H,device='cuda',generator=g)*.02).bfloat16()
    dn=(torch.randn(1,H,I,device='cuda',generator=g)*.02).bfloat16()
    prob=torch.linspace(.5,1.,rows,device='cuda')
    lids=torch.zeros(rows,dtype=torch.int32,device='cuda')
    dy=torch.randn(rows,H,device='cuda',generator=g).bfloat16()

    def chain():
        y,state=ep._owner_forward(x,lids,prob,gu,dn,10.)
        plan,xrows,gp,proj=state
        off=plan.grouped.offsets
        dyg=epf.to_grouped(dy,plan)
        dg,hb,dp=ops.act_backward_weighted(training.dgrad(dyg,dn,off),proj,gp,10.)
        dgu=training.wgrad_into(dg,xrows,off,1,gu.dtype)
        ddn=training.wgrad_into(dyg,hb,off,1,dn.dtype)
        dx=epf.to_receive_order(training.dgrad(dg,gu,off),plan)
        return y,dx,dgu,ddn,epf.to_receive_order(dp,plan)

    actual=chain()
    again=chain()
    for a,b in zip(actual,again):bits(a,b)
    del again
    # All master elements and route-weight derivative, independent CPU FP64.
    xx,gg,dd,pp=[v.cpu().double().requires_grad_() for v in (x,gu,dn,prob)]
    gate,up=F.linear(xx,gg[0]).chunk(2,-1)
    hidden=F.silu(gate.clamp_max(10.))*up.clamp(-10.,10.)*pp[:,None]
    yy=F.linear(hidden,dd[0])
    expected=(yy,*torch.autograd.grad(yy,(xx,gg,dd,pp),dy.cpu().double()))
    metrics={}
    for name,a,b in zip(('output','dx','gate_up','down','route_prob'),actual,expected):
        a=a.cpu().double()
        assert torch.isfinite(a).all() and torch.count_nonzero(b)>0
        err=float((a-b).norm()/b.norm())
        metrics[name]=err
        assert err<.03,(name,err)
    print('PUBLISHED_EXPERT_FP64_REL_L2',rows,metrics,flush=True)
    # The BF16 placeholder backward (rows.dgrad / wgrad) reads the expert offsets on the host, so it is not
    # graph-capturable; the repeat check above is its determinism gate.
