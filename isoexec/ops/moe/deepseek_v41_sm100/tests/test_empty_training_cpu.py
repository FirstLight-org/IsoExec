"""An empty actor microbatch has a defined VJP without launching CUDA kernels."""
import torch

from isoexec.ops.moe.deepseek_v41_sm100.training import moe_train


def test_empty_actor_vjp_preserves_every_trainable_input():
    x=torch.empty((0,32),dtype=torch.bfloat16,requires_grad=True)
    router=torch.ones((4,32),dtype=torch.bfloat16,requires_grad=True)
    gu=torch.ones((5,64,32),dtype=torch.bfloat16,requires_grad=True)
    dn=torch.ones((5,32,32),dtype=torch.bfloat16,requires_grad=True)
    bias=torch.zeros(4)
    for _ in range(2):
        y=moe_train(x,router,bias,gu,dn,top_k=2,routed_scale=1.,limit=10.)
        assert y.shape==(0,32) and y.dtype==torch.bfloat16
        gradients=torch.autograd.grad(y.sum(),(x,router,gu,dn))
        assert all(torch.equal(g,torch.zeros_like(p)) for g,p in zip(gradients,(x,router,gu,dn)))
