from __future__ import annotations

import torch

RMS_EPS, EPS, ITERS, H = 1e-20, 1e-6, 20, 5120
KW = dict(rms_eps=RMS_EPS, sinkhorn_eps=EPS, iterations=ITERS)


def make(T, seed=0, device="cuda", x_scale=2.0, w_dtype=torch.float32):
    g = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s, sc=1.0: torch.randn(*s, device=device, generator=g) * sc
    x = r(T, 4, H, sc=x_scale).bfloat16()
    w = r(24, 4 * H, sc=0.03).to(w_dtype)
    base = r(24, sc=0.5)
    scale = torch.tensor([0.7, 0.4, 0.9], device=device)
    return x, w, scale, base


def rnd(*shape, seed=0, dtype=torch.float32, sc=1.0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(*shape, device=device, generator=g) * sc).to(dtype)


def probs(T, seed):
    post = torch.sigmoid(rnd(T, 4, seed=seed)) * 2
    comb = torch.softmax(rnd(T, 16, seed=seed + 1), -1).view(T, 4, 4)
    return post, comb
