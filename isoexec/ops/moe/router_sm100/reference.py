"""Torch programs for the two routing names, operation for operation the kernel's (DESIGN.md
"Numerical program"), plus a float64 twin and the semantic references they are gated against.

``route_reference`` IS the declared program: run on CUDA tensors it is bit-equal to
``ops.route_select_only`` (Gate 2) -- every rounding site is one torch fp32 op in the kernel's
order (``1 / (1 + exp(-x))``, ``x > 20 ? x : log1p(exp(x))`` then ``sqrt``, one add for the key,
one add for the group score, the selected-set chain, ``+ 1e-20``, one divide, one multiply). The
selection is an exact total order and is computed here WITHOUT any tie-dependent library call:
every candidate gets one int64 sort key ``(allowed, key, -id)`` whose integer order is the declared
order, so ``torch.sort``'s stability and its treatment of ``-0.0`` cannot enter.

On CPU the scores come from torch's CPU ``exp``/``log1p``, which need not be CUDA's ``expf``/
``log1pf`` to the last bit; the CPU lane therefore gates SELECTION against the semantic references
(on the same CPU scores) and leaves weight bit-equality to the GPU lane.

Semantic references:

* ``hf_deepseek_v3`` -- ``transformers`` ``DeepseekV3MoE.route_tokens_to_experts`` transcribed
  literally (``torch.topk``, ``sorted=False``, ``-inf`` group masking, ``+1e-20``, scale).
* ``semantic_ordered`` -- the same formula with each ``torch.topk`` replaced by the declared tie rule
  (lower id / lower group id wins), which is what "selection identical to the semantic reference
  including planted ties" is measured against: HF/vLLM leave ties to ``torch.topk``, i.e. unstated.
* ``deepgemm_mega_gate`` -- DeepGEMM ``tests/test_mega_gate.py``'s oracle for routing B
  (``F.softplus(logits).sqrt()``, ranking ``scores + bias``, weights ``s / (sum + 1e-20) * scale``).
"""

from __future__ import annotations

import torch

GROUPED = "sigmoid_noaux_bias_select_grouped"
SQRTSOFTPLUS = "sqrtsoftplus_bias_select"


def score(logits: torch.Tensor, routing: str) -> torch.Tensor:
    """The score program: sigmoid as ``1 / (1 + exp(-x))``; sqrt-softplus with the x > 20 branch."""
    if routing == GROUPED:
        return 1.0 / (1.0 + torch.exp(-logits))
    if routing == SQRTSOFTPLUS:
        sp = torch.where(logits > 20.0, logits, torch.log1p(torch.exp(logits)))
        return torch.sqrt(sp)
    raise ValueError(routing)


def _canon(r: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(r), torch.full_like(r, float("-inf")), r)


def _ordered_bits(r: torch.Tensor) -> torch.Tensor:
    """A float32 -> int64 map whose integer order is the float order (``-0.0`` folded into ``+0.0``,
    which the kernel's ``>`` treats as equal)."""
    r = r.float() + 0.0                                    # -0.0 + 0.0 == +0.0, nothing else moves
    b = r.contiguous().view(torch.int32).long()
    return torch.where(b >= 0, b, b ^ 0x7FFFFFFF)


def selected_sum(v: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """THE renormalisation sum (schedule policy rule 2): ``v [T, K]`` are the scores of the experts ``ids [T, K]``
    in any order; one chain in ASCENDING EXPERT ID seeded with +0, one fp32 add per selected expert."""
    by_id = torch.gather(v, -1, torch.argsort(ids.long(), dim=-1, stable=True))
    acc = torch.zeros_like(by_id[..., 0])
    for j in range(by_id.size(-1)):
        acc = acc + by_id[..., j]
    return acc


def _selection(r: torch.Tensor, allowed: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k ids in rank order under (allowed first, r desc, id asc) -- exact, tie-free keys."""
    T, E = r.shape
    if r.dtype != torch.float32:
        # the float64 twin (report-only): two stable sorts give the same (allowed, r, id) order
        o1 = torch.sort(r + 0.0, dim=-1, descending=True, stable=True).indices
        o2 = torch.sort(torch.gather(allowed, 1, o1).to(torch.uint8), dim=-1, descending=True,
                        stable=True).indices
        return torch.gather(o1, 1, o2)[:, :k]
    ids = torch.arange(E, device=r.device, dtype=torch.int64).expand(T, E)
    key = (allowed.long() << 50) + (_ordered_bits(r) << 16) + (65535 - ids)
    return torch.sort(key, dim=-1, descending=True).indices[:, :k]


def group_scores(r: torch.Tensor, n_group: int) -> torch.Tensor:
    """G_g = the two largest keys of group g, added once (NaN -> -inf)."""
    T, E = r.shape
    top = torch.sort(r.view(T, n_group, E // n_group), dim=-1, descending=True).values
    return _canon(top[..., 0] + top[..., 1])


def selected_groups(G: torch.Tensor, topk_group: int) -> torch.Tensor:
    """bool [T, n_group]: the first topk_group groups under (G desc, g asc)."""
    T, ng = G.shape
    if G.dtype != torch.float32:
        top = torch.sort(G + 0.0, dim=-1, descending=True, stable=True).indices[:, :topk_group]
    else:
        gid = torch.arange(ng, device=G.device, dtype=torch.int64).expand(T, ng)
        key = (_ordered_bits(G) << 16) + (65535 - gid)
        top = torch.sort(key, dim=-1, descending=True).indices[:, :topk_group]
    out = torch.zeros(T, ng, dtype=torch.bool, device=G.device)
    return out.scatter_(1, top, True)


def route_reference(logits: torch.Tensor, *, routing: str, top_k: int, n_group: int = 1,
                    topk_group: int = 1, expert_bias: torch.Tensor | None = None,
                    routed_scale: float = 1.0, num_shared: int = 0,
                    mask: torch.Tensor | None = None, dtype: torch.dtype = torch.float32,
                    return_rank_order: bool = False):
    """The declared program. ``dtype=torch.float64`` is the float64 twin (report-only)."""
    x = logits.to(dtype)
    T, E = x.shape
    s = score(x, routing)
    r = _canon(s + expert_bias.to(dtype) if expert_bias is not None else s)
    if routing == GROUPED and topk_group < n_group:
        keep = selected_groups(group_scores(r, n_group), topk_group)
        allowed = keep.repeat_interleave(E // n_group, dim=1)
    else:
        allowed = torch.ones(T, E, dtype=torch.bool, device=x.device)
    ranked = _selection(r, allowed, top_k)                            # rank order
    sc = score(torch.gather(x, 1, ranked), routing)
    denom = selected_sum(sc, ranked) + 1e-20
    w = (sc / denom.unsqueeze(-1)) * routed_scale
    if return_rank_order:
        return ranked.to(torch.int32), w
    order = torch.argsort(ranked, dim=-1)                             # ascending expert id
    idx = torch.gather(ranked, 1, order).to(torch.int32)
    w = torch.gather(w, 1, order)
    if num_shared:
        sid = torch.arange(E, E + num_shared, device=x.device, dtype=torch.int32).expand(T, -1)
        idx = torch.cat([idx, sid], 1)
        w = torch.cat([w, torch.ones(T, num_shared, dtype=w.dtype, device=w.device)], 1)
    if mask is not None:
        idx = torch.where(mask[:, None], idx, torch.full_like(idx, -1))
        w = torch.where(mask[:, None], w, torch.zeros_like(w))
    return idx.contiguous(), w.contiguous()


# ------------------------------------------------------------------ semantic references ---


def hf_deepseek_v3(logits, expert_bias, n_group, topk_group, top_k, routed_scale):
    """``DeepseekV3MoE.route_tokens_to_experts`` verbatim (norm_topk_prob=True)."""
    router_logits = logits.sigmoid()
    choice = router_logits + expert_bias
    E = logits.size(-1)
    group_scores_ = choice.view(-1, n_group, E // n_group).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores_, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores_)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(-1, n_group, E // n_group).reshape(-1, E)
    choice = choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = router_logits.gather(1, topk_indices)
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_indices, topk_weights * routed_scale


def _ordered_topk(v: torch.Tensor, k: int) -> torch.Tensor:
    """``torch.topk(...)[1]`` with the declared tie rule (lower index first)."""
    n = v.size(-1)
    ids = torch.arange(n, device=v.device, dtype=torch.int64).expand_as(v)
    key = (_ordered_bits(_canon(v)) << 16) + (65535 - ids)
    return torch.sort(key, dim=-1, descending=True).indices[..., :k]


def semantic_ordered(logits, expert_bias, n_group, topk_group, top_k, routed_scale):
    """HF's formula with each ``torch.topk`` given the declared tie rule. Returns rank-ordered ids.

    The ``-inf`` fill that HF uses for the masked groups is kept, and so a masked expert could in
    principle tie an in-group ``-inf`` key; the declared program bypasses masked experts instead.
    The two differ only on rows with fewer than ``top_k`` finite keys inside the surviving groups,
    which the Gate-1 battery plants and reports separately."""
    router_logits = logits.sigmoid()
    choice = router_logits + expert_bias
    E = logits.size(-1)
    T = logits.size(0)
    top2 = torch.sort(_canon(choice).view(T, n_group, E // n_group), dim=-1, descending=True).values
    gsc = top2[..., 0] + top2[..., 1]
    group_idx = _ordered_topk(gsc, topk_group)
    group_mask = torch.zeros_like(gsc).scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(-1, n_group, E // n_group).reshape(-1, E)
    choice = choice.masked_fill(~score_mask.bool(), float("-inf"))
    return _ordered_topk(choice, top_k)


def deepgemm_mega_gate(logits, expert_bias, top_k, routed_scale):
    """DeepGEMM ``test_mega_gate.py`` oracle for routing B: (rank-ordered ids, weights)."""
    s = torch.nn.functional.softplus(logits).sqrt()
    ranking = s + expert_bias if expert_bias is not None else s
    idx = _ordered_topk(ranking, top_k)
    ts = s.gather(1, idx)
    return idx, ts / (ts.sum(dim=1, keepdim=True) + 1e-20) * routed_scale


__all__ = ["GROUPED", "SQRTSOFTPLUS", "deepgemm_mega_gate", "group_scores",
           "hf_deepseek_v3", "route_reference", "score", "selected_groups", "selected_sum", "semantic_ordered"]


# ------------------------------------------------------------------------------- backward ---


def route_backward_reference(dp, x, router_w, logits, *, routing, top_k, n_group=1, topk_group=1,
                             expert_bias=None, routed_scale=1.0, num_shared=0, mask=None,
                             seg_t=None):
    """The declared backward, op for op (DESIGN.md "Backward"); bit-equal to ops.route_backward on
    CUDA. Returns (dx bf16, dw fp32, dl fp32 [T, k], idx int32 [T, k+S])."""
    T, E = logits.shape
    if seg_t is None:
        seg_t = 512 if E <= 128 else 1024      # ops.seg_t_for, restated (no GPU import here)
    kw = dict(routing=routing, top_k=top_k, n_group=n_group, topk_group=topk_group,
              expert_bias=expert_bias, routed_scale=routed_scale)
    ranked, _ = route_reference(logits, return_rank_order=True, **kw)
    ranked = ranked.long()
    s_rank = score(torch.gather(logits, 1, ranked), routing)
    D = selected_sum(s_rank, ranked) + 1e-20
    order = torch.argsort(ranked, dim=-1)
    ids = torch.gather(ranked, 1, order)
    s = torch.gather(s_rank, 1, order)
    lsel = torch.gather(logits, 1, ids)
    A = torch.zeros(T, dtype=torch.float32, device=logits.device)
    for q in range(top_k):
        A = A + dp[:, q] * s[:, q]
    ds = routed_scale * ((dp * D[:, None] - A[:, None]) / (D * D)[:, None])
    if routing == GROUPED:
        dl = ds * (s * (1.0 - s))
    else:
        sig = torch.where(lsel > 20.0, torch.ones_like(lsel), 1.0 / (1.0 + torch.exp(-lsel)))
        dl = torch.where(s == 0, torch.zeros_like(s), (ds * sig) / (2.0 * s))
    if mask is not None:
        dl = torch.where(mask[:, None], dl, torch.zeros_like(dl))
    idx, _ = route_reference(logits, num_shared=num_shared, mask=mask, **kw)
    valid = idx[:, :top_k] >= 0
    W = router_w.float()
    acc = torch.zeros(T, x.shape[1], dtype=torch.float32, device=x.device)
    for q in range(top_k):
        e = ids[:, q].clamp(min=0)
        term = dl[:, q:q + 1] * W[e]
        acc = torch.where(valid[:, q:q + 1], acc + term, acc)
    dx = acc.to(torch.bfloat16)
    nseg = max(1, -(-T // seg_t))
    H = x.shape[1]
    xf = x.float()
    part = torch.zeros(nseg, E, H, dtype=torch.float32, device=x.device)
    cnt = torch.zeros(nseg, E, dtype=torch.int64, device=x.device)
    for t in range(T):
        sg = t // seg_t
        if not bool(valid[t].all()):
            continue
        e = ids[t]
        part[sg, e] = part[sg, e] + dl[t][:, None] * xf[t][None, :]
        cnt[sg, e] += 1
    v = [part[i] for i in range(nseg)]
    full = [cnt[i] > 0 for i in range(nseg)]
    while len(v) > 1:
        nv, nf = [], []
        for i in range(0, len(v), 2):
            if i + 1 == len(v):
                nv.append(v[i])
                nf.append(full[i])
                continue
            both = full[i] & full[i + 1]
            add = v[i] + v[i + 1]
            pick = torch.where(full[i][:, None], v[i], v[i + 1])
            nv.append(torch.where(both[:, None], add, pick))
            nf.append(full[i] | full[i + 1])
        v, full = nv, nf
    dw = torch.where(full[0][:, None], v[0], torch.zeros_like(v[0]))
    return dx, dw, dl, idx


__all__ += ["route_backward_reference"]
