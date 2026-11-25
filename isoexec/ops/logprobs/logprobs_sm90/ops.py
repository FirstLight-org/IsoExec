"""SM90 fp32 log-softmax: engine full rows and trainer sampled-token logprobs (TP-sharded vocab)."""

from __future__ import annotations

import torch
import torch.distributed as dist

# The recipe version is DECLARED in spec.py and imported here, never restated. The import runs in
# this direction because spec.py must stay import-light -- reading op metadata may not load a
# kernel -- so it cannot import this module, while this module already depends on the package.
from isoexec.ops.logprobs.logprobs_sm90._extension import extension
from isoexec.ops.logprobs.logprobs_sm90.spec import LEAVES, RECIPE_VERSION, STRIP  # noqa: F401  (RECIPE_VERSION is re-exported here)

_FLOATS = (torch.float32, torch.bfloat16)


def _check(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError("logprobs_sm90 requires CUDA tensors")
    if x.dtype not in _FLOATS:
        raise ValueError(f"logprobs_sm90: logits dtype {x.dtype} is not fp32/bf16")


def _rows(x: torch.Tensor) -> torch.Tensor:
    """[rows, V] with stride(1) == 1 on a 16-byte aligned base (copies only if necessary)."""
    x = x.reshape(-1, x.shape[-1])
    if (x.shape[0] > 1 and x.stride(1) != 1) or x.data_ptr() % 16 or (x.shape[0] > 1 and (x.stride(0) * x.element_size()) % 16):
        x = x.clone(memory_format=torch.contiguous_format)
    return x


def _temperature(t, device) -> torch.Tensor | None:
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        if t.numel() != 1:
            raise ValueError("logprobs_sm90: temperature must be a scalar")
        return t.reshape(1).to(device=device, dtype=torch.float32)
    return torch.tensor([float(t)], device=device, dtype=torch.float32)


def supports(vocab: int, world: int = 1) -> bool:
    return vocab % (LEAVES * STRIP) == 0 and world >= 1 and LEAVES % world == 0


def full_logprobs(logits: torch.Tensor, *, temperature=None, out: torch.Tensor | None = None) -> torch.Tensor:
    """Engine: full-row fp32 logprobs of ``logits [..., V]`` (world 1). No autograd."""
    x = logits
    # Fast path (the engine's steady state): aligned 2-D fp32/bf16 rows, tensor or no temperature.
    if (
        out is None
        and x.ndim == 2
        and x.is_cuda
        and (x.dtype is torch.float32 or x.dtype is torch.bfloat16)
        and (x.shape[0] <= 1 or x.stride(1) == 1)
        and x.data_ptr() % 16 == 0
        and x.shape[1] % (LEAVES * STRIP) == 0
        and (temperature is None or (isinstance(temperature, torch.Tensor) and temperature.dtype is torch.float32
                                     and temperature.is_cuda and temperature.numel() == 1))
    ):
        return extension().full_rows(x, temperature)
    _check(logits)
    shape = logits.shape
    V = shape[-1]
    if not supports(V):
        raise ValueError(f"logprobs_sm90: vocab {V} is not a multiple of {LEAVES * STRIP}")
    y = extension().full_rows(_rows(logits), _temperature(temperature, logits.device)).reshape(shape)
    if out is None:
        return y
    out.copy_(y)
    return out


def leaf_states(logits: torch.Tensor, leaves_local: int = LEAVES, *, temperature=None, entropy: bool = False):
    """``(M_g, S_g)`` fp32 ``[rows, leaves_local]`` for a row block holding ``leaves_local`` whole leaves;
    with ``entropy`` also ``A_g`` (the leaf's sum of exp(v - M_g) * (v - M_g)), from the same pass."""
    _check(logits)
    x = _rows(logits)
    rows = x.shape[0]
    m = torch.empty(rows, leaves_local, device=x.device, dtype=torch.float32)
    s = torch.empty_like(m)
    a = torch.empty_like(m) if entropy else None
    extension().leaf_states(x, leaves_local, m, s, _temperature(temperature, x.device), a)
    return (m, s, a) if entropy else (m, s)


@torch.no_grad()
def full_logprobs_sharded(logits: torch.Tensor, *, group) -> torch.Tensor:
    """Full logprob columns owned by this TP rank, using the sampled path's global leaf merge."""
    _check(logits)
    world = 1 if group is None else dist.get_world_size(group=group)
    vocab = logits.shape[-1] * world
    if not supports(vocab, world):
        raise ValueError(f"logprobs_sm90: vocab {vocab} / world {world} unsupported")
    if world == 1:
        return full_logprobs(logits)
    x = _rows(logits)
    if x.shape[0] == 0:
        return torch.empty(logits.shape, device=x.device, dtype=torch.float32)
    m, s = leaf_states(x, LEAVES // world)
    parts_m = [torch.empty_like(m) for _ in range(world)]
    parts_s = [torch.empty_like(s) for _ in range(world)]
    dist.all_gather(parts_m, m, group=group)
    dist.all_gather(parts_s, s, group=group)
    rM = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    rL = torch.empty_like(rM)
    ext = extension()
    ext.merge_rows(
        torch.cat(parts_m, dim=1), torch.cat(parts_s, dim=1), rM, rL,
        None, None, None, None, None, vocab, None,
    )
    # Two eager fp32 subtracts reproduce the CUDA __fsub_rn chain. The full-row writer
    # requires eight local leaves, which a TP shard need not contain.
    out = (x.float() - rM.unsqueeze(1)) - rL.unsqueeze(1)
    return out.reshape(logits.shape)


def _sampled_forward(logits, target, vocab_start, world, group, entropy: bool):
    """The sampled path's forward; with ``entropy`` the same leaf pass also carries the entropy terms.

    The leaf states (M_g, S_g) are evaluated by the same statements either way, so ``out`` does not
    depend on ``entropy``. Returns ``(x, out, lse, local_t, valid, ent)`` with ``ent`` None unless asked.
    """
    x = _rows(logits)
    t = target.reshape(-1).to(torch.int64).contiguous()
    rows, V_local = x.shape
    vocab = V_local * world
    leaves_local = LEAVES // world
    dev = x.device
    # [M | S | A] planes of one buffer: one all_gather moves every leaf term a rank owns.
    planes = 3 if entropy else 2
    st = torch.empty(planes, rows, leaves_local, device=dev, dtype=torch.float32)
    m, s = st[0], st[1]
    a = st[2] if entropy else None
    ext = extension()
    if rows:
        ext.leaf_states(x, leaves_local, m, s, None, a)
    valid = (t >= 0) & (t < vocab)
    in_shard = valid & (t >= vocab_start) & (t < vocab_start + V_local)
    local_t = torch.where(in_shard, t - vocab_start, torch.full_like(t, -1))
    ag = None
    if world > 1:
        parts = [torch.empty_like(st) for _ in range(world)]
        dist.all_gather(parts, st, group=group)
        mg = torch.cat([q[0] for q in parts], dim=1)
        sg = torch.cat([q[1] for q in parts], dim=1)
        if entropy:
            ag = torch.cat([q[2] for q in parts], dim=1)
        raw = torch.gather(x, 1, local_t.clamp(min=0).unsqueeze(1)).squeeze(1).float()
        raw = torch.where(in_shard, raw, torch.full_like(raw, -0.0))
        dist.all_reduce(raw, op=dist.ReduceOp.SUM, group=group)
    else:
        mg, sg, ag, raw = m, s, a, None
    rM = torch.empty(rows, device=dev, dtype=torch.float32)
    rL = torch.empty_like(rM)
    lse = torch.empty(rows, device=dev, dtype=torch.float64)
    out = torch.empty_like(rM)
    ent = torch.empty_like(rM) if entropy else None
    if rows:
        if world > 1:
            ext.merge_rows(mg, sg, rM, rL, lse, out, None, t, raw, vocab, None)
        else:
            ext.merge_rows(mg, sg, rM, rL, lse, out, x, t, None, vocab, None)
        if entropy:
            ext.entropy_rows(mg.contiguous(), sg.contiguous(), ag.contiguous(), ent)
    return x, out, lse, local_t, valid, ent


class _SampledFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, vocab_start, world, group):
        x, out, lse, local_t, valid, _ = _sampled_forward(logits, target, vocab_start, world, group, False)
        ctx.save_for_backward(x, lse, local_t, valid)
        ctx.shape = logits.shape
        ctx.dtype = logits.dtype
        ctx.leaves_local = LEAVES // world
        return out.reshape(target.shape)

    @staticmethod
    def backward(ctx, grad_out):
        x, lse, local_t, valid = ctx.saved_tensors
        g = grad_out.reshape(-1).to(torch.float32)
        g = torch.where(valid, g, torch.zeros_like(g)).contiguous()
        dx = torch.empty(x.shape, device=x.device, dtype=ctx.dtype)
        if x.shape[0]:
            extension().grad_rows(dx, x, lse, g, local_t, ctx.leaves_local)
        return dx.reshape(ctx.shape), None, None, None, None


@torch.no_grad()
def sampled_logprobs_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    vocab_start_index: int = 0,
    group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """No-grad ``(sampled logprobs, row entropy)`` from ONE leaf pass over ``logits [..., V_local]``.

    The logprobs are bit-identical to :func:`sampled_logprobs`. The entropy is fp32 per row, a metric
    outside the scoring contract: its leaf terms follow the fixed leaf/strip/lane order and the merge
    is the fixed lower-left tree over all eight leaves, so it is run-to-run deterministic and the same
    at every TP degree that holds whole leaves.
    """
    _check(logits)
    world = 1 if group is None else dist.get_world_size(group=group)
    V_local = logits.shape[-1]
    if not supports(V_local * world, world):
        raise ValueError(f"logprobs_sm90: vocab {V_local * world} / world {world} unsupported")
    if target.shape != logits.shape[:-1]:
        raise ValueError("logprobs_sm90: target must match logits[..., :-1]")
    _, out, _, _, _, ent = _sampled_forward(logits, target, int(vocab_start_index), world, group, True)
    return out.reshape(target.shape), ent.reshape(target.shape)


def sampled_logprobs(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    vocab_start_index: int = 0,
    group=None,
) -> torch.Tensor:
    """Trainer: logprob of `target` under `logits [..., V_local]` (this rank's contiguous vocab shard)."""
    _check(logits)
    world = 1 if group is None else dist.get_world_size(group=group)
    V_local = logits.shape[-1]
    if not supports(V_local * world, world):
        raise ValueError(f"logprobs_sm90: vocab {V_local * world} / world {world} unsupported")
    if target.shape != logits.shape[:-1]:
        raise ValueError("logprobs_sm90: target must match logits[..., :-1]")
    return _SampledFn.apply(logits, target, int(vocab_start_index), world, group)


@torch.no_grad()
def shard_sample(local_logits: torch.Tensor, *, group, req_of_row: torch.Tensor, req_temperature: torch.Tensor,
                 req_seed: torch.Tensor, row_pos: torch.Tensor, nan_col: int, temperature=None, ranks: bool = True):
    """Engine vocab-parallel sampling head over this rank's contiguous vocab shard ``local_logits
    [rows, V/world]`` (bf16 or fp32, rank-ordered shards, no padding columns).

    Returns ``(sampled int64 [rows], logprob fp32 [rows], rank int64 [rows])``, bitwise what the gathered
    path returns: vLLM's Gumbel-max token (``gumbel_sample``, fp32 noise, ``apply_temperature=False``;
    a row whose request temperature is 0 takes the plain argmax), the leaf8 recipe's logprob of that
    token (``full_logprobs(...)[row, token]``, ``temperature`` the scoring temperature) and vLLM's
    ``_ranks_kernel`` count of columns whose raw logit is >= the token's. Exchanges O(rows) data: one
    all-gather of the per-leaf states and candidates and one all-reduce of the integer counts.
    ``req_of_row`` (int32; < 0 = padding row) indexes ``req_temperature`` (fp32) and ``req_seed`` (int64);
    ``row_pos`` (int64) is the row's position (the Gumbel stream offset). ``ranks=False`` skips the
    all-reduce (the returned counts are then this shard's only). ``nan_col`` is the token the gathered path
    returns for a row that is NaN in every column (a function of the vocabulary and vLLM's kernel only; the
    engine probes it once at init); such a row gets that token, a NaN logprob and rank 0, as there. A row
    that is NaN in only some columns has no order-free argmax and may differ from the gathered path.
    """
    world = 1 if group is None else dist.get_world_size(group=group)
    rank = 0 if group is None else dist.get_rank(group=group)
    x = _rows(local_logits)
    rows, width = x.shape
    if not supports(width * world, world):
        raise ValueError(f"logprobs_sm90.shard_sample: vocab {width * world} / world {world} unsupported")
    ll = LEAVES // world
    dev = x.device
    t = _temperature(temperature, dev)
    part = torch.empty(5 * rows * ll, device=dev, dtype=torch.float32)
    from isoexec.ops.logprobs.logprobs_sm90._shard_extension import extension as shard_extension

    ext = shard_extension()
    ext.shard_pass(x, ll, rank * width, req_of_row, req_temperature, req_seed, row_pos, part, t)
    if world > 1:
        gathered = torch.empty(world * part.numel(), device=dev, dtype=torch.float32)
        dist.all_gather_into_tensor(gathered, part, group=group)
    else:
        gathered = part
    sampled = torch.empty(rows, device=dev, dtype=torch.int64)
    logprob = torch.empty(rows, device=dev, dtype=torch.float32)
    count = torch.empty(rows, device=dev, dtype=torch.int64)
    ext.shard_finish(gathered, world, ll, x, sampled, logprob, count, t, int(nan_col))
    if world > 1 and ranks:
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=group)
    return sampled, logprob, count
