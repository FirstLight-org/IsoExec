"""SM100 fp32 log-softmax: engine full rows and trainer sampled-token logprobs (TP-sharded vocab)."""

from __future__ import annotations

import torch
import torch.distributed as dist

from isoexec.ops.logprobs.logprobs_sm100._extension import extension
from isoexec.ops.logprobs.logprobs_sm100._extension_v2 import extension as extension_v2
from isoexec.ops.logprobs.logprobs_sm100.spec import (  # noqa: F401  (re-exported here)
    DEFAULT_RECIPE_BY_CLASS,
    IMPL_BY_RECIPE,
    LEAVES,
    RECIPE_VERSION,
    RECIPE_VERSION_V2,
    STRIP,
)

_FLOATS = (torch.float32, torch.bfloat16)

_EXTENSIONS = {RECIPE_VERSION: extension, RECIPE_VERSION_V2: extension_v2}

RECIPE: int = RECIPE_VERSION


def default_recipe(dtype, vocab: int) -> int:
    """The measured default for one deployment class, or v1 for a class v2 was never measured on."""
    name = {torch.bfloat16: "bf16", torch.float32: "fp32"}.get(dtype, dtype)
    return DEFAULT_RECIPE_BY_CLASS.get((name, int(vocab)), RECIPE_VERSION)


def select_recipe(recipe: int | None = None, *, dtype=None, vocab: int | None = None) -> int:
    """Pin this process's recipe, once, at deployment configuration time. Returns what was pinned."""
    global RECIPE
    if recipe is None:
        if dtype is None or vocab is None:
            raise ValueError("logprobs_sm100.select_recipe: give a recipe, or a (dtype, vocab) class")
        recipe = default_recipe(dtype, vocab)
    if recipe not in _EXTENSIONS:
        raise ValueError(f"logprobs_sm100: unknown recipe {recipe!r} (declared: {sorted(_EXTENSIONS)})")
    RECIPE = int(recipe)
    return RECIPE


def _ext(recipe: int | None):
    """The extension for ``recipe``, or for the process pin when the caller named none."""
    r = RECIPE if recipe is None else int(recipe)
    try:
        return _EXTENSIONS[r]()
    except KeyError:
        raise ValueError(f"logprobs_sm100: unknown recipe {r!r} (declared: {sorted(_EXTENSIONS)})") from None


def impl_id(recipe: int | None = None) -> str:
    """The declared impl id of ``recipe`` (default: the process pin)."""
    return IMPL_BY_RECIPE[RECIPE if recipe is None else int(recipe)]


def _check(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError("logprobs_sm100 requires CUDA tensors")
    if x.dtype not in _FLOATS:
        raise ValueError(f"logprobs_sm100: logits dtype {x.dtype} is not fp32/bf16")


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
            raise ValueError("logprobs_sm100: temperature must be a scalar")
        return t.reshape(1).to(device=device, dtype=torch.float32)
    return torch.tensor([float(t)], device=device, dtype=torch.float32)


def supports(vocab: int, world: int = 1) -> bool:
    return vocab % (LEAVES * STRIP) == 0 and world >= 1 and LEAVES % world == 0


def full_logprobs(logits: torch.Tensor, *, temperature=None, out: torch.Tensor | None = None,
                  recipe: int | None = None) -> torch.Tensor:
    """Engine: full-row fp32 logprobs of ``logits [..., V]`` (world 1). No autograd."""
    x = logits
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
        return _ext(recipe).full_rows(x, temperature)
    _check(logits)
    shape = logits.shape
    V = shape[-1]
    if not supports(V):
        raise ValueError(f"logprobs_sm100: vocab {V} is not a multiple of {LEAVES * STRIP}")
    y = _ext(recipe).full_rows(_rows(logits), _temperature(temperature, logits.device)).reshape(shape)
    if out is None:
        return y
    out.copy_(y)
    return out


def leaf_states(logits: torch.Tensor, leaves_local: int = LEAVES, *, temperature=None,
                recipe: int | None = None):
    """``(M_g, S_g)`` fp32 ``[rows, leaves_local]`` for a row block holding ``leaves_local`` whole leaves."""
    _check(logits)
    x = _rows(logits)
    rows = x.shape[0]
    m = torch.empty(rows, leaves_local, device=x.device, dtype=torch.float32)
    s = torch.empty_like(m)
    _ext(recipe).leaf_states(x, leaves_local, m, s, _temperature(temperature, x.device))
    return m, s


@torch.no_grad()
def full_logprobs_sharded(logits: torch.Tensor, *, group, recipe: int | None = None) -> torch.Tensor:
    """Full logprob columns owned by this TP rank, using the sampled path's global leaf merge."""
    _check(logits)
    world = 1 if group is None else dist.get_world_size(group=group)
    vocab = logits.shape[-1] * world
    if not supports(vocab, world):
        raise ValueError(f"logprobs_sm100: vocab {vocab} / world {world} unsupported")
    if world == 1:
        return full_logprobs(logits, recipe=recipe)
    x = _rows(logits)
    if x.shape[0] == 0:
        return torch.empty(logits.shape, device=x.device, dtype=torch.float32)
    m, s = leaf_states(x, LEAVES // world, recipe=recipe)
    parts_m = [torch.empty_like(m) for _ in range(world)]
    parts_s = [torch.empty_like(s) for _ in range(world)]
    dist.all_gather(parts_m, m, group=group)
    dist.all_gather(parts_s, s, group=group)
    rM = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    rL = torch.empty_like(rM)
    ext = _ext(recipe)
    ext.merge_rows(
        torch.cat(parts_m, dim=1), torch.cat(parts_s, dim=1), rM, rL,
        None, None, None, None, None, vocab, None,
    )
    out = (x.float() - rM.unsqueeze(1)) - rL.unsqueeze(1)
    return out.reshape(logits.shape)


def _sampled_forward(logits, target, vocab_start, world, group, *, need_grad, recipe=None):
    """The sampled-token forward, shared by the autograd wrapper and the no-grad fast path."""
    x = _rows(logits)
    t = target.reshape(-1).to(torch.int64).contiguous()
    rows, V_local = x.shape
    vocab = V_local * world
    leaves_local = LEAVES // world
    dev = x.device
    m = torch.empty(rows, leaves_local, device=dev, dtype=torch.float32)
    s = torch.empty_like(m)
    ext = _ext(recipe)
    if rows:
        ext.leaf_states(x, leaves_local, m, s, None)
    unsharded = world == 1 and vocab_start == 0
    if unsharded:
        valid, local_t = None, t
    else:
        valid = (t >= 0) & (t < vocab)
        in_shard = valid & (t >= vocab_start) & (t < vocab_start + V_local)
        local_t = torch.where(in_shard, t - vocab_start, torch.full_like(t, -1))
    if world > 1:
        parts_m = [torch.empty_like(m) for _ in range(world)]
        parts_s = [torch.empty_like(s) for _ in range(world)]
        dist.all_gather(parts_m, m, group=group)
        dist.all_gather(parts_s, s, group=group)
        mg = torch.cat(parts_m, dim=1)
        sg = torch.cat(parts_s, dim=1)
        raw = torch.gather(x, 1, local_t.clamp(min=0).unsqueeze(1)).squeeze(1).float()
        raw = torch.where(in_shard, raw, torch.full_like(raw, -0.0))
        dist.all_reduce(raw, op=dist.ReduceOp.SUM, group=group)
    else:
        mg, sg, raw = m, s, None
    rM = torch.empty(rows, device=dev, dtype=torch.float32)
    rL = torch.empty_like(rM)
    lse = torch.empty(rows, device=dev, dtype=torch.float64) if need_grad else None
    out = torch.empty_like(rM)
    if rows:
        if world > 1:
            ext.merge_rows(mg, sg, rM, rL, lse, out, None, t, raw, vocab, None)
        else:
            ext.merge_rows(mg, sg, rM, rL, lse, out, x, t, None, vocab, None)
    return out.reshape(target.shape), (x, lse, local_t, valid, leaves_local, vocab)


class _SampledFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, vocab_start, world, group, recipe):
        out, (x, lse, local_t, valid, leaves_local, vocab) = _sampled_forward(
            logits, target, vocab_start, world, group, need_grad=ctx.needs_input_grad[0], recipe=recipe
        )
        ctx.save_for_backward(x, lse, local_t, valid)
        ctx.shape = logits.shape
        ctx.dtype = logits.dtype
        ctx.leaves_local = leaves_local
        ctx.vocab = vocab
        ctx.recipe = recipe
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, lse, local_t, valid = ctx.saved_tensors
        if valid is None:
            valid = local_t.clamp(0, ctx.vocab - 1) == local_t
        g = grad_out.reshape(-1).to(torch.float32)
        g = torch.where(valid, g, 0.0).contiguous()
        dx = torch.empty(x.shape, device=x.device, dtype=ctx.dtype)
        if x.shape[0]:
            _ext(ctx.recipe).grad_rows(dx, x, lse, g, local_t, ctx.leaves_local)
        return dx.reshape(ctx.shape), None, None, None, None, None


def sampled_logprobs(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    vocab_start_index: int = 0,
    group=None,
    recipe: int | None = None,
) -> torch.Tensor:
    """Trainer: logprob of `target` under `logits [..., V_local]` (this rank's contiguous vocab shard)."""
    _check(logits)
    world = 1 if group is None else dist.get_world_size(group=group)
    V_local = logits.shape[-1]
    if not supports(V_local * world, world):
        raise ValueError(f"logprobs_sm100: vocab {V_local * world} / world {world} unsupported")
    if target.shape != logits.shape[:-1]:
        raise ValueError("logprobs_sm100: target must match logits[..., :-1]")
    if not (torch.is_grad_enabled() and logits.requires_grad):
        return _sampled_forward(logits, target, int(vocab_start_index), world, group, need_grad=False,
                               recipe=recipe)[0]
    return _SampledFn.apply(logits, target, int(vocab_start_index), world, group, recipe)
