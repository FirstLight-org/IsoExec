"""Autograd transport for replicated TP activations and unique EP token homes.

Every TP rank evaluates the same loss. Thus replicated outputs use identity
backwards, while disjoint contributions to an input are summed exactly once.
"""
import torch
import torch.distributed as dist
from contextlib import contextmanager
from .ordered_sum import ordered_sum


def validate_token_distribution(mode, *, tp_world, ep_world, tp_group=None, ep_group=None):
    """Validate the model's input ownership, independently from loss normalization.

    ``partitioned`` means EP ranks hold distinct complete sequences, while TP
    ranks still hold identical tokens. Its gradients represent the sum of the
    EP-local loss numerators; a caller wanting a mean supplies the global
    denominator. Such EP and TP groups must intersect in one rank only.
    """
    if mode not in ('replicated', 'partitioned'):
        raise ValueError('token_distribution must be replicated or partitioned')
    if mode == 'partitioned' and ep_world > 1:
        if not dist.is_initialized():
            raise ValueError('partitioned TP/EP requires initialized process groups')
        tp = (dist.get_process_group_ranks(tp_group if tp_group is not None else dist.group.WORLD)
              if tp_world > 1 else [dist.get_rank()])
        ep = dist.get_process_group_ranks(ep_group if ep_group is not None else dist.group.WORLD)
        if len(tp) != tp_world or len(ep) != ep_world or len(set(tp).intersection(ep)) != 1:
            raise ValueError('partitioned tokens require orthogonal TP and EP groups')
        coordinate = dist.get_rank(tp_group) if tp_world > 1 else 0
        members = [None] * ep_world
        dist.all_gather_object(members, (coordinate, tuple(tp)), group=ep_group)
        invalid = any(rank != coordinate for rank, _ in members)
        cohorts = tuple(sorted(ranks for _, ranks in members))
        invalid |= len(set(cohorts)) != ep_world
        if tp_world > 1:
            peers = [None] * tp_world
            dist.all_gather_object(peers, (invalid, cohorts), group=tp_group)
            invalid |= any(bad or other != cohorts for bad, other in peers)
        if invalid:
            raise ValueError('partitioned EP groups must align TP shard coordinates and replica cohorts')
    return mode


def validate_parallel_layer_ownership(owned, *, tp_world, ep_world, tp_group=None, ep_group=None):
    """A model-parallel group cannot combine parameters from different PP stages."""
    errors = []
    for label, world, group in (('TP', tp_world, tp_group), ('EP', ep_world, ep_group)):
        if world <= 1:
            continue
        inventories = [None] * world
        dist.all_gather_object(inventories, tuple(owned), group=group)
        if any(value != tuple(owned) for value in inventories):
            errors.append(label)
    if errors:
        raise ValueError('/'.join(errors) + ' groups must remain within one pipeline stage')


class _ParameterSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, *parameters):
        ctx.group = group
        ctx.save_for_backward(*parameters)
        ctx.set_materialize_grads(False)
        return tuple(parameter.view_as(parameter) for parameter in parameters)

    @staticmethod
    def backward(ctx, *gradients):
        # A single autograd node waits for every parameter use. Empty ranks or
        # branches with unused parameters therefore enter the same collective
        # sequence as nonempty ranks; per-parameter hooks cannot guarantee this.
        result = []
        for parameter, gradient in zip(ctx.saved_tensors, gradients):
            partial = torch.zeros_like(parameter) if gradient is None else gradient
            result.append(ordered_sum(partial, ctx.group))
        return (None, *result)


@contextmanager
def sum_parameter_gradients(module, names, group=None):
    """Temporarily expose differentiable aliases with one ordered gradient sum.

    Optimizer/DDP Parameter identities and storage remain unchanged. Restoration
    occurs before backward and on forward errors. Functional autograd.grad also
    receives the reduced derivatives, including zeros for locally unused leaves.
    Calls must not concurrently mutate the same module's parameter dictionaries.
    """
    if not torch.is_grad_enabled() or not names:
        yield
        return
    locations = []
    for name in sorted(names):
        parent_name, _, leaf = name.rpartition('.')
        owner = module.get_submodule(parent_name) if parent_name else module
        parameter = owner._parameters[leaf]
        locations.append((owner, leaf, parameter))
    aliases = _ParameterSum.apply(group, *(entry[2] for entry in locations))
    if len(locations) == 1 and isinstance(aliases, torch.Tensor):
        aliases = (aliases,)
    try:
        for (owner, leaf, _), alias in zip(locations, aliases):
            owner._parameters[leaf] = alias
        yield
    finally:
        for owner, leaf, parameter in locations:
            owner._parameters[leaf] = parameter


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        return ordered_sum(grad, ctx.group), None


def copy_to_parallel(x, group=None):
    return _Copy.apply(x, group)


class _Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, reducer):
        return reducer.gather_sum(x)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def reduce_from_parallel(x, reducer):
    return _Reduce.apply(x, reducer)


def _gather(x, spans, group):
    width = max(b - a for a, b in spans)
    send = x.new_zeros((width, *x.shape[1:]))
    send[:x.shape[0]].copy_(x)
    parts = [torch.empty_like(send) for _ in spans]
    dist.all_gather(parts, send, group=group)
    return torch.cat([p[:b-a] for p, (a,b) in zip(parts, spans)], 0)


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, spans, group):
        ctx.span = spans[dist.get_rank(group)]
        return _gather(x, spans, group)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.span
        return grad[a:b].contiguous(), None, None


def gather_home_rows(x, spans, group=None):
    return _Gather.apply(x, spans, group)


class _Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, spans, group):
        ctx.spans, ctx.group = spans, group
        a, b = spans[dist.get_rank(group)]
        return x[a:b].contiguous()

    @staticmethod
    def backward(ctx, grad):
        return _gather(grad, ctx.spans, ctx.group), None, None


def scatter_home_rows(x, spans, group=None):
    return _Scatter.apply(x, spans, group)
