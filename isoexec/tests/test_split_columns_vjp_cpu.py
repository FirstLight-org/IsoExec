"""Check fused split-column gradients against autograd slicing, including signed zeros and
exceptional values. Slice accumulation evaluates +0 + g, so copy/cat alone would incorrectly
preserve negative zero.
"""

from __future__ import annotations

import torch

from isoexec.core.linear import split_columns

# ``(label, column widths)`` -- the real packed layouts.
LAYOUTS = (
    ("gdn.in_proj q|k|v|z|beta|alpha (9B)", (2048, 2048, 4096, 4096, 32, 32)),
    ("gdn.in_proj (tiny MoE)", (128, 128, 256, 256, 8, 8)),
    ("attn.qkv q|k|v|gate (9B)", (4096, 1024, 1024, 4096)),
    ("mlp.fc1 gate|up (9B)", (12288, 12288)),
    ("mlp.fc1 gate|up (tiny)", (128, 128)),
)


class _W:
    """The only thing ``split_columns`` reads off a weight is ``shape[0]``."""

    def __init__(self, n):
        self.shape = (n,)


def _sliced(out, widths):
    parts, off = [], 0
    for n in widths:
        parts.append(out[:, off : off + n])
        off += n
    return tuple(parts)


def _cotangents(widths, T, seed):
    g = torch.Generator().manual_seed(seed)
    outs = []
    for i, n in enumerate(widths):
        c = torch.randn(T, n, generator=g, dtype=torch.float32)
        if i % 3 == 0:  # signed zeros, the class the spelling is about
            c[0, : min(n, 8)] = torch.tensor([-0.0, 0.0] * min(n, 8))[: min(n, 8)]
        if i % 3 == 1:  # subnormals
            c[0, : min(n, 4)] = torch.tensor([1e-42, -1e-42, 0.0, -0.0])[: min(n, 4)]
        if i % 3 == 2 and n >= 3:  # non-finite
            c[0, :3] = torch.tensor([float("nan"), float("inf"), float("-inf")])
        outs.append(c)
    return outs


def _bits(x):
    return x.contiguous().view(torch.int32)


def test_the_four_spellings_disagree_exactly_where_the_docstring_says():
    """The reason ``split_columns`` uses ``torch.add(g, 0, out=block)`` and not ``cat`` / ``copy_``."""
    y = torch.zeros(1, 4, requires_grad=True)
    a, b = y[:, :2], y[:, 2:]
    ga = torch.tensor([[-0.0, 1.0]])
    gb = torch.tensor([[-0.0, 2.0]])
    torch.autograd.backward([a, b], [ga, gb])
    sliced = _bits(y.grad).tolist()

    added = torch.empty(1, 4)
    torch.add(ga, 0, out=added[:, :2])
    torch.add(gb, 0, out=added[:, 2:])

    copied = torch.empty(1, 4)
    copied[:, :2].copy_(ga)
    copied[:, 2:].copy_(gb)

    catted = torch.cat([ga, gb], dim=1)

    assert sliced == [[0, 1065353216, 0, 1073741824]], sliced
    assert _bits(added).tolist() == sliced, "add-of-zero must reproduce autograd slicing"
    assert _bits(copied).tolist() != sliced, "copy_ was expected to PRESERVE -0.0 and differ"
    assert _bits(catted).tolist() != sliced, "cat was expected to PRESERVE -0.0 and differ"


def test_fused_vjp_is_bitwise_autograd_slicing():
    for label, widths in LAYOUTS:
        weights = tuple(_W(n) for n in widths)
        N = sum(widths)
        for T in (1, 7, 230):
            g = torch.Generator().manual_seed(hash((label, T)) & 0xFFFF)
            base = torch.randn(T, N, generator=g, dtype=torch.float32)
            cot = _cotangents(widths, T, seed=1234 + T)

            a = base.clone().requires_grad_(True)
            torch.autograd.backward(list(_sliced(a * 1.0, widths)), cot)

            b = base.clone().requires_grad_(True)
            torch.autograd.backward(list(split_columns(b * 1.0, weights)), cot)

            ndiff = int((_bits(a.grad) != _bits(b.grad)).sum())
            assert ndiff == 0, f"{label} T={T}: {ndiff} differing bit patterns in the assembled cotangent"


def test_forward_outputs_are_the_same_views():
    """The forward must not change: same values, same storage, no copy."""
    widths = (4096, 1024, 1024, 4096)
    weights = tuple(_W(n) for n in widths)
    out = torch.randn(11, sum(widths), requires_grad=True)
    parts = split_columns(out, weights)
    plain = _sliced(out, widths)
    for p, q in zip(parts, plain):
        assert p.shape == q.shape
        assert p.untyped_storage().data_ptr() == out.untyped_storage().data_ptr(), "a copy was made"
        assert torch.equal(p, q)


def test_no_grad_path_is_the_plain_slices():
    """With grad off -- every engine forward -- the custom Function must not be on the path at all."""
    widths = (128, 128, 256, 256, 8, 8)
    weights = tuple(_W(n) for n in widths)
    with torch.no_grad():
        out = torch.randn(5, sum(widths))
        parts = split_columns(out, weights)
    assert all(p.grad_fn is None for p in parts)
    assert all(not p.requires_grad for p in parts)


if __name__ == "__main__":  # pragma: no cover
    test_the_four_spellings_disagree_exactly_where_the_docstring_says()
    test_fused_vjp_is_bitwise_autograd_slicing()
    test_forward_outputs_are_the_same_views()
    test_no_grad_path_is_the_plain_slices()
    print("PASS test_split_columns_vjp_cpu")
