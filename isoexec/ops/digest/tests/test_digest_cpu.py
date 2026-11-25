"""Digest term semantics on the host (the torch reference the kernel must equal)."""

from __future__ import annotations

import torch

from isoexec.ops.digest import digest_terms, digest_terms_reference


def test_terms_by_hand():
    t = torch.tensor([1, 2, -1], dtype=torch.int16)  # -1 is the word 65535
    (s1, s2, s3), = digest_terms_reference([t])
    assert s1 == 1 + 2 + 65535
    assert s2 == 1 + 4 + 65535 * 65535


def test_cpu_tensors_take_the_reference():
    ts = [torch.randn(3, 5).to(torch.bfloat16), torch.randn(7)]
    assert digest_terms(ts) == digest_terms_reference(ts)


def test_position_sensitive_and_change_sensitive():
    t = torch.randn(64).to(torch.bfloat16)
    base, = digest_terms_reference([t])
    swapped = t.clone()
    swapped[[3, 9]] = t[[9, 3]]
    if not torch.equal(t[3].view(torch.int16), t[9].view(torch.int16)):
        moved, = digest_terms_reference([swapped])
        assert moved[:2] == base[:2] and moved[2] != base[2]
    bumped = t.clone()
    bumped.view(torch.int16)[5] += 1
    assert digest_terms_reference([bumped])[0] != base


def test_odd_byte_count_and_views():
    u = torch.arange(5, dtype=torch.uint8)
    assert digest_terms_reference([u])[0][0] == (0 | 1 << 8) + (2 | 3 << 8) + 4
    x = torch.randn(8, 6)
    assert digest_terms_reference([x.t()]) == digest_terms_reference([x.t().contiguous()])
