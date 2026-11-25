"""Require mm_trainer(x, W[a:b]) to equal mm_trainer(x, W)[:, a:b] bitwise.
Production per-rank projection shapes check that changing output width preserves the K-reduction result."""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

#: ``(K, N_full, tp)`` -- the full projection width and the TP degree that slices it.
CASES = (
    (256, 128, 2),  # tiny MoE, k/v: 2 kv heads x 64, TP=2 -> 64 cols per rank
    (256, 128, 4),  # tiny MoE at TP=4 -> 32 cols per rank (a FRACTION of a head)
    (256, 512, 8),  # h8 tiny value projection at TP=8 -> 64
    (4096, 1024, 8),  # 9B k/v: 4 kv heads x 256, TP=8 -> 128
    (2048, 512, 8),  # a narrower 9B-shaped block
)


@pytest.fixture(scope="module")
def installed():
    root = os.environ.get("ISOEXEC_ORACLE_ROOT")
    if not root:
        pytest.skip("ISOEXEC_ORACLE_ROOT is unset (the directory holding the tiny_qwen35_moe oracle checkpoint)")
    ckpt = os.path.join(root, "tiny_qwen35_moe")
    if not os.path.isdir(ckpt):
        pytest.skip(f"needs the tiny oracle checkpoint at {ckpt}")
    import json

    from isoexec.runtimes.trainer_installs import install_trainer_kernels

    with open(f"{ckpt}/config.json", encoding="utf-8") as fh:
        raw = json.load(fh)
    return install_trainer_kernels(ckpt, source="isoexec.models.qwen3_5_moe", raw_config=raw)


@pytest.mark.parametrize("K,N,tp", CASES)
def test_column_slice_is_exact(installed, K, N, tp):
    """Every rank's block of ``mm(x, W_full)`` equals its own ``mm(x, W_block)``, bit for bit."""
    from isoexec.ops.mm.trainer import mm_trainer
    del installed
    g = torch.Generator(device="cuda").manual_seed(20260907 + K + N + tp)
    for T in (1, 37, 230, 1024):
        x = (torch.randn(T, K, generator=g, device="cuda", dtype=torch.float32) * 0.05).to(torch.bfloat16)
        w = (torch.randn(N, K, generator=g, device="cuda", dtype=torch.float32) * 0.05).to(torch.bfloat16)
        full = mm_trainer(x, w)
        per = N // tp
        for r in range(tp):
            block = mm_trainer(x, w[r * per : (r + 1) * per].contiguous())
            want = full[:, r * per : (r + 1) * per]
            assert torch.equal(block.view(torch.int16), want.contiguous().view(torch.int16)), (
                f"K={K} N={N} tp={tp} T={T} rank={r}: the column block of the full GEMM is not the "
                f"GEMM of the column block -- max |d| "
                f"{float((block.float() - want.float()).abs().max())}"
            )


def test_the_provider_actually_used_is_reported(installed, capsys):
    """Which kernel served these shapes, in the report, so a green run says WHAT it proved.

    On sm90 the segment-tree GEMM (``ops/mm/mm_sm90``) serves every bf16 shape it expresses (K % 64 == 0)
    and the rest fall through to the vendored Triton ``matmul_persistent``; both are exact in N by
    construction (each output column is its own non-split K chain), and this records which one ran."""
    from isoexec.ops.mm.trainer import mm_impl_id

    assert installed  # the installs ran
    stats = {}
    if mm_impl_id() == "sm90_wgmma_segtree":
        from isoexec.ops.mm.mm_sm90.integration import stats as sm90_stats

        stats = sm90_stats()
    print(f"[mm-column-slice] mm provider {mm_impl_id()}, served counts: {stats}")
