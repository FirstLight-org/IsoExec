"""CPU contract of the DeepGEMM segment-tree GEMMs: which operand sets they serve, and that the row thresholds are package constants (no environment switch). The bits are
GPU-gated (``test_segtree_gpu.py``)."""
import torch

from isoexec.ops.mm.deepgemm_sm100 import segtree


def test_serves_whole_k_blocks_per_segment():
    assert segtree.serves(1280, 4096, 2) and segtree.serves(512, 4096, 1) and segtree.serves(4096, 4096, 8)
    assert not segtree.serves(1280, 4096, 3)        # K not cut evenly
    assert not segtree.serves(1280, 64 * 3, 2)      # a segment is not whole 64-wide K blocks
    assert not segtree.serves(1000, 4096, 2)        # N % 16
    assert not segtree.serves(1280, 4096, 0)


def test_thresholds_are_constants():
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops as attn_ops
    from isoexec.ops.moe.router_sm100 import ops as router
    assert attn_ops.SEGTREE_MM_ROWS == 128 and attn_ops.SEGTREE_MM_NARROW_ROWS == 1024
    assert attn_ops.SEGTREE_WOB_ROWS == 16 and attn_ops.ONCHIP_WOB_ROWS >= 1 << 20
    assert router.SEGTREE_ROWS >= 1 << 20                      # the segment-root selection at every row count
    assert not hasattr(segtree, "enabled") and not hasattr(segtree, "ENV")


def test_head_keeps_the_mm_sm100_plans(monkeypatch):
    """The FP32-root logprobs head shares the composition's ``linear`` but never takes the decode segment trees."""
    from isoexec.ops.attention.deepseek_v41_cache_sm100 import ops as attn_ops
    from isoexec.ops.logprobs import resolve
    seen = {}

    def fake(a, w, **kw):
        seen.update(kw)
        return a.new_zeros((a.shape[0], w.shape[0]), dtype=torch.float32)
    monkeypatch.setattr(attn_ops, "linear", fake)
    out = resolve._head_fp32_root(torch.zeros(2, 3, 64), torch.zeros(32, 64))
    assert out.shape == (2, 3, 32) and seen == {"root_fp32": True, "decode_tree": False}
