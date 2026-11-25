"""CPU contract of the SM100 DeepSeek-V4.1 BF16-actor recipe declarations."""
from isoexec.ops.moe.deepseek_v41_sm100 import spec


def test_declarations_cpu():
    assert spec.OP == "moe.deepseek_v41" and {i.op for i in spec.IMPLS} == {spec.OP}
    assert spec.ROUTING == "sqrtsoftplus_bias_select" and spec.FC2_LEAVES == 1
    assert all(i.supported_archs == frozenset({"sm100"}) for i in spec.IMPLS)


def test_exchange_pad_covers_every_swapped_token_tile_cpu():
    """``bi_tp`` asks the exchange's h region for ``Tk + tile_n`` rows under a swapped record; the region's headroom must
    cover the widest swapped token tile the vendored generator can produce."""
    from isoexec.ops._vendor.flashinfer_bi_moe.gemm.batch_invariant import generator
    from isoexec.ops.moe.deepseek_v41_sm100 import tp_exchange
    assert tp_exchange.H_PAD_ROWS >= max((*generator.TILES_N_NARROW, generator.SWAP_WIDE_TILE_N))
