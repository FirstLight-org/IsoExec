"""Abstract output contracts used by canonical AUTOFUSE capture around manual GDN ops."""

import sys

import torch

from isoexec.ops.gdn.shared.gdn_ops import gdn_core, gdn_l2norm

_GDN = "isoexec.ops.gdn.shared"


def test_gdn_l2norm_meta_contract_preserves_tensor_abi_without_loading_cuda_kernel():
    value = torch.empty_strided((7, 3, 128), (512, 128, 1), dtype=torch.bfloat16, device="meta")
    before = frozenset(sys.modules)
    result = gdn_l2norm(value)

    assert result.device.type == "meta"
    assert result.dtype is value.dtype
    assert result.shape == value.shape
    assert result.stride() == value.stride()
    newly_loaded = frozenset(sys.modules) - before
    # The l2norm kernel now lives in our tree; the meta branch must still not import it (it would
    # pull in Triton and pick a launch geometry for a tensor that has no storage). The old upstream
    # name is kept in the assertion so a half-finished revert cannot make this pass vacuously.
    assert f"{_GDN}.vendored_fla.l2norm" not in newly_loaded
    assert "vllm.third_party.flash_linear_attention.ops.l2norm" not in newly_loaded


def test_gdn_core_meta_contract_exposes_outputs_without_selecting_a_cuda_kernel():
    q = torch.empty((2, 7, 4, 128), dtype=torch.bfloat16, device="meta")
    k = torch.empty_like(q)
    v = torch.empty((2, 7, 4, 64), dtype=torch.bfloat16, device="meta")
    g = torch.empty((2, 7, 4), dtype=torch.float32, device="meta")
    beta = torch.empty((2, 7, 4), dtype=torch.bfloat16, device="meta")

    output, state = gdn_core(q, k, v, g, beta, output_final_state=True)

    assert output.shape == v.shape and output.dtype is v.dtype
    assert state.shape == (2, 4, 128, 64)
    assert state.dtype is torch.float32
    assert output.device.type == state.device.type == "meta"
