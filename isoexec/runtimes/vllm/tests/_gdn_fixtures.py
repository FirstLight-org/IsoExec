"""CPU collaborators for the native GDN layer and selected pool interfaces."""

import sys
from types import ModuleType

import torch

from isoexec.ops import gdn
from isoexec.ops.gdn.shared import gdn_ops
from isoexec.runtimes.vllm import gdn_engine_patch


class MambaBase:
    pass


class MambaAttentionBackendEnum:
    GDN_ATTN = object()


class MambaStateShapeCalculator:
    @staticmethod
    def gated_delta_net_state_shape(tp, key_heads, value_heads, key_dim, value_dim, window):
        channels = 2 * key_heads * key_dim + value_heads * value_dim
        return ((window - 1, channels // tp), (value_heads // tp, value_dim, key_dim))


class MambaStateDtypeCalculator:
    @staticmethod
    def gated_delta_net_state_dtype(model_dtype, conv_dtype, ssm_dtype):
        dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
        conv = model_dtype if conv_dtype == "auto" else dtypes[conv_dtype]
        return conv, conv if ssm_dtype == "auto" else dtypes[ssm_dtype]


class MambaStateCopyFuncCalculator:
    @staticmethod
    def gated_delta_net_state_copy_func():
        return conv_copy, ssm_copy


def conv_copy(*args, **kwargs):
    raise AssertionError("CPU protocol test executed native conv page copying")


def ssm_copy(*args, **kwargs):
    raise AssertionError("CPU protocol test executed native SSM page copying")


class Pool:
    def __init__(self, *, max_num_seqs, chunk_size, on_alloc, conv_weight, conv_bias,
                 A_log, dt_bias, num_k_heads, head_k_dim, num_v_heads, head_v_dim,
                 activation, dtype, device):
        self.capacity, self.chunk_size = max_num_seqs, chunk_size
        self._on_alloc = on_alloc
        self.conv_weight, self.conv_bias = conv_weight, conv_bias
        self.A_log, self.dt_bias = A_log, dt_bias
        self.num_k_heads, self.num_v_heads = num_k_heads, num_v_heads
        self.head_k_dim, self.head_v_dim = head_k_dim, head_v_dim
        self.activation = activation
        channels = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        self._publish("ssm_state", torch.zeros(max_num_seqs + 1, num_v_heads, 16, dtype=torch.uint8, device=device))
        self._publish("conv_state", torch.zeros(max_num_seqs + 1, channels, conv_weight.shape[-1] - 1,
                                               dtype=dtype, device=device))
        self._publish("slot2row", torch.zeros(8, dtype=torch.long, device=device))

    def _publish(self, name, tensor):
        setattr(self, name, tensor)
        self._on_alloc(name, tensor)
        return tensor


def install_gdn(monkeypatch):
    abstract = ModuleType("vllm.model_executor.layers.mamba.abstract")
    abstract.MambaBase = MambaBase
    utils = ModuleType("vllm.model_executor.layers.mamba.mamba_utils")
    utils.MambaStateDtypeCalculator = MambaStateDtypeCalculator
    utils.MambaStateShapeCalculator = MambaStateShapeCalculator
    utils.MambaStateCopyFuncCalculator = MambaStateCopyFuncCalculator
    registry = ModuleType("vllm.v1.attention.backends.registry")
    registry.MambaAttentionBackendEnum = MambaAttentionBackendEnum
    for module in (abstract, utils, registry):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    # GDNStateLayer subclasses MambaBase when runtimes.vllm._gdn is first imported. A test file that ran
    # earlier against real vLLM may have imported it already; drop it so it re-imports onto the stubs.
    monkeypatch.delitem(sys.modules, "isoexec.runtimes.vllm._gdn", raising=False)
    monkeypatch.setattr(gdn, "build_state_pool", Pool)
    monkeypatch.setitem(gdn.__dict__, "fla_chunk_size", lambda: 64)
    monkeypatch.setitem(gdn.__dict__, "gdn_cpr_min_pages", lambda: False)
    monkeypatch.setattr(gdn_ops, "gdn_cpr_min_pages", lambda: False)
    monkeypatch.setitem(gdn.__dict__, "GDN_CPR_MIN_STATE_SHAPES", ((1, 8), (1, 1, 4)))
    monkeypatch.setattr(gdn_engine_patch, "assert_engine_args_compatible", lambda *args, **kwargs: None)
    monkeypatch.setattr(gdn_engine_patch, "lift_gdn_batch_invariance_veto", lambda: None)
    monkeypatch.setattr(gdn_engine_patch, "install_cpr_lazy_driver", lambda: True)
    monkeypatch.setattr(gdn_engine_patch, "gdn_metadata", lambda prefix: None)
