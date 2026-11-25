"""The three class-level facts the CUSTOM attention backend owes vLLM, asserted against the wheel
that is actually installed (design section 3.1, "a registered attention backend").

Two of the three are INHERITED from ``FlashAttentionBackend`` rather than re-declared on our
subclass, which is deliberate: re-declaring them would make our copy agree with itself forever, and
the fact we need is that *the wheel* still says them. So the assertions live here instead.

  1. ``forward_includes_kv_cache_update is False``. vLLM then emits a separate
     ``unified_kv_cache_update`` -- a deterministic scatter -- and our ``forward`` reads the paged
     cache through the block table instead of writing KV itself. A wheel that flipped this to True
     would expect our impl to do the write, and the KV pages would silently be stale.
  2. ``supports_batch_invariance() is True``. Under ``VLLM_BATCH_INVARIANT=1`` the runner refuses a
     backend that does not claim it, so a flip here is an engine that will not start -- but it is
     also the wheel's own statement that this backend's reduction does not depend on batch shape,
     which is the property the whole gate rests on.
  3. ``_cudagraph_support is AttentionCGSupport.UNIFORM_BATCH`` -- ours, and a DOWNGRADE from what we
     inherit. See ``PyTorchVarlenAttentionMetadataBuilder``.

CPU-only: nothing here builds a backend instance, allocates, or touches CUDA. Importing the module
does register the backend under ``AttentionBackendEnum.CUSTOM`` (registration is an import side
effect), which is idempotent and is what every worker does anyway.
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm", reason="the CUSTOM backend contract is a statement about the pinned vLLM wheel")

from vllm.v1.attention.backend import AttentionCGSupport  # noqa: E402
from vllm.v1.attention.backends.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402

from isoexec.ops.attention.varlen_backend import (  # noqa: E402
    PyTorchVarlenAttentionBackend,
    PyTorchVarlenAttentionMetadataBuilder,
)


def test_forward_does_not_include_the_kv_cache_update():
    """Inherited from the wheel; our forward relies on vLLM emitting the separate scatter."""
    assert FlashAttentionBackend.forward_includes_kv_cache_update is False
    assert PyTorchVarlenAttentionBackend.forward_includes_kv_cache_update is False
    assert "forward_includes_kv_cache_update" not in vars(PyTorchVarlenAttentionBackend)


def test_the_backend_claims_batch_invariance():
    """Inherited from the wheel; ``VLLM_BATCH_INVARIANT=1`` refuses a backend that does not."""
    assert FlashAttentionBackend.supports_batch_invariance() is True
    assert PyTorchVarlenAttentionBackend.supports_batch_invariance() is True
    assert "supports_batch_invariance" not in vars(PyTorchVarlenAttentionBackend)


def test_cudagraph_support_is_uniform_batch():
    """Ours, and the one class attribute this backend overrides on the builder.

    Asserted as an identity against the wheel's enum member, not by name or value, so a wheel that
    renumbered ``AttentionCGSupport`` cannot let a stale integer pass.
    """
    builder = PyTorchVarlenAttentionBackend.get_builder_cls()
    assert builder is PyTorchVarlenAttentionMetadataBuilder
    assert issubclass(builder, FlashAttentionMetadataBuilder)
    assert builder._cudagraph_support is AttentionCGSupport.UNIFORM_BATCH
    assert vars(builder)["_cudagraph_support"] is AttentionCGSupport.UNIFORM_BATCH


def test_the_override_never_claims_more_than_the_wheel_does():
    """Two bounds, and the override has to sit between them.

    Above: design section 3.1 requires ``UNIFORM_BATCH`` or better, and it is strictly more than the
    ``UNIFORM_SINGLE_TOKEN_DECODE`` this backend used to declare -- query length k+1 (spec-decode
    verify) is admitted where only query length 1 was.

    Below: we may not claim MORE than ``FlashAttentionMetadataBuilder`` does, because every kernel we
    dispatch to is one of FA's. The base value is computed at class definition from
    ``get_flash_attn_version()`` (``flash_attn.py:313-317``), so it is ``ALWAYS`` on an FA3 host and
    ``UNIFORM_BATCH`` without one; both satisfy this bound, and a wheel that dropped below it would
    make the override an upgrade, which it must never be.
    """
    inherited = FlashAttentionMetadataBuilder._cudagraph_support
    ours = PyTorchVarlenAttentionMetadataBuilder._cudagraph_support
    assert ours.value > AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE.value
    assert ours.value <= inherited.value, (
        f"the wheel's FlashAttentionMetadataBuilder now claims {inherited.name}, less than the CUSTOM "
        f"backend's {ours.name}: the override would be an upgrade over the kernels it dispatches to"
    )


def test_get_builder_cls_returns_one_stable_class():
    """It used to synthesize a fresh class per call, so no two callers could compare builder types."""
    assert PyTorchVarlenAttentionBackend.get_builder_cls() is PyTorchVarlenAttentionBackend.get_builder_cls()


def test_the_backend_is_registered_under_the_custom_slot():
    """``register_backend(AttentionBackendEnum.CUSTOM)`` ran at import; the enum lookup is unconditional."""
    assert AttentionBackendEnum.CUSTOM.get_class() is PyTorchVarlenAttentionBackend
    assert PyTorchVarlenAttentionBackend.get_name() == "CUSTOM"


def test_the_paged_cache_contract_follows_the_arch_selected_package(monkeypatch):
    """SM100's kernel reads whole 128-key tiles out of token-major pages; SM90 keeps FA's declarations.

    Read off the host's arch tag (``core.arch.ARCH``, which needs no CUDA context), because vLLM
    asks while it configures the engine, before any worker exists.
    """
    from vllm.v1.attention.backend import MultipleOf
    from vllm.v1.kv_cache_layout import KVCacheLayout

    from isoexec.core import arch

    backend = PyTorchVarlenAttentionBackend
    monkeypatch.setattr(arch, "ARCH", "sm100")
    (sizes,) = backend.get_supported_kernel_block_sizes()
    assert isinstance(sizes, MultipleOf) and sizes.base == 128
    assert backend.get_preferred_block_size(16) == 128 and backend.get_preferred_block_size(544) == 640
    assert backend.supported_kv_cache_layouts() == (KVCacheLayout.LBNHC,)

    monkeypatch.setattr(arch, "ARCH", "sm90")
    ours = [getattr(s, "base", s) for s in backend.get_supported_kernel_block_sizes()]
    wheel = [getattr(s, "base", s) for s in FlashAttentionBackend.get_supported_kernel_block_sizes()]
    assert ours == wheel
    assert backend.get_preferred_block_size(16) == FlashAttentionBackend.get_preferred_block_size(16)
    assert backend.supported_kv_cache_layouts() == FlashAttentionBackend.supported_kv_cache_layouts()


def test_the_cache_contract_answers_vllms_per_spec_query(monkeypatch):
    """vLLM 0.33 asks per KV-cache spec (``get_supported_kernel_block_sizes(kv_cache_spec)``)."""
    from isoexec.core import arch

    backend = PyTorchVarlenAttentionBackend
    for host in ("sm100", "sm90"):
        monkeypatch.setattr(arch, "ARCH", host)
        assert backend.get_supported_kernel_block_sizes(None) == backend.get_supported_kernel_block_sizes()


def test_every_override_keeps_the_wheels_parameters():
    """An override that drops a parameter vLLM passes fails only at engine start; catch it here."""
    import inspect

    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadataBuilder

    from isoexec.ops.attention.varlen_backend import PyTorchVarlenAttentionImpl, PyTorchVarlenAttentionMetadataBuilder

    pairs = (
        (PyTorchVarlenAttentionBackend, FlashAttentionBackend),
        (PyTorchVarlenAttentionImpl, FlashAttentionImpl),
        (PyTorchVarlenAttentionMetadataBuilder, FlashAttentionMetadataBuilder),
    )
    for ours, wheel in pairs:
        for name in vars(ours):
            if name.startswith("__") or not callable(getattr(ours, name)) or not hasattr(wheel, name):
                continue
            mine = list(inspect.signature(getattr(ours, name)).parameters)
            theirs = list(inspect.signature(getattr(wheel, name)).parameters)
            assert mine == theirs, f"{ours.__name__}.{name}{mine} no longer matches vLLM's {theirs}"


@pytest.mark.parametrize("arch_name", ["sm90", "sm100"])
def test_native_cache_planning_supplies_the_cache_spec(monkeypatch, arch_name):
    """Exercise the V1 engine's real caller, including its spec-bearing block selection."""
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.utils import customize_attention_spec, select_common_block_size
    from isoexec.core import arch

    monkeypatch.setattr(arch, "ARCH", arch_name)
    spec = FullAttentionSpec(block_size=512, num_kv_heads=2, head_size=128, dtype=torch.bfloat16)
    customized = customize_attention_spec(PyTorchVarlenAttentionBackend, spec)
    assert customized == spec
    assert select_common_block_size(512, [PyTorchVarlenAttentionBackend], [customized]) == 512


def test_native_spec_dependent_rules_are_preserved(monkeypatch):
    """The inherited backend can require a different block size once the layer spec is known."""
    import torch
    from vllm.v1.attention.backend import MultipleOf
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.utils import customize_attention_spec, select_common_block_size
    from isoexec.core import arch

    monkeypatch.setattr(arch, "ARCH", "sm90")

    def sizes(cls, kv_cache_spec=None):
        return [32] if isinstance(kv_cache_spec, FullAttentionSpec) else [16]

    monkeypatch.setattr(FlashAttentionBackend, "get_supported_kernel_block_sizes", classmethod(sizes))
    spec = FullAttentionSpec(block_size=128, num_kv_heads=2, head_size=128, dtype=torch.bfloat16)
    customized = customize_attention_spec(PyTorchVarlenAttentionBackend, spec)
    assert customized.block_stride_alignment == MultipleOf(32)
    assert select_common_block_size(128, [PyTorchVarlenAttentionBackend], [customized]) == 32


def test_forward_consumes_the_native_flash_metadata(monkeypatch):
    """The current wheel's metadata has no cascade flag; use its live cache and valid rows."""
    from types import SimpleNamespace
    from typing import Any, cast
    import torch
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
    from isoexec.ops import _dispatch
    from isoexec.ops.attention import varlen_backend

    metadata = FlashAttentionMetadata(
        num_actual_tokens=2, max_query_len=2, query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        max_seq_len=2, seq_lens=torch.tensor([2], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32), slot_mapping=torch.tensor([0, 1]),
    )
    cache = torch.zeros(1, 1, 16, 256, dtype=torch.bfloat16)
    class Layer(torch.nn.Module):
        layer_name = "layer"

    layer = Layer()
    monkeypatch.setattr(varlen_backend, "get_attention_context",
                        lambda name: (metadata, layer, cache, metadata.slot_mapping))
    calls = []

    def paged(query, key, value, **kwargs):
        calls.append((key, value, kwargs))
        kwargs["out"].copy_(query)

    monkeypatch.setattr(_dispatch, "backend_module", lambda *args: SimpleNamespace(paged_core=paged))
    impl = cast(Any, object.__new__(varlen_backend.PyTorchVarlenAttentionImpl))
    impl.attn_type, impl.kv_cache_dtype = AttentionType.DECODER, "auto"
    impl.dcp_world_size = impl.pcp_world_size = 1  # set by AttentionImplBase.__new__ in production
    impl.alibi_slopes, impl.sliding_window = None, (-1, -1)
    impl.num_kv_heads, impl.head_size, impl.scale = 1, 128, 0.125
    query = torch.arange(4 * 2 * 128, dtype=torch.float32).reshape(4, 2, 128).bfloat16()
    output = torch.full_like(query, -1)
    result = impl.forward(layer, query, torch.empty(0), torch.empty(0), torch.empty(0), metadata, output=output)
    assert result is output and torch.equal(result[:2], query[:2]) and bool((result[2:] == -1).all())
    assert len(calls) == 1 and calls[0][0].data_ptr() == cache.data_ptr()
    assert calls[0][2]["cu_seqlens_q"] is metadata.query_start_loc


def test_the_backend_admits_pcp_and_dcp_for_the_isoexec_binding():
    # vLLM's check_attention_cp_compatibility reads these flags; the IsoExec binding runs CP attention.
    assert PyTorchVarlenAttentionBackend.supports_pcp() is True
    assert PyTorchVarlenAttentionBackend.supports_dcp() is True
    assert PyTorchVarlenAttentionBackend.get_impl_cls().can_return_lse_for_decode is True


@pytest.mark.parametrize(("pcp", "dcp", "size"), [(2, 2, 64), (1, 2, 32)])
def test_dcp_context_buffer_fits_the_pcp_local_batch(monkeypatch, pcp, dcp, size):
    from types import SimpleNamespace
    import torch

    def native_init(self, kv_cache_spec, layer_names, vllm_config, device):
        self.dcp_world_size = dcp
        self._dcp_context_kv_lens = torch.zeros(vllm_config.scheduler_config.max_num_seqs, dtype=torch.int32)

    monkeypatch.setattr(FlashAttentionMetadataBuilder, "__init__", native_init)
    config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
                             scheduler_config=SimpleNamespace(max_num_seqs=32))
    builder = PyTorchVarlenAttentionMetadataBuilder(None, [], config, torch.device("cpu"))
    assert builder._dcp_context_kv_lens.numel() == size
