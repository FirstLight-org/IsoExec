"""Check packed expert buffers against live parameter values, outputs and gradients.
Parameter rebinding, replaced watchers and EP-range changes must invalidate views; foreign storage must not be stolen,
and parameter identities, names and state_dict shapes must remain intact."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from isoexec.nn import MoE  # noqa: E402
from isoexec.ops.table import OpTable  # noqa: E402
from isoexec.schema import make_sequence_context  # noqa: E402

from isoexec.nn.tests import _moe_fixtures as F  # noqa: E402
from isoexec.ops.moe.shared import (  # noqa: E402
    packed_experts as P,
)
from isoexec.ops.moe.shared import (  # noqa: E402
    param_storage as PS,
)


class _Layers(nn.Module):
    """Two placed mixture layers on the reference table, the way a runtime holds them."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(F.component(F.spec(), device="cpu") for _ in range(2))
        for i, layer in enumerate(self.layers):
            layer.apply_placement(F.placement(), site=f"lm.layers.{i}.mlp", op_table=OpTable(mode="reference"))


@pytest.fixture
def model():
    m = _Layers()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g).mul_(0.05).to(p.dtype))
    # What every build path does once the parameters are final; without it every layer stacks.
    assert P.attach(m) == sum(1 for mod in m.modules() if isinstance(mod, MoE))
    try:
        yield m
    finally:
        # The attached views close over their layer, so a WeakSet entry can survive
        # until cyclic GC. Retire this fixture's aliases explicitly, including the
        # deliberately invalid foreign-storage case, before another test syncs.
        P.release_to_storage_owner(m)


@pytest.fixture
def layer(model):
    return model.layers[0]


def _stack_views(layer):
    """The exact expression ``layer.py`` falls back to."""
    held = layer.held_experts()
    return (
        torch.stack([torch.cat([e.gate, e.up], dim=0) for e in held], dim=0),
        torch.stack([e.down for e in held], dim=0),
    )


def _seq(T):
    return make_sequence_context(torch.arange(T), torch.tensor([0, T], dtype=torch.int32), max_q=T)


# the mandatory ones


def test_a_data_rebind_is_never_read_stale(layer):
    """A41, head on: repoint a parameter's storage and the next read must see the NEW bytes.

    Arm A is the steady state (re-packing rate-limited to one per weight sync), where the read has to
    fall back to ``torch.stack`` OVER THE NEW STORAGE. Arm B is the sync boundary, which licenses one
    re-pack and must rebuild the buffer from the current values -- never resurrect the old ones.
    """
    assert P.pack_layer(layer) is not None
    e0 = layer.held_experts()[0]
    buf = getattr(layer, P._STATE_ATTR)["gate_up"].buf
    fresh = torch.full_like(e0.gate, 0.125)

    layer.__dict__["_ix_packed_at"] = PS.SYNC_EPOCH  # arm A: this epoch's re-pack is spent
    with torch.no_grad():
        e0.gate.data = fresh.clone()
    gate_up, _ = layer.expert_weight_views()
    assert gate_up.data_ptr() != buf.data_ptr(), "a rebound parameter must NOT be served from the buffer"
    assert torch.equal(gate_up[0, : fresh.shape[0]], e0.gate), "stale read: the view is not the live parameter"
    assert P.check_layer(layer)[0] is False and "rebound" in P.check_layer(layer)[1]

    PS.bump_sync_epoch()  # arm B: the sync boundary
    gate_up2, _ = layer.expert_weight_views()
    assert torch.equal(gate_up2[0, : fresh.shape[0]], e0.gate), "the re-pack resurrected stale bytes"
    assert torch.equal(gate_up2, _stack_views(layer)[0])


def test_a_foreign_storage_is_refused(layer):
    """A DDP / distributed-optimizer bucket must be left alone -- packing it would sever THEIR alias."""
    held = layer.held_experts()
    flat = torch.zeros(sum(p.numel() for e in held for p in (e.gate, e.up, e.down)), dtype=torch.bfloat16)
    off, ptrs = 0, {}
    with torch.no_grad():
        for e in held:
            for a in ("gate", "up", "down"):
                p = getattr(e, a)
                v = flat[off : off + p.numel()].view(p.shape)
                v.copy_(p.data)
                p.data = v
                off += p.numel()
                ptrs[(id(e), a)] = p.data_ptr()
    before = P.stats()["refused_foreign_storage"]
    assert P.pack_layer(layer) is None, "packed a parameter whose storage belongs to someone else"
    assert P.stats()["refused_foreign_storage"] == before + 1
    for e in held:
        for a in ("gate", "up", "down"):
            assert getattr(e, a).data_ptr() == ptrs[(id(e), a)], "the bucket's aliasing was severed"
    # ... and the layer keeps serving correct views off the bucket.
    gate_up, down = layer.expert_weight_views()
    ref_gu, ref_dn = _stack_views(layer)
    assert torch.equal(gate_up, ref_gu) and torch.equal(down, ref_dn)


# bitwise


def test_views_are_bitwise_and_free(layer):
    ref_gu, ref_dn = _stack_views(layer)
    bufs = P.pack_layer(layer)
    assert bufs is not None
    gate_up, down = layer.expert_weight_views()
    assert torch.equal(gate_up, ref_gu) and torch.equal(down, ref_dn)
    assert gate_up.dtype is ref_gu.dtype and gate_up.shape == ref_gu.shape
    # free: the "stack" IS the buffer, and every parameter is a slice of it
    assert gate_up.data_ptr() == bufs["gate_up"].data_ptr()
    assert down.data_ptr() == bufs["down"].data_ptr()
    I = layer.intermediate
    for i, e in enumerate(layer.held_experts()):
        assert e.gate.data_ptr() == bufs["gate_up"][i, :I].data_ptr()
        assert e.up.data_ptr() == bufs["gate_up"][i, I:].data_ptr()
        assert e.down.data_ptr() == bufs["down"][i].data_ptr()
    assert layer.gate_up_view().data_ptr() == bufs["gate_up"].data_ptr()
    assert layer.down_view().data_ptr() == bufs["down"].data_ptr()


@pytest.mark.parametrize("T", [1, 5, 64])
def test_layer_output_is_bitwise(layer, T):
    x = torch.randn(T, layer.router.weight.shape[1], generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    with torch.no_grad():
        ref = layer(x, _seq(T)).clone()
    assert P.pack_layer(layer) is not None
    with torch.no_grad():
        got = layer(x, _seq(T))
    assert torch.equal(got, ref)


def test_expert_gradients_are_bitwise(layer):
    T = 32
    g = torch.Generator().manual_seed(11)
    x = torch.randn(T, layer.router.weight.shape[1], generator=g).to(torch.bfloat16)
    gout = torch.randn(T, x.shape[1], generator=g)

    def run():
        xx = x.clone().requires_grad_(True)
        layer(xx, _seq(T)).backward(gout)
        out = {n: p.grad.detach().clone() for n, p in layer.named_parameters() if p.grad is not None}
        for p in layer.parameters():
            p.grad = None
        return out, xx.grad.clone()

    ref, ref_x = run()
    assert P.pack_layer(layer) is not None
    got, got_x = run()
    assert set(got) == set(ref) and len(ref) > 3 * len(layer.held_experts())
    assert torch.equal(got_x, ref_x)
    bad = [n for n in ref if not torch.equal(got[n], ref[n])]
    assert not bad, f"{len(bad)} parameter grads moved, e.g. {bad[:4]}"


def test_disabled_by_env_is_the_untouched_path(layer, monkeypatch):
    monkeypatch.setenv("ISOEXEC_MOE_PACKED_EXPERTS", "0")
    ref_gu, ref_dn = _stack_views(layer)
    gate_up, down = layer.expert_weight_views()
    assert getattr(layer, P._STATE_ATTR, None) is None, "the env-off path must not pack"
    assert gate_up.data_ptr() != ref_gu.data_ptr() and torch.equal(gate_up, ref_gu)
    assert torch.equal(down, ref_dn)


# the watchers, and what unhooking one means


def test_an_in_place_write_is_visible_and_keeps_the_alias(layer):
    """The unified model's weight sync writes THROUGH the parameters (``dest.copy_``): the buffer
    receives the fresh bytes at the same address, which is also what a captured graph requires."""
    bufs = P.pack_layer(layer)
    ptr = bufs["gate_up"].data_ptr()
    e0 = layer.held_experts()[0]
    with torch.no_grad():
        e0.gate.copy_(torch.full_like(e0.gate, -0.25))
    gate_up, _ = layer.expert_weight_views()
    assert P.check_layer(layer) == (True, "")
    assert gate_up.data_ptr() == ptr, "an in-place write must not cost the alias, nor move the buffer"
    assert torch.equal(gate_up[0, : e0.gate.shape[0]], e0.gate)


def test_a_parameter_object_replacement_invalidates(layer):
    assert P.pack_layer(layer) is not None
    e0 = layer.held_experts()[0]
    layer.__dict__["_ix_packed_at"] = PS.SYNC_EPOCH
    e0.up = nn.Parameter(torch.full_like(e0.up, 0.5))  # what a Parameter-replacing loader does
    ok, why = P.check_layer(layer)
    assert not ok and ("REPLACED" in why or "rebound" in why), why
    gate_up, _ = layer.expert_weight_views()
    assert torch.equal(gate_up, _stack_views(layer)[0])


def test_unhooking_a_watcher_invalidates(layer):
    """A watch that is silently gone is the state in which a LATER rebind would not be observed."""
    assert P.pack_layer(layer) is not None
    e0 = layer.held_experts()[0]
    e0.__dict__["_parameters"] = dict(e0._parameters)  # wholesale replacement: the dict watch is gone
    ok, why = P.check_layer(layer)
    assert not ok and "REPLACED wholesale" in why, why
    layer.__dict__["_ix_packed_at"] = PS.SYNC_EPOCH
    assert layer.expert_weight_views()[0].data_ptr() != getattr(layer, P._STATE_ATTR)["gate_up"].buf.data_ptr()


def test_a_reverted_parameter_class_invalidates(layer):
    assert P.pack_layer(layer) is not None
    e0 = layer.held_experts()[0]
    e0.gate.__class__ = e0.gate.__class__._ix_unwatched  # the .data watch is gone
    ok, why = P.check_layer(layer)
    assert not ok and "__class__" in why, why


def test_packing_is_refused_during_a_cuda_graph_capture(layer, monkeypatch):
    """A pack allocates and COPIES, and the copy reads storage the rebind then frees.

    Captured, every replay would read that freed memory. vLLM's profile and warmup runs mean the
    capture is never a layer's first forward in practice; the refusal does not depend on that.
    """
    calls = []
    monkeypatch.setattr(P, "_capturing", lambda dev: (calls.append(dev), True)[1])
    assert P.pack_layer(layer) is None
    assert calls, "the capture guard was not consulted"
    assert getattr(layer, P._STATE_ATTR, None) is None
    monkeypatch.undo()
    assert P.pack_layer(layer) is not None


def test_the_capture_predicate_is_free_off_cuda(layer, monkeypatch):
    """It must not initialize CUDA to answer for a CPU parameter."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: pytest.fail("queried CUDA for a CPU tensor"))
    assert P._capturing(torch.device("cpu")) is False


def test_the_sync_boundary_refresh_never_moves_a_buffer(layer):
    """A CUDA-graph decode replay reads the capture-time address; a refresh must copy, not reallocate."""
    bufs = P.pack_layer(layer)
    ptrs = {k: v.data_ptr() for k, v in bufs.items()}
    allocs = P.stats()["buffer_allocs"]
    assert P.refresh_all_packed() >= 1
    after = getattr(layer, P._STATE_ATTR)
    assert {k: after[k].buf.data_ptr() for k in ptrs} == ptrs
    assert P.stats()["buffer_allocs"] == allocs, "the refresh reallocated a buffer a graph may point at"


# identity: nothing downstream may notice


def test_parameter_identity_names_and_count_survive(model, layer):
    before_objs = {n: id(p) for n, p in model.named_parameters()}
    before_shapes = {n: tuple(p.shape) for n, p in model.state_dict().items()}
    assert P.pack_layer(layer) is not None
    after_objs = {n: id(p) for n, p in model.named_parameters()}
    assert after_objs == before_objs, "packing must preserve Parameter OBJECT identity and every name"
    assert {n: tuple(p.shape) for n, p in model.state_dict().items()} == before_shapes
    assert len(list(model.parameters())) == len(before_objs)
    for e in layer.held_experts():
        assert e.gate.requires_grad and isinstance(e.gate, nn.Parameter)


def test_unpack_restores_private_storage(layer):
    ref_gu, _ = _stack_views(layer)
    assert P.pack_layer(layer) is not None
    assert P.unpack_layer(layer) is True
    assert getattr(layer, P._STATE_ATTR, None) is None
    gate_up, _ = _stack_views(layer)
    assert torch.equal(gate_up, ref_gu)
    seen = {p.untyped_storage().data_ptr() for e in layer.held_experts() for p in (e.gate, e.up, e.down)}
    assert len(seen) == 3 * len(layer.held_experts()), "unpack must give every parameter its own storage"


def test_ep_range_scopes_the_buffers(layer):
    """Pack only registered experts in the held EP range, and invalidate on later range changes.
    The fixture removes unowned module entries as shard_meta does."""
    n = len(layer.experts)
    half = n // 2
    for key in [k for k in list(layer.experts._modules) if int(k) >= half]:
        del layer.experts._modules[key]
    layer.ep_range = (0, half)
    bufs = P.pack_layer(layer)
    assert bufs is not None and bufs["gate_up"].shape[0] == half
    assert torch.equal(layer.expert_weight_views()[0], _stack_views(layer)[0])
    layer.ep_range = (half, n)
    assert P.packed_expert_weights(layer) is None, "a changed expert range must invalidate the buffers"
    assert torch.equal(layer.expert_weight_views()[0], _stack_views(layer)[0])


def test_packed_buffers_are_owned_nonpersistent_aliases(layer):
    packed = P.pack_layer(layer)
    assert packed is not None
    buffers = dict(layer.named_buffers())
    for role, value in packed.items():
        name = f"_isoexec_packed_{role}"
        assert buffers[name] is value
        assert name not in layer.state_dict()
        assert getattr(layer, P._STATE_ATTR)[role].buf is value
    assert P.unpack_layer(layer)
    assert not any(n.startswith("_isoexec_packed_") for n, _ in layer.named_buffers())


def test_sync_refuses_an_expert_rebound_into_foreign_storage(layer):
    assert P.pack_layer(layer) is not None
    param = layer.held_experts()[0].gate
    bucket = torch.empty(param.numel() + 1, dtype=param.dtype, device=param.device)
    param.data = bucket[: param.numel()].view(param.shape)
    with pytest.raises(RuntimeError, match="refresh failed"):
        P.refresh_all_packed()


def test_public_sync_boundary_refreshes_packed_buffers(layer):
    from isoexec.ops.moe import bump_sync_epoch

    assert bump_sync_epoch is PS.bump_sync_epoch
    assert P._ps is PS
    assert P.refresh_all_packed in PS._SYNC_HOOKS
    assert P.pack_layer(layer) is not None
    before = PS.SYNC_EPOCH
    assert bump_sync_epoch() == before + 1
    assert PS.SYNC_EPOCH == before + 1


def test_packed_buffers_use_the_injected_allocator(layer):
    """Engine sleep must own packed allocations; trainers retain the default allocator."""
    seen = []

    def recording(shape, dtype, device):
        seen.append(tuple(shape))
        return torch.empty(shape, dtype=dtype, device=device)

    PS.set_buffer_allocator(recording)
    try:
        assert P.pack_layer(layer, force=True) is not None
        assert seen, "packed weights bypassed the engine's allocator"
    finally:
        PS.set_buffer_allocator(None)
    assert PS.alloc_param_buffer((2, 3), torch.float32, torch.device("cpu")).shape == (2, 3)


def test_an_unattached_layer_still_produces_the_same_views():
    """The fallback is EXACT, which is why a missed build path is silent: only the launch count
    shows it."""
    m = _Layers()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g).mul_(0.05).to(p.dtype))
    layer = m.layers[0]
    assert layer.expert_views is None
    stacked = layer.expert_weight_views()
    try:
        assert P.attach(m) == len(m.layers)
        packed = layer.expert_weight_views()
        for a, b in zip(stacked, packed):
            assert torch.equal(a, b), "the packed view must be bit-identical to the stack it replaces"
    finally:
        P.release_to_storage_owner(m)
