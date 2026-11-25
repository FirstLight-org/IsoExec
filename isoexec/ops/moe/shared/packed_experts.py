"""Free expert-weight packing for the unified model's ``common/moe/layer.py``.

It lives in ``ops/moe/shared`` because it is not model math and no kernel package owns it: every
expert package consumes the same ``[E, 2I, H]`` / ``[E, H, I]`` layout. It decides how a layer's expert parameters are LAID OUT in memory and it shares its epoch / verify /
refresh machinery with packed projections through ``param_storage``. ``common/`` may not import ``ops/`` (lint L2), so the
dependency is inverted: :func:`attach` installs a closure on each ``MoE`` module's ``expert_views``
hook, and a layer with no hook installed stacks -- correct, just slower.

``MoE`` hands ``moe.expert_mlp`` two bind-time PHYSICAL views per forward::

    gate_up[e] = cat(gate_e, up_e)   [E, 2I, H]
    down[e]    = down_e              [E, H, I]

built as ``torch.stack([torch.cat(...) for e in experts])``. Those are not views, they are full
copies: 2E+2 kernel launches and a full re-materialization of the layer's expert weights on EVERY
forward (at the 35B production shape, 258 launches and ~200 MiB written per layer per forward).

The epoch, mutation watches and refresh callbacks come from ``param_storage.py``,
shared with packed projections. One mutation epoch and sync boundary cover all
aliased parameter buffers.

The layout is the target one: ONE contiguous ``[E, 2I, H]`` buffer with ``gate_e = buf[e, :I]`` and
``up_e = buf[e, I:]``, and one ``[E, H, I]`` buffer with ``down_e = buf[e]``. Every ``_Expert``'s
parameter OBJECT is preserved (only ``.data`` is repointed), so ``named_parameters()``, the logical
vocabulary names, optimizer state keyed on parameter identity and ``requires_grad`` all survive.
``gate_up_view()`` then returns the buffer itself.

A41 -- "DDP replaces ``param.data``, so a cached stack would alias stale storage" -- is answered in
three places, none of which is a cache:

  * The pack is REFUSED outright unless every expert parameter privately owns its whole storage
    (``storage_offset == 0`` and ``storage.nbytes() == numel * itemsize``, plain strided
    ``torch.Tensor`` data).  A parameter that DDP, a distributed optimizer or FSDP has already
    re-pointed into a flat bucket is a view into a LARGER storage, so it fails this test and the
    layer stays on ``torch.stack`` forever.  We never steal storage another owner handed out.
  * Once packed, every mutation that could break the aliasing is observable: Parameter OBJECT
    replacement (``WatchedParamDict``) and ``param.data = ...`` / ``set_`` / ``resize_`` /
    ``detach_`` (``DataWatch``) both bump ``ops/moe/shared/param_storage.py``'s single ``EPOCH``.  :func:`packed_expert_weights`
    verifies before every read -- O(1) while the epoch is unchanged, a full O(E) scan after any
    recorded mutation -- and returns ``None`` (-> exact ``torch.stack`` fallback) the moment the
    alias cannot be trusted.  A stale read is not reachable: the fallback reads the parameters.
  * At the weight-sync boundary the unified model writes THROUGH the parameters
    (``sync/weights.py::load_weights`` and the vLLM wrapper both do ``dest.copy_(t)``), so the
    packed buffer receives the fresh bytes in place, at the same address, with the alias intact --
    which is also what a captured CUDA-graph decode requires, since a replay reads the capture-time
    pointer.  :func:`refresh_all_packed` registers with ``param_storage.on_sync_boundary`` as belt
    and braces for a
    sync that replaces a parameter instead of writing into it, and it refuses to move a buffer.

Under grad the read routes through :class:`_PackedView`, which returns the buffer as a free alias and
in backward hands each expert parameter exactly the slice ``torch.cat``/``torch.stack`` would have --
same values, same order, so gradients are bitwise what the copy path produced.
"""

from __future__ import annotations

import logging
import os
import weakref
from typing import Sequence

import torch

from . import param_storage as _ps

logger = logging.getLogger(__name__)

_STATE_ATTR = "_isoexec_packed_experts"
#: Set on a layer that genuinely cannot be packed (foreign storage); it stacks from then on.
_GIVEUP_ATTR = "_ix_pack_giveup"

# Default ON. Off keeps the exact `torch.stack([torch.cat(...)])` path, byte for byte.
_ENV = "ISOEXEC_MOE_PACKED_EXPERTS"


def enabled() -> bool:
    return os.environ.get(_ENV, "1") == "1"


# Every layer with live packed buffers, so a sync boundary can refresh them all. WeakSet, so a torn
# down layer drops out with no bookkeeping.
_PACKED_LAYERS: "weakref.WeakSet" = weakref.WeakSet()
_ATTACHED: "weakref.WeakSet" = weakref.WeakSet()

VIOLATION_COUNT = 0
ALLOC_COUNT = 0
REUSE_COUNT = 0
REPACK_COUNT = 0
REPACK_SKIPPED = 0
REFUSED_FOREIGN_STORAGE = 0
_LOG_FIRST_N = 8


def stats() -> dict:
    return {
        "violations": VIOLATION_COUNT,
        "repacks": REPACK_COUNT,
        "repacks_rate_limited": REPACK_SKIPPED,
        "refused_foreign_storage": REFUSED_FOREIGN_STORAGE,
        "buffer_allocs": ALLOC_COUNT,
        "buffer_reuses": REUSE_COUNT,
        "layers": len(_PACKED_LAYERS),
    }


class _PackedView(torch.autograd.Function):
    """``buf`` in, ``buf`` out (free alias); backward scatters the grad onto the expert params.

    ``torch.stack``/``torch.cat``'s autograd behaviour without their forward copies. The params are
    inputs only so autograd records the edges -- their values are never read, because their storage
    IS ``buf``. ``per_slot`` is 1 (``down``) or 2 (``gate_up``: ``[:I]`` then ``[I:]``, the order
    ``cat(gate, up)`` concatenated them in).
    """

    @staticmethod
    def forward(ctx, buf, per_slot, *params):  # noqa: D102
        ctx.per_slot = per_slot
        # .view() (not the input object) so autograd owns a distinct output tensor, sharing storage.
        return buf.view(buf.shape)

    @staticmethod
    def backward(ctx, *grad_outputs):  # noqa: D102
        (grad_out,) = grad_outputs
        if ctx.per_slot == 1:
            return (None, None, *grad_out.unbind(0))
        half = grad_out.shape[1] // 2
        out = []
        for slot in grad_out.unbind(0):
            out.append(slot[:half])
            out.append(slot[half:])
        return (None, None, *out)


class _PackBuffer:
    """One packed buffer plus everything needed to re-verify the alias."""

    __slots__ = (
        "role",
        "layer",
        "buffer_name",
        "buf_obj",
        "params",
        "owners",
        "per_slot",
        "probes",
        "shapes",
        "dtype",
        "buf_ptr",
        "buf_bytes",
        "poisoned",
        "reason",
        "epoch",
    )

    def __init__(self, layer, role, buf, params, owners, per_slot, slots):
        self.role = role
        self.layer = weakref.ref(layer)
        self.buffer_name = f"_isoexec_packed_{role}"
        layer.register_buffer(self.buffer_name, buf, persistent=False)
        self.buf_obj = buf
        self.params = params  # flat list[nn.Parameter], slot-major then within-slot order
        self.owners = owners  # flat list[(module, attr)]
        self.per_slot = per_slot
        self.dtype = buf.dtype
        self.buf_ptr = buf.data_ptr()
        self.buf_bytes = buf.untyped_storage().nbytes()
        self.poisoned = False
        self.reason = ""
        self.shapes = [s.shape for s in slots]
        # (owning module's _parameters dict, attr, recorded Parameter, expected data_ptr, shape).
        self.probes = [
            (mod._parameters, attr, p, s.data_ptr(), s.shape) for (mod, attr), p, s in zip(owners, params, slots)
        ]
        self.epoch = _ps.EPOCH

    @property
    def buf(self) -> torch.Tensor:
        layer = self.layer()
        if layer is None or self.buffer_name not in layer._buffers:
            raise RuntimeError(f"packed expert buffer {self.buffer_name} has lost its owning module")
        buf = layer._buffers[self.buffer_name]
        if buf is None:
            raise RuntimeError(f"packed expert buffer {self.buffer_name} was released before use")
        return buf


class _Watch:
    """The layer-level half of the invariant: the mutation watchers themselves are still armed.

    ``WatchedParamDict`` observes Parameter OBJECT replacement and ``DataWatch`` observes
    ``.data`` rebinds; both bump the global epoch, which is what makes ``_PackBuffer``'s O(1) fast
    path sound. Neither can observe being REMOVED -- a module whose ``_parameters`` was replaced
    wholesale, or a parameter whose ``__class__`` was reverted, is silently unwatched from then on.
    So that is checked directly, on every read, and it is why this is per LAYER and deduplicated by
    module rather than per role: both roles ride the same E expert modules, so the steady state is
    one C-level pass over E dict ids and 3E parameter classes instead of two overlapping ones.
    """

    __slots__ = ("mod_dicts", "pdicts", "pdict_ids", "params", "classes")

    def __init__(self, mods, params):
        self.mod_dicts = [m.__dict__ for m in mods]
        # ``pdicts`` holds strong references on purpose: ``pdict_ids`` compares by id(), which would
        # be unsound if a watched dict could be freed and a fresh one land at the same address. The
        # comparison MUST be by identity -- two dicts with equal contents compare equal, so a
        # wholesale replacement by a copy is invisible to ``==``, and that replacement is precisely
        # the mutation this check exists to catch.
        self.pdicts = [m._parameters for m in mods]
        self.pdict_ids = [id(d) for d in self.pdicts]
        self.params = params
        self.classes = list(map(type, params))

    def ok(self) -> bool:
        return [id(d["_parameters"]) for d in self.mod_dicts] == self.pdict_ids and list(
            map(type, self.params)
        ) == self.classes

    def why(self) -> str:
        for i, (d, pd) in enumerate(zip(self.mod_dicts, self.pdicts)):
            if d.get("_parameters") is not pd:
                return f"expert {i}: module._parameters REPLACED wholesale (mutation watch lost)"
        for p, cls in zip(self.params, self.classes):
            if type(p) is not cls:
                return "a Parameter __class__ was reverted (.data watch lost)"
        return ""


def _slots(buf: torch.Tensor, per_slot: int) -> list[torch.Tensor]:
    """The destination view for each parameter, in the order the params are recorded."""
    if per_slot == 1:
        return [buf[i] for i in range(buf.shape[0])]
    half = buf.shape[1] // 2
    out = []
    for i in range(buf.shape[0]):
        out.append(buf[i, :half])
        out.append(buf[i, half:])
    return out


def _private_storage(p: torch.nn.Parameter) -> bool:
    """True iff ``p`` owns its whole storage -- nobody else has a claim on those bytes.

    This is the A41 guard. Megatron's DDP / distributed optimizer and FSDP hand a parameter a view
    into a flat multi-parameter bucket; packing there would sever THEIR aliasing, which is why
    the alias guard keeps the trainer on ``torch.stack``. A view into a larger storage, a
    non-zero storage offset, a tensor subclass (DTensor) or a non-strided layout all fail here.
    """
    d = p.data
    if type(d) is not torch.Tensor or d.layout is not torch.strided:
        return False
    if d.is_meta or d.device.type == "meta":
        return False
    if not d.is_contiguous() or d.storage_offset() != 0:
        return False
    try:
        return d.untyped_storage().nbytes() == d.numel() * d.element_size()
    except Exception:  # pragma: no cover - exotic storages
        return False


def _capturing(device: torch.device) -> bool:
    """Is a CUDA graph being captured on this device's stream right now?

    Split out so the refusal it guards is testable without a capture, and so the CUDA-init question
    is answered once: the query is skipped entirely for a non-CUDA parameter, and for a CUDA one CUDA
    is already initialized by definition.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    return bool(torch.cuda.is_current_stream_capturing())


def _in_buffer(p: torch.nn.Parameter, buf: torch.Tensor) -> bool:
    try:
        lo = buf.data_ptr()
        return lo <= p.data.data_ptr() < lo + buf.untyped_storage().nbytes()
    except Exception:  # pragma: no cover
        return False


def _full_check(pb: _PackBuffer) -> bool:
    """The alias invariant in full. Buffer storage first: if it moved (a cumem sleep), the per-param
    pointer comparisons would compare two addresses that both moved and could agree by accident."""
    buf = pb.buf
    if buf.data_ptr() != pb.buf_ptr or buf.untyped_storage().nbytes() != pb.buf_bytes:
        return False
    dtype = pb.dtype
    for pdict, attr, p, ptr, shape in pb.probes:
        cur = pdict.get(attr)
        if cur is not p or cur.data_ptr() != ptr or cur.shape != shape or cur.dtype is not dtype:
            return False
    return True


def _check(pb: _PackBuffer) -> bool:
    """Hot-path check: same verdict as :func:`_full_check`, much cheaper in the steady state.

    Call only with the layer's :class:`_Watch` already proven intact -- with the watchers armed, an
    unchanged global epoch proves no parameter was rebound since the last full verification.
    """
    buf = pb.buf
    if buf.data_ptr() != pb.buf_ptr or buf.untyped_storage().nbytes() != pb.buf_bytes:
        return False
    if pb.epoch == _ps.EPOCH:
        return True
    if not _full_check(pb):
        return False
    pb.epoch = _ps.EPOCH
    return True


def _explain(pb: _PackBuffer) -> str:
    buf = pb.buf
    if buf.data_ptr() != pb.buf_ptr:
        return "packed buffer moved (data_ptr changed)"
    if buf.untyped_storage().nbytes() != pb.buf_bytes:
        return "packed buffer storage resized/freed (cumem sleep?)"
    for i, (pdict, attr, p, ptr, shape) in enumerate(pb.probes):
        cur = pdict.get(attr)
        if cur is None:
            return f"expert {i // pb.per_slot}: parameter {attr!r} deleted"
        if cur is not p:
            return f"expert {i // pb.per_slot}: parameter {attr!r} object REPLACED"
        if cur.data_ptr() != ptr:
            return f"expert {i // pb.per_slot}: parameter {attr!r} .data rebound off the packed buffer"
        if cur.shape != shape:
            return f"expert {i // pb.per_slot}: parameter {attr!r} shape drifted {tuple(cur.shape)} != {tuple(shape)}"
        if cur.dtype is not pb.dtype:
            return f"expert {i // pb.per_slot}: parameter {attr!r} dtype drifted"
    return "no violation found (stale poison?)"


def _state(layer) -> dict | None:
    return getattr(layer, _STATE_ATTR, None)


_ROLES: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    # (role, params per expert slot, attribute names in slot order)
    ("gate_up", 2, ("gate", "up")),
    ("down", 1, ("down",)),
)


def materialize_meta_experts(model, device) -> int:
    """Allocate engine expert storage packed from birth, before loading any bytes.

    The caller owns the allocation context (including a runtime's weights pool)
    and must materialize remaining meta tensors without replacing these views.
    No existing parameter storage is stolen or copied. This avoids temporarily
    retaining a complete per-parameter copy in a sleepable allocator's old pool.
    """
    global ALLOC_COUNT
    if not enabled():
        return 0
    count = 0
    for layer in model.modules():
        if not (hasattr(layer, 'expert_views') and hasattr(layer, 'held_experts')):
            continue
        experts = layer.held_experts()
        if len(experts) < 2:
            continue
        plan = []
        for role, per_slot, attrs in _ROLES:
            owners = [(expert, attr) for expert in experts for attr in attrs]
            params = [owner._parameters[attr] for owner, attr in owners]
            ref = params[0]
            if any(not p.is_meta or p.shape != ref.shape or p.dtype != ref.dtype for p in params):
                raise ValueError('direct expert allocation requires uniform, entirely meta parameters')
            shape = (len(experts), per_slot * ref.shape[0], *ref.shape[1:])
            if per_slot == 1:
                shape = (len(experts), *ref.shape)
            plan.append((role, per_slot, owners, params, shape, ref.dtype))
        state = {'_ep_range': layer.ep_range}
        for role, per_slot, owners, previous, shape, dtype in plan:
            # Plain allocation deliberately inherits the caller's existing pool;
            # entering a second pluggable allocator context here is invalid.
            buf = torch.empty(shape, dtype=dtype, device=device)
            ALLOC_COUNT += 1
            slots = _slots(buf, per_slot)
            params = []
            for (owner, attr), old, slot in zip(owners, previous, slots):
                param = torch.nn.Parameter(slot, requires_grad=old.requires_grad)
                owner._parameters[attr] = param
                if not isinstance(owner._parameters, _ps.WatchedParamDict):
                    owner.__dict__['_parameters'] = _ps.WatchedParamDict(owner._parameters)
                param.__class__ = _ps.watched_class(param.__class__)
                params.append(param)
            state[role] = _PackBuffer(layer, role, buf, params, owners, per_slot, slots)
        state['_watch'] = _Watch(experts, [p for role, _, _ in _ROLES for p in state[role].params])
        setattr(layer, _STATE_ATTR, state)
        _PACKED_LAYERS.add(layer)
        count += 1
    return count


def pack_layer(layer, *, force: bool = False) -> dict[str, torch.Tensor] | None:
    """Pack ``layer``'s HELD experts into one ``[E,2I,H]`` and one ``[E,H,I]`` buffer.

    Idempotent: a second call on an already-packed, still-valid layer is a no-op returning the
    existing buffers. ``None`` means "cannot pack" -- fewer than two held experts, ragged
    shapes/dtypes/devices, meta parameters, or (the A41 guard) any parameter whose storage is not
    privately its own. The caller then keeps the exact ``torch.stack`` path.

    ``force`` re-packs from the CURRENT parameter values, reusing the existing buffers whenever they
    are intact and the right shape/dtype/device, so a layer allocates its buffers exactly once for
    the life of the process -- a CUDA-graph replay bakes the buffer address at capture time.
    """
    global ALLOC_COUNT, REUSE_COUNT, REFUSED_FOREIGN_STORAGE
    experts: Sequence = layer.held_experts()
    if len(experts) < 2:
        return None
    ep_range = layer.ep_range

    old_state = _state(layer)
    if old_state is not None and not force:
        if old_state.get("_ep_range") != ep_range or not old_state["_watch"].ok():
            return None
        out = {}
        for role, _, _ in _ROLES:
            pb = old_state.get(role)
            if pb is None or pb.poisoned:
                return None
            if not _check(pb):
                pb.poisoned, pb.reason = True, _explain(pb)
                return None
            out[role] = pb.buf
        return out

    # Pass 1: gather and validate every role BEFORE mutating anything, so a refusal on the second
    # role can never leave the first one half-repointed into an untracked buffer.
    plan = []
    for role, per_slot, attrs in _ROLES:
        params, owners = [], []
        for e in experts:
            for a in attrs:
                p = e._parameters.get(a)
                if p is None:
                    return None
                params.append(p)
                owners.append((e, a))

        ref = params[0].data
        if any(
            p.data.shape != ref.shape or p.data.dtype is not ref.dtype or p.data.device != ref.device for p in params
        ):
            logger.warning("[isoexec-moe-pack] ragged expert weights for role %s -- not packing", role)
            return None

        old = old_state.get(role) if old_state else None
        # A41: refuse if anything but us owns these bytes. On the re-pack path a parameter already
        # sitting inside OUR buffer is ours, not foreign.
        for p in params:
            if _private_storage(p):
                continue
            if old is not None and _in_buffer(p, old.buf):
                continue
            REFUSED_FOREIGN_STORAGE += 1
            logger.info(
                "[isoexec-moe-pack] %s/%s: expert parameter storage is not privately owned "
                "(DDP / distributed optimizer / FSDP bucket?) -- NOT packing; keeping torch.stack",
                getattr(layer, "ns", "?"),
                role,
            )
            return None
        plan.append((role, per_slot, params, owners, old, ref))

    # A pack ALLOCATES and COPIES. Both are captured if this forward is the capture, and the copies
    # read each parameter's pre-pack storage -- which the rebind then frees, so every replay would
    # read freed memory. vLLM's profile and warmup runs mean the capture is never a layer's first
    # forward in practice; refuse anyway rather than depend on that, and leave the layer on
    # torch.stack for this forward. (A capture that runs UNPACKED bakes a torch.stack temp the
    # sync-boundary refresh cannot reach, which is what the ACTIVE banner exists to make visible.)
    if _capturing(plan[0][5].device):
        logger.warning(
            "[isoexec-moe-pack] %s: refusing to pack DURING a CUDA-graph capture; the copy would be "
            "replayed against freed storage. This layer keeps torch.stack, and the captured graph "
            "will too.",
            getattr(layer, "ns", "?"),
        )
        return None

    # Pass 2: allocate (or reuse) and repoint.
    state: dict = {"_ep_range": ep_range}
    n = len(experts)
    for role, per_slot, params, owners, old, ref in plan:
        shape = (n, per_slot * ref.shape[0], *ref.shape[1:]) if per_slot > 1 else (n, *ref.shape)
        buf = None
        if (
            old is not None
            and old.buf.shape == shape
            and old.buf.dtype is ref.dtype
            and old.buf.device == ref.device
            and old.buf.data_ptr() == old.buf_ptr
            and old.buf.untyped_storage().nbytes() == old.buf_bytes
        ):
            buf = old.buf
            REUSE_COUNT += 1
        if buf is None:
            buf = _ps.alloc_param_buffer(shape, ref.dtype, ref.device)
            ALLOC_COUNT += 1

        slots = _slots(buf, per_slot)
        # Copy-then-rebind one parameter at a time: the rebind drops the last reference to that
        # parameter's old storage, so the transient peak is (buffer + one expert), not 2x.
        with torch.no_grad():
            for p, slot in zip(params, slots):
                cur = p.data
                if cur.data_ptr() != slot.data_ptr():
                    slot.copy_(cur)
                    p.data = slot
                elif cur.shape != slot.shape or cur.stride() != slot.stride() or cur.dtype is not slot.dtype:
                    # Same address, wrong view: the bytes are ours already, only the metadata is stale.
                    p.data = slot

        # Arm the mutation watch AFTER the rebinds -- each of them would bump the epoch otherwise.
        for (mod, _), p in zip(owners, params):
            d = mod.__dict__.get("_parameters")
            if not isinstance(d, _ps.WatchedParamDict):
                mod.__dict__["_parameters"] = _ps.WatchedParamDict(d)
            p.__class__ = _ps.watched_class(p.__class__)

        state[role] = _PackBuffer(layer, role, buf, params, owners, per_slot, slots)

    state["_watch"] = _Watch(experts, [p for role, _, _ in _ROLES for p in state[role].params])
    setattr(layer, _STATE_ATTR, state)
    _PACKED_LAYERS.add(layer)
    _report_first_pack(layer, state)
    return {role: state[role].buf for role, _, _ in _ROLES}


_REPORTED = [False]


def _report_first_pack(layer, state) -> None:
    """Say ONCE per process that the alias is live, and with what. Capture-time visibility: if the
    forward being CUDA-graph captured runs on the fallback instead, the graph bakes a ``torch.stack``
    temp that the sync-boundary refresh cannot reach, and this line is how that is noticed."""
    if _REPORTED[0]:
        return
    _REPORTED[0] = True
    gu, dn = state["gate_up"].buf, state["down"].buf
    mib = (gu.numel() * gu.element_size() + dn.numel() * dn.element_size()) / 2**20
    print(
        f"[ISOEXEC-MOE-PACK] packed expert weights ACTIVE on {getattr(layer, 'ns', '?')}: "
        f"gate_up{tuple(gu.shape)} + down{tuple(dn.shape)} = {mib:.1f} MiB, "
        f"{len(state['_watch'].params)} parameters aliased onto them; the per-forward "
        f"stack/cat of the expert weights is now free.",
        flush=True,
    )


def packed_expert_weights(layer) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The hot-path read: ``(gate_up[E,2I,H], down[E,H,I])``, or ``None`` to fall back.

    ``None`` always means "use ``torch.stack([torch.cat(...)])``", which is always correct. Under
    grad the result is autograd-connected to the expert parameters (see :class:`_PackedView`), so
    expert weight gradients land exactly where ``cat``/``stack`` put them.
    """
    global VIOLATION_COUNT
    state = _state(layer)
    if state is None:
        return None
    if state.get("_ep_range") != layer.ep_range:
        return None
    watch_ok = state["_watch"].ok()
    out = []
    for role, _, _ in _ROLES:
        pb = state.get(role)
        if pb is None or pb.poisoned:
            return None
        if not (watch_ok and _check(pb)):
            why = state["_watch"].why() or _explain(pb)
            pb.poisoned, pb.reason = True, why
            VIOLATION_COUNT += 1
            if VIOLATION_COUNT <= _LOG_FIRST_N:
                print(
                    f"[ISOEXEC-MOE-PACK] packed expert weights INVALIDATED for role={role} on "
                    f"{getattr(layer, 'ns', '?')}: {why}. Falling back to torch.stack for this layer "
                    f"(re-pack is rate-limited to one per weight sync). violation #{VIOLATION_COUNT}",
                    flush=True,
                )
            return None
        out.append(pb)
    if not torch.is_grad_enabled() or not any(p.requires_grad for pb in out for p in pb.params):
        return out[0].buf, out[1].buf
    return tuple(_PackedView.apply(pb.buf, pb.per_slot, *pb.params) for pb in out)


def repack_if_synced(layer) -> dict[str, torch.Tensor] | None:
    """:func:`pack_layer` (force), at most once per :data:`param_storage.SYNC_EPOCH` per layer.

    ``None`` means "not this time" as well as "cannot pack" -- both leave the caller on
    ``torch.stack``, which is always correct. Rate-limited because a re-pack is far more expensive
    than the stack it replaces: doing it per forward would be a regression, not an optimisation.
    """
    global REPACK_COUNT, REPACK_SKIPPED
    if layer.__dict__.get("_ix_packed_at") == _ps.SYNC_EPOCH and _state(layer) is not None:
        REPACK_SKIPPED += 1
        return None
    layer.__dict__["_ix_packed_at"] = _ps.SYNC_EPOCH
    REPACK_COUNT += 1
    return pack_layer(layer, force=True)


def _refresh_packed_layers(layers) -> int:
    """Refresh the selected packed layer objects without changing buffer addresses.

    The unified model's weight sync writes THROUGH the parameters (``dest.copy_``), so the buffers
    already hold the fresh bytes and this is a no-op verification. It exists for a sync that replaces
    a Parameter instead: a CUDA-graph decode replay reads the address baked at capture time, so a
    refresh must copy fresh bytes into the SAME buffer. A refresh that would move or reallocate a
    buffer raises rather than swapping the pointer under a live graph.
    """
    n = 0
    for layer in layers:
        state = _state(layer)
        if state is None:
            continue
        layer.__dict__["_ix_packed_at"] = _ps.SYNC_EPOCH
        previous = {role: state[role].buf for role, _, _ in _ROLES}
        for role, buf in previous.items():
            recorded = state[role]
            if (buf is not recorded.buf_obj or buf.data_ptr() != recorded.buf_ptr
                    or buf.untyped_storage().nbytes() != recorded.buf_bytes
                    or buf.dtype is not recorded.dtype or not buf.is_contiguous()):
                raise RuntimeError(f"[isoexec-moe-pack] GRAPH-UNSAFE eager refresh: captured buffer "
                                   f"role={role} was replaced or changed on {getattr(layer, 'ns', '?')}")
        # Keep each old Tensor and allocation alive until this owner's comparison.
        previous_storage = {role: buf.untyped_storage() for role, buf in previous.items()}
        pre = {role: _ps.buffer_signature(buf) for role, buf in previous.items()}
        out = pack_layer(layer, force=True)
        if out is None:
            raise RuntimeError(
                f"[isoexec-moe-pack] refresh failed for {getattr(layer, 'site', None) or type(layer).__name__}: pack_layer returned None at the sync "
                "boundary; captured decode cannot read stale expert weights"
            )
        for role, buf in out.items():
            if (buf is not previous.get(role) or _ps.buffer_signature(buf) != pre.get(role)
                    or buf.untyped_storage() is not previous_storage.get(role)):
                raise RuntimeError(
                    f"[isoexec-moe-pack] GRAPH-UNSAFE eager refresh: packed buffer role={role} was "
                    f"REALLOCATED or its address / layout changed on {getattr(layer, 'ns', '?')}. "
                    "A captured decode graph requires the original buffer object, storage, shape, "
                    "dtype, device and strides."
                )
        n += 1
    return n


@_ps.on_sync_boundary
def refresh_all_packed() -> int:
    """Refresh every live packed layer for legacy process-wide sync boundaries."""
    return _refresh_packed_layers(list(_PACKED_LAYERS))


@_ps.on_scoped_sync_boundary
def refresh_packed_for(root: torch.nn.Module) -> int:
    """Refresh packed layers owned by ``root`` and leave other models untouched."""
    return _refresh_packed_layers(layer for layer in root.modules() if layer in _PACKED_LAYERS)


def check_layer(layer) -> tuple[bool, str]:
    """Diagnostics: the validity verdict WITHOUT poisoning or logging."""
    state = _state(layer)
    if state is None:
        return False, "not packed"
    if state.get("_ep_range") != layer.ep_range:
        return False, "ep_range changed since packing"
    why = state["_watch"].why()
    if why:
        return False, why
    for role, _, _ in _ROLES:
        pb = state.get(role)
        if pb is None:
            return False, f"role {role} not packed"
        if pb.poisoned:
            return False, f"{role}: poisoned: {pb.reason}"
        if not _check(pb):
            return False, f"{role}: {_explain(pb)}"
    return True, ""


def unpack_layer(layer) -> bool:
    """Give every expert parameter back its own storage (a clone of its slice) and drop the buffers.

    Costs one full copy of the expert weights and transiently 2x their memory; teardown / A-B only.
    """
    state = _state(layer)
    if state is None:
        return False
    with torch.no_grad():
        for role, _, _ in _ROLES:
            pb = state.get(role)
            if pb is None:
                continue
            for (mod, attr), p in zip(pb.owners, pb.params):
                cur = mod._parameters.get(attr)
                if cur is None:
                    continue
                cur.data = cur.data.clone()
                cur.__class__ = getattr(cur.__class__, "_ix_unwatched", cur.__class__)
                d = mod.__dict__.get("_parameters")
                if isinstance(d, _ps.WatchedParamDict):
                    mod.__dict__["_parameters"] = dict(d)
    for role, _, _ in _ROLES:
        pb = state.get(role)
        if pb is not None:
            delattr(layer, pb.buffer_name)
    delattr(layer, _STATE_ATTR)
    _PACKED_LAYERS.discard(layer)
    return True


def _views_for(layer) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The free alias for one layer, packing it on first use. ``None`` -> the caller stacks.

    Re-packing is rate-limited to one attempt per weight sync (:func:`repack_if_synced`): a re-pack
    copies the whole expert weight set, so doing it per forward would be slower than the stack it
    replaces. ``_ix_pack_giveup`` is set only when the layer genuinely cannot be packed -- most
    importantly when a parameter's storage belongs to someone else (a DDP / distributed-optimizer
    bucket), where repointing it would sever THEIR aliasing.
    """
    if layer.__dict__.get(_GIVEUP_ATTR) or not enabled():
        return None
    w = packed_expert_weights(layer)
    if w is None:
        # First forward, or the alias was broken by whatever ran since the last one. Re-pack from the
        # CURRENT parameter values (so this can never resurrect stale bytes) and retry once.
        if repack_if_synced(layer) is None:
            if _state(layer) is None:
                layer.__dict__[_GIVEUP_ATTR] = True
        else:
            w = packed_expert_weights(layer)
    return w


@_ps.on_packing_prepare
def prepare_packed_for(root: torch.nn.Module) -> None:
    """Initialize attached owned layers before a native storage snapshot."""
    if not enabled():
        return
    with torch.no_grad():
        for layer in root.modules():
            if layer not in _ATTACHED or layer.expert_views is None:
                continue
            parameter = next(layer.parameters(), None)
            if parameter is not None and _capturing(parameter.device):
                raise RuntimeError("Expert packing preparation must run outside CUDA graph capture")
            if _views_for(layer) is None and _state(layer) is not None:
                layer.__dict__["_ix_packed_at"] = _ps.SYNC_EPOCH
                if pack_layer(layer, force=True) is None:
                    # DDP already owns these parameters. Dropping unused packing
                    # must not clone/rebind them through unpack_layer().
                    state = _state(layer)
                    assert state is not None
                    for role, _, _ in _ROLES:
                        packed = state.get(role)
                        if packed is not None:
                            layer._buffers.pop(packed.buffer_name, None)
                    delattr(layer, _STATE_ATTR)
                    _PACKED_LAYERS.discard(layer)
                    layer.__dict__[_GIVEUP_ATTR] = True


def release_to_storage_owner(model) -> None:
    """Retire expert aliases before a sharding host owns their parameter storage."""
    for mod in model.modules():
        if hasattr(mod, "expert_views") and hasattr(mod, "held_experts"):
            unpack_layer(mod)
            mod.expert_views = None
            _ATTACHED.discard(mod)


def attach(model) -> int:
    """Install the packing hook on every mixture layer of ``model``; returns how many were attached.

    Every build path calls this once, after the parameters exist and after any sharding has re-pointed
    them (``storage.packing.attach_packed_experts``, called by the runtimes after materialization). A layer whose hook was never installed reads ``expert_views is None`` and
    takes the exact ``torch.stack`` path -- the same bits, more launches -- so a missed build path is a
    performance bug the ``bench_step`` pack stats catch, never a correctness one.

    Duck-typed on purpose: this module names ``held_experts`` / ``ep_range`` / ``experts`` and nothing
    else about the layer, and it imports nothing from ``common/``.
    """
    n = 0
    for mod in model.modules():
        if hasattr(mod, "expert_views") and hasattr(mod, "held_experts"):
            mod.expert_views = lambda layer=mod: _views_for(layer)
            _ATTACHED.add(mod)
            n += 1
    return n


__all__ = [
    "enabled",
    "attach",
    "materialize_meta_experts",
    "release_to_storage_owner",
    "pack_layer",
    "packed_expert_weights",
    "repack_if_synced",
    "refresh_all_packed",
    "refresh_packed_for",
    "prepare_packed_for",
    "check_layer",
    "unpack_layer",
    "stats",
]
