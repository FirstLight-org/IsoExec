"""Free packing for the COLUMN-parallel projection weights the flat kinds hold as separate params.

It lives HERE, beside the GEMM that consumes the buffer it builds, for the reason
``ops/moe/shared/packed_experts.py`` lives in the MoE family: it is not model math, it decides how
a module's projection parameters are LAID OUT in memory. ``common/`` may not import ``ops/``
(lint L2), so the dependency is inverted exactly as it is there -- :func:`attach` installs a closure
on each module's ``packed_proj`` hook, and a module with no hook installed takes the unfused path
(``torch.cat`` of the weights, or several GEMMs), which is correct, just slower.

Three sites use it:

    ``GDN.in_proj``        q | k | v | z | beta | alpha   -- ONE GEMM already; the operand was built
                                                             by a literal ``torch.cat`` PER FORWARD
    ``Attention.qkv``      q | k | v | gate               -- four GEMMs before

The GDN case was pure loss: ``common/linear.py::packed_view`` re-materialized the six weight blocks
on every forward (measured at the 9B: 24 ``aten::cat`` launches and 2316 MiB written per forward).
The attention case buys a PROVIDER: the split q/gate (N=4096) and k/v (N=1024) widths are in no
census entry and therefore ran vLLM's Triton ``matmul_persistent``, while the fused
``attn.qkv (4096, 10240)`` got a faster provider. Measured 9B forward: -1.82 ms. On sm90 the
segment-tree GEMM (``ops/mm/mm_sm90``) now serves split and packed widths alike.

``Mlp.fc1`` (gate | up) is packed where it pays, and only there. The GEMM is column-separable, so the
packed and split forms give the same bits; the question is cost. On a training host (9B, 3584 tokens) it
was a REGRESSION: the GEMM gained nothing (2 x N=12288 = 0.958 ms vs 1 x N=24576 = 1.018 ms) while the
activation read two STRIDED halves of one [T, 2I] row (``silu_mul_megatron`` 0.214 -> 0.354 ms), for
+2.50 ms forward and +10.45 ms forward+backward. A wrapped trainer (DDP buckets, FSDP) never packs
anyway: its parameters live in foreign storage and the pack gives up. On the engine the strided rows
are read by ``silu_mul_rows`` at contiguous cost, so one GEMM launch instead of two is a net decode win
there. The hook contract below
is generic, so a site whose split width is genuinely uncovered can be added by declaring two
methods -- but a site must be MEASURED before it is declared, not derived from the census's coverage
banner, which reports the census's shapes rather than the ones the model issues (section 6.5).

The mechanism is ``packed_experts``', not a variant of it
=========================================================

One contiguous ``[sum(N_i), H]`` buffer per site; every member parameter's ``.data`` is REPOINTED at
its row block of that buffer. The parameter OBJECTS are preserved, so ``named_parameters()``, the
logical vocabulary names, ``requires_grad`` and optimizer state keyed on parameter identity all
survive; the bytes are copied once, at pack time, and never again. ``packed_weight()`` then returns
the buffer itself and the per-forward ``cat`` disappears rather than being cached.

An ALIAS, deliberately, and not a cached COPY. A cache of the packed bytes would have to be
invalidated whenever a member's VALUE changes, and there is no sound key for that:

    >>> p = nn.Parameter(torch.zeros(4, 4)); p._version
    0
    >>> p.data.copy_(fresh); p._version          # the weight-sync spelling, sync/weights.py
    0

``param.data.copy_(t)`` -- which is how ``sync/weights.py::load_weights`` and the vLLM
wrapper deliver fresh weights -- moves neither the data pointer nor ``_version``, so a copy keyed on
``(data_ptr, _version)`` would serve STALE weights after every sync. (``ops/moe/shared/param_storage.py``
records the same fact from the other side: ``_version`` does not bump on a ``.data`` rebind either.)
An alias has no staleness class at all: the parameter's bytes ARE the buffer, so a write anywhere --
weight sync, in-place optimizer step, a bucket owner writing through its own view -- is seen with no
bookkeeping. It also costs no memory, where a copy of these three sites at the 9B shape would be
+9.3 GiB against ~18 GiB of weights. The evidence script is
the packed-projection version-counter regression tests.

A41 -- "DDP replaces ``param.data``, so a cached view would alias stale storage" -- is answered the
way ``packed_experts`` answers it, in the same three places, with the same shared bookkeeping from
``ops/moe/shared/param_storage.py`` (one mutation epoch, one sync boundary, one buffer allocator):

  * The pack is REFUSED unless every member parameter privately owns its whole storage. A parameter
    that Megatron's DDP, its distributed optimizer or FSDP has already re-pointed into a flat bucket
    is a view into a LARGER storage and fails that test, and the site stays unfused forever
    (:data:`_GIVEUP_ATTR`). We never steal storage another owner handed out. This is why the win is
    NOT realized on a Megatron-DDP trainer -- the same limitation the packed-expert alias guard exists to
    work around for mcore's own layout, and ``smoke_worker_isoexec`` at TP=4 is the gate that
    exercises the give-up path.
  * Once packed, every mutation that could break the aliasing is observable: Parameter OBJECT
    replacement (``WatchedParamDict``) and ``param.data = ...`` / ``set_`` / ``resize_`` /
    ``detach_`` (``DataWatch``) both bump ``param_storage.EPOCH``. :func:`packed_weight` verifies
    before every read -- an integer compare while the epoch is unchanged, a full rescan after any
    recorded mutation -- and returns ``None`` (-> the exact unfused path) the moment the alias
    cannot be trusted. A stale read is not reachable: the fallback reads the parameters.
  * At the weight-sync boundary the model writes THROUGH the parameters, so the buffer receives the
    fresh bytes in place, at the same address, with the alias intact -- which is what a captured
    CUDA-graph decode requires, since a replay reads the capture-time pointer.
    :func:`refresh_all_packed` registers with ``param_storage.on_sync_boundary`` as belt and braces
    for a sync that REPLACES a parameter instead of writing into it, and it refuses to move a buffer.

Bit-neutrality
==============

Nothing here is arithmetic. The forward change is that ``mm`` reads one operand whose bytes are the
concatenation of the operands it read before, which is the ``mm`` region's own contract: fixed
``BLOCK_SIZE_K`` and no split-K, so an output element does not depend on which columns share its
launch; on sm90 the segment-tree GEMM's segmentation reads K alone (``mm_sm90/segments.py``), and
``mm_sm90/tests/test_mm_sm90_gpu.py`` checks packed against unpacked bit for bit. That is the
argument ``8ef83c9`` used for the GDN one-GEMM in_proj; ``ops/mm/tests/mm_column_slice_gpu.py``
proves the column-slice half on the live providers. On SM100 the same holds because the
``mm_sm100`` segment rule reads K alone (``mm_sm100/segments.py``, policy v4): the packed weight and
each member share K, so they share every output column's K reduction. Under grad the read routes through
:class:`_PackedProj`, which returns the buffer as a free alias and hands each member parameter
exactly the row block ``torch.cat`` would have -- same values, same order, so weight gradients are
bitwise what the cat path produced.

The hook contract
=================

Duck-typed, so this module imports nothing from ``common/``. A packable module declares

    ``packed_proj``        attribute, ``None`` until :func:`attach` installs a closure; the closure
                           returns the ``[sum(N_i), H]`` operand or ``None`` ("take the slow path")
    ``packed_proj_roles()``method returning the member parameter attribute names IN CONCAT ORDER,
                           or ``()`` for "do not pack this instance" (an ``Attention`` row whose
                           ``kv_gather`` is installed, say, where a gather sits between the GEMM and
                           the k/v consumers)
    ``packed_proj_name``   class attribute: the name the packed buffer is REGISTERED under on that
                           module (``in_proj_packed``, ``qkv_packed``, ``fc1_packed``)
"""

from __future__ import annotations

import logging
import weakref

import torch

from ..moe import param_storage as _ps

logger = logging.getLogger(__name__)

_STATE_ATTR = "_isoexec_packed_proj"
#: Set on a module that genuinely cannot be packed (foreign storage); it stays unfused from then on.
_GIVEUP_ATTR = "_ix_projpack_giveup"

# Every module with a live packed buffer, so a sync boundary can refresh them all. WeakSet, so a
# torn-down module drops out with no bookkeeping.
_PACKED: "weakref.WeakSet" = weakref.WeakSet()
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
        "sites": len(_PACKED),
    }


class _PackedProj(torch.autograd.Function):
    """``buf`` in, ``buf`` out (free alias); backward slices the grad back onto the member params.

    ``torch.cat``'s autograd behaviour without its forward copy. The params are inputs only so that
    autograd records the edges -- their values are never read here, because their storage IS ``buf``.
    ``offsets`` is the cumulative row offset of each member, so member ``i`` receives
    ``grad_out[offsets[i]:offsets[i+1]]``: exactly the block ``CatBackward0`` narrows out.
    """

    @staticmethod
    def forward(ctx, buf, offsets, *params):  # noqa: D102
        ctx.offsets = offsets
        # .view() (not the input object) so autograd owns a distinct output tensor sharing storage.
        return buf.view(buf.shape)

    @staticmethod
    def backward(ctx, *grad_outputs):  # noqa: D102
        (grad_out,) = grad_outputs
        off = ctx.offsets
        return (None, None, *(grad_out[off[i] : off[i + 1]] for i in range(len(off) - 1)))


class _Pack:
    """One packed buffer plus everything needed to re-verify the alias.

    The buffer is not held here: it is a REGISTERED non-persistent buffer on the owning module, and
    :attr:`buf` reads it back out of ``mod._buffers`` on every access. Design v4.1 section 2.2 --
    "every tensor the model owns is a Parameter or a registered buffer" -- and reading it back is
    also what makes ``nn.Module._apply`` (a ``.to()`` / ``to_empty()`` that would replace the buffer
    object under us) observable: :func:`_full_check` compares identity, not just the pointer.
    """

    __slots__ = ("site", "bufs", "bufname", "buf_obj", "params", "attrs", "pdict", "offsets", "dtype", "buf_ptr", "buf_bytes", "poisoned", "reason", "probes", "epoch")

    def __init__(self, site, bufs, bufname, params, attrs, pdict, offsets, slots):
        buf = bufs[bufname]
        self.site = site
        self.bufs = bufs  # the owning module's ``_buffers`` dict
        self.bufname = bufname
        self.buf_obj = buf
        self.params = params
        self.attrs = attrs
        self.pdict = pdict
        self.offsets = offsets
        self.dtype = buf.dtype
        self.buf_ptr = buf.data_ptr()
        self.buf_bytes = buf.untyped_storage().nbytes()
        self.poisoned = False
        self.reason = ""
        # (attr, recorded Parameter, expected data_ptr, shape).
        self.probes = [(a, p, s.data_ptr(), s.shape) for a, p, s in zip(attrs, params, slots)]
        self.epoch = _ps.EPOCH

    @property
    def buf(self):
        return self.bufs.get(self.bufname)


class _Watch:
    """The module-level half of the invariant: the mutation watchers themselves are still armed.

    ``WatchedParamDict`` observes Parameter OBJECT replacement and ``DataWatch`` observes ``.data``
    rebinds; both bump the global epoch, which is what makes :meth:`_Pack`'s O(1) fast path sound.
    Neither can observe being REMOVED -- a module whose ``_parameters`` was replaced wholesale, or a
    parameter whose ``__class__`` was reverted, is silently unwatched from then on -- so that is
    checked directly on every read.
    """

    __slots__ = ("mod_dict", "pdict", "pdict_id", "params", "classes")

    def __init__(self, mod, params):
        self.mod_dict = mod.__dict__
        # A strong reference on purpose: ``pdict_id`` compares by id(), which would be unsound if a
        # watched dict could be freed and a fresh one land at the same address. The comparison MUST
        # be by identity -- two dicts with equal contents compare equal, so a wholesale replacement
        # by a copy is invisible to ``==``, and that is precisely the mutation this check catches.
        self.pdict = mod._parameters
        self.pdict_id = id(self.pdict)
        self.params = params
        self.classes = list(map(type, params))

    def ok(self) -> bool:
        return id(self.mod_dict.get("_parameters")) == self.pdict_id and list(map(type, self.params)) == self.classes

    def why(self) -> str:
        if self.mod_dict.get("_parameters") is not self.pdict:
            return "module._parameters REPLACED wholesale (mutation watch lost)"
        for p, cls in zip(self.params, self.classes):
            if type(p) is not cls:
                return "a Parameter __class__ was reverted (.data watch lost)"
        return ""


def _private_storage(p: torch.nn.Parameter) -> bool:
    """True iff ``p`` owns its whole storage -- nobody else has a claim on those bytes.

    The A41 guard, verbatim from ``packed_experts``. Megatron's DDP / distributed optimizer and FSDP
    hand a parameter a view into a flat multi-parameter bucket; packing there would sever THEIR
    aliasing. A view into a larger storage, a non-zero storage offset, a tensor subclass (DTensor) or
    a non-strided layout all fail here.
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

    A pack ALLOCATES and COPIES; both would be captured, and the copies read each parameter's
    pre-pack storage, which the rebind then frees -- so every replay would read freed memory.
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


def _full_check(pk: _Pack) -> bool:
    """The alias invariant in full. Buffer storage first: if it moved (a cumem sleep), the per-param
    pointer comparisons would compare two addresses that both moved and could agree by accident."""
    buf = pk.buf
    if buf is not pk.buf_obj or buf.data_ptr() != pk.buf_ptr or buf.untyped_storage().nbytes() != pk.buf_bytes:
        return False
    dtype = pk.dtype
    for attr, p, ptr, shape in pk.probes:
        cur = pk.pdict.get(attr)
        if cur is not p or cur.data_ptr() != ptr or cur.shape != shape or cur.dtype is not dtype:
            return False
    return True


def _check(pk: _Pack) -> bool:
    """Hot-path check: same verdict as :func:`_full_check`, much cheaper in the steady state.

    Call only with the module's :class:`_Watch` already proven intact -- with the watchers armed, an
    unchanged global epoch proves no parameter was rebound since the last full verification.
    """
    buf = pk.buf
    if buf is not pk.buf_obj or buf.data_ptr() != pk.buf_ptr or buf.untyped_storage().nbytes() != pk.buf_bytes:
        return False
    if pk.epoch == _ps.EPOCH:
        return True
    if not _full_check(pk):
        return False
    pk.epoch = _ps.EPOCH
    return True


def _explain(pk: _Pack) -> str:
    buf = pk.buf
    if buf is None:
        return f"registered buffer {pk.bufname!r} was DEREGISTERED"
    if buf is not pk.buf_obj:
        return f"registered buffer {pk.bufname!r} was REPLACED (a .to() / to_empty() ran?)"
    if buf.data_ptr() != pk.buf_ptr:
        return "packed buffer moved (data_ptr changed)"
    if buf.untyped_storage().nbytes() != pk.buf_bytes:
        return "packed buffer storage resized/freed (cumem sleep?)"
    for attr, p, ptr, shape in pk.probes:
        cur = pk.pdict.get(attr)
        if cur is None:
            return f"parameter {attr!r} deleted"
        if cur is not p:
            return f"parameter {attr!r} object REPLACED"
        if cur.data_ptr() != ptr:
            return f"parameter {attr!r} .data rebound off the packed buffer"
        if cur.shape != shape:
            return f"parameter {attr!r} shape drifted {tuple(cur.shape)} != {tuple(shape)}"
        if cur.dtype is not pk.dtype:
            return f"parameter {attr!r} dtype drifted"
    return "no violation found (stale poison?)"


def _state(mod) -> _Pack | None:
    return getattr(mod, _STATE_ATTR, None)


def _roles(mod) -> tuple[str, ...]:
    fn = getattr(mod, "packed_proj_roles", None)
    return tuple(fn()) if fn is not None else ()


def pack_site(mod, *, force: bool = False) -> torch.Tensor | None:
    """Pack ``mod``'s declared projection members into one contiguous ``[sum(N_i), H]`` buffer.

    Idempotent: a second call on an already-packed, still-valid module is a no-op returning the
    existing buffer. ``None`` means "cannot pack" -- fewer than two members, a member that is not a
    2-D parameter, ragged K / dtype / device, a meta parameter, or (the A41 guard) any member whose
    storage is not privately its own. The caller then keeps the exact unfused path.

    ``force`` re-packs from the CURRENT parameter values, reusing the existing buffer whenever it is
    intact and the right shape/dtype/device, so a site allocates its buffer exactly once for the life
    of the process -- a CUDA-graph replay bakes the buffer address at capture time.
    """
    global ALLOC_COUNT, REUSE_COUNT, REFUSED_FOREIGN_STORAGE
    attrs = _roles(mod)
    if len(attrs) < 2:
        return None

    old = _state(mod)
    if old is not None and not force:
        if old.poisoned or old.attrs != list(attrs):
            return None
        watch = mod.__dict__.get("_ix_projpack_watch")
        if watch is None or not watch.ok() or not _check(old):
            old.poisoned, old.reason = True, _explain(old)
            return None
        return old.buf

    params, pdict = [], mod._parameters
    for a in attrs:
        p = pdict.get(a)
        if p is None or p.dim() != 2:
            return None
        params.append(p)

    ref = params[0].data
    if any(p.data.shape[1] != ref.shape[1] or p.data.dtype is not ref.dtype or p.data.device != ref.device for p in params):
        logger.warning("[isoexec-proj-pack] ragged projection members on %s -- not packing", getattr(mod, "ns", "?"))
        return None
    if any(p.data.is_meta or p.data.device.type == "meta" for p in params):
        return None

    # A41: refuse if anything but us owns these bytes. On the re-pack path a parameter already
    # sitting inside OUR buffer is ours, not foreign.
    for p in params:
        if _private_storage(p):
            continue
        if old is not None and _in_buffer(p, old.buf):
            continue
        REFUSED_FOREIGN_STORAGE += 1
        logger.info(
            "[isoexec-proj-pack] %s: projection parameter storage is not privately owned "
            "(DDP / distributed optimizer / FSDP bucket?) -- NOT packing; keeping the unfused path",
            getattr(mod, "ns", "?"),
        )
        return None

    if _capturing(ref.device):
        logger.warning(
            "[isoexec-proj-pack] %s: refusing to pack DURING a CUDA-graph capture; the copy would be "
            "replayed against freed storage. This site keeps the unfused path, and so will the graph.",
            getattr(mod, "ns", "?"),
        )
        return None

    rows = [int(p.data.shape[0]) for p in params]
    offsets = [0]
    for r in rows:
        offsets.append(offsets[-1] + r)
    shape = (offsets[-1], int(ref.shape[1]))

    bufname = mod.packed_proj_name
    prev = old.buf if old is not None else None
    buf = None
    if (
        old is not None
        and prev is not None
        and prev.shape == shape
        and prev.dtype is ref.dtype
        and prev.device == ref.device
        and prev.data_ptr() == old.buf_ptr
        and prev.untyped_storage().nbytes() == old.buf_bytes
    ):
        buf = prev
        REUSE_COUNT += 1
    if buf is None:
        # Through ``param_storage``'s allocator, so a vLLM worker gets it from the sleepable CuMem
        # pool (level-1 sleep must be able to offload these bytes) and the post-pack allocator
        # fragmentation is released at the next sync boundary. Explicit dtype / device, never
        # inferred from a global default.
        buf = _ps.alloc_param_buffer(shape, dtype=ref.dtype, device=ref.device)
        ALLOC_COUNT += 1
    # Design v4.1 section 2.2: every tensor the model owns is a Parameter or a REGISTERED buffer.
    # Non-persistent, so it never reaches a state_dict or a distributed checkpoint, and outside
    # the weight-sync allowlist, so it never rides a weight sync -- it holds the
    # parameters' own bytes, which ride it as parameters.
    if mod._buffers.get(bufname) is not buf:
        if bufname in mod._buffers:
            del mod._buffers[bufname]
        mod.register_buffer(bufname, buf, persistent=False)

    slots = [buf[offsets[i] : offsets[i + 1]] for i in range(len(rows))]
    # Copy-then-rebind one parameter at a time: the rebind drops the last reference to that
    # parameter's old storage, so the transient peak is (buffer + one member), not 2x.
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
    d = mod.__dict__.get("_parameters")
    if not isinstance(d, _ps.WatchedParamDict):
        mod.__dict__["_parameters"] = _ps.WatchedParamDict(d)
    pdict = mod._parameters
    for p in params:
        p.__class__ = _ps.watched_class(p.__class__)

    pk = _Pack(f"{getattr(mod, 'ns', '?')}.{bufname}", mod._buffers, bufname, params, list(attrs), pdict, offsets, slots)
    setattr(mod, _STATE_ATTR, pk)
    mod.__dict__["_ix_projpack_watch"] = _Watch(mod, params)
    _PACKED.add(mod)
    _report_first_pack(mod, pk)
    return buf


_REPORTED = [False]


def _report_first_pack(mod, pk) -> None:
    """Say ONCE per process that the alias is live, and with what. Capture-time visibility: if the
    forward being CUDA-graph captured runs on the fallback instead, the graph bakes an unfused
    operand the sync-boundary refresh cannot reach, and this line is how that is noticed."""
    if _REPORTED[0]:
        return
    _REPORTED[0] = True
    mib = pk.buf.numel() * pk.buf.element_size() / 2**20
    print(
        f"[ISOEXEC-PROJ-PACK] packed projection weights ACTIVE on {pk.site}: "
        f"{'|'.join(pk.attrs)} -> {tuple(pk.buf.shape)} = {mib:.1f} MiB, {len(pk.params)} parameters "
        f"aliased onto it; the per-forward cat / the split GEMMs at this site are now one operand.",
        flush=True,
    )


def packed_weight(mod) -> torch.Tensor | None:
    """The hot-path read: the ``[sum(N_i), H]`` operand, or ``None`` to fall back.

    ``None`` always means "take the unfused path", which is always correct. Under grad the result is
    autograd-connected to the member parameters (see :class:`_PackedProj`), so their gradients land
    exactly where ``cat`` put them.
    """
    global VIOLATION_COUNT
    pk = _state(mod)
    if pk is None or pk.poisoned:
        return None
    watch = mod.__dict__.get("_ix_projpack_watch")
    if watch is None or not watch.ok() or not _check(pk):
        why = (watch.why() if watch is not None else "mutation watch missing") or _explain(pk)
        pk.poisoned, pk.reason = True, why
        VIOLATION_COUNT += 1
        if VIOLATION_COUNT <= _LOG_FIRST_N:
            print(
                f"[ISOEXEC-PROJ-PACK] packed projection weights INVALIDATED on {pk.site}: {why}. "
                f"Falling back to the unfused path for this site (re-pack is rate-limited to one per "
                f"weight sync). violation #{VIOLATION_COUNT}",
                flush=True,
            )
        return None
    if not torch.is_grad_enabled() or not any(p.requires_grad for p in pk.params):
        return pk.buf
    return _PackedProj.apply(pk.buf, pk.offsets, *pk.params)


def repack_if_synced(mod) -> torch.Tensor | None:
    """:func:`pack_site` (force), at most once per :data:`param_storage.SYNC_EPOCH` per module.

    ``None`` means "not this time" as well as "cannot pack" -- both leave the caller unfused, which
    is always correct. Rate-limited because a re-pack copies the whole projection: doing it per
    forward would be a regression, not an optimisation.
    """
    global REPACK_COUNT, REPACK_SKIPPED
    if mod.__dict__.get("_ix_projpacked_at") == _ps.SYNC_EPOCH and _state(mod) is not None:
        REPACK_SKIPPED += 1
        return None
    mod.__dict__["_ix_projpacked_at"] = _ps.SYNC_EPOCH
    REPACK_COUNT += 1
    return pack_site(mod, force=True)


def _refresh_packed_sites(sites) -> int:
    """Refresh the selected packed module objects without changing buffer addresses.

    The model's weight sync writes THROUGH the parameters (``dest.copy_``), so the buffer already
    holds the fresh bytes and this is a no-op verification. It exists for a sync that REPLACES a
    Parameter instead: a CUDA-graph decode replay reads the address baked at capture time, so a
    refresh must copy fresh bytes into the SAME buffer. A refresh that would move or reallocate a
    buffer raises rather than swapping the pointer under a live graph.
    """
    n = 0
    for mod in sites:
        pk = _state(mod)
        if pk is None:
            continue
        mod.__dict__["_ix_projpacked_at"] = _ps.SYNC_EPOCH
        previous = pk.buf
        if (previous is not pk.buf_obj or previous.data_ptr() != pk.buf_ptr
                or previous.untyped_storage().nbytes() != pk.buf_bytes
                or previous.dtype is not pk.dtype or not previous.is_contiguous()):
            raise RuntimeError(f"[isoexec-proj-pack] GRAPH-UNSAFE eager refresh: the captured buffer "
                               f"was replaced or changed on {pk.site}")
        # Retain both objects across refresh so an address cannot be freed and
        # reused by an unrelated allocation while we compare this owner's buffer.
        previous_storage = previous.untyped_storage()
        pre = _ps.buffer_signature(previous)
        buf = pack_site(mod, force=True)
        if buf is None:
            raise RuntimeError(
                f"[isoexec-proj-pack] refresh refused for {pk.site}: projection parameters cannot "
                "be rebound to the captured buffer at the weight-sync boundary"
            )
        if (buf is not previous or _ps.buffer_signature(buf) != pre
                or buf.untyped_storage() is not previous_storage):
            raise RuntimeError(
                f"[isoexec-proj-pack] GRAPH-UNSAFE eager refresh: the packed buffer was REALLOCATED "
                f"or its address / layout changed on {pk.site}. A captured decode graph requires "
                "the original buffer object, storage, shape, dtype, device and strides."
            )
        n += 1
    return n


@_ps.on_sync_boundary
def refresh_all_packed() -> int:
    """Refresh every live packed site for legacy process-wide sync boundaries."""
    return _refresh_packed_sites(list(_PACKED))


@_ps.on_scoped_sync_boundary
def refresh_packed_for(root: torch.nn.Module) -> int:
    """Refresh packed sites owned by ``root`` and leave other models untouched."""
    return _refresh_packed_sites(mod for mod in root.modules() if mod in _PACKED)


def check_module(mod) -> tuple[bool, str]:
    """Diagnostics: the validity verdict WITHOUT poisoning or logging."""
    pk = _state(mod)
    if pk is None:
        return False, "not packed"
    if pk.poisoned:
        return False, f"poisoned: {pk.reason}"
    watch = mod.__dict__.get("_ix_projpack_watch")
    if watch is None:
        return False, "mutation watch missing"
    why = watch.why()
    if why:
        return False, why
    if not _check(pk):
        return False, _explain(pk)
    return True, ""


def unpack(mod) -> bool:
    """Give every member parameter back its own storage (a clone of its block) and drop the buffer.

    Costs one full copy of the projection and transiently 2x its memory; teardown / A-B only.
    """
    pk = _state(mod)
    if pk is None:
        return False
    with torch.no_grad():
        for attr in pk.attrs:
            cur = mod._parameters.get(attr)
            if cur is None:
                continue
            cur.data = cur.data.clone()
            cur.__class__ = getattr(cur.__class__, "_ix_unwatched", cur.__class__)
    d = mod.__dict__.get("_parameters")
    if isinstance(d, _ps.WatchedParamDict):
        mod.__dict__["_parameters"] = dict(d)
    mod._buffers.pop(pk.bufname, None)
    delattr(mod, _STATE_ATTR)
    mod.__dict__.pop("_ix_projpack_watch", None)
    _PACKED.discard(mod)
    return True


def _giveup(mod, why: str) -> None:
    """This site cannot be packed. Drop the state, DEREGISTER the buffer, and stop trying.

    Deregistering matters: a buffer left registered after the alias was severed is a dead copy of the
    projection that ``named_buffers()`` consumers would still carry -- the FSDP2 host broadcasts every
    buffer to the DP replicas, SGLang's release/resume clones every buffer, and both would be paying
    for bytes nothing reads. Any parameter still aliasing those bytes keeps the STORAGE alive on its
    own, so this frees memory exactly when it is safe to.
    """
    pk = _state(mod)
    if pk is not None:
        mod._buffers.pop(pk.bufname, None)
        try:
            delattr(mod, _STATE_ATTR)
        except AttributeError:  # pragma: no cover
            pass
    _PACKED.discard(mod)
    mod.__dict__[_GIVEUP_ATTR] = True
    logger.info("[isoexec-proj-pack] %s: giving up on packing -- %s", getattr(mod, "ns", "?"), why)


def _weight_for(mod) -> torch.Tensor | None:
    """The free alias for one site, packing it on first use. ``None`` -> the caller stays unfused.

    Packing is attempted at most ONCE per :data:`param_storage.SYNC_EPOCH` (the same rate limit
    :func:`repack_if_synced` applies, inlined here so the give-up decision can tell "refused" apart
    from "not this time"): a pack copies the whole projection, so doing it per forward would be
    slower than the cat it replaces. ``_ix_projpack_giveup`` is set only when the site genuinely
    cannot be packed -- most importantly when a member's storage belongs to someone else (a DDP /
    distributed-optimizer bucket), where repointing it would sever THEIR aliasing. A capture in
    progress is transient and does NOT give up.
    """
    global REPACK_COUNT, REPACK_SKIPPED
    if mod.__dict__.get(_GIVEUP_ATTR):
        return None
    w = packed_weight(mod)
    if w is not None:
        return w
    # First forward, or the alias was broken by whatever ran since the last one.
    if _state(mod) is not None and mod.__dict__.get("_ix_projpacked_at") == _ps.SYNC_EPOCH:
        REPACK_SKIPPED += 1  # already tried at this sync epoch; unfused this forward, retry after the next
        return None
    if _capturing(_device_of(mod)):
        return None  # transient, and pack_site would refuse anyway -- do NOT give up on it
    mod.__dict__["_ix_projpacked_at"] = _ps.SYNC_EPOCH
    REPACK_COUNT += 1
    # Re-pack from the CURRENT parameter values, so this can never resurrect stale bytes.
    if pack_site(mod, force=True) is None:
        _giveup(mod, "pack_site refused (foreign storage, ragged members, or a meta parameter)")
        return None
    return packed_weight(mod)


def _device_of(mod):
    for p in mod._parameters.values():
        if p is not None:
            return p.data.device
    return torch.device("cpu")


@_ps.on_packing_prepare
def prepare_packed_for(root: torch.nn.Module) -> None:
    """Initialize only this owner's attached sites before native admission.

    A host may have rebound previously packed parameters into DDP storage. Retire
    our stale buffer without moving that host's parameters, retaining the normal
    unfused fallback. Legacy first-forward packing remains lazy.
    """
    with torch.no_grad():
        for mod in root.modules():
            if mod not in _ATTACHED or mod.packed_proj is None:
                continue
            if _capturing(_device_of(mod)):
                raise RuntimeError("Projection packing preparation must run outside CUDA graph capture")
            if _weight_for(mod) is None and _state(mod) is not None:
                mod.__dict__["_ix_projpacked_at"] = _ps.SYNC_EPOCH
                if pack_site(mod, force=True) is None:
                    _giveup(mod, "native admission found foreign or unsupported projection storage")


def release_to_storage_owner(model) -> None:
    """Retire aliases before a sharding host takes exclusive parameter storage ownership.

    FSDP presents temporary full parameters during forward. Those may look like
    standalone storage, but rebinding them would detach the host's copy-out buffer.
    This transition runs before wrapping, never on the forward or update path.
    """
    for mod in model.modules():
        if hasattr(mod, "packed_proj") and hasattr(mod, "packed_proj_roles"):
            unpack(mod)
            mod.packed_proj = None
            _ATTACHED.discard(mod)


def attach(model) -> int:
    """Install the packing hook on every packable module of ``model``; returns how many were attached.

    Called from ``runtimes/parallel.load()`` beside ``packed_experts.attach``, when every parameter
    is at its final storage and before any DDP wrap takes views of it. A module whose hook was never
    installed reads ``packed_proj is None`` and takes the unfused path -- the same bits, more
    launches -- so a missed build path is a performance bug the ``bench_step`` pack stats catch,
    never a correctness one.

    The pack itself is LAZY -- the first forward that reads the hook does it -- and that is not an
    accident of the ``packed_experts`` idiom being copied. ``load()`` runs BEFORE a host wraps the
    model: the FSDP2 host's sequence is ``shard()`` (which is where this attaches) -> ``fully_shard``
    (which re-points every parameter into a DTensor) -> ``broadcast_buffers`` (which copies rank 0's
    bytes over every registered buffer of every DP replica). Packing at attach time would register
    multi-GiB buffers that ``fully_shard`` then orphans and ``broadcast_buffers`` then broadcasts for
    nothing. Packing at the first forward instead means the storage test runs AFTER the wrap, so a
    wrapped host simply gives up and never registers a buffer at all.

    Duck-typed on purpose: this module names ``packed_proj`` / ``packed_proj_roles`` and nothing else
    about the module, and it imports nothing from ``common/``.
    """
    n = 0
    for mod in model.modules():
        if not (hasattr(mod, "packed_proj") and hasattr(mod, "packed_proj_roles")):
            continue
        if len(_roles(mod)) < 2:
            continue
        mod.packed_proj = lambda m=mod: _weight_for(m)
        _ATTACHED.add(mod)
        mod.__dict__.pop(_GIVEUP_ATTR, None)
        n += 1
    return n


# ------------------------------------------------------------------ adjacent rows of one bucket
#
# The trainer case the private-storage guard above refuses. Megatron's DDP re-points every parameter
# into a flat bucket, so a member never owns its storage -- but when the host lays the bucket out so
# a site's members are ADJACENT ROWS IN ROW ORDER (``runtimes/megatron/layout.py`` does), the packed
# operand already exists: it is the ``[sum(N_i), H]`` view of those rows. Nothing is copied, rebound
# or registered; the bytes belong to the bucket owner and stay there. The alias is re-verified on
# EVERY read by pointer arithmetic over the live parameters (one integer compare per member), so a
# host that moves its storage (offload / reload resizes the bucket in place, a re-wrap re-points the
# parameters) is followed, and a layout that is not adjacent simply returns ``None`` -> the ``cat``
# path. Same bytes in the same order as ``torch.cat`` would produce, hence the same GEMM bits.
#
# The members' fp32 ``main_grad`` views live at the same offsets of the grad buffer, so when they are
# adjacent too the packed operand carries them as a wgrad sink (``ops/mm/wgrad.SINK_ATTR``): the one
# packed weight-gradient GEMM accumulates straight into all members' main_grads.

_BUCKET_REPORTED = [False, False]  # (active reported, fallback reported): one line each per process


def adjacent_rows(tensors) -> torch.Tensor | None:
    """The ``[sum(N_i), H]`` view over 2-D row-major ``tensors`` that are adjacent rows of one storage,
    in order; ``None`` when they are not (any gap, reorder, other storage, dtype or width)."""
    first = tensors[0]
    if first.dim() != 2 or not first.is_contiguous():
        return None
    width, dtype, storage = first.shape[1], first.dtype, first.untyped_storage().data_ptr()
    end = first.data_ptr() + first.numel() * first.element_size()
    rows = first.shape[0]
    for t in tensors[1:]:
        if (t.dim() != 2 or not t.is_contiguous() or t.shape[1] != width or t.dtype is not dtype
                or t.untyped_storage().data_ptr() != storage or t.data_ptr() != end):
            return None
        end += t.numel() * t.element_size()
        rows += t.shape[0]
    return first.detach().as_strided((rows, width), (width, 1))


def bucket_weight(mod) -> torch.Tensor | None:
    """The packed operand as a view of adjacent bucket rows, or ``None`` for the unfused path."""
    params = tuple(getattr(mod, role) for role in _roles(mod))
    if len(params) < 2:
        return None
    view = adjacent_rows([p.data for p in params])
    if view is None:
        if not _BUCKET_REPORTED[1]:
            _BUCKET_REPORTED[1] = True
            print(f"[ISOEXEC-PROJ-PACK] bucket alias NOT adjacent on {getattr(mod, 'site', '?')} "
                  f"(members {[tuple(p.shape) for p in params]} are not consecutive rows of one bucket); "
                  f"falling back to cat", flush=True)
        return None
    if not _BUCKET_REPORTED[0]:
        _BUCKET_REPORTED[0] = True
        print(f"[ISOEXEC-PROJ-PACK] bucket alias ACTIVE on {getattr(mod, 'site', '?')}: "
              f"{tuple(view.shape)} view of adjacent bucket rows", flush=True)
    if not torch.is_grad_enabled() or not any(p.requires_grad for p in params):
        return view
    offsets = [0]
    for p in params:
        offsets.append(offsets[-1] + int(p.shape[0]))
    out = _PackedProj.apply(view, offsets, *params)
    from . import wgrad

    grads = [getattr(p, "main_grad", None) for p in params]
    if wgrad.enabled() and all(g is not None and hasattr(p, "grad_added_to_main_grad")
                               and not getattr(p, wgrad.OPT_OUT_ATTR, False)
                               for g, p in zip(grads, params)):
        sink = adjacent_rows(grads)
        if sink is not None and sink.dtype is torch.float32:
            setattr(out, wgrad.SINK_ATTR, (sink, params))
    return out


def attach_bucket_alias(modules) -> int:
    """Install :func:`bucket_weight` as the ``packed_proj`` hook of each of ``modules`` (the host picks
    which sites -- a trainer must not pack a site measured slower packed). Returns how many."""
    n = 0
    for mod in modules:
        if not (hasattr(mod, "packed_proj") and hasattr(mod, "packed_proj_roles")) or len(_roles(mod)) < 2:
            continue
        mod.packed_proj = lambda m=mod: bucket_weight(m)
        n += 1
    return n


__all__ = [
    "adjacent_rows",
    "attach",
    "attach_bucket_alias",
    "bucket_weight",
    "release_to_storage_owner",
    "check_module",
    "pack_site",
    "packed_weight",
    "repack_if_synced",
    "refresh_all_packed",
    "refresh_packed_for",
    "prepare_packed_for",
    "stats",
    "unpack",
]
