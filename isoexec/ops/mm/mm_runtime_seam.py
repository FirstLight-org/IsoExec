"""Install OUR vendored ``mm`` into a live vLLM process. The inverted direction, and only that.

After `mm_batch_invariant.py` the arithmetic is ours -- pure torch+triton, no runtime import. This
file is the other half: the engine has already registered `aten::mm` / `aten::addmm` / `aten::bmm`
against ITS module, and something has to point those registrations at our copy. That is the
"inverted direction" the exemption ledger's own README calls permanent:

    "c -- process/memory/registration plumbing. Permanent: a host may import its own runtime.
     Also covers the inverted direction, where we patch the runtime to install OUR numerics --
     vendoring can never remove those imports."

So `ops/mm/` ends up with exactly one vLLM import instead of three, and the one that remains is
category (c) by construction rather than by argument: it borrows no kernel, it overwrites one.
Kernel-ownership.md §6's acceptance -- "`grep` returns only category (c)" -- is met in the shape it
describes.

WHY THE REBIND IS LOAD-BEARING AND NOT COSMETIC. On SM90, which is what this cluster runs,
`enable_batch_invariant_mode` does NOT register mm/addmm; `ops/moe/moe_sm90/moe_batch_invariant.py` does,
and it has two branches. The scoped branch (`mm_fwd_scope.install`) now calls our vendored
functions directly. The unscoped fallback registers vLLM's `bi.mm_batch_invariant` BY VALUE, and
that function resolves `matmul_persistent` through vLLM's module globals at call time -- so after
this rebind, that path runs our kernel too. Without the rebind, ownership of `mm` would be
conditional on which branch won, which is precisely the kind of "depends how it installed" claim
the contract is supposed to eliminate.

WHAT IT DOES NOT REACH, said plainly. `enable_batch_invariant_mode` captures `bmm_batch_invariant`
BY VALUE into `aten::bmm` and rebinds `torch.bmm` to it, so a rebind of the module attribute does
not retake those two if it runs afterwards. `install_bmm_scope` (flag `..._MM_FWD_ONLY_BMM=1`, on
in every production launcher) re-registers `aten::bmm` against our vendored bmm and is verified
post-registration, which is what actually owns the bmm dispatch. The module-attribute rebind here
covers everything that resolves `bi.<name>` at call time, and is honest about covering no more.

Bitwise-neutral by construction: after the rebind the engine and the trainer resolve the same vendored
callables (the installed provider -- on sm90 the segment-tree GEMM, ``ops/mm/mm_sm90`` -- is reached
through the same late-resolving global from both entry points).
"""

from __future__ import annotations

import sys

from . import mm_batch_invariant as V

_INSTALLED = False
_LOG_ONCE = False

#: The three vendored wrappers, bound by value. Safe to bind by value because each resolves
#: `matmul_persistent` through the vendored module's globals at call time -- exactly the structure
#: upstream has, and the structure the installed providers (`mm_sm90`, `mm_sm100`) rely on to compose.
_REBIND_WRAPPERS = ("mm_batch_invariant", "addmm_batch_invariant", "bmm_batch_invariant")


class _MatmulPersistentProxy:
    """Late-resolving, attribute-TRANSPARENT stand-in for ``bi.matmul_persistent``.

    NOT a by-value bind, and the difference is load-bearing. `install_mm_sm90` (and sm100's
    `install_mm_sm100`) REBIND `mm_batch_invariant.matmul_persistent` after this seam may already
    have run; a by-value copy taken before them would leave the engine's own entry point pointing at
    the un-provided kernel while ours was wrapped -- the two runtimes would then be running
    different kernels, which is the exact split-brain the manifest handshake exists to refuse.

    ATTRIBUTE TRANSPARENCY is the second half, and it is not cosmetic. The contract's `mm` impl id
    is a LIVE PREDICATE, not a literal: `ops/mm/trainer.mm_impl_id()` reads
    ``bi.matmul_persistent._isoexec_sm90_segtree`` to decide whether this process owes
    ``sm90_wgmma_segtree`` or ``triton_batch_invariant``, precisely so a process cannot attest a
    provider it does not run. A plain function trampoline answers False to that question whatever is
    installed underneath, so the engine would have attested Triton while running another kernel --
    `fingerprint:mm:engine_{prefill,decode}` violating on every engine run, which is exactly the
    thing the predicate was introduced to catch. Delegating ``__getattr__`` to the live callable
    keeps every such marker (`_isoexec_sm90_segtree`, `_isoexec_sm100`, `__name__`, ...) answering about
    the kernel that will actually run.

    Cost: one extra Python frame per GEMM, against a kernel whose cheapest shape here is 60 us.
    """

    __slots__ = ()

    #: Our own marker, answered before delegation so it is true of the proxy itself.
    _isoexec_vendored = True

    def __call__(self, a, b, bias=None):
        return V.matmul_persistent(a, b, bias=bias)

    def __getattr__(self, name):
        # Only reached for attributes the proxy does not define itself.
        return getattr(V.matmul_persistent, name)

    def __repr__(self) -> str:  # pragma: no cover -- diagnostics only
        return f"<isoexec vendored matmul_persistent -> {V.matmul_persistent!r}>"


_mm_provider = _MatmulPersistentProxy()


def _runtime_module():
    """vLLM's batch_invariant module, or None. Never raises: a trainer-only process legitimately
    has no engine to patch, and the vendored kernels work perfectly well unpatched."""
    return sys.modules.get("vllm.model_executor.determinism.batch_invariant")


def runtime_batch_invariant_mode_on() -> bool:
    """Has vLLM's ``enable_batch_invariant_mode`` already run in this process?

    A REGISTRATION-ORDER query, not a numerics one: `install_bmm_scope` must not register
    ``aten::bmm`` before vLLM does, because vLLM registers it unconditionally and the later
    registration wins. Reading a boolean flag off the engine is the only way to ask, and it is the
    same class of question as "is the model-parallel state initialised".
    """
    from .batch_invariant_install import installed

    bi = _runtime_module()
    return installed() or (bool(getattr(bi, "_batch_invariant_MODE", False)) if bi is not None else False)


def install_vendored_mm() -> bool:
    """Point vLLM's batch-invariant module at our vendored callables. Idempotent; True if done.

    Preserves ``bi._isoexec_stock_matmul_persistent`` with its existing meaning -- *whatever
    ``matmul_persistent`` was before IsoExec touched it*, i.e. genuinely upstream's -- so the gate
    tests can still reach the function they are supposed to compare against. Under `ops/mm/` the
    fallthrough and the bit-equality reference are now ``mm_batch_invariant.matmul_persistent``, our
    copy, which is the point.
    """
    global _INSTALLED, _LOG_ONCE

    if _INSTALLED:
        return True
    bi = _runtime_module()
    if bi is None:
        return False

    # Reproduce upstream's runtime-configured fp16 tile constant on OUR module, so the copy is
    # launched from the same constants and not merely written from the same source. fp16 is off
    # every live path on this branch; doing it anyway is what keeps "copy" true rather than
    # approximately true.
    try:
        import torch

        if torch.cuda.is_available():
            V.configure_fp16_block_size_n(torch.cuda.current_device())
    except Exception:  # noqa: BLE001
        pass

    # Record the genuinely-upstream function once, before anything of ours is in the chain.
    if not hasattr(bi, "_isoexec_stock_matmul_persistent"):
        bi._isoexec_stock_matmul_persistent = bi.matmul_persistent

    for name in _REBIND_WRAPPERS:
        ours = getattr(V, name)
        ours._isoexec_vendored = True
        setattr(bi, name, ours)
    bi.matmul_persistent = _mm_provider

    _INSTALLED = True
    if not _LOG_ONCE:
        _LOG_ONCE = True
        prov = V.provenance()
        print(
            f"[ISOEXEC-MM-VENDOR] vLLM's batch-invariant GEMM entry points now resolve to the "
            f"VENDORED copy in ops/mm/mm_batch_invariant.py "
            f"(matmul_persistent, {', '.join(_REBIND_WRAPPERS)}). Copied "
            f"byte-for-byte from {prov['project']} {prov['version']} ({prov['commit']}), "
            f"{len(prov['symbols'])} symbols / {prov['vendored_lines']} of "
            f"{prov['upstream_total_lines']} upstream lines. `mm` is OURS: an "
            f"upstream patch release can no longer change this rounding schedule under the "
            f"contract. NOTE this rebind does not retake aten::bmm or torch.bmm, which vLLM "
            f"captures by value -- ISOEXEC_MM_FWD_ONLY_BMM=1 does that.",
            flush=True,
        )
    return True


def vendored_mm_installed() -> bool:
    """Diagnostic: is the live ``bi.matmul_persistent`` chain rooted in our copy?"""
    return _INSTALLED
