"""The mla_sm100 forward as a ``Backend`` for the mla_sm90 conformance relations."""

from __future__ import annotations

import hashlib
from pathlib import Path

from isoexec.ops.attention.mla_sm90.tests._harness import SCALE, Backend

__all__ = ["SM100Backend", "PLAN_VARIANTS"]

PLAN_VARIANTS = [{"_plan": "exact"}, {"_plan": "capacity"}, {"_plan": "capacity_loose"},
                 {"_plan": "capacity_loose", "num_parts": 1000}]


class SM100Backend(Backend):
    """The Blackwell deterministic absorbed-MLA forward."""

    name = "mla_sm100"
    has_seg = True
    has_backward = False   # the backward stays the sm90 FA3-basis path; see README.md

    def __init__(self, seg_tokens: int = 512, recipe: int = 1):
        super().__init__(seg_tokens)
        from isoexec.ops.attention.mla_sm100 import fwd as fwd_mod

        #: Round 5: which recipe the relations drive (1 = sm100_absorbed_segtree, 2 = _v2).
        self.recipe = int(recipe)
        if self.recipe != 1:
            self.name = f"mla_sm100_v{self.recipe}"

        self._mod = fwd_mod
        # NOTE deliberately NOT named ``_fwd_mod``: mla_sm90's ``rel_cuda_graph`` gates on that
        # attribute and then hard-imports the sm90 kernel, which cannot build on a Blackwell box.
        # That relation reports n/a for this backend, and this package states the same claim itself
        # in ``tests/_relations_sm100.py::rel_cuda_graph_sm100``.
        self._fwd = fwd_mod.mla_sm100_fwd
        self._plan = fwd_mod.mla_sm100_plan
        self.has_overrides = True

    def fwd(self, b, exec_override=None, seg_tokens=None):
        """``seg_tokens`` overrides the deployment SEG for one call: used ONLY by the sensitivity
        emulation of a phase-dependent program, never by a conformance relation."""
        kw = {}
        if exec_override is not None:
            ov = dict(exec_override)
            plan_mode = ov.pop("_plan", None)
            if plan_mode is not None:
                cap = None if plan_mode == "exact" else {
                    "max_seqlen_k": int(b.seqused_k.max()) + (0 if plan_mode == "capacity" else 4096),
                    "num_tokens": b.q.shape[0] + (0 if plan_mode == "capacity" else 777)}
                kw["plan"] = self._plan(b.cu_seqlens_q, b.seqused_k, self.seg_tokens, b.q.shape[1],
                                        b.max_seqlen_q, capacity=cap, exec_override=ov or None)
            else:
                kw["exec_override"] = ov
        return self._fwd(b.q, b.kv_cache, b.block_table, b.cu_seqlens_q, b.seqused_k, SCALE,
                         seg_tokens if seg_tokens is not None else self.seg_tokens,
                         b.max_seqlen_q, recipe=self.recipe, **kw)

    def identity(self) -> dict:
        """Digests of what compiled the arithmetic, so a result is attributable to one build."""
        ident = {}
        try:
            ident["fwd_profile_digest"] = self._mod.profile_digest()
            from isoexec.ops.attention.mla_sm100._extension import cutlass_id, profile_flags

            ident["cutlass"] = cutlass_id()
            ident["nvcc_flags"] = " ".join(profile_flags())
        except Exception as exc:  # pragma: no cover
            ident["fwd_profile_digest"] = f"unavailable: {exc!r}"
        ident["seg_table_digest"] = self._mod.seg_table_digest()
        h = hashlib.sha256()
        root = Path(self._mod.__file__).resolve().parent
        for name in ("fwd.py", "_extension.py", "spec.py"):
            p = root / name
            if p.is_file():
                h.update(name.encode())
                h.update(p.read_bytes())
        ident["fwd_source_digest"] = h.hexdigest()[:16]
        ident["fwd_signature"] = "role-free"
        ident["recipe"] = self._mod.RECIPE_IDS[self.recipe]
        return ident

    def overrides(self) -> list:
        return list(self._mod.EXEC_OVERRIDE_VARIANTS) + PLAN_VARIANTS
