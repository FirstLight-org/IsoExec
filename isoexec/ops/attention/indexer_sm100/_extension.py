"""Build and load the indexer_sm100 extensions (lightning indexer, index logits, top-k) under the SM100 numerical
profile of attention/sparse_sm100 (the same flags, in the same order; the two packages were one until the split).

The profile is mla_sm100's, verbatim: ``build_cache.NUMERICS_FLAGS``
(``--ftz=false --fmad=false``) then ``--ftz=true -DNDEBUG --prec-div=false`` -- the later
``--ftz=true`` wins, ``--fmad=false`` is the load-bearing flag (no FMA contraction on any fold path,
in any translation unit).  One profile for the top-k, the indexer and the index logits.  Each sub-op is its own extension (its own build directory), so iterating on one kernel does
not rebuild the others; ``profile_digest(sub)`` folds the flags, the CUTLASS identity and the bytes of
that sub-op's sources (``csrc/<sub>`` plus ``csrc/common``) into one id.

Importing this module builds nothing; ``extension(sub)`` is the cached entry point.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

from isoexec.ops.build_cache import (
    NUMERICS_FLAGS,
    arch_build_dir,
    cutlass_identity,
    require_pinned_cutlass,
    sm100_cuda_flags,
)

__all__ = ["extension", "profile_digest", "profile_flags", "cutlass_id", "SOURCES", "build_dir"]

_PKG = Path(__file__).resolve().parent
_SRC = _PKG / "csrc"

#: Sources per sub-op extension (relative to csrc/).
SOURCES = {
    "topk": ("topk/bindings.cpp", "topk/topk.cu"),
    "indexer": ("indexer/bindings.cpp", "indexer/indexer.cu"),
    # the index scores on DeepGEMM's SM100 MQA logits (isoexec/ops/_vendor/deep_gemm, MIT; PROVENANCE.md there)
    "mqa": ("mqa/bindings.cpp", "mqa/mqa.cu", "mqa/gather.cu"),
}

#: The vendored DeepGEMM include root (its own CUTLASS/CuTe closure at the revision it vendors).
_DEEP_GEMM_INCLUDE = _PKG.parent.parent / "_vendor" / "deep_gemm" / "include"

#: Per-sub-op additions (include roots of vendored code, its build macros). Part of the digest via the sources.
SUB_INCLUDES: dict = {}
#: Sub-ops built against a vendored tree's own CUTLASS instead of the pinned one (include roots, absolute; hashed
#: into the digest file by file).
SUB_VENDOR_INCLUDES = {"mqa": (_DEEP_GEMM_INCLUDE,)}
SUB_CUDA_FLAGS = {
                  # DeepJIT's front-end flags for these kernels (C++20 templated lambdas, constexpr in device code);
                  # host symbols hidden and no STB_GNU_UNIQUE, so the copy never binds to a host DeepGEMM
                  "mqa": ("--expt-relaxed-constexpr", "--expt-extended-lambda", "-U__CUDA_NO_HALF_OPERATORS__",
                          "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
                          "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-Xcompiler=-fvisibility=hidden",
                          "-Xcompiler=-fvisibility-inlines-hidden", "-Xcompiler=-fno-gnu-unique")}
SUB_CFLAGS = {"mqa": ("-fvisibility=hidden", "-fvisibility-inlines-hidden", "-fno-gnu-unique")}

#: Flags appended to the shared SM100 profile.  Order is part of the digest.  mla_sm100's exactly.
EXTRA_CUDA_FLAGS = (
    "--ftz=true",
    "-DNDEBUG",
    "--prec-div=false",
    "-lineinfo",
    "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills",
    "--threads", "8",
)

assert NUMERICS_FLAGS == ("--ftz=false", "--fmad=false"), "the shared numerics profile moved"


def profile_flags() -> tuple[str, ...]:
    return sm100_cuda_flags(extra=EXTRA_CUDA_FLAGS)


def _source_dirs(sub: str) -> list[Path]:
    return [_SRC / sub, _SRC / "common"]


def profile_digest(sub: str = "indexer") -> str:
    """Identity of one build: the flag profile, the CUTLASS headers and this sub-op's sources."""
    h = hashlib.sha256()
    h.update(" ".join(profile_flags() + SUB_CUDA_FLAGS.get(sub, ())).encode())
    h.update(cutlass_identity("indexer").encode())
    for d in _source_dirs(sub):
        if not d.is_dir():
            continue
        for path in sorted(d.rglob("*")):
            if path.is_file() and path.suffix in {".cu", ".cuh", ".h", ".hpp", ".cpp", ".inc"}:
                h.update(os.fsencode(str(path.relative_to(_SRC))))
                h.update(path.read_bytes())
    h.update(" ".join(SUB_CFLAGS.get(sub, ())).encode())
    for root in SUB_VENDOR_INCLUDES.get(sub, ()):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                h.update(os.fsencode(str(path.relative_to(root))))
                h.update(path.read_bytes())
    return h.hexdigest()[:16]


def cutlass_id() -> str:
    return cutlass_identity("indexer")


def build_dir(sub: str) -> Path:
    import sys

    import torch

    tag = f"torch-{torch.__version__}-py{sys.version_info.major}{sys.version_info.minor}"
    return arch_build_dir(f"ISOEXEC_INDEXER_{sub.upper()}_SM100_BUILD_DIR", f"indexer-{sub}-sm100",
                          tag + "-" + profile_digest(sub), _SRC / sub)


@lru_cache(None)
def extension(sub: str):
    """Build (once) and load the ``sub`` extension (a key of ``SOURCES``)."""
    mod = _load(sub)
    if sub in SUB_VENDOR_INCLUDES:
        # a vendored copy must not share symbols with a host copy of the same library (e.g. vLLM's DeepGEMM)
        from isoexec.ops._vendor.deep_gemm._extension import isolation_violations
        bad = isolation_violations(mod.__file__)
        if bad:
            raise RuntimeError(f"indexer_sm100 {sub}: {mod.__file__} exports vendored symbols: {bad[:8]}")
    return mod


def _load(sub: str):
    from unittest.mock import patch

    from isoexec.ops.prebuilt import load_extension

    build = build_dir(sub)
    build.mkdir(parents=True, exist_ok=True)
    includes = [str(_SRC)]
    if sub in SUB_VENDOR_INCLUDES:
        # the vendored tree's own include root (with its CUTLASS / CuTe closure), never the pinned CUTLASS
        includes += [str(p) for p in SUB_VENDOR_INCLUDES[sub]]
    else:
        cutlass = require_pinned_cutlass("indexer", "SM100")
        includes += [str(cutlass), str(cutlass.parent / "tools" / "util" / "include")]
    includes += [str(_SRC / d) for d in SUB_INCLUDES.get(sub, ())]
    with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "10.0a"}):
        return load_extension(
            name=f"isoexec_indexer_{sub}_sm100_cuda",
            sources=[str(_SRC / f) for f in SOURCES[sub]],
            build_directory=str(build),
            extra_include_paths=[p for p in includes if Path(p).is_dir()],
            extra_cflags=["-O3", "-std=c++20", *[f for f in SUB_CUDA_FLAGS.get(sub, ()) if f.startswith("-D")],
                          *SUB_CFLAGS.get(sub, ())],
            extra_ldflags=["-lcuda"],
            extra_cuda_cflags=list(profile_flags()) + list(SUB_CUDA_FLAGS.get(sub, ())),
            verbose=os.environ.get(f"ISOEXEC_INDEXER_{sub.upper()}_SM100_BUILD_VERBOSE") == "1",
        )
