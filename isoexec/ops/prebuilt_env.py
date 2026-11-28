"""Environment side of the prebuilt cache (``ops/prebuilt.py``), importable without the rest of it.

``isoexec/__init__`` calls :func:`activate` when a prebuilt cache or prebuilt-only mode is configured, before any JIT
can run; keeping this module separate lets ``python -m isoexec.ops.prebuilt`` run as ``__main__`` without a second
copy of the store module.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

ENV_DIR = "ISOEXEC_PREBUILT_DIR"
ENV_ONLY = "ISOEXEC_PREBUILT_ONLY"
MARKER = "ISOEXEC_PREBUILT_MISSING"


def prebuilt_only() -> bool:
    return os.environ.get(ENV_ONLY, "0") == "1"


def cache_root() -> Path:
    """The prebuilt cache root (``ISOEXEC_PREBUILT_DIR``, else under the torch extension cache)."""
    configured = os.environ.get(ENV_DIR)
    if configured:
        return Path(configured).expanduser().absolute()
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    base = Path(root).expanduser() if root else Path(os.environ.get("XDG_CACHE_HOME")
                                                     or Path.home() / ".cache") / "torch_extensions"
    return base.absolute() / "isoexec" / "prebuilt"


def _real_nvcc() -> str | None:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if home and os.access(os.path.join(home, "bin", "nvcc"), os.X_OK):
        return os.path.join(home, "bin", "nvcc")
    return shutil.which("nvcc")


_NVCC_SHIM = """#!/bin/sh
# IsoExec prebuilt-only mode (ISOEXEC_PREBUILT_ONLY=1): the DeepGEMM JIT must not compile in this process.
# --version is answered by the real nvcc so the JIT's cache keys are those of the toolchain.
case "$1" in --version) exec "{nvcc}" "$@";; esac
kernel=$(tr -s ' \\n' ' ' < "$1" 2>/dev/null | grep -o 'reinterpret_cast<void\\*>(&[^;]*' | head -c 800)
[ -n "$kernel" ] || kernel=$(grep -m1 '#include <deep_gemm/impls/' "$1" 2>/dev/null)
echo "{marker} deepgemm-jit kernel: $kernel; not in the prebuilt cache {where}; {only}=1 forbids compiling it in this process (source $1). Prebuild with \\`python -m isoexec.ops.prebuilt build --dir <cache> --warm <cmd>\\` on a GPU of this arch."
exit 86
"""


def _nvcc_refusal_shim(where: str) -> str:
    nvcc = _real_nvcc() or "nvcc"
    text = _NVCC_SHIM.format(nvcc=nvcc, marker=MARKER, where=where, only=ENV_ONLY)
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    directory = Path(os.environ.get("TMPDIR") or "/tmp") / f"isoexec-prebuilt-{os.getuid()}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"nvcc-refuse-{digest}"
    if not path.exists():
        tmp = directory / f".{path.name}.{os.getpid()}"
        tmp.write_text(text)
        tmp.chmod(0o755)
        os.replace(tmp, path)
    return str(path)


def activate() -> None:
    """Point the DeepGEMM JIT and Triton caches at the prebuilt root and, in prebuilt-only mode, install the
    refusals. Environment only (child processes inherit it); idempotent; cheap enough to call from import."""
    configured = os.environ.get(ENV_DIR)
    only = prebuilt_only()
    if configured:
        root = cache_root()
        dg = str(root / "dg-jit")
        if only:
            # A miss creates its temporary entry in the FIRST directory before compiling; keep that one writable so
            # a read-only shared cache still ends in the named refusal, not a filesystem error.
            scratch = Path(os.environ.get("TMPDIR") or "/tmp") / f"isoexec-prebuilt-{os.getuid()}" / "dg-jit-scratch"
            dg = f"{scratch}:{dg}"
        os.environ["ISOEXEC_DG_JIT_CACHE_DIR"] = dg
        os.environ["TRITON_CACHE_DIR"] = str(root / "triton")
    if only:
        os.environ["TRITON_CACHE_MANAGER"] = "isoexec.ops.prebuilt:TritonPrebuiltOnly"
        where = os.environ.get("ISOEXEC_DG_JIT_CACHE_DIR", "<default deep_gemm_jit>")
        os.environ["ISOEXEC_DG_JIT_NVCC_COMPILER"] = _nvcc_refusal_shim(where)
