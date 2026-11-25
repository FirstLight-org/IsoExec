"""Build the vendored FA3 library lazily with PyTorch's CUDA C++ loader; importing this builds nothing.

The build is keyed by every input that can change the generated code: the vendored sources (hashed
by ``sm90_build_dir``), the flag profile in ``spec.py``, the CUTLASS revision, torch and nvcc. The
operators register under ``torch.ops.isoexec_gqa_sm90``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock
from unittest.mock import patch

from isoexec.ops.attention.gqa_sm90 import spec
from isoexec.ops.build_cache import cutlass_revision, require_pinned_cutlass, sm90_build_dir

FA3_DIR = Path(__file__).resolve().parent / "fa3"
NAMESPACE = "isoexec_gqa_sm90"

#: The translation units: the host API, the forward kernels (plain, PackGQA, paged per head-dim
#: pair; split and paged split per segment-split head dim), FA3's own backward kernels, the varlen
#: scheduler, and the segment-split combine. What hopper/setup.py compiles under ``spec.BUILD_DEFINES``
#: (its split instantiations restricted to ``spec.SEG_HEAD_DIMS``), plus ``seg_combine.cu`` in place of
#: its combine kernels.
_FWD_DIMS = ("64", "128", "256", "64_256", "64_512")
SOURCES = (
    "flash_api_stable.cpp",
    *(f"instantiations/flash_fwd_hdim{d}_bf16{v}_sm90.cu" for d in _FWD_DIMS for v in ("", "_packgqa", "_paged")),
    *(f"instantiations/flash_fwd_hdim{d}_bf16{v}_sm90.cu" for d in spec.SEG_HEAD_DIMS for v in ("_split", "_paged_split")),
    *(f"instantiations/flash_bwd_hdim{d}_bf16_sm90.cu" for d in spec.HEAD_DIMS),
    "flash_prepare_scheduler.cu",
    "seg_combine.cu",
)
_LOCK = Lock()


def _defines() -> list[str]:
    return [f"-D{name}" for name in spec.BUILD_DEFINES]


def _nvcc_version(cuda_home: str) -> str:
    out = subprocess.run([str(Path(cuda_home) / "bin" / "nvcc"), "--version"], capture_output=True, text=True)
    return out.stdout.strip().splitlines()[-1] if out.returncode == 0 and out.stdout else "nvcc-unknown"


def profile_digest(cuda_home: str, cutlass_identity: str) -> str:
    """Digest of everything but the source bytes (``sm90_build_dir`` hashes those)."""
    import torch

    h = hashlib.sha256()
    for part in (spec.FA3_COMMIT, " ".join(spec.NVCC_FLAGS), " ".join(spec.CXX_FLAGS), " ".join(_defines()),
                 " ".join(SOURCES), cutlass_identity, torch.__version__, _nvcc_version(cuda_home)):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


@lru_cache(None)
def extension():
    """Build (once per profile) and load the library; return ``torch.ops.isoexec_gqa_sm90``."""
    import torch
    import torch.utils.cpp_extension as cpp

    cuda_home = cpp.CUDA_HOME
    if not cuda_home:
        raise RuntimeError("gqa_sm90 found no CUDA toolkit; set CUDA_HOME")
    cutlass_inc = require_pinned_cutlass("gqa", "SM90")
    identity = f"{cutlass_revision(cutlass_inc) or 'unrecorded'}@{cutlass_inc.resolve()}"
    tag = (f"torch-{torch.__version__}-py{sys.version_info.major}{sys.version_info.minor}-"
           f"{profile_digest(cuda_home, identity)}")
    build = sm90_build_dir("ISOEXEC_GQA_SM90_BUILD_DIR", "gqa-sm90", tag, FA3_DIR, append_tag_to_override=True)
    build.mkdir(parents=True, exist_ok=True)
    includes = [str(FA3_DIR), str(cutlass_inc)]
    cccl = Path(cuda_home) / "include" / "cccl"  # CUDA 13 keeps CCCL here (hopper/setup.py adds it too)
    if cccl.is_dir():
        includes.append(str(cccl))
    with _LOCK, patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "9.0a"}):
        cpp.load(
            name="isoexec_gqa_sm90_fa3",
            sources=[str(FA3_DIR / f) for f in SOURCES],
            build_directory=str(build),
            extra_include_paths=includes,
            extra_cflags=[*spec.CXX_FLAGS, *_defines()],
            extra_cuda_cflags=["--threads", "2", *spec.NVCC_FLAGS, *_defines()],
            is_python_module=False,
            verbose=os.environ.get("ISOEXEC_GQA_SM90_BUILD_VERBOSE") == "1",
        )
    ops = getattr(torch.ops, NAMESPACE)
    for name in ("fwd", "bwd"):
        if not hasattr(ops, name):
            raise RuntimeError(f"gqa_sm90: the built library does not register {NAMESPACE}::{name}")
    return ops


__all__ = ["NAMESPACE", "SOURCES", "extension", "profile_digest"]
