"""IsoExec build adapter for the attributed DeepGEMM C++/JIT source snapshot.

Build products stay in PyTorch's extension cache, never the source tree. CUDA,
PyTorch, a C++20 host compiler with std::format, and Ninja are toolchain/runtime
dependencies; DeepGEMM, DeepJIT and CUTLASS are supplied by this package.

Isolation: other DeepGEMM builds (vLLM's bundled upstream, a pip ``deep_gemm``)
compile the same C++ namespaces. GCC gives inline/static-member globals such as
``deep_gemm::jit`` STB_GNU_UNIQUE binding, which the dynamic loader unifies
process-wide even under RTLD_LOCAL, so whichever ``_C.init`` ran last chose the
JIT include root for both copies. This extension is therefore built with hidden
visibility and ``-fno-gnu-unique`` and refused at load if any ``deep_gemm``/
``deep_jit`` symbol is still exported or any symbol has GNU_UNIQUE binding.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent

#: Host flags that keep every C++ symbol of this copy private to its shared object.
ISOLATION_CFLAGS = ("-fvisibility=hidden", "-fvisibility-inlines-hidden", "-fno-gnu-unique")

_STB_GNU_UNIQUE = 10
_SHN_UNDEF = 0
_STV_DEFAULT, _STV_PROTECTED = 0, 3
#: Itanium-mangled fragments of the namespaces shared with upstream DeepGEMM.
_PRIVATE_NAMESPACES = (b"9deep_gemm", b"8deep_jit")


@lru_cache(None)
def include_digest() -> str:
    """SHA-256 over the JIT include tree (deep_gemm, deep_jit, cute, cutlass headers)."""
    return _tree_digest(("include",))


def _tree_digest(directories) -> str:
    digest = hashlib.sha256()
    for directory in directories:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(ROOT).as_posix().encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def isolation_violations(path) -> list[str]:
    """Dynamic symbols of an ELF64 shared object that would let another DeepGEMM share state.

    Returns GNU_UNIQUE symbols (unified across RTLD_LOCAL objects by the loader) and exported
    definitions in the ``deep_gemm``/``deep_jit`` namespaces (interposable by an earlier object).
    Pure Python so the check runs wherever the extension loads, without binutils.
    """
    data = Path(path).read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise RuntimeError(f"{path} is not a little-endian ELF64 object")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    violations = []
    for _name, sh_type, _flags, _addr, offset, size, link, _info, _align, entsize in sections:
        if sh_type != 11:  # SHT_DYNSYM
            continue
        str_offset = sections[link][4]
        for index in range(1, size // entsize):
            st_name, st_info, st_other, st_shndx = struct.unpack_from("<IBBH", data, offset + index * entsize)
            end = data.index(b"\0", str_offset + st_name)
            name = data[str_offset + st_name:end]
            binding, visibility = st_info >> 4, st_other & 3
            if binding == _STB_GNU_UNIQUE:
                violations.append("GNU_UNIQUE " + name.decode())
            elif (st_shndx != _SHN_UNDEF and visibility in (_STV_DEFAULT, _STV_PROTECTED)
                  and any(fragment in name for fragment in _PRIVATE_NAMESPACES)):
                violations.append("exported " + name.decode())
    return violations


def source_digest() -> str:
    """Hash every vendored build input, including transitive header contents."""
    digest = hashlib.sha256(_tree_digest(("csrc", "include")).encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


@lru_cache(None)
def extension():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    from isoexec.ops.prebuilt import load_extension

    if CUDA_HOME is None:
        raise RuntimeError("Vendored DeepGEMM requires a CUDA toolkit; set CUDA_HOME before warmup")
    # The module name carries no path (the toolkit location and CXX were in it): the prebuilt store's key covers
    # the toolchain by version, so a checkout or toolkit at another path reuses the same build.
    identity = hashlib.sha256((source_digest() + torch.__version__).encode()).hexdigest()[:20]
    module = load_extension(
        name="_isoexec_deep_gemm_C_" + identity,
        sources=[str(ROOT / "csrc/python_api.cpp")],
        extra_include_paths=[str(ROOT / "include"), str(Path(CUDA_HOME) / "include/cccl")],
        extra_cflags=["-std=c++20", "-O3", "-fPIC", "-Wno-psabi", "-Wno-deprecated-declarations",
                      *ISOLATION_CFLAGS],
        extra_ldflags=["-lcudart"],
        with_cuda=True,
        verbose=os.environ.get("ISOEXEC_DEEPGEMM_BUILD_VERBOSE", "0") == "1",
    )
    violations = isolation_violations(module.__file__)
    if violations:
        raise RuntimeError(
            f"Vendored DeepGEMM extension {module.__file__} is not symbol-isolated from other DeepGEMM "
            f"builds ({len(violations)} violations, first: {violations[:5]}); refusing to use it")
    return module
