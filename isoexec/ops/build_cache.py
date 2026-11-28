import hashlib
import os
import sys
from pathlib import Path
from threading import Lock
from typing import Sequence
from unittest.mock import patch


def sm90_build_dir(
    override: str, family: str, tag: str, source: Path, *, append_tag_to_override: bool = False,
    extra_sources: Sequence[Path] = (),
) -> Path:
    """Honor an explicit cache path, else use PyTorch's cache root and isolate sources.

    ``extra_sources``: further source trees the build includes by relative path (another package's
    csrc, :data:`SM100_COMMON`). Their bytes join the key so an edit there rebuilds this extension;
    the default (none) leaves every existing key unchanged."""
    configured = os.environ.get(override)
    if configured:
        path = Path(configured).expanduser()
        return path / tag if append_tag_to_override else path

    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if root:
        cache = Path(root).expanduser()
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "torch_extensions"

    # The job checkout itself is ephemeral; hash source bytes, not its path, so
    # identical bundles share a compiled cache without mixing different kernels.
    source = source.resolve()
    digest = hashlib.sha256()

    def record(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix in {
            ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".inc", ".inl"
        }:
            record(os.fsencode(path.relative_to(source)))
            record(path.read_bytes())
    for extra in extra_sources:
        extra = Path(extra).resolve()
        for path in sorted(extra.rglob("*")):
            if path.is_file() and path.suffix in _SOURCE_SUFFIXES:
                record(os.fsencode(path.relative_to(extra)))
                record(path.read_bytes())
    return cache / "isoexec" / family / f"{tag}-{digest.hexdigest()[:16]}"


#: Headers shared by SM100 kernel packages (``tmem_pair.cuh``: the tensor-memory allocation and
#: CTA-pair handshake of the tcgen05 kernels). Included by relative path; a package that includes
#: it passes it as ``extra_sources`` so its build key and profile digest fold it.
SM100_COMMON = Path(__file__).resolve().parent / "sm100_common" / "csrc"


# Preserve subnormals and separate multiply/add rounding across both architectures.
NUMERICS_FLAGS = ("--ftz=false", "--fmad=false")

DIAGNOSTIC_FLAGS = ("-lineinfo", "-Xptxas=-v,-warn-spills")

_BUILD_LOCK = Lock()


def build_sm90_extension(
    family: str,
    module_file: str,
    sources: Sequence[str],
    *,
    include_csrc: bool = False,
    include_cccl: bool = False,
    include_cutlass: bool = False,
    extra_cuda_flags: Sequence[str] = (),
):
    """Build and load sources from the caller's sibling csrc directory with the shared SM90 profile.

    ``include_cutlass`` adds the pinned CUTLASS include root (``require_pinned_cutlass``); its revision
    is folded into the build directory so a different checkout never reuses another's objects.
    """
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    from isoexec.ops.prebuilt import load_extension

    upper = family.upper()
    src = Path(module_file).resolve().parent / "csrc"
    tag = f"torch-{torch.__version__}-py{sys.version_info.major}{sys.version_info.minor}"
    cutlass_inc = None
    if include_cutlass:
        cutlass_inc = require_pinned_cutlass(family, "SM90")
        tag += f"-cutlass-{(cutlass_revision(cutlass_inc) or 'unrecorded')[:12]}"
    build = sm90_build_dir(f"ISOEXEC_{upper}_SM90_BUILD_DIR", f"{family}-sm90", tag, src,
                           append_tag_to_override=include_cutlass)
    build.mkdir(parents=True, exist_ok=True)
    version = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
    standard = "c++20" if version >= (2, 11) else "c++17"

    includes = []
    if include_csrc:
        includes.append(str(src))
    if cutlass_inc is not None:
        includes.append(str(cutlass_inc))
    if include_cccl and CUDA_HOME:
        cccl = Path(CUDA_HOME) / "include" / "cccl"
        if cccl.is_dir():
            includes.append(str(cccl))

    with _BUILD_LOCK, patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "9.0a"}):
        return load_extension(
            name=f"isoexec_{family}_sm90_cuda",
            sources=[str(src / f) for f in sources],
            build_directory=str(build),
            extra_include_paths=includes,
            extra_cflags=["-O3", f"-std={standard}"],
            extra_ldflags=["-lcuda"],
            extra_cuda_cflags=[
                "-O3",
                f"-std={standard}",
                "-gencode=arch=compute_90a,code=sm_90a",
                "--expt-relaxed-constexpr",
                *NUMERICS_FLAGS,
                *extra_cuda_flags,
            ],
            verbose=os.environ.get(f"ISOEXEC_{upper}_SM90_BUILD_VERBOSE") == "1",
        )


_SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".inc", ".inl"})
arch_build_dir = sm90_build_dir


#: The CUTLASS commit ``make setup`` installs ([cutlass] in environments/default.toml; a test keeps
#: the two equal). It is FlashMLA ba89a34's csrc/cutlass submodule, which mla_sm90 was derived from
#: and qualified against. It reports 4.3.5 but is not the v4.3.5 tag: SM90 TMA copy headers differ.
CUTLASS_REVISION = "147f5673d0c1c3dcf66f78d677fd647e4a020219"
#: Where setup puts that commit's headers, inside the managed venv, with its revision recorded.
CUTLASS_INSTALLED = Path(sys.prefix) / "share" / "isoexec" / "cutlass"
CUTLASS_REVISION_FILE = ".isoexec-revision"
#: The same commit's headers that a built wheel bundles (setup.py), for an environment managed setup
#: did not build, such as SkyRL's own venv.
CUTLASS_BUNDLED = Path(__file__).resolve().parents[1] / "_runtime" / "cutlass"

CUTLASS_CANDIDATES = (
    str(CUTLASS_INSTALLED / "include"),
    str(CUTLASS_BUNDLED / "include"),
    str(Path(__file__).resolve().parents[2] / "third_party" / "cutlass" / "include"),
)


def _cutlass_tree(path: Path) -> Path | None:
    """``path`` as a CUTLASS include root; accepts either ``<cutlass>`` or ``<cutlass>/include``."""
    for include in (path, path / "include"):
        if (include / "cutlass" / "version.h").is_file():
            return include
    return None


def cutlass_revision(include: Path) -> str | None:
    """The commit setup recorded for an installed tree, or ``None`` for a tree setup did not install."""
    marker = include.parent / CUTLASS_REVISION_FILE
    return marker.read_text().strip() if marker.is_file() else None


def cutlass_include(family: str = "", arch: str = "SM100") -> Path | None:
    """The CUTLASS include root for a build, or ``None`` when no checkout is present.

    ``ISOEXEC_<FAMILY>_<ARCH>_CUTLASS`` or ``ISOEXEC_<ARCH>_CUTLASS`` select a tree explicitly;
    otherwise the first of ``CUTLASS_CANDIDATES`` that holds one: the setup-installed tree, then the
    wheel's bundled tree.
    """
    overrides = [f"ISOEXEC_{arch}_CUTLASS"]
    if family:
        overrides.insert(0, f"ISOEXEC_{family.upper()}_{arch}_CUTLASS")
    for name in overrides:
        configured = os.environ.get(name)
        if configured:
            include = _cutlass_tree(Path(configured).expanduser())
            if include is None:
                raise RuntimeError(
                    f"{name}={configured!r} does not hold a CUTLASS include tree "
                    f"(no cutlass/version.h under it or its include/); point it at a CUTLASS checkout"
                )
            return include
    for candidate in CUTLASS_CANDIDATES:
        path = Path(candidate)
        if (path / "cutlass" / "version.h").is_file():
            return path
    return None


def require_pinned_cutlass(family: str, arch: str) -> Path:
    """The CUTLASS include root at ``CUTLASS_REVISION``, refusing a missing or differently pinned tree.

    An explicit override that setup did not install carries no revision and is taken as the
    caller's responsibility; a setup-installed tree at another commit is refused, since other
    CuTe headers can change the generated kernel.
    """
    include = cutlass_include(family, arch)
    if include is None:
        raise RuntimeError(
            f"ops/{family} needs CUTLASS headers at {CUTLASS_REVISION[:8]} and none were found "
            f"(searched {list(CUTLASS_CANDIDATES)}). Install the isoexec wheel (it bundles them) or run "
            f"`make setup` (it installs them into the venv), "
            f"or set ISOEXEC_{family.upper()}_{arch}_CUTLASS to a CUTLASS checkout at that commit."
        )
    revision = cutlass_revision(include)
    if revision is not None and revision != CUTLASS_REVISION:
        raise RuntimeError(
            f"CUTLASS at {include.parent} is {revision[:8]}, not {CUTLASS_REVISION[:8]}; "
            f"rerun `make setup` with a new ISOEXEC_ROOT"
        )
    return include


def cutlass_identity(family: str = "") -> str:
    """``cutlass-<major>.<minor>.<patch>@<path>`` for the resolved checkout, or ``cutlass-absent``."""
    include = cutlass_include(family)
    if include is None:
        return "cutlass-absent"
    fields = {}
    for line in (include / "cutlass" / "version.h").read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "#define" and parts[1].startswith("CUTLASS_"):
            fields[parts[1]] = parts[2]
    version = ".".join(fields.get(f"CUTLASS_{k}", "?") for k in ("MAJOR", "MINOR", "PATCH"))
    return f"cutlass-{version}@{include}"


def sm100_cuda_flags(standard: str = "c++20", extra: Sequence[str] = ()) -> tuple[str, ...]:
    """The SM100 nvcc arithmetic profile, in the order it reaches the command line."""
    return (
        "-O3",
        f"-std={standard}",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--expt-relaxed-constexpr",
        *NUMERICS_FLAGS,
        *extra,
    )


def sm100_profile_digest(family: str, source: Path, *, extra_cuda_flags: Sequence[str] = (),
                         extra_sources: Sequence[Path] = ()) -> str:
    """Identity of one SM100 build: the flag profile, the CUTLASS headers and the kernel sources."""
    digest = hashlib.sha256()
    digest.update(" ".join(sm100_cuda_flags(extra=extra_cuda_flags)).encode())
    digest.update(cutlass_identity(family).encode())
    for root in (source, *extra_sources):
        for path in sorted(Path(root).resolve().rglob("*")):
            if path.is_file() and path.suffix in _SOURCE_SUFFIXES:
                digest.update(os.fsencode(path.name))
                digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def build_sm100_extension(
    family: str,
    module_file: str,
    sources: Sequence[str],
    *,
    include_csrc: bool = False,
    include_cccl: bool = False,
    include_cutlass: bool = False,
    extra_cuda_flags: Sequence[str] = (),
    extra_sources: Sequence[Path] = (),
):
    """Build and load sources from the caller's sibling ``csrc`` directory with the SM100 profile.

    ``extra_sources`` names other source trees the build includes by relative path; they join the
    build key."""
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    from isoexec.ops.prebuilt import load_extension

    upper = family.upper()
    src = Path(module_file).resolve().parent / "csrc"
    tag = f"torch-{torch.__version__}-py{sys.version_info.major}{sys.version_info.minor}"
    build = arch_build_dir(f"ISOEXEC_{upper}_SM100_BUILD_DIR", f"{family}-sm100", tag, src,
                           extra_sources=extra_sources)
    build.mkdir(parents=True, exist_ok=True)
    version = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
    standard = "c++20" if version >= (2, 11) else "c++17"

    includes = []
    if include_csrc:
        includes.append(str(src))
    if include_cccl and CUDA_HOME:
        cccl = Path(CUDA_HOME) / "include" / "cccl"
        if cccl.is_dir():
            includes.append(str(cccl))
    if include_cutlass:
        cutlass = cutlass_include(family)
        if cutlass is None:
            raise RuntimeError(
                f"ops/{family} asked for CUTLASS headers and none were found (searched "
                f"{list(CUTLASS_CANDIDATES)}); set ISOEXEC_{upper}_SM100_CUTLASS or "
                f"ISOEXEC_SM100_CUTLASS to a <cutlass>/include directory"
            )
        includes += [str(cutlass), str(cutlass.parent / "tools" / "util" / "include")]

    with _BUILD_LOCK, patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "10.0a"}):
        # Through the content-keyed prebuilt store (``ops/prebuilt.py``): a hit loads the stored object, a miss
        # builds exactly this command and publishes it.
        return load_extension(
            name=f"isoexec_{family}_sm100_cuda",
            sources=[str(src / f) for f in sources],
            build_directory=str(build),
            extra_include_paths=[p for p in includes if Path(p).is_dir()],
            extra_cflags=["-O3", f"-std={standard}"],
            extra_ldflags=["-lcuda"],
            extra_cuda_cflags=list(sm100_cuda_flags(standard, extra_cuda_flags)),
            verbose=os.environ.get(f"ISOEXEC_{upper}_SM100_BUILD_VERBOSE") == "1",
            extra_trees=[str(p) for p in extra_sources],
        )


load_sm100_extension = build_sm100_extension
