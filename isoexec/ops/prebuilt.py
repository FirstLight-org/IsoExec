"""A relocatable, content-keyed store of IsoExec's compiled extensions, and the prebuilt-only mode.

Every IsoExec torch extension is loaded through :func:`load_extension` (``build_cache.build_sm100_extension`` /
``build_sm90_extension``, ``sparse_sm100``, the vendored DeepGEMM host module). Its KEY is a digest of what determines
the machine code, never of where the checkout or the cache lives:

- the toolchain: ``nvcc --version``, the host compiler's ``--version`` (program name dropped), torch and its CUDA,
  Python ``major.minor``, the C++11 ABI, the machine, ``TORCH_CUDA_ARCH_LIST``;
- the flags, with absolute roots replaced by tokens (``<isoexec>`` = this package, ``<cuda>``, ``<prefix>``);
- the bytes of every source and of every include tree the build can read: each include directory outside the
  CUDA toolkit (CUTLASS, vendored trees, ``csrc``), the ``csrc`` tree of every source, and ``extra_trees`` (sources
  a build reaches by a relative include); files are recorded by path RELATIVE to their tree.

A build's real header closure (ninja's dependency log) is checked against the hashed set after every build;
anything outside it, the toolkit, torch, Python or ``/usr`` is recorded as ``uncovered`` in the entry and printed.

The store, ``$ISOEXEC_PREBUILT_DIR/ext`` (default ``<TORCH_EXTENSIONS_DIR | $XDG_CACHE_HOME/torch_extensions>/
isoexec/prebuilt/ext``), holds ``<name>/<key>/{<name>.so, entry.json, .committed}``, published atomically (a whole
directory renamed into place after ``.committed`` is written, as ``deep_jit``'s ``DiskCache`` publishes JIT
kernels). A hit imports the stored shared object directly (``importlib``, as torch's own loader does), so neither
ninja nor the build directory is consulted and a checkout at another path, or another host with the same
toolchain, reuses it. A miss builds exactly as before (``torch.utils.cpp_extension.load``, same arguments) and
publishes the result.

``ISOEXEC_PREBUILT_DIR=<dir>`` also places the DeepGEMM JIT cache (``<dir>/dg-jit``) and the Triton cache
(``<dir>/triton``) under the same root, so one directory is the whole prebuilt cache.

``ISOEXEC_PREBUILT_ONLY=1`` (prebuilt-only mode, for booked clusters): nothing compiles in the process. A missing or
stale extension raises :class:`PrebuiltMissing` (``ISOEXEC_PREBUILT_MISSING ...``) naming the artefact, the key it
expected and, when the store holds the extension under another key, which inputs differ; a DeepGEMM JIT miss fails
through a refusing ``nvcc`` (``ISOEXEC_DG_JIT_NVCC_COMPILER``; ``--version`` is answered by the real nvcc so every
cache key is unchanged) with the same marker and the kernel it would have compiled; a Triton miss raises
:class:`PrebuiltMissing` from the cache manager before the compiler runs.

Prebuild, manifest and verification: ``python -m isoexec.ops.prebuilt --help``.

Mature reference: ``ops/_vendor/deep_gemm/include/deep_jit/cache/disk.hpp`` (content-keyed, atomically published
cache entries, path-free keys) and ``ops/build_cache.py`` ``sm90_build_dir`` (source bytes by relative path).
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

# the environment half (isoexec/__init__ imports it alone, before any JIT can run)
from isoexec.ops.prebuilt_env import (  # noqa: F401  (re-exported)
    _NVCC_SHIM,
    ENV_DIR,
    ENV_ONLY,
    MARKER,
    _nvcc_refusal_shim,
    _real_nvcc,
    activate,
    cache_root,
    prebuilt_only,
)

#: v2 (2026-11-20): toolchain identity extended (torch git version, nvcc host compiler, glibc, compiler env), the
#: include-path environment hashed, uncovered headers refused. Every v1 entry is a MISS under v2 (the schema is a key
#: input), never a hit.
SCHEMA = "isoexec-prebuilt-v2"
COMMITTED = ".committed"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]          # .../isoexec
_IGNORED_DIRS = frozenset({"__pycache__", ".git"})
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})


#: Environment the build reads that can change the compiled bytes, keyed by value (paths tokenised): nvcc's own
#: flag injection, the compilers torch / nvcc pick (also keyed by their --version), gcc's driver search paths, the
#: link search path. Include-path variables are keyed by the CONTENT of their directories (``INCLUDE_PATH_ENV``).
COMPILER_ENV = ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "CC", "CXX", "PYTORCH_NVCC",
                "GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH")
#: Environment that adds include directories (gcc: CPATH, C(PLUS)_INCLUDE_PATH; torch: CUDA_INC_PATH, CUDNN_HOME/PATH).
INCLUDE_PATH_ENV = ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "CUDA_INC_PATH", "CUDNN_HOME", "CUDNN_PATH")


class PrebuiltUncovered(RuntimeError):
    """A build read a header outside every hashed tree and outside the toolchain's own directories: an edit there
    would not change the key, so the result is not published (a hit must never depend on an unhashed file)."""


class PrebuiltMissing(RuntimeError):
    """Prebuilt-only mode met an artefact the cache does not hold (or holds only for other inputs)."""

    def __init__(self, kind: str, name: str, *, key: str, expected: str, detail: str = "", hint: str = ""):
        self.kind, self.name, self.key, self.expected, self.detail = kind, name, key, expected, detail
        text = (f"{MARKER} {kind} {name}: key {key} is not in the prebuilt cache (expected {expected}); "
                f"{ENV_ONLY}=1 forbids compiling it in this process.")
        if detail:
            text += " " + detail
        text += " " + (hint or f"Prebuild it with `python -m isoexec.ops.prebuilt build --dir <cache>` on the same "
                                f"toolchain and sources, then check with `python -m isoexec.ops.prebuilt verify`.")
        super().__init__(text)


def ext_root() -> Path:
    return cache_root() / "ext"


def __getattr__(name):
    if name == "TritonPrebuiltOnly":
        cls = _triton_manager()
        globals()[name] = cls
        return cls
    raise AttributeError(name)


def _triton_manager():
    from triton.runtime.cache import FileCacheManager

    class TritonPrebuiltOnly(FileCacheManager):
        """Triton's file cache, refusing (before the compiler runs) a kernel whose metadata it does not hold."""

        def __init__(self, key, override=False, dump=False):
            try:
                super().__init__(key, override=override, dump=dump)
            except OSError as exc:          # a read-only cache cannot create the directory of a missing kernel
                raise PrebuiltMissing("triton", f"kernel {key}", key=key, expected=os.environ.get(
                    "TRITON_CACHE_DIR", "<triton cache>"), detail=f"({exc})") from exc

        def get_group(self, filename):
            group = super().get_group(filename)
            if group is None and filename.endswith(".json") and not filename.startswith("__grp__"):
                raise PrebuiltMissing("triton", filename[:-5], key=self.key, expected=self.cache_dir)
            return group

    return TritonPrebuiltOnly


# ----------------------------------------------------------------------------------------------- the key

def _run_version(cmd: Sequence[str]) -> str:
    try:
        out = subprocess.run(list(cmd), capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {type(exc).__name__}"


@lru_cache(None)
def toolchain() -> dict:
    """What, besides sources and flags, determines the compiled bytes (no path of any tool)."""
    import torch
    from torch.utils import cpp_extension

    nvcc = os.environ.get("PYTORCH_NVCC") or (
        os.path.join(cpp_extension.CUDA_HOME, "bin", "nvcc") if cpp_extension.CUDA_HOME else "nvcc")
    nvcc_version = _run_version([nvcc, "--version"])
    # nvcc's host compiler: torch passes -ccbin $CC; nvcc honours NVCC_CCBIN; else the gcc on PATH
    host = os.environ.get("CC") or os.environ.get("NVCC_CCBIN") or "gcc"
    return {
        "nvcc": "\n".join(line for line in nvcc_version.splitlines() if not line.startswith("Built on ")),
        "cxx": _compiler_version(os.environ.get("CXX", "c++")),
        "nvcc_host": _compiler_version(host),
        "torch": torch.__version__,
        "torch_git": str(getattr(torch.version, "git_version", "")),
        "torch_cuda": str(torch.version.cuda),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "cxx11_abi": bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", True)),
        "glibc": "-".join(platform.libc_ver()),
        "machine": platform.machine(),
        "env": {name: _normalize(os.environ[name]) for name in COMPILER_ENV if os.environ.get(name)},
    }


def _compiler_version(program: str) -> str:
    """First line of ``program --version`` without the program name (g++-13 and c++ may be one compiler)."""
    lines = _run_version([program, "--version"]).splitlines()
    return lines[0].split(" ", 1)[-1] if lines else ""


@lru_cache(None)
def _system_include_dirs() -> tuple[str, ...]:
    """The host compilers' own search directories (``-E -v``): headers there are keyed by the compiler and glibc
    versions, not by content."""
    dirs: set[str] = set()
    for program in {os.environ.get("CXX", "c++"), os.environ.get("CC") or os.environ.get("NVCC_CCBIN") or "gcc"}:
        try:
            out = subprocess.run([program, "-E", "-x", "c++", "-", "-v"], input="", capture_output=True, text=True,
                                 timeout=60).stderr
        except (OSError, subprocess.SubprocessError):
            continue
        inside = False
        for line in out.splitlines():
            if line.startswith("#include <...> search starts here"):
                inside = True
            elif line.startswith("End of search list"):
                inside = False
            elif inside and line.strip():
                dirs.add(os.path.realpath(line.strip()))
    return tuple(sorted(dirs))


def _roots() -> list[tuple[str, str]]:
    from torch.utils import cpp_extension

    roots = [(str(PACKAGE_ROOT), "<isoexec>")]
    if cpp_extension.CUDA_HOME:
        roots.append((os.path.realpath(cpp_extension.CUDA_HOME), "<cuda>"))
        roots.append((os.path.abspath(cpp_extension.CUDA_HOME), "<cuda>"))
    roots.append((os.path.realpath(sys.prefix), "<prefix>"))
    roots.append((os.path.abspath(sys.prefix), "<prefix>"))
    # longest first so a nested root (a venv inside the checkout) wins
    return sorted(set(roots), key=lambda r: -len(r[0]))


def _normalize(text: str) -> str:
    for root, token in _roots():
        text = text.replace(root, token)
    return text


def _label(path: Path) -> str:
    """A path-free name for ``path``: relative to this package, the CUDA toolkit or the Python prefix."""
    resolved = str(Path(path).resolve())
    for root, token in _roots():
        if resolved == root or resolved.startswith(root + os.sep):
            return token + resolved[len(root):]
    return "<abs>" + resolved


def _is_toolkit(path: Path) -> bool:
    return _label(path).startswith("<cuda>")


def _files(tree: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(tree):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORED_DIRS)
        for name in sorted(filenames):
            if Path(name).suffix not in _IGNORED_SUFFIXES:
                yield Path(dirpath) / name


@lru_cache(None)
def _tree_digest(tree: str) -> tuple[str, int]:
    """SHA-256 over (relative path, bytes) of every file under ``tree``, and the file count."""
    digest = hashlib.sha256()
    root = Path(tree)
    count = 0
    for path in _files(root):
        rel = os.fsencode(path.relative_to(root).as_posix())
        data = path.read_bytes()
        for part in (rel, data):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
        count += 1
    return digest.hexdigest(), count


@lru_cache(None)
def _file_digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csrc_root(source: Path) -> Path:
    """The ``csrc`` directory a source belongs to (its include universe), else its own directory."""
    for parent in source.resolve().parents:
        if parent.name == "csrc":
            return parent
        if parent == PACKAGE_ROOT:
            break
    return source.resolve().parent


def extension_inputs(name: str, sources: Sequence[str], *, extra_include_paths: Sequence[str] = (),
                     extra_cflags: Sequence[str] = (), extra_ldflags: Sequence[str] = (),
                     extra_cuda_cflags: Sequence[str] = (), with_cuda: bool | None = None,
                     extra_trees: Sequence[str | Path] = ()) -> dict:
    """The key inputs of one extension build (JSON-serialisable, no absolute path unless marked ``<abs>``)."""
    trees: list[Path] = []
    for include in extra_include_paths:
        path = Path(include)
        if path.is_dir() and not _is_toolkit(path):
            trees.append(path.resolve())
    for source in sources:
        trees.append(_csrc_root(Path(source)))
    for extra in extra_trees:
        if Path(extra).is_dir():
            trees.append(Path(extra).resolve())
    for var in INCLUDE_PATH_ENV:                   # include directories the environment adds, by content
        for item in filter(None, os.environ.get(var, "").split(os.pathsep)):
            base = Path(item) / "include" if var.startswith("CUDNN") else Path(item)
            if base.is_dir():
                trees.append(base.resolve())
    seen, ordered = set(), []
    for tree in trees:
        if str(tree) not in seen:
            seen.add(str(tree))
            ordered.append(tree)
    return {
        "schema": SCHEMA,
        "name": name,
        "toolchain": toolchain(),
        "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
        "with_cuda": with_cuda,
        "cflags": [_normalize(f) for f in extra_cflags],
        "cuda_cflags": [_normalize(f) for f in extra_cuda_cflags],
        "ldflags": [_normalize(f) for f in extra_ldflags],
        "include_paths": [_label(Path(p)) for p in extra_include_paths],
        "sources": [[_label(Path(s)), _file_digest(str(Path(s).resolve()))] for s in sources],
        "trees": [[_label(t), *_tree_digest(str(t))] for t in ordered],
    }


def key_of(inputs: dict) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:24]


def path_dependent_inputs(inputs: dict) -> list[str]:
    """Inputs that still name an absolute location (an include tree or source outside every known root)."""
    text = json.dumps(inputs)
    return sorted({part.split('"')[0] for part in text.split("<abs>")[1:]})


def describe_difference(old: dict, new: dict) -> list[str]:
    """Which key inputs differ between two entries (for a stale-entry refusal)."""
    out = []
    for field in ("schema", "arch_list", "with_cuda", "cflags", "cuda_cflags", "ldflags", "include_paths"):
        if old.get(field) != new.get(field):
            out.append(field)
    for field, value in (new.get("toolchain") or {}).items():
        if (old.get("toolchain") or {}).get(field) != value:
            out.append(f"toolchain.{field}")
    for field in ("sources", "trees"):
        before = {row[0]: row[1:] for row in old.get(field, [])}
        after = {row[0]: row[1:] for row in new.get(field, [])}
        for label in sorted(set(before) | set(after)):
            if before.get(label) != after.get(label):
                what = "added" if label not in before else "removed" if label not in after else "changed"
                out.append(f"{field[:-1]} {label} ({what})")
    return out


# ----------------------------------------------------------------------------------------------- the store

class _Probe(Exception):
    def __init__(self, name: str, key: str, inputs: dict):
        super().__init__(name)
        self.name, self.key, self.inputs = name, key, inputs


_PROBING = [False]


def _entry(name: str, key: str) -> Path:
    return ext_root() / name / key


def verify_on_load() -> bool:
    return os.environ.get("ISOEXEC_PREBUILT_VERIFY", "0") == "1"


def _stored(name: str, key: str, *, deep: bool | None = None) -> Path | None:
    """The committed shared object for (name, key), or None. Always: the file exists with its recorded size.
    ``deep`` (default ``ISOEXEC_PREBUILT_VERIFY=1``): also its SHA-256 and the digest of the toolchain headers the
    build read (what ninja's mtime check would have caught and the version-level key does not)."""
    entry = _entry(name, key)
    if not (entry / COMMITTED).exists():
        return None
    meta = json.loads((entry / "entry.json").read_text())
    so = entry / meta["file"]
    if not so.is_file() or so.stat().st_size != meta.get("bytes"):
        return None
    if deep if deep is not None else verify_on_load():
        if _sha(so) != meta.get("sha256"):
            return None
        if _external_digest(meta.get("toolchain_deps", [])) != meta.get("toolchain_deps_sha256"):
            return None
    return so


def _import(name: str, so: Path):
    spec = importlib.util.spec_from_file_location(name, str(so))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stale_detail(name: str, inputs: dict) -> str:
    folder = ext_root() / name
    if not folder.is_dir():
        return f"The cache holds no build of {name}."
    notes = []
    entries = sorted((p for p in folder.iterdir() if (p / COMMITTED).exists()), key=lambda p: -p.stat().st_mtime)
    for entry in entries[:3]:
        try:
            old = json.loads((entry / "entry.json").read_text())["inputs"]
        except (OSError, KeyError, ValueError):
            continue
        diff = describe_difference(old, inputs)
        notes.append(f"stale entry {entry.name} differs in: {', '.join(diff[:12]) or 'nothing recorded'}")
    return ("The cache holds other builds of it: " + "; ".join(notes) + ".") if notes else \
        f"The cache holds no committed build of {name}."


def _uncovered_dependencies(build_directory: str, inputs: dict, sources: Sequence[str]) -> list[str]:
    """Headers the build read (ninja's dependency log) that no hashed tree, source, toolkit, torch, Python or
    system directory covers: an edit there would not change the key."""
    import sysconfig

    import torch

    missing, _ = _classify_dependencies(build_directory, inputs, sources)
    return missing


def _classify_dependencies(build_directory: str, inputs: dict, sources: Sequence[str]) -> tuple[list[str], list[str]]:
    """(uncovered headers, toolchain headers) of a build from ninja's dependency log. Toolchain headers live in the
    CUDA toolkit, torch's include tree, Python's include directory or a host compiler's own search directories:
    the key covers them by version (nvcc build, torch version + git hash, compiler, glibc), not by content."""
    import sysconfig

    import torch

    covered = [str(PACKAGE_ROOT / t[0][len("<isoexec>/"):]) for t in inputs["trees"] if t[0].startswith("<isoexec>/")]
    covered += [str(Path(p).resolve()) for p in sources]
    toolchain_dirs = [os.path.realpath(os.path.dirname(torch.__file__)),
                      os.path.realpath(sysconfig.get_paths()["include"]), *_system_include_dirs()]
    tree_labels = [t[0] for t in inputs["trees"]]
    try:
        out = subprocess.run(["ninja", "-C", build_directory, "-t", "deps"], capture_output=True, text=True,
                             timeout=120)
    except (OSError, subprocess.SubprocessError):
        return ["<ninja deps unavailable>"], []
    if out.returncode != 0 or not any(line.startswith("    ") for line in out.stdout.splitlines()):
        return ["<ninja deps unavailable>"], []
    missing, external = set(), set()
    for line in out.stdout.splitlines():
        if not line.startswith("    "):
            continue
        dep = os.path.realpath(os.path.join(build_directory, line.strip()))
        if any(dep == c or dep.startswith(c + os.sep) for c in covered):
            continue
        if any(_label(Path(dep)).startswith(t + "/") for t in tree_labels):
            continue
        if _is_toolkit(Path(dep)) or any(dep.startswith(d + os.sep) for d in toolchain_dirs):
            external.add(dep)
            continue
        missing.add(_label(Path(dep)))
    return sorted(missing), sorted(external)


def _external_digest(paths: Sequence[str]) -> str:
    """SHA-256 over (path, bytes) of toolchain headers a build read (``ISOEXEC_PREBUILT_VERIFY=1`` rechecks it)."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode() + b"\0")
        try:
            digest.update(hashlib.sha256(Path(path).read_bytes()).digest())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


def _publish(name: str, key: str, inputs: dict, module, build_directory: str | None, sources: Sequence[str],
             build_seconds: float) -> None:
    so = Path(getattr(module, "__file__", "") or "")
    if not so.is_file():
        return
    folder = ext_root() / name
    if (folder / key / COMMITTED).exists():
        return
    uncovered, external = (_classify_dependencies(build_directory, inputs, sources) if build_directory
                           else (["<no build directory: dependencies unknown>"], []))
    if uncovered:
        raise PrebuiltUncovered(
            f"ISOEXEC_PREBUILT_UNCOVERED torch-extension {name} (key {key}): the build read headers outside every "
            f"hashed tree and outside the toolchain's directories: {uncovered[:12]}; an edit there would not change "
            f"the key, so the build is NOT published to {folder}. Add the tree to the build's extra_trees / include "
            f"paths (build_cache.build_sm100_extension(extra_sources=...)).")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f".tmp-{uuid.uuid4().hex}"
        tmp.mkdir()
        shutil.copy2(so, tmp / so.name)
        meta = {"name": name, "key": key, "file": so.name, "sha256": _file_digest(str(so)),
                "bytes": so.stat().st_size, "build_seconds": round(build_seconds, 2), "built_at": time.time(),
                "uncovered_deps": uncovered, "path_dependent": path_dependent_inputs(inputs), "inputs": inputs,
                "toolchain_deps": external, "toolchain_deps_sha256": _external_digest(external)}
        (tmp / "entry.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
        (tmp / COMMITTED).write_text("")
        try:
            os.rename(tmp, folder / key)
        except OSError:                       # another process published the same key first
            shutil.rmtree(tmp, ignore_errors=True)
    except OSError as exc:
        print(f"[isoexec-prebuilt] {name}: not published to {folder} ({exc})", file=sys.stderr, flush=True)


def load_extension(*, name: str, sources: Sequence[str], build_directory: str | None = None,
                   extra_include_paths: Sequence[str] = (), extra_cflags: Sequence[str] = (),
                   extra_ldflags: Sequence[str] = (), extra_cuda_cflags: Sequence[str] = (),
                   with_cuda: bool | None = None, verbose: bool = False,
                   extra_trees: Sequence[str | Path] = ()):
    """``torch.utils.cpp_extension.load`` through the content-keyed store (same arguments, same build).

    ``extra_trees``: source trees the build reaches by a relative include outside its ``csrc``; they join the key.
    """
    from isoexec.core import load_timers

    activate()
    start = time.monotonic()
    inputs = extension_inputs(name, sources, extra_include_paths=extra_include_paths, extra_cflags=extra_cflags,
                              extra_ldflags=extra_ldflags, extra_cuda_cflags=extra_cuda_cflags,
                              with_cuda=with_cuda, extra_trees=extra_trees)
    key = key_of(inputs)
    if _PROBING[0]:
        raise _Probe(name, key, inputs)
    so = _stored(name, key)
    if so is not None:
        module = _import(name, so)
        load_timers.add("ext.store_hit", time.monotonic() - start)
        return module
    if prebuilt_only():
        detail = _stale_detail(name, inputs)
        if (_entry(name, key) / COMMITTED).exists():
            detail = ("The entry for this key exists but fails the load check (recorded size"
                      + (", SHA-256 or toolchain-header digest" if verify_on_load() else "") + "): rebuild it.")
        raise PrebuiltMissing("torch-extension", name, key=key, expected=str(_entry(name, key)), detail=detail)
    from torch.utils import cpp_extension

    kwargs = dict(name=name, sources=list(sources), extra_include_paths=list(extra_include_paths),
                  extra_cflags=list(extra_cflags), extra_ldflags=list(extra_ldflags),
                  extra_cuda_cflags=list(extra_cuda_cflags), verbose=verbose)
    if build_directory is not None:
        kwargs["build_directory"] = build_directory
    if with_cuda is not None:
        kwargs["with_cuda"] = with_cuda
    module = cpp_extension.load(**kwargs)
    seconds = time.monotonic() - start
    load_timers.add("ext.build", seconds)
    so_dir = os.path.dirname(getattr(module, "__file__", "") or "") or build_directory
    _publish(name, key, inputs, module, so_dir, sources, seconds)
    return module


# ----------------------------------------------------------------------------------------------- the registry

#: Every IsoExec SM100 torch extension: (label, module, entry point, args). ``build`` makes them all (CPU only);
#: ``verify`` recomputes each key from the checkout and the toolchain at hand.
EXTENSIONS: tuple[tuple[str, str, str, tuple], ...] = (
    ("deep_gemm_C", "isoexec.ops._vendor.deep_gemm._extension", "extension", ()),
    ("deepseek_v41_rows", "isoexec.ops.attention.deepseek_v41_cache_sm100._extension", "extension", ()),
    *((f"indexer_{sub}", "isoexec.ops.attention.indexer_sm100._extension", "extension", (sub,))
      for sub in ("topk", "indexer", "mqa")),
    *((f"sparse_{sub}", "isoexec.ops.attention.sparse_sm100._extension", "extension", (sub,))
      for sub in ("attn", "prims", "bwd", "dualrail")),
    ("gqa_sm100", "isoexec.ops.attention.gqa_sm100._extension", "extension", ()),
    ("mla_sm100", "isoexec.ops.attention.mla_sm100._extension", "extension", ()),
    ("rope_sm100", "isoexec.ops.attention.rope_sm100._extension", "extension", ()),
    ("detreduce_sm100", "isoexec.ops.collectives.detreduce_sm100._extension", "extension", ()),
    ("digest_sm100", "isoexec.ops.digest._extension", "extension", ("sm100",)),
    ("engram_sm100", "isoexec.ops.embed.engram_sm100._extension", "extension", ()),
    ("fused_sm100", "isoexec.ops.fused_sm100._extension", "extension", ()),
    ("conv_sm100", "isoexec.ops.gdn.conv_sm100._extension", "extension", ()),
    ("gdn_cpr_sm100", "isoexec.ops.gdn.gdn_cpr_sm100._extension", "extension", ()),
    ("kda_glm_sm100", "isoexec.ops.kda.glm_sm100._extension", "extension", ()),
    ("kda_glm_rec_sm100", "isoexec.ops.kda.glm_rec_sm100._extension", "extension", ()),
    ("kda_slot_reset_sm100", "isoexec.ops.kda.slot_reset_sm100", "extension", ()),
    ("logprobs_sm100", "isoexec.ops.logprobs.logprobs_sm100._extension", "extension", ()),
    ("logprobs_sm100_v2", "isoexec.ops.logprobs.logprobs_sm100._extension_v2", "extension", ()),
    ("logprobs_sm100_sampler", "isoexec.ops.logprobs.logprobs_sm100._extension_sampler", "extension", ()),
    ("mlp_sm100", "isoexec.ops.mlp.mlp_sm100._extension", "extension", ()),
    ("mm_sm100", "isoexec.ops.mm.mm_sm100._extension", "extension", ()),
    ("mmgen_sm100", "isoexec.ops.mm.mm_sm100.general._extension", "extension", ()),
    ("deepgemm_mega_ep_io", "isoexec.ops.moe.deepgemm_sm100._extension", "extension", ()),
    ("moe_sm100", "isoexec.ops.moe.moe_sm100._extension", "extension", ()),
    ("router_sm100", "isoexec.ops.moe.router_sm100._extension", "extension", ()),
    ("norms_sm100", "isoexec.ops.norms.norms_sm100._extension", "extension", ()),
    ("mhc_sm100", "isoexec.ops.residual.mhc_sm100._extension", "extension", ()),
)


def _entry_point(module: str, attr: str):
    fn = getattr(importlib.import_module(module), attr)
    return getattr(fn, "__wrapped__", fn)     # past lru_cache: a probe must reach the loader every time


def probe(label: str) -> tuple[str, str, dict]:
    """(extension name, key, inputs) the registry entry ``label`` would load, computed without building."""
    _, module, attr, args = next(e for e in EXTENSIONS if e[0] == label)
    _PROBING[0] = True
    try:
        _entry_point(module, attr)(*args)
    except _Probe as hit:
        return hit.name, hit.key, hit.inputs
    finally:
        _PROBING[0] = False
    raise RuntimeError(f"{label}: {module}.{attr} did not load through isoexec.ops.prebuilt.load_extension")


def _build_one(label: str) -> dict:
    """Subprocess body of ``build``: load (build if needed) one registry extension; report what happened."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from isoexec.core import load_timers

    start = time.monotonic()
    try:
        name, key, _ = probe(label)
        hit = _stored(name, key) is not None
        _, module, attr, args = next(e for e in EXTENSIONS if e[0] == label)
        _entry_point(module, attr)(*args)
        state = "hit" if hit else "built"
        return {"label": label, "name": name, "key": key, "state": state,
                "seconds": round(time.monotonic() - start, 1), "timers": load_timers.snapshot()}
    except Exception as exc:  # noqa: BLE001 -- reported per extension
        return {"label": label, "state": "FAILED", "seconds": round(time.monotonic() - start, 1),
                "error": f"{type(exc).__name__}: {str(exc)[-1500:]}"}


# ----------------------------------------------------------------------------------------------- manifest

def _dg_entries(root: Path) -> list[dict]:
    out = []
    folder = root / "dg-jit" / "cache"
    if folder.is_dir():
        for entry in sorted(folder.iterdir()):
            if (entry / COMMITTED).exists():
                files = {p.name: _sha(p) for p in sorted(entry.iterdir())
                         if p.is_file() and p.name in ("kernel.cubin", "kernel.cu")}
                out.append({"entry": entry.name, "files": files})
    return out


def _triton_entries(root: Path) -> list[dict]:
    out = []
    folder = root / "triton"
    if folder.is_dir():
        for entry in sorted(folder.iterdir()):
            if entry.is_dir():
                files = {p.name: _sha(p) for p in sorted(entry.iterdir()) if p.is_file() and p.name != "lock"
                         and not p.name.startswith(".") and ".tmp." not in p.name}
                if files:
                    out.append({"key": entry.name, "files": files})
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(root: Path) -> dict:
    """Every committed artefact under ``root`` with its digest (extensions, DeepGEMM JIT kernels, Triton)."""
    ext = []
    folder = root / "ext"
    if folder.is_dir():
        for name_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            for entry in sorted(p for p in name_dir.iterdir() if (p / COMMITTED).exists()):
                meta = json.loads((entry / "entry.json").read_text())
                ext.append({"name": meta["name"], "key": meta["key"], "file": meta["file"],
                            "sha256": _sha(entry / meta["file"]), "bytes": meta["bytes"],
                            "uncovered_deps": meta.get("uncovered_deps", []),
                            "path_dependent": meta.get("path_dependent", [])})
    body = {"schema": SCHEMA, "toolchain": toolchain(), "ext": ext, "dg_jit": _dg_entries(root),
            "triton": _triton_entries(root)}
    body["digest"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    body["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return body


def write_manifest(root: Path) -> dict:
    data = manifest(root)
    path = root / "MANIFEST.json"
    tmp = root / f".MANIFEST.{os.getpid()}.json"
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    os.replace(tmp, path)
    return data


def verify(root: Path, *, load: bool = False, labels: Sequence[str] | None = None, deep: bool = False) -> list[str]:
    """Problems with the cache at ``root`` for this checkout and toolchain (empty: the cache is complete)."""
    problems: list[str] = []
    recorded = root / "MANIFEST.json"
    if recorded.is_file():
        data = json.loads(recorded.read_text())
        for e in data.get("ext", []):
            path = root / "ext" / e["name"] / e["key"] / e["file"]
            if not path.is_file():
                problems.append(f"{MARKER} torch-extension {e['name']} {e['key']}: manifest artefact {path} missing")
            elif _sha(path) != e["sha256"]:
                problems.append(f"STALE torch-extension {e['name']} {e['key']}: {path} digest differs from manifest")
        for e in data.get("dg_jit", []):
            for fname, sha in e["files"].items():
                path = root / "dg-jit" / "cache" / e["entry"] / fname
                if not path.is_file() or _sha(path) != sha:
                    problems.append(f"{MARKER} deepgemm-jit {e['entry']}/{fname}: missing or digest differs")
        for e in data.get("triton", []):
            for fname, sha in e["files"].items():
                path = root / "triton" / e["key"] / fname
                if not path.is_file() or _sha(path) != sha:
                    problems.append(f"{MARKER} triton {e['key']}/{fname}: missing or digest differs")
    else:
        problems.append(f"no manifest at {recorded} (artefact digests unchecked)")
    old = os.environ.get(ENV_DIR)
    os.environ[ENV_DIR] = str(root)
    try:
        for label in labels or [e[0] for e in EXTENSIONS]:
            try:
                name, key, inputs = probe(label)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{label}: key not computable ({type(exc).__name__}: {exc})")
                continue
            so = _stored(name, key, deep=deep)
            if so is None:
                problems.append(f"{MARKER} torch-extension {name} ({label}): key {key} missing or failing the "
                                f"{'deep ' if deep else ''}load check. "
                                + _stale_detail(name, inputs))
            elif load:
                try:
                    _import(name, so)
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{label}: {so} does not import ({type(exc).__name__}: {exc})")
    finally:
        if old is None:
            os.environ.pop(ENV_DIR, None)
        else:
            os.environ[ENV_DIR] = old
    return problems


# ----------------------------------------------------------------------------------------------- CLI

def _warm_deepgemm(root: Path, max_m: int, max_m_grad: int) -> int:
    """GPU: the DeepGEMM row sweep (``ops/mm/deepgemm_sm100/jit_warm``) into the cache at ``root``."""
    os.environ[ENV_DIR] = str(root)
    os.environ.pop(ENV_ONLY, None)
    activate()
    from isoexec.ops.mm.deepgemm_sm100 import jit_warm

    folder = root / "dg-jit" / "cache"
    before = len(list(folder.iterdir())) if folder.is_dir() else 0
    start = time.monotonic()
    rows = jit_warm.shapes(root)
    if not rows:
        print(f"[dg-sweep] no recorded signatures in {root / 'dg-shapes.jsonl'} (run a --warm command first)")
        return 1
    jit_warm.sweep(root, max_m, max_m_grad=max_m_grad or max_m)
    after = len(list(folder.iterdir())) if folder.is_dir() else 0
    print(f"[dg-sweep] {len(rows)} signatures, DeepGEMM JIT entries {before} -> {after}, "
          f"{time.monotonic() - start:.0f}s", flush=True)
    return 0


def _summary(data: dict) -> str:
    lines = [f"MANIFEST digest {data['digest']}  ({len(data['ext'])} extensions, {len(data['dg_jit'])} DeepGEMM JIT "
             f"kernels, {len(data['triton'])} Triton kernels)"]
    for e in data["ext"]:
        lines.append(f"  ext    {e['name']:<40} {e['key']}  sha256 {e['sha256'][:16]}  {e['bytes']:>10} B"
                     + (f"  UNCOVERED {e['uncovered_deps']}" if e["uncovered_deps"] else ""))
    for e in data["dg_jit"]:
        lines.append(f"  dg-jit {e['entry'][:64]:<64} cubin {e['files'].get('kernel.cubin', '')[:16]}")
    for e in data["triton"]:
        lines.append(f"  triton {e['key'][:64]:<64} {len(e['files'])} files")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import concurrent.futures
    import multiprocessing

    parser = argparse.ArgumentParser(prog="python -m isoexec.ops.prebuilt", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build every extension (CPU), run each --warm command against the cache "
                                     "(DeepGEMM JIT + Triton products of a model and layout; needs its GPUs), "
                                     "write and print MANIFEST.json")
    b.add_argument("--dir", required=True, type=Path)
    b.add_argument("--parallel", type=int, default=6, help="extensions built at once (each uses MAX_JOBS)")
    b.add_argument("--only", default="", help="comma-separated registry labels (default: all)")
    b.add_argument("--warm", action="append", default=[], help="shell command run with ISOEXEC_PREBUILT_DIR set")
    b.add_argument("--sweep-m", type=int, default=0, help="after the --warm commands, compile every DeepGEMM "
                   "configuration of each recorded GEMM signature for M = 1..N (needs a GPU; see warm-deepgemm)")
    b.add_argument("--sweep-m-grad", type=int, default=0, help="the same bound for grad-mode (trainer) signatures")
    w = sub.add_parser("warm-deepgemm", help="on a GPU: call every DeepGEMM GEMM signature recorded in the cache "
                       "(dg-shapes.jsonl, written by --warm runs) at M = 1..--max-m, compiling every kernel "
                       "configuration the per-M heuristics can choose")
    w.add_argument("--dir", required=True, type=Path)
    w.add_argument("--max-m", type=int, required=True)
    w.add_argument("--max-m-grad", type=int, default=0)
    m = sub.add_parser("manifest", help="write and print MANIFEST.json of a cache")
    m.add_argument("--dir", required=True, type=Path)
    v = sub.add_parser("verify", help="check a cache against its manifest and this checkout's keys")
    v.add_argument("--dir", required=True, type=Path)
    v.add_argument("--load", action="store_true", help="also import every extension")
    v.add_argument("--deep", action="store_true", help="also re-hash every shared object and the toolchain headers "
                   "each build read (what ISOEXEC_PREBUILT_VERIFY=1 does at every load)")
    v.add_argument("--only", default="")
    sub.add_parser("list", help="the registry labels")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        print("\n".join(e[0] for e in EXTENSIONS))
        return 0
    root = args.dir.expanduser().absolute()
    if args.cmd == "warm-deepgemm":
        return _warm_deepgemm(root, args.max_m, args.max_m_grad)
    if args.cmd == "manifest":
        print(_summary(write_manifest(root)))
        return 0
    if args.cmd == "verify":
        labels = [x for x in args.only.split(",") if x] or None
        problems = verify(root, load=args.load, labels=labels, deep=args.deep)
        for p in problems:
            print(p)
        count = len(labels or EXTENSIONS)
        print(f"VERIFY {'OK' if not problems else 'FAILED'}: {count} extension keys checked, "
              f"{len(problems)} problem(s), cache {root}")
        return 0 if not problems else 1
    # build
    os.environ[ENV_DIR] = str(root)
    os.environ.pop(ENV_ONLY, None)
    root.mkdir(parents=True, exist_ok=True)
    labels = [x for x in args.only.split(",") if x] or [e[0] for e in EXTENSIONS]
    unknown = sorted(set(labels) - {e[0] for e in EXTENSIONS})
    if unknown:
        parser.error(f"unknown labels {unknown}")
    start = time.monotonic()
    failed = 0
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.parallel), mp_context=ctx) as pool:
        for result in pool.map(_build_one, labels):
            failed += result["state"] == "FAILED"
            print(f"[build] {result['label']:<26} {result['state']:<6} {result['seconds']:>7.1f}s "
                  f"{result.get('key', '')} {result.get('error', '')[:600]}", flush=True)
    print(f"[build] extensions: {len(labels) - failed}/{len(labels)} in {time.monotonic() - start:.0f}s", flush=True)
    for command in args.warm:
        t = time.monotonic()
        rc = subprocess.run(command, shell=True, env=dict(os.environ)).returncode
        print(f"[warm] rc={rc} {time.monotonic() - t:.0f}s: {command}", flush=True)
        failed += rc != 0
    if args.sweep_m:
        cmd = [sys.executable, "-m", "isoexec.ops.prebuilt", "warm-deepgemm", "--dir", str(root), "--max-m",
               str(args.sweep_m), "--max-m-grad", str(args.sweep_m_grad or args.sweep_m)]
        failed += subprocess.run(cmd, env=dict(os.environ)).returncode != 0
    data = write_manifest(root)
    print(_summary(data))
    return 0 if not failed else 1


activate()

if __name__ == "__main__":
    # Run the canonical module's main: under ``-m`` this file is also ``__main__``, a second copy whose state
    # (the probe flag) the loaders, which import ``isoexec.ops.prebuilt``, would not see.
    sys.exit(importlib.import_module("isoexec.ops.prebuilt").main())
