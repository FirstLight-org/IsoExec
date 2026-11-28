"""The vendored DeepGEMM must stay isolated from any other DeepGEMM loaded in the process.

Regression for a warmup failure: vLLM's bundled upstream DeepGEMM and this vendored copy
compile the same C++ namespaces. GCC emitted ``deep_gemm::jit`` (the JIT runtime holding the include
root) as an STB_GNU_UNIQUE symbol, which the loader unified across both RTLD_LOCAL extensions, so
the later ``_C.init`` chose the include root and cache for both: vendored kernels compiled against
upstream headers. No GPU is needed: JIT identity is read without constructing the CUDA runtime.

The extension tests build the vendored extension, so they run only when TORCH_EXTENSIONS_DIR names
an explicit (non-shared) build directory and a CUDA toolkit is present; the symbol-binding test
needs only a host C++ compiler.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

from isoexec.ops._vendor.deep_gemm import _extension

REPO = Path(__file__).resolve().parents[3]

_TOY = textwrap.dedent("""
    namespace deep_gemm {
    struct Runtime { int include_root = 0; };
    inline Runtime jit;
    }
    #define TOY_API extern "C" __attribute__((visibility("default")))  // like PyInit_*
    TOY_API void toy_init(int root) { deep_gemm::jit.include_root = root; }
    TOY_API int toy_root() { return deep_gemm::jit.include_root; }
""")


def _cxx():
    for candidate in (os.environ.get("CXX"), "g++-13", "g++"):
        if candidate and shutil.which(candidate):
            return shutil.which(candidate)
    pytest.skip("no host C++ compiler")


def _toy_pair(tmp_path, tag, flags):
    source = tmp_path / "toy.cpp"
    source.write_text(_TOY)
    libs = []
    for copy in ("vendored", "upstream"):
        out = tmp_path / f"libtoy_{tag}_{copy}.so"
        subprocess.run([_cxx(), "-std=c++17", "-O2", "-fPIC", "-shared", *flags, str(source), "-o", str(out)],
                       check=True)
        libs.append(out)
    return libs


@pytest.mark.parametrize("isolated", [False, True])
def test_isolation_flags_stop_the_loader_sharing_inline_globals(tmp_path, isolated):
    flags = list(_extension.ISOLATION_CFLAGS) if isolated else []
    vendored, upstream = _toy_pair(tmp_path, "iso" if isolated else "default", flags)
    violations = _extension.isolation_violations(vendored)
    a = ctypes.CDLL(str(vendored), mode=os.RTLD_LOCAL)
    b = ctypes.CDLL(str(upstream), mode=os.RTLD_LOCAL)
    a.toy_init(1)
    b.toy_init(2)  # the later init, as vLLM's _C.init was in that failure
    if isolated:
        assert violations == []
        assert (a.toy_root(), b.toy_root()) == (1, 2)
    else:
        # The failure mode this package must never be built with: one object for both copies.
        assert any(v.startswith("GNU_UNIQUE _ZN9deep_gemm3jitE") for v in violations), violations
        assert (a.toy_root(), b.toy_root()) == (2, 2)


def test_isolation_check_reports_exported_private_namespaces(tmp_path):
    source = tmp_path / "exported.cpp"
    source.write_text("namespace deep_jit { int compile(int x) { return x + 1; } }\n")
    out = tmp_path / "libexported.so"
    subprocess.run([_cxx(), "-fPIC", "-shared", str(source), "-o", str(out)], check=True)
    assert _extension.isolation_violations(out) == ["exported _ZN8deep_jit7compileEi"]
    hidden = tmp_path / "libhidden.so"
    subprocess.run([_cxx(), "-fPIC", "-shared", "-fvisibility=hidden", str(source), "-o", str(hidden)], check=True)
    assert _extension.isolation_violations(hidden) == []


def _extension_build_available():
    if not os.environ.get("TORCH_EXTENSIONS_DIR"):
        return "set TORCH_EXTENSIONS_DIR to a private build dir to build the vendored extension"
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if not os.path.exists(os.path.join(cuda, "bin", "nvcc")):
        return "CUDA toolkit required to build the vendored extension"
    return None


_NEEDS_BUILD = pytest.mark.skipif(_extension_build_available() is not None,
                                  reason=_extension_build_available() or "")


def _upstream_module():
    import importlib.util
    for name in ("vllm.third_party.deep_gemm", "deep_gemm"):
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ValueError):
            continue
    return None


_PROBE = textwrap.dedent("""
    import ctypes, importlib, json, os, sys
    order, upstream_name, scratch = sys.argv[1], sys.argv[2], sys.argv[3]

    def ours():
        from isoexec.ops._vendor import deep_gemm as vendored
        return vendored.load()

    def upstream():
        return importlib.import_module(upstream_name) if upstream_name != "-" else None

    if order == "vendored-first":
        api = ours(); up = upstream()
    else:
        up = upstream(); api = ours()
    identity = api.check_jit_identity()

    # The JIT runtime and its config are private to the vendored object.
    lib = ctypes.CDLL(api._C.__file__)
    exported = [s for s in ("_ZN9deep_gemm3jitE", "_ZN9deep_gemm10jit_configE", "_ZN9deep_gemm7runtimeE")
                if hasattr(lib, s)]

    # Re-initialization with another identity is refused; the same identity is accepted.
    try:
        api._C.init(scratch, "isoexec-other")
        reinit_refused = False
    except RuntimeError:
        reinit_refused = True
    api._C.init(api._ROOT, api._JIT_SIGNATURE)

    # check_jit_identity fails closed on a foreign include root.
    real = api.jit_identity
    api.jit_identity = lambda: dict(real(), include_dirs=[os.path.join(scratch, "include")])
    try:
        api.check_jit_identity()
        foreign_refused = False
    except RuntimeError:
        foreign_refused = True
    api.jit_identity = real
    print(json.dumps({"identity": identity, "exported": exported, "reinit_refused": reinit_refused,
                      "foreign_refused": foreign_refused, "root": api._ROOT,
                      "upstream": getattr(up, "__name__", None)}))
""")


@_NEEDS_BUILD
@pytest.mark.parametrize("order", ["vendored-first", "upstream-first"])
def test_vendored_jit_identity_survives_either_init_order(tmp_path, order):
    upstream = _upstream_module()
    if upstream is None and order == "upstream-first":
        pytest.skip("no upstream DeepGEMM installed to load alongside")
    cache = tmp_path / "jit-cache"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]),
               ISOEXEC_DG_JIT_CACHE_DIR=str(cache), DG_JIT_CACHE_DIR=str(tmp_path / "upstream-cache"))
    env.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")  # a GPU-free build names its target explicitly
    run = subprocess.run([sys.executable, "-c", _PROBE, order, upstream or "-", str(tmp_path)],
                         env=env, capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0, run.stderr[-4000:]
    result = json.loads(run.stdout.strip().splitlines()[-1])
    identity = result["identity"]
    assert identity["include_dirs"] == [os.path.join(result["root"], "include")]
    assert identity["env_prefix"] == "ISOEXEC_DG"
    assert identity["extra_signature"].startswith("isoexec-deep_gemm-")
    assert identity["cache_dirs"] == [str(cache)]
    assert result["exported"] == []
    assert result["reinit_refused"] and result["foreign_refused"]
    assert result["upstream"] == upstream


@_NEEDS_BUILD
def test_built_extension_exports_no_deepgemm_state(monkeypatch):
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "10.0a")
    module = _extension.extension()  # raises if the built object is not isolated
    assert _extension.isolation_violations(module.__file__) == []
