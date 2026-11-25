"""The index-score path needs no external kernel library (executor rules 2b).

* In a child process whose import system refuses ``vllm``, ``deep_gemm``, ``flash_mla``, ``flashinfer`` and
  ``flash_attn`` (meta-path blocker, proven to block first), every module of sparse_sm100 and of the V4.1 cache
  composition imports, and the ``mqa`` sub-extension's build inputs resolve inside the isoexec package.
* The ``mqa`` sources include only the vendored DeepGEMM tree, CUDA and torch headers (no host copy's headers).
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[5]
_PKG = pathlib.Path(__file__).resolve().parents[1]

_CHILD = r'''
import sys

BLOCKED = ("vllm", "deep_gemm", "flash_mla", "flashinfer", "flash_attn")

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is blocked in this process")
        return None

sys.meta_path.insert(0, _Block())
for name in BLOCKED:
    sys.modules[name] = None
for name in BLOCKED:
    try:
        __import__(name)
    except ImportError:
        pass
    else:
        raise SystemExit(f"BLOCKER FAILED: import {name} succeeded")

import importlib, pathlib
mods = ["isoexec.ops.attention.sparse_sm100", "isoexec.ops.attention.sparse_sm100._extension",
        "isoexec.ops.attention.sparse_sm100.index_logits", "isoexec.ops.attention.sparse_sm100.indexer",
        "isoexec.ops.attention.sparse_sm100.topk", "isoexec.ops.attention.sparse_sm100.dualrail",
        "isoexec.ops.attention.deepseek_v41_cache_sm100", "isoexec.ops.attention.deepseek_v41_cache_sm100.ops",
        "isoexec.ops.attention.deepseek_v41_cache_sm100.paged", "isoexec.ops.attention.deepseek_v41_cache_sm100.cache",
        "isoexec.ops.attention.deepseek_v41_cache_sm100.tp", "isoexec.ops._vendor.deep_gemm._extension"]
for m in mods:
    importlib.import_module(m)
from isoexec.ops.attention.sparse_sm100 import _extension as ext
import isoexec
root = pathlib.Path(isoexec.__file__).resolve().parent
for inc in ext.SUB_VENDOR_INCLUDES["mqa"]:
    assert pathlib.Path(inc).resolve().is_relative_to(root), inc
    assert (pathlib.Path(inc) / "deep_gemm" / "impls" / "sm100_mqa_logits.cuh").is_file()
for src in ext.SOURCES["mqa"]:
    assert (ext._SRC / src).resolve().is_relative_to(root), src
digest = ext.profile_digest("mqa")
assert len(digest) == 16
leaked = sorted(n for n in sys.modules if n.split(".")[0] in BLOCKED and sys.modules[n] is not None)
assert not leaked, leaked
print("OK", digest)
'''


def test_index_path_imports_without_external_kernel_libraries():
    res = subprocess.run([sys.executable, "-c", _CHILD], cwd=_REPO, capture_output=True, text=True, timeout=300,
                         env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    assert res.returncode == 0 and "OK" in res.stdout, res.stdout + res.stderr


def test_mqa_sources_include_only_the_vendored_tree():
    allowed = re.compile(r'#include\s+[<"](deep_gemm/|mqa/|torch/|c10/|cuda|cuda_runtime|atomic|cstdio|cstdlib|cstdint)')
    for src in (_PKG / "csrc" / "mqa").iterdir():
        for line in src.read_text().splitlines():
            if line.lstrip().startswith("#include"):
                assert allowed.match(line.strip()), f"{src.name}: {line}"
