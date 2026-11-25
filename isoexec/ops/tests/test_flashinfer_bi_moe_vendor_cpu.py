"""The vendored FlashInfer batch-invariant BF16 MoE (``_vendor/flashinfer_bi_moe``): provenance and isolation, CPU only.

* every vendored file is listed in ``upstream_manifest.json`` with its local sha256 (an unrecorded edit fails);
* the tree imports, and its dispatcher selects records, in a child process in which ``flashinfer``, ``deep_gemm``,
  ``flash_mla``, ``tilelang`` and ``vllm`` are unimportable (meta-path blocker): no run-time dependency on an external
  kernel library;
* no FP8 / FP4 operand format remains in the copied GEMM-class tables.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

VENDOR = pathlib.Path(__file__).resolve().parents[1] / "_vendor" / "flashinfer_bi_moe"
REPO = pathlib.Path(__file__).resolve().parents[3]


def test_manifest_covers_every_file():
    manifest = json.loads((VENDOR / "upstream_manifest.json").read_text())
    listed = {r["path"]: r for r in manifest["files"]}
    on_disk = {str(p.relative_to(VENDOR)) for p in VENDOR.rglob("*.py")}
    assert on_disk == set(listed), sorted(on_disk ^ set(listed))
    for path, row in listed.items():
        assert hashlib.sha256((VENDOR / path).read_bytes()).hexdigest() == row["local_sha256"], path
    for name in ("flashinfer-LICENSE.txt", "flashinfer-NOTICE.txt", "LICENSE.batch-invariant-ops.txt"):
        assert (VENDOR / "licenses" / name).is_file(), name


_CHILD = r'''
import sys
BLOCKED = ("flashinfer", "deep_gemm", "flash_mla", "tilelang", "vllm")

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is blocked in this process")
        return None

sys.meta_path.insert(0, _Block())
for name in BLOCKED:
    sys.modules[name] = None
import importlib, pathlib
root = pathlib.Path(sys.argv[1])
for p in sorted(root.rglob("*.py")):
    mod = "isoexec.ops._vendor.flashinfer_bi_moe." + ".".join(p.relative_to(root).with_suffix("").parts)
    importlib.import_module(mod.removesuffix(".__init__"))
from isoexec.ops._vendor.flashinfer_bi_moe.fused_moe.batch_invariant.schedule import default_configs
for M in (1, 128, 8192):
    default_configs(M, 7, 5120, 288, 385, 148, "sm100")
    default_configs(M, 7, 640, 2304, 385, 148, "sm100")
leaked = [m for m in sys.modules if m.split(".")[0] in BLOCKED and sys.modules[m] is not None]
assert not leaked, leaked
print("OK")
'''


def test_imports_without_external_kernel_libraries():
    out = subprocess.run([sys.executable, "-c", _CHILD, str(VENDOR)], cwd=REPO, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO)},
                         timeout=600)
    assert out.returncode == 0 and out.stdout.strip().endswith("OK"), out.stderr[-3000:]


def test_no_fp8_fp4_formats_in_the_copied_tables():
    from isoexec.ops._vendor.flashinfer_bi_moe.gemm.batch_invariant import generator, spec
    assert spec.IN_FMTS == ("bf16",)
    for table in (spec.MMA_K_OF, spec.MMA_OP_OF, spec.ELEM_BITS_OF, spec.MMA_RATE_OF, generator.CTA_CHUNKS_MAX_TN):
        assert set(table) == {"bf16"}, table
    for empty in (spec.FP8_FMTS, spec.FP8_BLOCKSCALE_FMTS, spec.BLOCK_SCALED, spec.ALPHA_GIVEN, spec.SF_VEC,
                  spec.SF_FMT, spec.CLASS_TAG, generator.SWAP_WIDE_FMTS, generator.BREUSE_FMTS):
        assert not empty
