"""Subprocess checks that component imports initialize no serving runtime or CUDA context. Meta
construction must import no operation package or architecture probe.
"""

from __future__ import annotations

import subprocess
import sys

_RUNTIME_ROOTS = ("vllm", "megatron", "sglang")

_IMPORT_PROBE = """
import sys
import torch
import isoexec.nn
leaked = sorted({m.split(".", 1)[0] for m in sys.modules} & set(%(roots)r))
print("LEAKED:" + ",".join(leaked))
print("CUDA_INIT:" + str(torch.cuda.is_initialized()))
print("EXPORTS:" + ",".join(sorted(isoexec.nn.__all__)))
"""

_CONSTRUCT_PROBE = """
import sys
import isoexec.nn as nn
attn = nn.Attention(hidden=256, heads=4, kv_heads=2, head_dim=64, rope_theta=1e7, rotary_fraction=0.25)
norm = nn.RMSNorm(hidden=256)
rows = nn.inventory(attn, prefix="lm.layers.0.mixer")
assert len(rows) == 7 and all(p.device.type == "meta" for p in attn.parameters())
assert norm.weight.device.type == "meta"
print("OPS:" + ",".join(sorted(m for m in sys.modules if m.split(".")[:2] == ["isoexec", "ops"])))
print("ARCH:" + ",".join(sorted(m for m in sys.modules if m == "isoexec.core.arch")))
"""


def _run(probe: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=600, check=False
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


def test_importing_the_component_tier_imports_no_runtime_and_initializes_no_cuda():
    lines = _run(_IMPORT_PROBE % {"roots": _RUNTIME_ROOTS}).splitlines()
    assert "LEAKED:" in lines, lines
    assert "CUDA_INIT:False" in lines, lines
    assert any(line.startswith("EXPORTS:Attention,") for line in lines), lines


def test_constructing_a_component_on_meta_never_triggers_arch_selection():
    lines = _run(_CONSTRUCT_PROBE).splitlines()
    assert "OPS:" in lines, lines  # no isoexec.ops module imported at all
    assert "ARCH:" in lines, lines  # and therefore no ARCH probe either
