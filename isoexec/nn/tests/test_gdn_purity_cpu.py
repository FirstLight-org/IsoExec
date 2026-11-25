"""Subprocess checks that meta GDN and MLP construction imports no operation package or
architecture probe.
"""

from __future__ import annotations

import subprocess
import sys

_PROBE = """
import sys
from isoexec.nn.gdn import GDN
from isoexec.nn.mlp import MLP
from isoexec.nn import inventory

gdn = GDN(hidden=256, key_heads=4, value_heads=8, key_head_dim=32, value_head_dim=32, conv_kernel=4)
mlp = MLP(hidden=256, intermediate=512)
assert len(inventory(gdn, prefix="lm.layers.0.mixer")) == 11
assert len(inventory(mlp, prefix="lm.layers.0.mlp")) == 3
assert all(p.device.type == "meta" for p in (*gdn.parameters(), *mlp.parameters()))
assert len(gdn.state_specs()) == 4 and mlp.state_specs() == ()
print("OPS:" + ",".join(sorted(m for m in sys.modules if m.split(".")[:2] == ["isoexec", "ops"])))
print("ARCH:" + ",".join(sorted(m for m in sys.modules if m == "isoexec.core.arch")))
print("TRITON:" + ",".join(sorted({m.split(".", 1)[0] for m in sys.modules} & {"triton", "vllm"})))
"""


def test_constructing_a_mixer_or_an_mlp_on_meta_never_triggers_arch_selection():
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=600, check=False)
    assert out.returncode == 0, out.stderr[-3000:]
    lines = out.stdout.splitlines()
    assert "OPS:" in lines, lines  # no isoexec.ops module imported at all
    assert "ARCH:" in lines, lines  # and therefore no ARCH probe either
    assert "TRITON:" in lines, lines  # and no kernel runtime behind it
