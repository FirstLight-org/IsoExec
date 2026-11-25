"""Require public imports and plugin registration to leave CUDA uninitialized for forked workers.
A deliberate vendored_fla.utils import is the positive control that the context probe can detect initialization."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

#: Everything a host reaches before it forks: the three tiers, and -- where vLLM is installed -- the
#: plugin surface, whose registration call is the last thing to run in the parent.
_SURFACE = """
import torch
import isoexec.ops, isoexec.nn
try:
    from isoexec.runtimes.vllm import registration  # noqa: F401
    registration.register_plugin()
except Exception as exc:  # vLLM is not installed in every validation environment
    print("PLUGIN_SKIPPED:" + type(exc).__name__)
%(extra)s
print("CUDA_INIT:" + str(torch.cuda.is_initialized()))
print("FLA_LOADED:" + str("isoexec.ops.gdn.shared.vendored_fla.utils" in __import__("sys").modules))
"""

#: The positive control's one extra line: the module whose import-time ``_check_platform()`` is the
#: hazard. Written out here rather than in the probe so the two probes differ by exactly it.
_HOIST = "import isoexec.ops.gdn.shared.vendored_fla.utils  # noqa: F401"


def _run(extra: str = "") -> dict[str, str]:
    """The probe, in a subprocess with the GPUs VISIBLE. Hiding them would answer another question."""
    out = subprocess.run(
        [sys.executable, "-c", _SURFACE % {"extra": extra}], capture_output=True, text=True,
        timeout=900, check=False,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return dict(line.split(":", 1) for line in out.stdout.splitlines() if ":" in line)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a context can only be created where a device is")
def test_importing_the_public_surface_creates_no_cuda_context():
    """The property, and then the proof that the probe can see it fail.

    If a module-scope ``vendored_fla`` import is ever hoisted anywhere onto this path -- into a
    family facade, into a spec, into the plugin -- the first assertion goes red with the module
    named, rather than the fork hazard being discovered by a worker that dies at launch.
    """
    clean = _run()
    assert clean["FLA_LOADED"] == "False", "the vendored fla utils are on the import path"
    assert clean["CUDA_INIT"] == "False", (
        "importing the public surface initialized CUDA; a context in the parent breaks vLLM's "
        "forked workers. Look for a new module-scope import that reaches a triton driver."
    )

    hoisted = _run(_HOIST)
    assert hoisted["FLA_LOADED"] == "True"
    assert hoisted["CUDA_INIT"] == "True", (
        "the positive control did not create a context, so the probe above cannot detect one and "
        "would report the property it guards whatever the import graph did"
    )
