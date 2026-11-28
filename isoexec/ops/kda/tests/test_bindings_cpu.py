"""Every KDA binding that launches a kernel runs on its inputs' device and checks the launch (no GPU, no build).

The neighbouring operators (``norms_sm100/csrc/host.cpp``, ``gdn_cpr_sm100/csrc/rows.cu``, ``slot_reset_sm100``) open
a ``CUDAGuard`` on the inputs' device before they ask for the current stream, and follow each launch with
``C10_CUDA_KERNEL_LAUNCH_CHECK()``. Without the guard a call on another device than the current one launches on the
wrong device's stream; without the check a launch that fails (e.g. a grid axis over its limit) returns with the
output never written and no error.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

KDA = Path(__file__).resolve().parents[1]
BINDINGS = (KDA / "glm_sm100" / "csrc" / "bindings.cpp", KDA / "glm_rec_sm100" / "csrc" / "bindings.cpp",
            KDA / "slot_reset_sm100" / "csrc" / "reset.cu")
LAUNCH = re.compile(r"\b(\w+_launch|launch_layers<[^>]*>|launch<[^>]*>)\(")


def _bound_functions(text: str) -> dict[str, str]:
    """Body of every function the module binds (``m.def("name", &ns::fn)``) and this file defines."""
    names = re.findall(r'm\.def\("\w+",\s*&(?:\w+::)?(\w+)\)', text)
    assert names
    bodies = {}
    for name in names:
        m = re.search(rf"^[^\n;{{}}]*\b{name}\([^;{{]*\)\s*\{{", text, re.M)
        if m is None:            # declared here, defined in a kernel file (e.g. chunk_len in plan.cu): no launch
            continue
        depth, i = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        bodies[name] = text[m.end():i - 1]
    return bodies


@pytest.mark.parametrize("path", BINDINGS, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_every_launching_binding_guards_its_device_and_checks_each_launch_cpu(path):
    bodies = _bound_functions(path.read_text())
    launching = {n: b for n, b in bodies.items() if LAUNCH.search(b)}
    assert launching, path
    for name, body in launching.items():
        first_launch = LAUNCH.search(body).start()
        guard = body.find("CUDAGuard")
        assert 0 <= guard < first_launch, f"{path.name}:{name}: no CUDAGuard before the first launch"
        stream = body.find("getCurrentCUDAStream")
        assert stream < 0 or guard < stream, f"{path.name}:{name}: the stream is read before the guard"
        for m in LAUNCH.finditer(body):      # every launch call has a check after it (branches may share one)
            assert "C10_CUDA_KERNEL_LAUNCH_CHECK()" in body[m.end():], f"{path.name}:{name}: {m.group(1)} unchecked"
