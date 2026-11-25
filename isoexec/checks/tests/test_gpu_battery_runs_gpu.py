"""Run each GPU battery in a separate process. Missing CUDA skips unless
ISOEXEC_GPU_BATTERIES=required; incomplete or failed batteries fail the test.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from isoexec.checks.tests.test_gpu_battery_manifest_cpu import (
    BATTERIES,
    REQUIRE_ENV,
    Battery,
    _REPO_ROOT,
)


@pytest.mark.parametrize("battery", BATTERIES, ids=lambda b: b.path.split("/")[-1][: -len("_gpu.py")])
def test_battery(battery: Battery, record_property):
    record_property("claim", battery.claim)
    import torch

    if not torch.cuda.is_available():
        reason = f"NO GPU -> UNWITNESSED: {battery.claim}"
        if os.environ.get(REQUIRE_ENV) == "required":
            pytest.fail(f"{REQUIRE_ENV}=required but no CUDA device is present. {reason}")
        pytest.skip(reason)

    env = dict(os.environ, **battery.extra_env)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    try:
        done = subprocess.run(
            battery.command(), cwd=str(_REPO_ROOT), env=env,
            capture_output=True, text=True, timeout=battery.timeout_s,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{battery.path} exceeded {battery.timeout_s}s. UNWITNESSED: {battery.claim}")
    tail = "\n".join((done.stdout + done.stderr).splitlines()[-40:])
    if done.returncode == 2:
        pytest.fail(
            f"{battery.path} exited 2 = INCOMPLETE: the claim could not be made on this node, which "
            f"is a gap, not a pass. UNWITNESSED: {battery.claim}\n{tail}"
        )
    assert done.returncode == 0, f"{battery.path} exited {done.returncode}\n{tail}"
