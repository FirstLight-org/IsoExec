"""Validate the public NCCL ABI and refusal boundary without creating a group."""

import ctypes
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from isoexec.checks import Refusal
from isoexec.runtimes.vllm import nccl_channels as channels


@pytest.fixture(autouse=True)
def clear_pins(monkeypatch):
    monkeypatch.setattr(channels, "_RECEIPTS", [])
    for name in channels._PROCESS_PINS:
        monkeypatch.delenv(name, raising=False)


class UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def test_config_matches_the_pinned_public_header(tmp_path):
    python = os.environ.get("ISOEXEC_PY")
    compiler = shutil.which("cc")
    if not python or not compiler:
        pytest.skip("pinned ISOEXEC_PY and C compiler are required for the ABI comparison")
    packages = Path(python).parent.parent / "lib/python3.12/site-packages"
    header = packages / "nvidia/nccl/include/nccl.h"
    if not header.exists() or "#define NCCL_VERSION_CODE 23007" not in header.read_text():
        pytest.skip("pinned NCCL 2.30.7 header is not installed in ISOEXEC_PY")
    statements = []
    for name, field_type in channels._NcclConfig._fields_:
        statements.append(f'printf("{name}.offset %zu\\n", offsetof(ncclConfig_t, {name}));')
        value = f"(long long)config.{name}" if field_type is not ctypes.c_void_p else f"(long long)(intptr_t)config.{name}"
        statements.append(f'printf("{name}.value %lld\\n", {value});')
    source = tmp_path / "nccl_abi.c"
    binary = tmp_path / "nccl_abi"
    source.write_text('#include <stdio.h>\n#include <stddef.h>\n#include <stdint.h>\n#include <nccl.h>\n'
                      'int main(void) { ncclConfig_t config = NCCL_CONFIG_INITIALIZER;\n'
                      + "\n".join(statements) + '\nreturn 0; }\n')
    subprocess.run([compiler, "-I", str(header.parent), "-I", str(packages / "nvidia/cu13/include"),
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    values = dict(line.split() for line in subprocess.check_output([str(binary)], text=True).splitlines())
    actual = channels._NcclConfig(16, 23007)
    actual.minCTAs = channels._UNDEFINED
    actual.maxCTAs = channels._UNDEFINED
    for name, _ in channels._NcclConfig._fields_:
        assert int(values[f"{name}.offset"]) == getattr(channels._NcclConfig, name).offset
        if name == "size":
            # The runtime advertises only its supported prefix; the public
            # header initializes the complete, extensible configuration struct.
            assert actual.size == ctypes.sizeof(channels._NcclConfig)
            assert actual.size <= int(values["size.value"])
        else:
            assert int(values[f"{name}.value"]) == (getattr(actual, name) or 0)


def test_init_uses_config_on_the_original_communicator_and_records_acceptance():
    calls = []

    class Initialize:
        def __call__(self, output, world, unique_id, rank, config):
            config = ctypes.cast(config, ctypes.POINTER(channels._NcclConfig)).contents
            calls.append((world, rank, config.minCTAs, config.maxCTAs))
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 123
            return 0

    library = SimpleNamespace(lib=SimpleNamespace(ncclCommInitRankConfig=Initialize()),
                              ncclGetRawVersion=lambda: 23007,
                              NCCL_CHECK=lambda code: None if code == 0 else pytest.fail(str(code)))
    result = channels._initialize_rank(library, 2, UniqueId(), 1, 16)
    assert result.value == 123
    assert calls == [(2, 1, 1, 16)]
    assert channels._RECEIPTS[-1]["max_ctas"] == 16
    json.dumps(channels._RECEIPTS)


@pytest.mark.parametrize("version", [21699, 22601, 23008, 24000])
def test_abi_outside_the_verified_range_refuses_before_entering_nccl(version):
    """Outside the range someone read a header for, the field order is unknown, so it refuses.

    The refusal names the capability (``nccl.cta_bounds``) rather than a version number, because a
    verl or slime deployment cannot act on "requires 23007": it does not choose that pin. It can act
    on "confirm these seven fields, then widen the range".
    """
    library = SimpleNamespace(ncclGetRawVersion=lambda: version)
    with pytest.raises(Refusal, match="nccl.cta_bounds"):
        channels._initialize_rank(library, 2, UniqueId(), 0, 16)


@pytest.mark.parametrize("version", [22602, 22809, 22907, 23007])
def test_every_verified_abi_passes_the_version_gate(version):
    """The measured versions get PAST the gate -- 22907 is the one a hard pin on 23007 refused.

    The stub library has no ``ncclCommInitRankConfig``, so reaching that refusal is the evidence the
    version gate let it through; a version gate that rejected would raise ``nccl.cta_bounds`` first.
    """
    library = SimpleNamespace(ncclGetRawVersion=lambda: version, lib=SimpleNamespace())
    with pytest.raises(Refusal, match="ncclCommInitRankConfig"):
        channels._initialize_rank(library, 2, UniqueId(), 0, 16)


def test_the_mirror_is_the_prefix_it_writes_and_nothing_more():
    """The struct stops at ``maxCTAs``, and ``size`` says so.

    This is what makes one build span NCCL releases: the library reads at most ``size`` bytes, and
    ``ncclConfig_t`` grows by APPENDING. Declaring fields past the ones written would inflate
    ``size`` past an older library's own struct -- which is how a 2.30.7-shaped mirror came to be
    refused by 2.29.7 whose first seven fields are byte-identical.
    """
    names = [name for name, _ in channels._NcclConfig._fields_]
    assert names == ["size", "magic", "version", "blocking", "cgaClusterSize", "minCTAs", "maxCTAs"]
    # Offsets read off the C headers at 2.26.2 / 2.28.9 / 2.29.7 / 2.30.7, which agree.
    assert [channels._NcclConfig.__dict__[n].offset for n in names] == [0, 8, 12, 16, 20, 24, 28]
    assert ctypes.sizeof(channels._NcclConfig) == 32
    built = channels._NcclConfig(16, 22907)
    assert built.size == 32 and built.magic == 0xCAFE_BEEF
    assert (built.minCTAs, built.maxCTAs) == (1, 16)
    assert built.version == 22907, "the struct targets the RUNNING library, not one release"


@pytest.mark.parametrize("configured, expected", [(None, 16), ("8", 8), ("16", 16)])
def test_trainer_transport_resolves_the_same_engine_channel_cap(monkeypatch, configured, expected):
    class Library:
        def ncclCommInitRank(self, world, unique_id, rank):
            pytest.fail("legacy initializer must be replaced")

    monkeypatch.setitem(sys.modules, "vllm.distributed.device_communicators.pynccl_wrapper",
                        SimpleNamespace(NCCLLibrary=Library))
    monkeypatch.setattr(channels, "_CAP", None)
    monkeypatch.setattr(channels, "_initialize_rank", lambda library, world, uid, rank, cap: cap)
    if configured is None:
        monkeypatch.delenv("ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS", raising=False)
    else:
        monkeypatch.setenv("ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS", configured)
    channels.install_pynccl_channel_policy()
    assert Library().ncclCommInitRank(2, UniqueId(), 0) == expected


@pytest.mark.parametrize("name", channels._PROCESS_PINS)
def test_process_pin_refuses_before_entering_nccl(monkeypatch, name):
    monkeypatch.setenv(name, "1")
    with pytest.raises(Refusal, match=name):
        channels._initialize_rank(None, 2, UniqueId(), 0, 16)


def test_environment_clear_changes_only_the_three_process_pins():
    env = {**{name: "1" for name in channels._PROCESS_PINS}, "NCCL_DEBUG": "INFO"}
    channels.clear_process_pins(env)
    assert env == {"NCCL_DEBUG": "INFO"}


@pytest.mark.parametrize("fails", [False, True])
def test_vllm_override_preserves_inherited_transport_settings(monkeypatch, fails):
    from isoexec.runtimes.vllm.vllm_patches import (
        _VLLM_NCCL_TRANSPORT_OVERRIDES, _without_nccl_overrides,
    )

    for name in _VLLM_NCCL_TRANSPORT_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth0")
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "1")

    def native_override():
        for name in (*_VLLM_NCCL_TRANSPORT_OVERRIDES, *channels._PROCESS_PINS):
            os.environ[name] = "engine-only"
        os.environ["VLLM_USE_AOT_COMPILE"] = "0"
        if fails:
            raise RuntimeError("native initialization failed")
        return 42

    override = _without_nccl_overrides(native_override)
    if fails:
        with pytest.raises(RuntimeError, match="native initialization failed"):
            override()
    else:
        assert override() == 42
    for name in _VLLM_NCCL_TRANSPORT_OVERRIDES:
        assert os.environ.get(name) == ("1" if name == "NCCL_NVLS_ENABLE" else None)
    assert not any(name in os.environ for name in channels._PROCESS_PINS)
    assert os.environ["NCCL_SOCKET_IFNAME"] == "eth0"
    assert os.environ["VLLM_USE_AOT_COMPILE"] == "0"


def test_override_inventory_covers_every_pinned_runtime_nccl_write():
    from isoexec.runtimes.vllm.vllm_patches import _VLLM_NCCL_TRANSPORT_OVERRIDES

    python = os.environ.get("ISOEXEC_PY")
    if not python:
        pytest.skip("pinned ISOEXEC_PY is required")
    source = (Path(python).parent.parent / "lib/python3.12/site-packages/vllm/"
              "model_executor/determinism/batch_invariant.py")
    module = ast.parse(source.read_text())
    override = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == "override_envs_for_invariance")
    writes = {target.slice.value for node in ast.walk(override) if isinstance(node, ast.Assign)
              for target in node.targets if isinstance(target, ast.Subscript)
              and ast.unparse(target.value) == "os.environ" and isinstance(target.slice, ast.Constant)
              and str(target.slice.value).startswith("NCCL_")}
    assert writes == set(_VLLM_NCCL_TRANSPORT_OVERRIDES) | set(channels._PROCESS_PINS)


def test_original_torch_group_receives_options_without_extra_groups(monkeypatch):
    import torch.distributed as dist

    calls = []

    def original(ranks=None, backend=None, pg_options=None):
        calls.append((ranks, backend, pg_options))
        return SimpleNamespace(_get_backend=lambda device: SimpleNamespace(options=pg_options))

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    create = channels._wrap_torch_group(original, 16)
    create([0, 1], "nccl")
    assert len(calls) == 1
    assert calls[0][0:2] == ([0, 1], "nccl")
    assert (calls[0][2].config.min_ctas, calls[0][2].config.max_ctas) == (1, 16)
    create([0, 1], "gloo")
    assert len(calls) == 2
    assert calls[-1][2] is None


def test_conflicting_native_options_refuse_before_group_creation():
    import torch.distributed as dist

    def original(ranks=None, backend=None, pg_options=None):
        pytest.fail("must refuse before creating a communicator")

    options = dist.ProcessGroupNCCL.Options()
    options.config.max_ctas = 8
    with pytest.raises(Refusal, match="pg_options.max_ctas"):
        channels._wrap_torch_group(original, 16)(backend="nccl", pg_options=options)
