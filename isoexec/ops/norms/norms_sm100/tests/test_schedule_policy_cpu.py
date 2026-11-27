"""norms_sm100 forward: ``schedule.py`` against the policy and against what the kernels' plan code uses.

CPU only, no extension build: the plan values are read from ``csrc/plan.h`` and ``csrc/launch_fwd.cu`` as text.
Fails if the segment differs from :func:`isoexec.ops.schedule_policy.reduce`, if the lane count, the lane-cap
table or the recipe binding in the header differ from ``schedule.py``, or if the stated DAG differs from the
tree's executable statements (``reference.py`` / ``reference_v2.py``).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from isoexec.ops import schedule_policy as policy
from isoexec.ops._testing import bitwise_equal
from isoexec.ops.norms.norms_sm90 import reference as R1
from isoexec.ops.norms.norms_sm100 import reference_v2 as R2
from isoexec.ops.norms.norms_sm100 import schedule, spec

PKG = Path(__file__).resolve().parents[1]
PLAN = (PKG / "csrc" / "plan.h").read_text()
LAUNCH = (PKG / "csrc" / "launch_fwd.cu").read_text()

#: Every width the forward supports up to 16384 (plan.h::fwd_supported): the table's entries.
SUPPORTED = (128, 384) + tuple(range(256, 16384 + 1, 256))


def _body(name: str, text: str = PLAN) -> str:
    """The body of the inline function ``name`` in plan.h, whitespace collapsed."""
    m = re.search(r"inline\s+[\w:]+\s+" + name + r"\s*\([^)]*\)\s*\{", text)
    assert m, name
    depth, i = 1, m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return " ".join(text[m.end() : i - 1].split())


def _switch(name: str) -> tuple[dict[int, int], int]:
    """``({case label: returned value}, default value)`` of a ``switch (n)`` function in plan.h."""
    body = re.sub(r"//[^\n]*", "", PLAN[PLAN.index(name + "(int64_t n)") :].split("\n}\n", 1)[0])
    table: dict[int, int] = {}
    pending: list[int] = []
    default = None
    for label, value in re.findall(r"case (\d+):|return (\d+);", body):
        if label:
            pending.append(int(label))
        elif pending:
            table.update(dict.fromkeys(pending, int(value)))
            pending = []
        else:
            default = int(value)
    assert table and default is not None and re.search(r"default:\s*return " + str(default), body), name
    return table, default


def _header_v1(n: int) -> tuple[int, int]:
    """``fwd_lanes`` with its constants read from the header."""
    body = _body("fwd_lanes")
    m = re.fullmatch(
        r"if \(n % (\d+) == 0\) return \{(\d+), int\(n / (\d+)\)\}; TORCH_CHECK\(n == (\d+) \|\| n == (\d+),.*\); "
        r"return \{(\d+), int\(n / (\d+)\)\};", body)
    assert m, body
    mod, lanes_a, div_a, n1, n2, lanes_b, div_b = map(int, m.groups())
    if n % mod == 0:
        return lanes_a, n // div_a
    assert n in (n1, n2)
    return lanes_b, n // div_b


def _header_v2(n: int) -> tuple[int, int]:
    """``fwd_lanes_v2(n, fwd_v2_lmax(n))`` with the cap table read from the header."""
    body = _body("fwd_lanes_v2")
    assert "int64_t e = n / 8;" in body and "int L = 1; while (L * 2 <= lmax && e % int64_t(L * 2) == 0) L *= 2; " \
        "return {L, int(e / L)};" in body, body
    table, default = _switch("fwd_v2_lmax")
    cap, e, lanes = table.get(n, default), n // 8, 1
    while lanes * 2 <= cap and e % (lanes * 2) == 0:
        lanes *= 2
    return lanes, e // lanes


def _instantiated(macro: str) -> set[tuple[int, int]]:
    text = LAUNCH[LAUNCH.index("#define " + macro + "(X)") :].split("\n\n", 1)[0].split("#define GATED", 1)
    block = text[0] if macro == "FWD_WIDTHS" else "#define GATED" + text[1]
    return {(int(k), int(ell)) for k, ell in re.findall(r"X\((\d+), (\d+)\)", block)}


@pytest.mark.parametrize("n", SUPPORTED)
def test_the_segment_is_the_policy_output(n):
    """One segment per shard, d_max = 1: G = 1, S = width, and the lanes partition exactly that segment."""
    seg = schedule.segment(n)
    assert seg == policy.reduce(n // schedule.HIDDEN_SHARDS) == policy.Segments(n, 1, "tree")
    red = schedule.reducer(n)
    ell, k = schedule.lanes(n)
    assert ell * k * schedule.LEAF == seg.S
    cols = [c for lane in range(ell) for c in red.members(seg.S, lane)]
    assert sorted(cols) == list(range(seg.S))


def test_supported_widths_are_the_headers():
    assert "return n > 0 && (n % 256 == 0 || n == 128 || n == 384);" in _body("fwd_supported")
    for n in range(0, 16385):
        assert schedule.V1.supported(n) == (n in SUPPORTED), n


def test_the_lane_function_equals_the_v1_table_on_every_entry():
    for n in SUPPORTED:
        assert (schedule.V1.lanes(n), schedule.V1.visits(n)) == _header_v1(n) == R1.lanes(n), n


def test_the_lane_function_equals_the_v2_table_on_every_entry():
    table, default = _switch("fwd_v2_lmax")
    assert table == dict.fromkeys(schedule.V2_SWEPT, schedule.V2.cap) and default == schedule.V2.cap_elsewhere
    assert R2.LMAX == table
    for n in SUPPORTED:
        assert (schedule.V2.lanes(n), schedule.V2.visits(n)) == _header_v2(n) == R2.lanes(n), n
        if n not in schedule.V2_SWEPT:
            assert schedule.V2.lanes(n) == schedule.V1.lanes(n), n


def test_the_lane_count_is_the_policys_and_every_width_keeps_its_lanes():
    """The most lanes are ``policy.lanes(accumulators)``, one accumulator per thread sharing the row: 32 (one warp)
    for v1, 128 (four warps) for v2 at the swept
    widths. On every width of :data:`SUPPORTED` the lanes are what they were: the largest power of two <= the earlier
    literal cap (32, or 128 at a swept width) that divides n / 8, for both reducers and for the bound one."""
    assert (schedule.V1.accumulators, schedule.V1.accumulators_elsewhere) == (32, 32)
    assert (schedule.V2.accumulators, schedule.V2.accumulators_elsewhere) == (128, 32)
    assert schedule.V1.cap == schedule.V1.cap_elsewhere == schedule.V2.cap_elsewhere == policy.lanes(32) == 32
    assert schedule.V2.cap == policy.lanes(128) == 128
    for n in SUPPORTED:
        was = {1: schedule.pow2_dividing(n // 8, 32),
               2: schedule.pow2_dividing(n // 8, 128 if n in schedule.V2_SWEPT else 32)}
        assert schedule.V1.lanes(n) == was[1] and schedule.V2.lanes(n) == was[2], n
        assert schedule.lanes(n)[0] == was[schedule.BINDING.get(n, schedule.DEFAULT_VERSION)], n


def test_the_binding_is_the_headers():
    table, default = _switch("fwd_recipe_default")
    assert table == schedule.BINDING and default == schedule.DEFAULT_VERSION
    assert set(schedule.BINDING) == set(R2.DEFAULT_V2)
    assert "return fwd_recipe_default(n);" in _body("fwd_recipe")
    active = _body("fwd_lanes_active")
    assert active == "if (fwd_recipe(n) != 2) return fwd_lanes(n); int lmax = fwd_v2_lmax(n); return fwd_lanes_v2(n, lmax);"
    assert schedule.V1.impl_id == spec.RMS_IMPL and schedule.V2.impl_id == spec.RMS_IMPL_V2
    assert (schedule.V1.version, schedule.V2.version) == spec.RECIPE_VERSIONS


def test_the_launch_code_uses_the_bound_lanes():
    """The forward launch reads the active plan, and every width of the tree's batteries has its kernel."""
    rms = LAUNCH[LAUNCH.index("void launch_rms_fwd(") : LAUNCH.index("void launch_gated_fwd(")]
    assert "Lanes l = fwd_lanes_active(x.size(1));" in rms and "fwd_lanes(" not in rms
    assert "Lanes l = fwd_lanes(x.size(1));" in LAUNCH[LAUNCH.index("void launch_gated_fwd(") :]
    built = _instantiated("FWD_WIDTHS")
    for n in schedule.WIDTHS:
        ell, k = schedule.lanes(n)
        assert (k, ell) in built, (n, ell, k)
        assert (ell, k) == (_header_v2(n) if schedule.BINDING.get(n) == 2 else _header_v1(n)), n
    for k, ell in _instantiated("GATED_WIDTHS"):
        assert schedule.V1.lanes(8 * ell * k) == ell, (k, ell)


def test_the_kernel_text_is_the_stated_lane_program():
    """Lane membership, chain step and butterfly as ``schedule.py`` states them (text of the base sources)."""
    fwd = " ".join((PKG / "csrc" / "fwd.cuh").read_text().split())
    common = " ".join((PKG / "csrc" / "common.cuh").read_text().split())
    assert "int lane = threadIdx.x % L;" in fwd and "float acc = 0.f;" in fwd
    assert fwd.count("acc = __fadd_rn(acc, __fmul_rn(") == 3 and "acc = lane_fold<L>(acc);" in fwd
    assert "k * 8 * L + 8 * lane" in fwd
    assert "for (int s = L >> 1; s > 0; s >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, s));" in common
    assert "v[q] = __fadd_rn(v[q], v[q + s]);" in common and "return warp_fold(v[0], 32);" in common


@pytest.mark.parametrize("n", schedule.WIDTHS + (256, 1024, 8192))
@pytest.mark.parametrize("version", (1, 2))
def test_the_stated_dag_is_the_executable_statement(n, version):
    """Members and strides evaluated from the data = the tree's reference, bit for bit (CPU fp32)."""
    red = schedule.REDUCERS[version]
    ref = R1 if version == 1 else R2
    idx = torch.tensor([red.members(n, lane) for lane in range(red.lanes(n))])
    assert torch.equal(idx, ref._swizzle(n))
    g = torch.Generator().manual_seed(n + version)
    x = (torch.randn(37, n, generator=g) * 3.0).bfloat16()
    x[3], x[5, ::7], x[6, 1] = 0.0, -0.0, float("inf")
    x[7, n // 2] = float("nan")
    want = red.reference_fn()(x)
    assert bitwise_equal(schedule.evaluate(x, red), want), (n, version)
    if schedule.reducer(n) is red:
        assert bitwise_equal(schedule.evaluate(x), want)


def test_the_policy_is_not_on_a_call_path():
    """No module of the package imports ``schedule`` or the policy; importing the ops does not load them."""
    for path in sorted(PKG.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".h", ".cpp") and "tests" not in path.parts \
                and path.name != "schedule.py":
            text = path.read_text()
            assert "schedule_policy" not in text and "import schedule" not in text and "schedule." not in text, path
    code = ("import sys; import isoexec.ops.norms.norms_sm100.ops, isoexec.ops.norms.norms_sm100.integration; "
            "assert 'isoexec.ops.norms.norms_sm100.schedule' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
