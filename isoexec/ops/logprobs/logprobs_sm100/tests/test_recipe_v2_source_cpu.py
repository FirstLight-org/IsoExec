"""Recipe v2 changes the leaf state and NOTHING else -- checked mechanically, not by review."""

from __future__ import annotations

import difflib
import re
from pathlib import Path

_CSRC = Path(__file__).resolve().parents[1] / "csrc"
_V1 = _CSRC / "logprobs.cu"
_V2 = _CSRC / "logprobs_v2.cu"

_SPELLINGS = (("logprobs_sm100_v2", "logprobs_smNS"), ("logprobs_sm100", "logprobs_smNS"))

_BLOCK_START = "// Leaf state (M_g, S_g)"
_BLOCK_END = "// The row merge (shared by merge_rows"


def _lines(path: Path) -> list[str]:
    text = path.read_text()
    for old, new in _SPELLINGS:
        text = text.replace(old, new)
    return text.splitlines()


def _block(lines: list[str]) -> tuple[int, int]:
    starts = [i for i, l in enumerate(lines) if l.startswith(_BLOCK_START)]
    ends = [i for i, l in enumerate(lines) if l.startswith(_BLOCK_END)]
    assert len(starts) == 1 and len(ends) == 1, (starts, ends)
    assert starts[0] < ends[0]
    return starts[0], ends[0]


def test_v2_differs_from_v1_only_inside_the_leaf_state_block():
    a, b = _lines(_V1), _lines(_V2)
    a_lo, a_hi = _block(a)
    b_lo, b_hi = _block(b)
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    outside: list[str] = []
    changed_regions = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_regions += 1
        for i in range(i1, i2):
            if not (a_lo <= i < a_hi or re.match(r"^\s*//", a[i])):
                outside.append(f"v1:{i + 1}: {a[i]}")
        for j in range(j1, j2):
            if not (b_lo <= j < b_hi or re.match(r"^\s*//", b[j])):
                outside.append(f"v2:{j + 1}: {b[j]}")
    assert not outside, outside
    assert changed_regions >= 2, changed_regions
    assert a != b


def test_the_two_recipes_share_the_merge_write_and_vjp_text_verbatim():
    a, b = _lines(_V1), _lines(_V2)
    assert a[_block(a)[1]:] == b[_block(b)[1]:]


def test_everything_above_the_leaf_state_is_shared_too():
    """The geometry constants, the bf16 widening, pack8 and the temperature divide: identical text."""
    a, b = _lines(_V1), _lines(_V2)
    head_a, head_b = a[2:_block(a)[0]], b[2:_block(b)[0]]  # line 2 is the version in the header comment
    assert head_a == head_b
    for lines in (a, b):
        assert "constexpr int STRIP = 8;    // columns per strip (recipe)" in lines
        assert "constexpr int LANES = 256;  // lanes per leaf (recipe); strip q of a leaf belongs to lane q % LANES" in lines
        assert "constexpr int G = 8;        // leaves per full vocabulary row (recipe)" in lines


def test_v2_leaf_state_is_branchless_and_v1_is_not():
    a, b = _lines(_V1), _lines(_V2)
    v1_body = "\n".join(l for l in a[slice(*_block(a))] if not l.lstrip().startswith("//"))
    v2_body = "\n".join(l for l in b[slice(*_block(b))] if not l.lstrip().startswith("//"))
    assert "if (v > m)" in v1_body and "else if (v != NINF)" in v1_body
    assert "if (v > m)" not in v2_body and "NINF)" not in v2_body.replace("m == NINF", "").replace("mm == NINF", "")
    conds = [l.strip() for l in v2_body.splitlines() if re.search(r"\b(if|\?)\b|\?", l)]
    assert all(("threadIdx.x == 0" in c) or ("NINF" in c) or ("constexpr (TEMP)" in c)
               or ("constexpr (PACKED)" in c) or ("lmax[l] > mm" in c)
               or ("constexpr (ENT)" in c) or ("[ENT ? LANES : 1]" in c) for c in conds), conds  # compile time


def test_each_extension_compiles_only_its_own_recipe():
    from isoexec.ops.logprobs.logprobs_sm100 import _extension, _extension_sampler, _extension_v2

    assert _extension.SOURCES == ("bindings.cpp", "logprobs.cu")
    assert _extension_v2.SOURCES == ("bindings_v2.cpp", "logprobs_v2.cu")
    # The decode sampler builds its own module and reaches the recipes through ops._ext only.
    assert _extension_sampler.SOURCES == ("sampler_bindings.cpp", "sampler.cu")
    sources = (_extension.SOURCES, _extension_v2.SOURCES, _extension_sampler.SOURCES)
    assert sum(len(s) for s in sources) == len(set().union(*sources))
    assert sorted(p.name for p in _CSRC.iterdir() if p.suffix in (".cu", ".cpp", ".cuh", ".h")) == [
        "bindings.cpp", "bindings_v2.cpp", "logprobs.cu", "logprobs_v2.cu", "sampler.cu", "sampler_bindings.cpp"
    ]


def test_the_v2_bindings_export_exactly_v1s_entry_points():
    """Same six names, so ops.py routes between recipes by swapping one object (see _extension_v2)."""
    def exported(path: Path) -> list[str]:
        return sorted(re.findall(r'm\.def\("([a-z_]+)"', path.read_text()))

    assert exported(_CSRC / "bindings_v2.cpp") == exported(_CSRC / "bindings.cpp")
    assert exported(_CSRC / "bindings.cpp") == ["entropy_rows", "full_rows", "grad_rows", "leaf_states", "merge_rows",
                                                     "write_rows"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
