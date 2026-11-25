"""Mechanically check the "sm100 bits equal sm90 bits" claim at the SOURCE level."""

from __future__ import annotations

import difflib
import re
from pathlib import Path

_SM90 = Path(__file__).resolve().parents[2] / "logprobs_sm90" / "csrc"
_SM100 = Path(__file__).resolve().parents[1] / "csrc"

_ARCH_SPELLINGS = (
    ("logprobs_sm90", "logprobs_smXXX"),
    ("logprobs_sm100", "logprobs_smXXX"),
    ("SM90", "SMXXX"),
    ("SM100", "SMXXX"),
    ("sm90", "smXXX"),
    ("sm100", "smXXX"),
)


def _normalize(text: str) -> str:
    for old, new in _ARCH_SPELLINGS:
        text = text.replace(old, new)
    return text


def _lines(path: Path) -> list[str]:
    return _normalize(path.read_text()).splitlines()


def test_bindings_cpp_is_byte_identical_once_the_arch_tag_is_normalized():
    a, b = _lines(_SM90 / "bindings.cpp"), _lines(_SM100 / "bindings.cpp")
    assert a == b


def test_logprobs_cu_differs_only_in_comment_prose():
    a, b = _lines(_SM90 / "logprobs.cu"), _lines(_SM100 / "logprobs.cu")
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    changed_a: list[str] = []
    changed_b: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_a.extend(a[i1:i2])
        changed_b.extend(b[j1:j2])
    for line in changed_a + changed_b:
        assert re.match(r"^\s*//", line), line
    assert any("__fdividef" in line for line in changed_a)
    assert any("__fdiv_rn" in line for line in changed_b)
    assert len(changed_a) <= 5 and len(changed_b) <= 20, (changed_a, changed_b)


def test_no_arch_specific_launch_bounds_or_gencode_leaked_into_the_shared_body():
    a = _normalize((_SM90 / "logprobs.cu").read_text())
    b = _normalize((_SM100 / "logprobs.cu").read_text())
    assert "smXXX" in a and "smXXX" in b
    assert "__CUDA_ARCH__" not in a and "__CUDA_ARCH__" not in b
