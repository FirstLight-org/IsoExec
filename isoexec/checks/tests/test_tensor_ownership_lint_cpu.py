"""Allocation defaults and auxiliary Tensor attributes fail the production lint."""

from isoexec.checks.lint import (
    scan_tensor_ownership,
    tensor_ownership_violations,
)


def test_tensor_ownership_tree_is_clean():
    """Clean, AND the walk that decided so actually visited files.

    ``tensor_ownership_violations`` walks ``TENSOR_ROOTS`` through ``_walk_py``, which skips a root
    that is not a directory without raising, and then drops every path with ``tests`` in it. Both
    are silent, so ``== []`` is equally the answer for "nothing is wrong" and "nothing was read".
    """
    import isoexec.checks.lint as lint

    scanned = [
        rel for _, rel in lint._walk_py(lint.ISOEXEC_DIR, lint.TENSOR_ROOTS, ())
        if "tests" not in rel.split("/") and not rel.split("/")[-1].startswith("test_")
    ]
    assert len(scanned) >= 100, f"the tensor-ownership walk visited {len(scanned)} production files"
    assert tensor_ownership_violations() == []


def test_allocation_and_tensor_alias_negatives(tmp_path):
    path = tmp_path / "kernel.py"
    path.write_text('''
import torch

def kernel(x: torch.Tensor):
    a = torch.empty(4, dtype=torch.float32)
    b = torch.zeros(4, device="cpu")
    good = torch.ones(4, dtype=torch.float32, device="cpu")
    alias = x
    setattr(alias, "rows", good)
    setattr(torch.zeros(4, dtype=torch.int64, device="cpu"), "metadata", 1)
    return a, b

def model(x: ModelConfig):
    setattr(x, "metadata", 1)
''')
    problems = scan_tensor_ownership(str(path), "ops/kernel.py")
    assert sum("IMPLICIT TENSOR ALLOCATION" in p for p in problems) == 2
    assert sum("TENSOR ATTRIBUTE" in p for p in problems) == 2
    assert len(problems) == 4
