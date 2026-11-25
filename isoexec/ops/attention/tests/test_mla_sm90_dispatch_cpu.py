"""The MLA wrappers forward the role-free kernel ABI without device-side metadata."""

import sys
from types import SimpleNamespace


def test_role_free_kernel_calls(monkeypatch):
    from isoexec.ops.attention.mla_sm90.ops import kernel_fwd, kernel_plan

    operands = tuple(object() for _ in range(9))
    seen = []
    result = object()

    def fwd(q, kv, table, cu, lengths, scale, segment, maximum, *, out):
        seen.append((q, kv, table, cu, lengths, scale, segment, maximum, out))
        return result

    def plan(cu, lengths, segment, heads, maximum, *, workspace):
        seen.append((cu, lengths, segment, heads, maximum, workspace))
        return result

    monkeypatch.setitem(sys.modules, "isoexec.ops.attention.mla_sm90.fwd",
                        SimpleNamespace(mla_sm90_fwd=fwd, mla_sm90_plan=plan))
    assert kernel_fwd(*operands[:8], out=operands[8]) is result
    assert seen.pop() == operands
    assert kernel_plan(*operands[:5], workspace=operands[5]) is result
    assert seen.pop() == operands[:6]
