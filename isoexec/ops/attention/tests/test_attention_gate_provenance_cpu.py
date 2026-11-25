"""Check the vendored gate's own copy: provenance record, recorded digests, the recorded jit_fuser
substitution and the compile decoration. The copy was qualified when it was vendored; it is not
re-diffed against an installed Megatron."""

from __future__ import annotations

import importlib

from isoexec.ops.tests._vendor_copy import (
    assert_copy_matches_recorded_digests,
    assert_provenance_record_is_complete,
)

V = importlib.import_module("isoexec.ops.attention.attention_gate")

UPSTREAM_MODULE = "megatron.core.transformer.attention"


def test_provenance_record_is_complete():
    assert_provenance_record_is_complete(
        V,
        upstream_module=UPSTREAM_MODULE,
        keys=("project", "module", "version", "commit", "url", "vendored_on", "symbols", "substitutions"),
        holder_key="vendored_symbol_holder",
    )


def test_vendored_copy_matches_its_own_recorded_digests():
    assert_copy_matches_recorded_digests(
        V, requalify="the full-shape run's exact-zero rollout/trainer logprob gate (the gate is elementwise and runs in both)"
    )


def test_the_decoration_is_still_applied():
    """An undecorated copy is a different function, and only a GPU gate would notice.

    `torch.compile` returns a wrapper carrying `_torchdynamo_orig_callable` (older torch:
    `_torchdynamo_inline`); a plain `def` carries neither. The check is deliberately structural
    rather than behavioural so it runs on a CPU box with no GPU and no megatron.
    """
    holder = V.provenance()["vendored_symbol_holder"]
    fn = getattr(getattr(V, holder), "_apply_output_gate")
    marks = ("_torchdynamo_orig_callable", "_torchdynamo_inline", "get_compiler_config", "_torchdynamo_disable")
    assert any(hasattr(fn, m) for m in marks), (
        "the vendored `_apply_output_gate` is no longer torch.compile-decorated. That is a "
        "ROUNDING CHANGE, not a packaging change: inductor's sigmoid and ATen's disagree by one "
        "ulp on some inputs, so the undecorated function is a different impl and `attention.gate`'s "
        "recorded evidence no longer describes it. See the module docstring's measurement."
    )
    assert V.jit_fuser is __import__("torch").compile, "jit_fuser is no longer torch.compile"


def test_the_substitution_table_records_the_jit_fuser_rule():
    """`jit_fuser` is re-expressed, not copied: the table must record the rule and its upstream text."""
    assert V.provenance()["substitutions"], "the substitution table no longer records the jit_fuser rule"
    assert "jit_fuser" in V._JIT_FUSER_UPSTREAM_STMT


if __name__ == "__main__":  # pragma: no cover
    import sys

    from isoexec.ops.tests._vendor_copy import run_as_script

    sys.exit(run_as_script(globals()))
