"""Test engine-config field diagnostics against stub runtime trees."""

from __future__ import annotations

import os
import textwrap

from isoexec.checks.lint import (
    refusal_field_violations,
    scan_field_reads,
)

STUB = textwrap.dedent(
    '''
    """A stub runtime module."""

    def install_refusals(parallel_config, vllm_config):
        out = []
        if not parallel_config.enable_deterministic_inference:
            out.append("deterministic mode must be on")
        if getattr(vllm_config, "enforce_eager", False):
            out.append("enforce_eager is not admissible")
        return out
    '''
)


def _write(tmp_path, body: str):
    root = tmp_path / "isoexec"
    (root / "runtimes" / "vllm").mkdir(parents=True)
    (root / "runtimes" / "vllm" / "stub.py").write_text(body)
    return str(root)


def test_a_defaulted_getattr_over_an_engine_config_is_a_violation(tmp_path):
    root = _write(tmp_path, STUB)
    problems = refusal_field_violations(root, roots=(os.path.join("runtimes", "vllm"),))
    assert any("DEFAULTED REFUSAL FIELD" in p and "enforce_eager" in p for p in problems)


def test_a_plain_attribute_read_is_not_a_violation_by_itself(tmp_path):
    root = _write(tmp_path, STUB)
    problems = refusal_field_violations(root, roots=(os.path.join("runtimes", "vllm"),))
    # Require the stub violation to be visited before trusting an empty result.
    assert problems, "the refusal-field scan produced nothing at all; it read no file"
    assert not any("enable_deterministic_inference" in p and "DEFAULTED" in p for p in problems)


def test_an_unverifiable_run_is_a_failure_not_a_pass(tmp_path):
    """Neither engine wheel is installed in the CPU job, so the rule must say so out loud."""
    from isoexec.checks.lint import _engine_config_fields

    root = _write(tmp_path, STUB)
    problems = refusal_field_violations(root, roots=(os.path.join("runtimes", "vllm"),))
    if _engine_config_fields() is None:
        assert any("UNVERIFIABLE REFUSAL FIELDS" in p for p in problems)
    else:
        # When engine fields are available, require validation rather than an unverifiable
        # report.
        assert not any("UNVERIFIABLE REFUSAL FIELDS" in p for p in problems), (
            "the engine config fields are readable here, so the lint must not report itself "
            f"unverifiable: {problems}"
        )


def test_the_scanner_reads_both_forms(tmp_path):
    root = _write(tmp_path, STUB)
    reads = scan_field_reads(os.path.join(root, "runtimes", "vllm", "stub.py"), "runtimes/vllm/stub.py")
    got = {(r.obj, r.field, r.defaulted) for r in reads}
    assert ("parallel_config", "enable_deterministic_inference", False) in got
    assert ("vllm_config", "enforce_eager", True) in got


def test_native_config_types_are_scoped_and_cover_local_annotations(tmp_path, monkeypatch):
    from isoexec.checks import lint

    source = tmp_path / "native.py"
    source.write_text(textwrap.dedent('''
        def native(cfg: TransformerConfig, opt: "OptimizerConfig"):
            if getattr(cfg, "sequence_parallel", False):
                return opt.optimizer_cpu_offload
            return cfg.sequnce_parallel

        def model(cfg: ModelConfig):
            return cfg.hidden_size, cfg.norm

        def local(router):
            config: TransformerConfig = router.config
            return config.moe_router_fusion
    '''))
    reads = lint.scan_field_reads(str(source), "ops/new_native.py")
    assert {(r.obj, r.field, r.defaulted, r.config_type) for r in reads} == {
        ("cfg", "sequence_parallel", True, "TransformerConfig"),
        ("cfg", "sequnce_parallel", False, "TransformerConfig"),
        ("opt", "optimizer_cpu_offload", False, "OptimizerConfig"),
        ("config", "moe_router_fusion", False, "TransformerConfig"),
    }
    monkeypatch.setattr(lint, "_engine_config_fields", lambda: {
        "TransformerConfig": {"sequence_parallel", "moe_router_fusion"},
        "OptimizerConfig": {"optimizer_cpu_offload"},
    })
    problems = lint._unknown_field_problems(reads)
    assert len(problems) == 1 and "sequnce_parallel" in problems[0]


def test_required_dataclass_fields_are_declared():
    from dataclasses import dataclass

    from isoexec.checks.lint import _declared_fields

    @dataclass
    class Native:
        required: int
        optional: bool = False

    assert {"required", "optional"} <= _declared_fields(Native)
