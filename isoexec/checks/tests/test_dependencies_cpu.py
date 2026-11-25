"""The dependency envelope: ranges are self-consistent, and a refusal names the surface."""

import dataclasses

import pytest

from isoexec.checks import Refusal, dependencies


def test_every_verified_point_lies_inside_its_accepted_range():
    """``verified`` is where the gate RAN; it cannot sit outside what IsoExec consents to run on.

    A verified point outside the accepted range would mean the gate proved something about a
    version the code then refuses -- evidence for a configuration nobody can deploy.
    """
    for dep in dependencies.DEPENDENCIES:
        low, high = dep.accepted
        assert low <= high, dep.name
        for point in dep.verified:
            assert low <= point <= high, f"{dep.name}: verified {point} outside accepted {dep.accepted}"


def test_every_dependency_says_what_it_is_used_for():
    """``uses`` is the instruction for widening the range, so it must name a surface, not a version."""
    for dep in dependencies.DEPENDENCIES:
        assert len(dep.uses) > 30, dep.name
        assert not dep.uses.strip().startswith("version"), dep.name


def test_survey_never_raises_even_when_a_library_is_absent(monkeypatch):
    """A survey is diagnosis; it must work on a box where half the stack is missing."""
    monkeypatch.setattr(dependencies, "DEPENDENCIES", (
        dependencies.Dependency(name="absent_thing", uses="a surface that does not exist anywhere here",
                                accepted=((1, 0), (2, 0)), verified=((1, 0),),
                                read=lambda: None, required=False),
    ))
    rows = dependencies.survey()
    assert rows[0]["found"] is None and rows[0]["in_accepted"] is False
    dependencies.check()


def test_a_version_below_or_above_the_range_refuses_and_names_the_surface(monkeypatch):
    dep = dependencies.Dependency(
        name="nccl", uses="ncclConfig_t's first seven fields (through maxCTAs) for channel bounds",
        accepted=((2, 26), (2, 30)), verified=((2, 30),), read=lambda: (2, 31),
    )
    monkeypatch.setattr(dependencies, "DEPENDENCIES", (dep,))
    with pytest.raises(Refusal, match="ncclConfig_t"):
        dependencies.check()
    monkeypatch.setattr(dependencies, "DEPENDENCIES",
                        (dataclasses.replace(dep, read=lambda: (2, 25)),))
    with pytest.raises(Refusal, match="dependency.nccl"):
        dependencies.check()


def test_a_required_library_that_is_absent_refuses(monkeypatch):
    monkeypatch.setattr(dependencies, "DEPENDENCIES", (
        dependencies.Dependency(name="torch", uses="aten op semantics and the cuBLAS workspace",
                                accepted=((2, 13), (2, 14)), verified=((2, 14),),
                                read=lambda: None, required=True),))
    with pytest.raises(Refusal, match="dependency.torch: not importable"):
        dependencies.check()


def test_unverified_reports_accepted_but_ungated_versions(monkeypatch):
    """The gap this module exists to SURFACE: runnable, but not where the evidence was measured."""
    monkeypatch.setattr(dependencies, "DEPENDENCIES", (
        dependencies.Dependency(name="torch", uses="aten op semantics and the cuBLAS workspace",
                                accepted=((2, 13), (2, 14)), verified=((2, 14),),
                                read=lambda: (2, 13), required=True),))
    assert dependencies.unverified() == ["torch"]
    dependencies.check()


def test_the_nccl_range_covers_both_environments_this_repo_is_run_in():
    """Regression guard for the failure that motivated this module.

    The certified venv ships NCCL 2.30 and SkyRL's pinned torch ships 2.29. Both must be accepted:
    their ``ncclConfig_t`` prefixes were read and are byte-identical, and a range that excluded
    either would refuse a run over a difference that touches no bit.
    """
    nccl = next(d for d in dependencies.DEPENDENCIES if d.name == "nccl")
    low, high = nccl.accepted
    for shipped in ((2, 29), (2, 30)):
        assert low <= shipped <= high, shipped


def test_an_advisory_dependency_reports_instead_of_refusing(monkeypatch, capsys):
    """A requirement only some compositions reach must not be refused by a table that cannot see
    the model.

    The shipped compositions bind arch norm kernels at every norm site and never call F.rms_norm,
    so refusing them over nvidia-cutlass-dsl would block a run on a library they do not execute.
    """
    monkeypatch.setattr(dependencies, "DEPENDENCIES", (
        dependencies.Dependency(
            name="nvidia-cutlass-dsl", uses="the reduction behind torch.nn.functional.rms_norm",
            accepted=((4, 5, 2), (4, 5, 2)), verified=((4, 5, 2),), read=lambda: (4, 6, 2),
            required=False, kind="numerics",
            advisory="only the eager torch RMSNorm expressions reach this reduction"),))
    dependencies.check(host="test")
    assert "not refused because only the eager" in capsys.readouterr().out


def test_a_numerics_mismatch_without_an_advisory_refuses(monkeypatch):
    """Without a stated reason to only report, the table must refuse."""
    monkeypatch.setattr(dependencies, "DEPENDENCIES", (
        dependencies.Dependency(
            name="nvidia-cutlass-dsl", uses="the reduction behind torch.nn.functional.rms_norm",
            accepted=((4, 5, 2), (4, 5, 2)), verified=((4, 5, 2),), read=lambda: (4, 6, 2),
            required=False, kind="numerics"),))
    with pytest.raises(Refusal, match="nvidia-cutlass-dsl"):
        dependencies.check()
