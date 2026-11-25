"""Every prefill claim holds in all three prefill modes (ops.choose_prefill_mode): the tests of the modules below
run with two-pass, fully fused and prepared prefill forced. A test about the two-pass schedule
alone is marked ``two_pass_only``."""

from __future__ import annotations

import pytest

from isoexec.ops.gdn.gdn_cpr_sm90 import ops as G

_BOTH_MODES = {"test_identity_gpu", "test_splits_and_slots_gpu", "test_packs_gpu", "test_composition_gpu",
               "test_poisoned_records_gpu", "test_stress_gpu", "test_engine_state_gpu", "test_backward_gpu", "test_record_encoding_gpu"}


def pytest_configure(config):
    config.addinivalue_line("markers", "two_pass_only: a test of the two-pass prefill's schedule")


def pytest_generate_tests(metafunc):
    if metafunc.module.__name__.rsplit(".", 1)[-1] in _BOTH_MODES:
        modes = ["two_pass"] if metafunc.definition.get_closest_marker("two_pass_only") else ["two_pass", "fused", "prepared"]
        metafunc.fixturenames.append("prefill_mode")
        metafunc.parametrize("prefill_mode", modes, indirect=True)


@pytest.fixture
def prefill_mode(request):
    with G._prefill_mode(request.param):
        yield request.param
