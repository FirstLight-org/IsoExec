"""After-FC2 deployment flags stay off by default and reach their consumers."""

import pytest

from isoexec.core import flags


@pytest.mark.parametrize("name", ["ISOEXEC_MOE_AFTER_FC2_OWNER", "ISOEXEC_MOE_AFTER_FC2_GATHER"])
def test_after_fc2_flags_have_explicit_forwarding_and_numerics_census(name):
    flag = flags.get(name)
    assert flag.default == "0"
    assert flag.disposition == flags.DEPLOYMENT
    assert set(flag.forwarded_by) == {flags.TRAIN, flags.ENGINE, flags.VLLM_ENV}
    assert name in flags.NUMERICS_SWITCHES
