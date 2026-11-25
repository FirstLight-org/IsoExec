"""L0 (CPU): the ENGINE's NCCL channel policy -- when the floor comes off, and at what ceiling.

M7 V10 made the neutralizer default ON *where ``Placement`` owns the row-parallel reduce*, which is
the condition design section 3.1 states and which this module's ``unpinned`` docstring argues: 11 of
the 12 NCCL primitives in a decode step are AllGathers, the 12th is an order-free bf16 all-reduce,
and the reduce that WOULD be channel-sensitive is pik's leaf tree, which never enters NCCL. So the
default reads ``ISOEXEC_PIK`` and nothing else, and an explicit
``ISOEXEC_ENGINE_NCCL_UNPIN`` still wins in both directions.

Pure environment logic: no CUDA, no communicator, no vLLM.
"""

from __future__ import annotations

import pytest

from isoexec.ops.collectives import nccl_identity as ni


def test_the_default_follows_pik_because_pik_owns_the_reduce():
    assert ni.engine_channel_pin_neutralized({"ISOEXEC_PIK": "1"}) is True
    assert ni.engine_channel_pin_neutralized({"ISOEXEC_PIK": "0"}) is False
    assert ni.engine_channel_pin_neutralized({}) is False


@pytest.mark.parametrize("pik", ["0", "1"])
def test_an_explicit_flag_wins_in_both_directions(pik):
    env = {"ISOEXEC_PIK": pik}
    assert ni.engine_channel_pin_neutralized({**env, "ISOEXEC_ENGINE_NCCL_UNPIN": "1"}) is True
    assert ni.engine_channel_pin_neutralized({**env, "ISOEXEC_ENGINE_NCCL_UNPIN": "0"}) is False


def test_the_default_cap_is_the_one_every_shipped_recipe_sets():
    """It used to be 8, which no deployment has ever run: all five recipes/benches say 16."""
    assert ni.ENGINE_CHANNEL_CAP_DEFAULT == "16"
    assert ni.engine_channel_cap({"ISOEXEC_PIK": "1"}) == "16"
    assert ni.engine_channel_cap({"ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS": "8"}) == "8"
    assert ni.engine_channel_cap({"ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS": ""}) is None


def test_the_requested_identity_is_the_policy_the_worker_installs(monkeypatch):
    env = {"ISOEXEC_PIK": "1"}
    assert ni.requested_engine_identity(env) == ni.ENGINE_PER_COMMUNICATOR
    monkeypatch.setattr(ni, "_DECLARED_ENGINE_CAP", None)
    ni.declare_engine_channels(int(ni.engine_channel_cap(env)))
    identity, constants = ni.effective_identity({})
    assert identity == ni.ENGINE_PER_COMMUNICATOR
    assert constants == {**ni.UNPINNED_CONSTANTS, ni.MAX_CTAS: "16"}


def test_the_cap_still_selects_the_eight_channel_identity_when_asked_for():
    env = {"ISOEXEC_PIK": "1", "ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS": "8"}
    assert ni.requested_engine_identity(env) == ni.ENGINE_PER_COMMUNICATOR


def test_pik_off_leaves_vllms_pin_alone():
    """NCCL owns the reduce then, and its channel count is numerics rather than a knob."""
    assert ni.requested_engine_identity({"ISOEXEC_PIK": "0"}) == ni.UNPINNED
    assert ni.engine_channel_pin_neutralized({"ISOEXEC_PIK": "0"}) is False


def test_an_unknown_cap_is_refused_rather_than_named():
    """The identity table is exact: a tuple it does not name is not a deployment we can attest."""
    with pytest.raises(ValueError, match="unsupported engine per-communicator channel ceiling"):
        ni.requested_engine_identity({"ISOEXEC_PIK": "1", "ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS": "5"})
