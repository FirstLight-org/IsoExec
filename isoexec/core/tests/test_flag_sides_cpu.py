"""Check that every flag declaration names its reader sides from the documented vocabulary."""

from __future__ import annotations

import dataclasses

import pytest

from isoexec.core import flags as F


def test_no_declaration_has_malformed_sides():
    assert F.malformed_sides() == []


def test_the_vocabulary_is_enforced_at_import():
    """A hard raise, not a warning: a token outside the vocabulary matches no reader."""
    good = F.Flag("X", "0", ("both",), "s", F.FUNCTION, (F.TRAIN, F.ENGINE))
    bad = F.Flag("X", "0", ("train",), "s", F.FUNCTION, (F.TRAIN,))
    assert not good.has_malformed_sides and bad.has_malformed_sides
    original = list(F.FLAGS)
    try:
        F.FLAGS.append(bad)
        with pytest.raises(ValueError, match="Flag.sides must be a subset"):
            F._validate_sides()
    finally:
        F.FLAGS[:] = original
    F._validate_sides()


def test_flag_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        F.Flag("X", "0", ("both",), "s", F.FUNCTION).name = "other"
