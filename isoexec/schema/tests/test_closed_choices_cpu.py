"""Foreign config values refuse at construction."""

import pytest

from isoexec.schema import Mlp, Mtp


@pytest.mark.parametrize("field,value", [("activation", "gelu"), ("routing", "sigmoid_topk")])
def test_unsupported_mlp_choice_refuses_at_construction(field, value):
    with pytest.raises(ValueError, match=field):
        Mlp(intermediate_size=64, **{field: value})


def test_unsupported_mtp_projection_and_removed_chaining_refuse():
    with pytest.raises(ValueError, match="proj"):
        Mtp(depth=1, proj="separate_e_h")
    with pytest.raises(TypeError, match="chaining"):
        Mtp(depth=1, chaining="parallel")
