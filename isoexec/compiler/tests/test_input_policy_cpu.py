"""Token normalization precedes dtype-specialized capture and exact guards."""

from dataclasses import replace

import pytest
import torch

from isoexec.compiler.capture.input_policy import _normalize_token_ids
from isoexec.compiler.tests.test_capture_cpu import example_inputs


def test_default_token_dtype_preserves_aliases_and_rejects_unsupported_inputs():
    ids, seq = example_inputs()
    ids = ids.to(torch.int32)
    seq = replace(seq, input_ids=ids)
    shared = [ids]
    untouched = {"positions": seq.positions}
    values = {"input_ids": ids, "seq": seq, "shared": (shared, shared), "untouched": untouched}
    normalized = _normalize_token_ids(values)
    assert normalized["input_ids"].dtype == torch.int64
    assert normalized["seq"].input_ids is normalized["input_ids"] is normalized["shared"][0][0]
    assert normalized["shared"][0] is normalized["shared"][1]
    assert normalized["untouched"] is untouched and normalized["seq"].positions is seq.positions
    assert seq.input_ids is ids and ids.dtype == torch.int32
    assert _normalize_token_ids(normalized)["seq"] is normalized["seq"]
    for invalid in (ids.float(), ids.bool(), None):
        with pytest.raises(ValueError, match="input_ids.*int32 or int64"):
            _normalize_token_ids({"input_ids": invalid})
    with pytest.raises(ValueError, match="storage aliases"):
        _normalize_token_ids({"input_ids": ids, "view": ids.view_as(ids)})
    hidden = torch.ones((4, 16), dtype=torch.bfloat16)
    assert _normalize_token_ids({"hidden": hidden})["hidden"] is hidden
