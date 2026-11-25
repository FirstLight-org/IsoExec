"""IsoExec's contiguous CP packer equals verl's ``preprocess_thd_engine(cp_layout="contiguous")`` row for row.

verl's packer refuses contiguous CP on the shipped megatron-core (no ``PackedSeqParams.cp_partition_mode``);
the IsoExec engine packs those rows itself. Here verl's packer runs with that metadata check forced
through, so the two can be compared: ids, rolled labels and positions on every CP rank, plus the
global padded boundaries.
"""

from __future__ import annotations

import pytest
import torch

from isoexec.integrations.verl.thd import contiguous_thd


def _verl_thd_util():
    """verl's ``models/mcore/util.py`` loaded on its own: the package ``__init__`` applies a Megatron patch
    that needs a Megatron module this CPU lane does not import; ``util.py`` itself needs only mpu."""
    import importlib.util
    import sys
    from pathlib import Path

    import verl

    name = "_isoexec_test_verl_mcore_util"
    if name not in sys.modules:
        path = Path(verl.__file__).parent / "models" / "mcore" / "util.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _nested(lengths, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.nested.nested_tensor([torch.randint(1, 1000, (n,), generator=g) for n in lengths],
                                      layout=torch.jagged)


@pytest.mark.parametrize("tp,cp", [(1, 2), (2, 2), (1, 4), (2, 4), (1, 8)])
@pytest.mark.parametrize("lengths", [[37, 5, 120, 64], [9], [300, 1, 2, 17]])
def test_matches_verl_contiguous_packing(monkeypatch, tp, cp, lengths):
    from types import SimpleNamespace

    util = _verl_thd_util()

    monkeypatch.setattr(util, "_PACKED_SEQ_PARAMS_HAS_CP_PARTITION_MODE", True)
    monkeypatch.setattr(util, "_packed_seq_params_supports", lambda name: False)
    monkeypatch.setattr(util, "PackedSeqParams", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(util, "preprocess_for_mindspeed", lambda *a, **k: None, raising=False)
    values = _nested(lengths, seed=tp * 10 + cp)
    for rank in range(cp):
        monkeypatch.setattr(util.mpu, "get_tensor_model_parallel_world_size", lambda: tp)
        monkeypatch.setattr(util.mpu, "get_context_parallel_world_size", lambda: cp)
        monkeypatch.setattr(util.mpu, "get_context_parallel_rank", lambda rank=rank: rank)
        for roll in (False, True):
            want_rows, want_params, want_pos = util.preprocess_thd_engine(
                values, pre_process=True, need_roll=roll, cp_layout="contiguous")
            rows, params, pos = contiguous_thd(values, tp=tp, cp=cp, cp_rank=rank, need_roll=roll)
            assert torch.equal(rows, want_rows), (rank, roll)
            assert torch.equal(pos, want_pos), (rank, roll)
            assert torch.equal(params.cu_seqlens_q_padded, want_params.cu_seqlens_q_padded)
            assert params.max_seqlen_q == want_params.max_seqlen_q
            assert not hasattr(params, "cp_partition_mode") or params.cp_partition_mode is None
