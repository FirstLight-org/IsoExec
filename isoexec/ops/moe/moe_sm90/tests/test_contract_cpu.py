"""The standalone provider declares its own arithmetic and never installs itself."""

import pytest
import torch

from isoexec.core.registry import Registry
from isoexec.ops.moe.moe_sm90 import fc1, make_plan, route
from isoexec.ops.moe.moe_sm90.spec import IMPLS

#: The table ops of the moe family, whose declarations this package owns. They are the WHOLE of what
#: it declares: no op id here names an architecture. An ``moe_sm90.*`` id would restate a kernel a
#: family op id already declares, and no host binds one, so none would ever reach a digest.
FAMILY = {"moe.route_score", "moe.route_mechanics", "moe.expert_mlp", "moe.combine", "moe.shared_expert"}


def test_standalone_declarations():
    reg = Registry.from_impls(IMPLS)
    assert set(reg.ops) == FAMILY
    # Expert GEMMs are covered by moe.expert_mlp with its leaf-plan, dtype and activation inputs.
    assert not {op for op in reg.ops if op.startswith("moe_sm90.")}
    # The family's impls carry the twin classes check 2 sees the host asymmetry through, so they are
    # NOT required to be classes of one.
    for name in FAMILY:
        for impl in reg.ops[name].impls.values():
            assert impl.supported_archs == frozenset({"sm90"})


def test_cpu_inputs_are_rejected_before_loading_extension():
    with pytest.raises(ValueError, match="CUDA"):
        make_plan(torch.tensor([0, 1], dtype=torch.int32), 1)
    with pytest.raises(ValueError, match="CUDA"):
        route(torch.zeros((1, 1), dtype=torch.int32), 1)
    with pytest.raises(ValueError, match="CUDA"):
        fc1(torch.zeros((1, 8)), torch.zeros((1, 8, 8)), None)
